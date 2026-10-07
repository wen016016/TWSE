"""行情與籌碼資料。

- K 線：Yahoo Finance (yfinance)，台股免費盤中資料約有延遲
- 三大法人、融資融券：FinMind 免費 API
- 大戶持股 (集保股權分散表，每週更新)：集保結算所開放資料
"""
import datetime as dt
import threading
import time

import pandas as pd
import requests
import yfinance as yf

from .config import DATA_DIR, FINMIND_TOKEN

TZ = "Asia/Taipei"
_cache: dict = {}
_lock = threading.Lock()


def cached(key, ttl, fn):
    now = time.time()
    with _lock:
        hit = _cache.get(key)
        if hit and now - hit[0] < ttl:
            return hit[1]
    val = fn()
    with _lock:
        _cache[key] = (now, val)
    return val


# ---------------------------------------------------------------- FinMind
FINMIND_URL = "https://api.finmindtrade.com/api/v4/data"


def finmind(dataset, **params) -> pd.DataFrame:
    p = {"dataset": dataset, **params}
    if FINMIND_TOKEN:
        p["token"] = FINMIND_TOKEN
    r = requests.get(FINMIND_URL, params=p, timeout=30)
    j = r.json()
    if j.get("status") != 200:
        raise RuntimeError(f"FinMind {dataset}: {j.get('msg')}")
    return pd.DataFrame(j.get("data", []))


def stock_info() -> pd.DataFrame:
    path = DATA_DIR / "stock_info.csv"

    def load():
        if path.exists() and time.time() - path.stat().st_mtime < 86400:
            return pd.read_csv(path, dtype=str)
        df = finmind("TaiwanStockInfo")
        df = df[df["type"].isin(["twse", "tpex"])]
        # 一檔股票可能有多個分類：優先保留具體產業，籠統的「電子工業」放最後
        df = df.assign(_g=(df["industry_category"] == "電子工業").astype(int)).sort_values("_g").drop_duplicates("stock_id")
        df = df[["stock_id", "stock_name", "type", "industry_category"]]
        df.to_csv(path, index=False)
        return df.astype(str)

    try:
        return cached("info", 86400, load)
    except Exception:
        if path.exists():
            return pd.read_csv(path, dtype=str)
        return pd.DataFrame(columns=["stock_id", "stock_name", "type", "industry_category"])


def resolve(code: str) -> dict:
    """'2330' / '台積電' / '2330.TW' -> {code, name, yf, market}"""
    code = code.strip().upper().replace(".TWO", "").replace(".TW", "")
    info = stock_info()
    row = info[info["stock_id"] == code]
    if row.empty and not code.isdigit():
        row = info[info["stock_name"] == code]
        if row.empty:
            row = info[info["stock_name"].str.contains(code, na=False)]
    if row.empty:
        if not code[:4].isdigit():
            raise ValueError(f"找不到股票：{code}")
        return {"code": code, "name": code, "yf": f"{code}.TW", "market": "上市"}
    r = row.iloc[0]
    otc = r["type"] == "tpex"
    return {"code": r["stock_id"], "name": r["stock_name"], "yf": f"{r['stock_id']}.{'TWO' if otc else 'TW'}",
            "market": "上櫃" if otc else "上市", "industry": r.get("industry_category", "")}


# ---------------------------------------------------------------- K 線
# timeframe -> (yfinance period, 快取秒數)
TF = {"1m": ("5d", 30), "5m": ("30d", 60), "1d": ("2y", 300), "1wk": ("5y", 1800)}
INDEX = "^TWII"


def _normalize(df: pd.DataFrame, is_index: bool) -> pd.DataFrame:
    df = df.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]].dropna()
    df[["open", "high", "low", "close"]] = df[["open", "high", "low", "close"]].round(2)
    df.index = df.index.tz_convert(TZ) if df.index.tz is not None else df.index.tz_localize(TZ)
    if not is_index:
        df["volume"] = df["volume"] / 1000  # 股 -> 張
    return df


def get_kline(code: str, tf: str, live=True) -> pd.DataFrame:
    """K 線 (Yahoo) + 用證交所即時報價補上 / 更新最新一根 (Yahoo 指數日K常缺當天、盤中有延遲)"""
    is_index = code.startswith("^")
    sym = code if is_index else resolve(code)["yf"]
    period, ttl = TF[tf]

    def load():
        df = yf.Ticker(sym).history(period=period, interval=tf, auto_adjust=False)
        if df.empty:
            raise ValueError(f"查無 {code} 的 {tf} K 線資料")
        return _normalize(df, is_index)

    df = cached(f"k:{sym}:{tf}", ttl, load)
    return _live(df, code, tf, is_index) if live else df


def _live(df, code, tf, is_index):
    try:
        from . import realtime
        q = realtime.index_quote(code) if is_index else realtime.quote(code)
    except Exception:  # noqa: BLE001
        return df
    if not q or not q.get("price") or not q.get("date"):
        return df
    px = float(q["price"])
    day = pd.Timestamp(q["date"]).date()
    if day < df.index[-1].date():
        return df
    df = df.copy()
    last = df.index[-1]
    hi = max(px, q.get("high") or px)
    lo = min(px, q.get("low") or px)
    if tf in ("1d", "1wk"):
        same = last.date() == day if tf == "1d" else last.isocalendar()[:2] == pd.Timestamp(day).isocalendar()[:2]
        if same:
            df.loc[last, "close"] = px
            df.loc[last, "high"] = max(df.loc[last, "high"], hi if tf == "1d" else px)
            df.loc[last, "low"] = min(df.loc[last, "low"], lo if tf == "1d" else px)
            if tf == "1d" and not is_index and q.get("volume"):
                df.loc[last, "volume"] = max(df.loc[last, "volume"], q["volume"])
        else:
            label = pd.Timestamp(day, tz=TZ)
            if tf == "1wk":
                label = label - pd.Timedelta(days=label.weekday())
            df.loc[label] = [q.get("open") or px, hi, lo, px, 0 if is_index else (q.get("volume") or 0)]
    else:
        mins = 1 if tf == "1m" else 5
        t = q.get("time") or "13:30:00"
        bucket = pd.Timestamp(f"{q['date']} {t}", tz=TZ).floor(f"{mins}min")
        if bucket == last:
            df.loc[last, "close"] = px
            df.loc[last, "high"] = max(df.loc[last, "high"], px)
            df.loc[last, "low"] = min(df.loc[last, "low"], px)
        elif bucket > last and bucket.date() == day:
            vol = 0
            if not is_index and q.get("volume"):
                vol = max(q["volume"] - df.loc[df.index.date == day, "volume"].sum(), 0)
            df.loc[bucket] = [px, px, px, px, vol]
    return df


def download_daily(codes: list[str], period="1y") -> dict[str, pd.DataFrame]:
    """批次下載日 K (選股掃描用)"""
    syms = {resolve(c)["yf"]: c for c in codes}
    raw = yf.download(list(syms), period=period, interval="1d", group_by="ticker",
                      auto_adjust=False, threads=True, progress=False)
    out = {}
    for s, c in syms.items():
        try:
            df = _normalize(raw[s].copy(), False)
            if len(df) > 60:
                out[c] = df
        except Exception:
            pass
    return out


# ---------------------------------------------------------------- 籌碼
def _start(days):
    return (dt.date.today() - dt.timedelta(days=int(days * 1.6))).isoformat()


def institutional(code: str, days=40) -> pd.DataFrame:
    """三大法人每日買賣超 (張)，欄位：外資/投信/自營商/合計"""
    code = resolve(code)["code"]

    def load():
        df = finmind("TaiwanStockInstitutionalInvestorsBuySell", data_id=code, start_date=_start(days))
        if df.empty:
            return pd.DataFrame(columns=["外資", "投信", "自營商", "合計"])
        df["net"] = (df["buy"] - df["sell"]) / 1000
        who = {"Foreign_Investor": "外資", "Foreign_Dealer_Self": "外資", "Investment_Trust": "投信",
               "Dealer_self": "自營商", "Dealer_Hedging": "自營商"}
        df["who"] = df["name"].map(who)
        p = df.pivot_table(index="date", columns="who", values="net", aggfunc="sum").fillna(0)
        for c in ["外資", "投信", "自營商"]:
            if c not in p:
                p[c] = 0.0
        p["合計"] = p[["外資", "投信", "自營商"]].sum(axis=1)
        p.index = pd.to_datetime(p.index)
        return p.sort_index()

    return cached(f"inst:{code}", 3600, load)


def market_institutional(days=15) -> pd.DataFrame:
    """大盤三大法人買賣超 (億元)"""
    def load():
        df = finmind("TaiwanStockTotalInstitutionalInvestors", start_date=_start(days))
        df["net"] = (df["buy"] - df["sell"]) / 1e8
        who = {"Foreign_Investor": "外資", "Foreign_Dealer_Self": "外資", "Investment_Trust": "投信",
               "Dealer_self": "自營商", "Dealer_Hedging": "自營商"}
        df["who"] = df["name"].map(who)
        p = df.dropna(subset=["who"]).pivot_table(index="date", columns="who", values="net", aggfunc="sum").fillna(0)
        p.index = pd.to_datetime(p.index)
        return p.sort_index()

    return cached("minst", 3600, load)


def margin(code: str, days=30) -> pd.DataFrame:
    """融資融券餘額 (張)"""
    code = resolve(code)["code"]

    def load():
        df = finmind("TaiwanStockMarginPurchaseShortSale", data_id=code, start_date=_start(days))
        if df.empty:
            return pd.DataFrame(columns=["融資餘額", "融券餘額"])
        df = df.set_index(pd.to_datetime(df["date"])).sort_index()
        return pd.DataFrame({"融資餘額": df["MarginPurchaseTodayBalance"], "融券餘額": df["ShortSaleTodayBalance"]})

    return cached(f"margin:{code}", 3600, load)


# ---------------------------------------------------------------- 集保大戶
TDCC_URL = "https://opendata.tdcc.com.tw/getOD.ashx?id=1-5"


def _tdcc_files():
    files = sorted(DATA_DIR.glob("tdcc_*.csv"))
    if not files or time.time() - files[-1].stat().st_mtime > 6 * 3600:
        r = requests.get(TDCC_URL, headers={"User-Agent": "Mozilla/5.0"}, timeout=90)
        text = r.content.decode("utf-8-sig")
        date = text.splitlines()[1].split(",")[0].strip()
        (DATA_DIR / f"tdcc_{date}.csv").write_text(text, encoding="utf-8")
        files = sorted(DATA_DIR.glob("tdcc_*.csv"))
    return files


def _tdcc_table(path) -> pd.DataFrame:
    def load():
        df = pd.read_csv(path, dtype={"證券代號": str})
        df["證券代號"] = df["證券代號"].str.strip()
        return df
    return cached(f"tdcc:{path.name}", 86400, load)


def big_holders(code: str) -> dict | None:
    """400 張以上大戶 / 千張大戶 / 50 張以下散戶 持股比例，並與上一份資料比較"""
    code = resolve(code)["code"]
    files = cached("tdcc_files", 3600, _tdcc_files)

    def ratios(path):
        df = _tdcc_table(path)
        g = df[df["證券代號"] == code].set_index("持股分級")["占集保庫存數比例%"]
        if g.empty:
            return None
        return {"date": path.stem.split("_")[1],
                "big400": float(g.reindex([12, 13, 14, 15]).sum()),
                "big1000": float(g.get(15, 0)),
                "retail": float(g.reindex(range(1, 9)).sum())}

    cur = ratios(files[-1])
    if not cur:
        return None
    if len(files) > 1:
        prev = ratios(files[-2])
        if prev:
            cur["prev_date"] = prev["date"]
            for k in ("big400", "big1000", "retail"):
                cur[f"chg_{k}"] = cur[k] - prev[k]
    return cur
