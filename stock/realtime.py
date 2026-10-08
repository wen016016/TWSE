"""即時報價 (證交所 MIS / 期交所 MIS，免費) + 成交明細收集 + 即時推送

⚠ 這兩個免費 API 有頻率限制，問太快會被暫時封鎖 IP。所以：
  - 全系統所有證交所請求共用一個限速器 (MIS_GAP 秒一次)，指數 + 個股合併成一次請求 (最多 100 檔)
  - 期交所只抓目前正在交易的盤別，另一盤每 60 秒補一次
  - 被擋 (連線被斷 / 回傳不是 JSON) → 自動退讓 60 秒，期間用最後一次的資料
  - 報價結果放共用快取，K 線補最新價、停損監控、網頁推送、成交明細都讀同一份，不重複請求

成交明細 (大戶/散戶)：
1. FinMind TaiwanStockPriceTick (需付費會員 token) → 完整逐筆
2. 否則：每次刷新報價時比對累計量，自己累積成交明細
   - tv = 該次快照的最新一筆成交張數，v = 累計成交量
   - 兩次快照之間累計量增加超過 tv → 中間還有其他筆成交 (筆數/大小未知) → 標記「合併量」，不列入大戶/散戶分類
   - 成交價 >= 前一刻委賣價 = 外盤 (主動買)；<= 前一刻委買價 = 內盤 (主動賣)
"""
import csv
import datetime as dt
import threading
import time

import pandas as pd
import requests

from . import data
from .config import DATA_DIR, FINMIND_TOKEN

MIS_URL = "https://mis.twse.com.tw/stock/api/getStockInfo.jsp"
TAIFEX_URL = "https://mis.taifex.com.tw/futures/api/getQuoteList"
MIS_GAP = 1.8        # 證交所：兩次請求最少間隔 (秒) ≈ 每 5 秒 3 次以內
TAIFEX_GAP = 2.0     # 期交所
BATCH = 100          # 一次請求最多幾檔
BLOCK_BACKOFF = 60   # 被擋後退讓秒數

TICK_DIR = DATA_DIR / "ticks"
TICK_DIR.mkdir(exist_ok=True)
_session = requests.Session()
_session.headers["User-Agent"] = "Mozilla/5.0"


def _now():
    return dt.datetime.now(dt.timezone(dt.timedelta(hours=8)))


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _arr(s):
    return [float(v) for v in (s or "").split("_") if v not in ("", "-")]


# ----------------------------------------------------------------- 限速
class Limiter:
    def __init__(self, name, gap):
        self.name, self.gap = name, gap
        self.lock = threading.Lock()
        self.last = 0.0
        self.blocked_until = 0.0
        self.errors = 0
        self.requests = 0

    def blocked(self):
        return time.time() < self.blocked_until

    def wait(self):
        if self.blocked():
            raise RuntimeError(f"{self.name} 暫時限制連線，{int(self.blocked_until - time.time())} 秒後重試")
        with self.lock:
            d = self.last + self.gap - time.time()
            if d > 0:
                time.sleep(d)
            self.last = time.time()
            self.requests += 1

    def fail(self):
        self.errors += 1
        if self.errors >= 2:  # 連續失敗 → 視為被擋，退讓
            self.blocked_until = time.time() + BLOCK_BACKOFF
            self.errors = 0

    def ok(self):
        self.errors = 0

    def status(self):
        return {"blocked": self.blocked(), "retry_in": max(0, int(self.blocked_until - time.time())),
                "requests": self.requests}


MIS = Limiter("證交所即時行情", MIS_GAP)
FUT = Limiter("期交所即時行情", TAIFEX_GAP)


def health():
    return {"mis": MIS.status(), "taifex": FUT.status()}


# ----------------------------------------------------------------- 證交所報價 (個股 + 指數共用)
INDEX_EX = {"^TWII": ("tse_t00.tw", "t00", "加權指數"), "^TWOII": ("otc_o00.tw", "o00", "櫃買指數")}
_IDX_BY_C = {v[1]: k for k, v in INDEX_EX.items()}
_cache: dict[str, tuple[float, dict]] = {}   # code / ^TWII -> (時間, quote)
_last_price: dict = {}
_ingest_hooks = []                            # 每次抓到新報價時呼叫 (成交明細收集)


def _exch(code):
    if code in INDEX_EX:
        return INDEX_EX[code][0]
    info = data.resolve(code)
    return f"{'otc' if info['market'] == '上櫃' else 'tse'}_{info['code']}.tw"


def _parse(m):
    c = m.get("c")
    bid, ask = _arr(m.get("b")), _arr(m.get("a"))
    trade = m.get("trade") if isinstance(m.get("trade"), dict) else {}
    # 證交所新格式：z 常是 "-"，最新一筆成交改放在 trade {t, v, z}
    price = _f(m.get("z")) or _f(trade.get("z"))
    key = _IDX_BY_C.get(c, c)
    if price is None:  # 真的還沒成交 (盤前 / 整天無量) → 上次成交價，再不行委買賣中價 / 昨收
        mid = round((bid[0] + ask[0]) / 2, 2) if bid and ask else (bid[0] if bid else (ask[0] if ask else None))
        price = _last_price.get(key) or mid or _f(m.get("y"))
    _last_price[key] = price
    prev = _f(m.get("y"))
    last_vol = _f(m.get("tv")) or _f(trade.get("v")) or 0
    q = {"code": key, "name": m.get("n"), "price": price, "last_vol": last_vol,
         "volume": _f(m.get("v")) or 0, "bid": bid, "ask": ask,
         "bid_vol": _arr(m.get("g")), "ask_vol": _arr(m.get("f")),
         "open": _f(m.get("o")), "high": _f(m.get("h")), "low": _f(m.get("l")),
         "prev_close": prev, "limit_up": _f(m.get("u")), "limit_down": _f(m.get("w")),
         "time": trade.get("t") or m.get("t"), "date": m.get("d")}
    if key in INDEX_EX:
        q.update(name=INDEX_EX[key][2], volume=0, limit_up=None, limit_down=None, bid=[], ask=[],
                 chg=round(price - prev, 2) if price and prev else None,
                 chg_pct=round((price / prev - 1) * 100, 2) if price and prev else None)
    return key, q


def _fetch(keys: list[str]):
    """透過限速器向證交所抓一批 (個股代號或 ^TWII/^TWOII)，結果寫進快取"""
    for i in range(0, len(keys), BATCH):
        chunk = keys[i:i + BATCH]
        MIS.wait()
        try:
            r = _session.get(MIS_URL, params={"ex_ch": "|".join(_exch(k) for k in chunk), "json": "1", "delay": "0",
                                              "_": int(time.time() * 1000)}, timeout=8)
            arr = r.json().get("msgArray", [])
            MIS.ok()
        except Exception as e:  # noqa: BLE001
            MIS.fail()
            raise RuntimeError(f"證交所即時行情連線失敗：{e}") from e
        now = time.time()
        got = {}
        for m in arr:
            k, q = _parse(m)
            _cache[k] = (now, q)
            got[k] = q
        for hook in _ingest_hooks:
            try:
                hook(got)
            except Exception:  # noqa: BLE001
                pass


def quotes(codes: list[str], max_age=3.0) -> dict:
    """{code: quote}；max_age 秒內的快取直接用，其餘一次合併抓"""
    keys = []
    for c in dict.fromkeys(codes):
        if c in INDEX_EX:
            keys.append(c)
        else:
            try:
                keys.append(data.resolve(c)["code"])
            except Exception:  # noqa: BLE001
                pass
    now = time.time()
    stale = [k for k in keys if k not in _cache or now - _cache[k][0] > max_age]
    if stale:
        try:
            _fetch(stale)
        except RuntimeError:
            # 證交所連不上 → 沒有快取的改用 Yahoo (延遲數分鐘)，最多 30 檔避免太慢
            missing = [k for k in keys if k not in _cache][:30]
            for k in missing:
                q = _yahoo_quote(k)
                if q:
                    _cache[k] = (time.time() - max_age + 30, q)  # 30 秒後再試證交所
            if not any(k in _cache for k in keys):
                raise
    return {k: _cache[k][1] for k in keys if k in _cache}


WWW = Limiter("證交所網站", 3.0)


def _twse_index_quote():
    """加權指數備援：證交所網站「每 5 秒指數」(和即時行情不同主機)，09:00 那筆 = 昨收"""
    WWW.wait()
    try:
        r = _session.get("https://www.twse.com.tw/rwd/zh/TAIEX/MI_5MINS_INDEX",
                         params={"date": _now().strftime("%Y%m%d"), "response": "json"}, timeout=10)
        rows = r.json().get("data") or []
        WWW.ok()
    except Exception:  # noqa: BLE001
        WWW.fail()
        return None
    if not rows:
        return None
    vals = [float(x[1].replace(",", "")) for x in rows]
    price, prev = vals[-1], vals[0]
    return {"code": "^TWII", "name": "加權指數", "price": price, "prev_close": prev, "last_vol": 0, "volume": 0,
            "bid": [], "ask": [], "bid_vol": [], "ask_vol": [], "open": vals[1] if len(vals) > 1 else price,
            "high": max(vals[1:] or vals), "low": min(vals[1:] or vals), "limit_up": None, "limit_down": None,
            "chg": round(price - prev, 2), "chg_pct": round((price / prev - 1) * 100, 2),
            "time": rows[-1][0] + " (證交所網站)", "date": _now().strftime("%Y%m%d"), "source": "twse_www"}


def _yahoo_quote(key):
    """備援：證交所即時行情被擋時，加權用證交所網站，其他用 Yahoo 1分K 最後一根 (會延遲)"""
    if key == "^TWII":
        q = _twse_index_quote()
        if q:
            return q
    try:
        m1 = data.get_kline(key, "1m", live=False)
        d1 = data.get_kline(key, "1d", live=False)
    except Exception:  # noqa: BLE001
        return None
    last = m1.iloc[-1]
    today = m1.index[-1].date()
    day = m1[m1.index.date == today]
    prev = d1[d1.index.date < today]["close"]
    prev_close = float(prev.iloc[-1]) if len(prev) else None
    price = float(last["close"])
    name = INDEX_EX[key][2] if key in INDEX_EX else data.resolve(key)["name"]
    q = {"code": key, "name": name, "price": price, "last_vol": 0, "volume": float(day["volume"].sum()),
         "bid": [], "ask": [], "bid_vol": [], "ask_vol": [], "open": float(day["open"].iloc[0]),
         "high": float(day["high"].max()), "low": float(day["low"].min()), "prev_close": prev_close,
         "limit_up": round(prev_close * 1.1, 2) if prev_close and key not in INDEX_EX else None,
         "limit_down": round(prev_close * 0.9, 2) if prev_close and key not in INDEX_EX else None,
         "time": m1.index[-1].strftime("%H:%M:%S") + " (Yahoo延遲)", "date": today.strftime("%Y%m%d"),
         "source": "yahoo"}
    if key in INDEX_EX:
        q["chg"] = round(price - prev_close, 2) if prev_close else None
        q["chg_pct"] = round((price / prev_close - 1) * 100, 2) if prev_close else None
    return q


def quote(code: str, max_age=3.0) -> dict:
    c = data.resolve(code)["code"]
    q = quotes([c], max_age).get(c)
    if not q:
        raise RuntimeError(f"查無 {code} 即時報價")
    return q


def index_quotes(max_age=3.0) -> dict:
    return quotes(list(INDEX_EX), max_age)


def index_quote(sym: str) -> dict | None:
    return index_quotes().get(sym)


# ----------------------------------------------------------------- 台指期
_fut = {"day": None, "night": None, "t": {"day": 0, "night": 0}}


def _fut_one(market_type):
    body = {"MarketType": market_type, "SymbolType": "F", "KindID": "1", "CID": "TXF", "ExpireMonth": "",
            "RowSize": "全部", "PageNo": "", "SortColumn": "", "AscDesc": "A"}
    FUT.wait()
    try:
        rows = _session.post(TAIFEX_URL, json=body, timeout=8).json()["RtData"]["QuoteList"]
        FUT.ok()
    except Exception as e:  # noqa: BLE001
        FUT.fail()
        raise RuntimeError(f"期交所即時行情連線失敗：{e}") from e
    fut = [x for x in rows if x["SymbolID"].startswith("TXF") and x["SymbolID"].endswith(("-F", "-M"))]
    if not fut:
        return None
    x = fut[0]
    t = x.get("CTime") or ""
    return {"symbol": x["SymbolID"], "name": x["DispCName"], "price": _f(x.get("CLastPrice")), "ref": _f(x.get("CRefPrice")),
            "chg": _f(x.get("CDiff")), "chg_pct": _f(x.get("CDiffRate")),
            "high": _f(x.get("CHighPrice")), "low": _f(x.get("CLowPrice")),
            "volume": _f(x.get("CTotalVolume")), "date": x.get("CDate"),
            "time": f"{t[:2]}:{t[2:4]}:{t[4:6]}" if len(t) >= 6 else t}


def futures(max_age=3.0) -> dict:
    """台指期近月：日盤 (08:45~13:45)、夜盤 (15:00~05:00)；只頻繁抓正在交易的那一盤"""
    now = _now()
    hm = now.hour * 100 + now.minute
    session = "night" if (hm >= 1500 or hm < 500) else "day"
    for k, mt, age in (("day", "0", max_age if session == "day" else 60), ("night", "1", max_age if session == "night" else 60)):
        if time.time() - _fut["t"][k] > age and not FUT.blocked():
            try:
                v = _fut_one(mt)
                if v and v.get("price"):
                    _fut[k] = v
                _fut["t"][k] = time.time()
            except RuntimeError:
                pass
    out = {"day": _fut["day"], "night": _fut["night"], "session": session}
    out["current"] = out.get(session) or out.get("day")
    spot = _cache.get("^TWII", (0, None))[1]
    if out["current"] and out["current"].get("price") and spot and spot.get("price"):
        out["basis"] = round(out["current"]["price"] - spot["price"], 2)
    return out


# ----------------------------------------------------------------- 成交明細收集
FIELDS = ["time", "price", "vol", "side", "merged"]


class TickCollector:
    """不自己發請求：掛在報價抓取上 (每次抓到新報價就比對)，盤中收集被查詢過的股票與持倉"""

    def __init__(self):
        self.watch: dict[str, float] = {}       # code -> 最後一次被要求的時間
        self.prev: dict[str, dict] = {}         # code -> 上一個快照
        self.ticks: dict[tuple, list] = {}      # (date, code) -> [tick]
        self.lock = threading.Lock()
        self.last_error = None
        _ingest_hooks.append(self._ingest)

    def add(self, code):
        c = data.resolve(code)["code"]
        with self.lock:
            self.watch[c] = time.time()

    def _active(self):
        from . import broker
        held = {p["code"] for p in broker.open_positions()}
        with self.lock:  # 被查詢過的股票追蹤 6 小時；持倉一直追蹤
            return sorted(held | {c for c, t in self.watch.items() if time.time() - t < 6 * 3600})

    def _ingest(self, got: dict):
        now = _now()
        hm = now.hour * 100 + now.minute
        if now.weekday() >= 5 or not (859 <= hm <= 1331):
            return
        active = set(self._active())
        for code, q in got.items():
            if code not in active:
                continue
            p = self.prev.get(code)
            self.prev[code] = q
            if not p or q["date"] != p["date"] or q["volume"] <= p["volume"]:
                continue
            delta = q["volume"] - p["volume"]
            tv = min(q["last_vol"], delta)
            price = q["price"]
            if p["ask"] and price >= p["ask"][0]:
                side = "B"
            elif p["bid"] and price <= p["bid"][0]:
                side = "S"
            elif q["ask"] and price >= q["ask"][0]:
                side = "B"
            elif q["bid"] and price <= q["bid"][0]:
                side = "S"
            else:
                side = "N"
            rows = [{"time": q["time"], "price": price, "vol": tv, "side": side, "merged": 0}]
            if delta - tv > 0:
                rows.append({"time": q["time"], "price": price, "vol": delta - tv, "side": "U", "merged": 1})
            self._store(q["date"], code, rows)

    def _store(self, date, code, rows):
        key = (date, code)
        with self.lock:
            self.ticks.setdefault(key, self._load(date, code)).extend(rows)
        d = TICK_DIR / date
        d.mkdir(exist_ok=True)
        f = d / f"{code}.csv"
        new = not f.exists()
        with f.open("a", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, FIELDS)
            if new:
                w.writeheader()
            w.writerows(rows)

    def _load(self, date, code):
        f = TICK_DIR / date / f"{code}.csv"
        if not f.exists():
            return []
        return pd.read_csv(f).to_dict("records")

    def get(self, code, date=None) -> pd.DataFrame:
        date = date or _now().strftime("%Y%m%d")
        key = (date, code)
        with self.lock:
            rows = self.ticks.get(key)
            if rows is None:
                rows = self.ticks[key] = self._load(date, code)
            return pd.DataFrame(list(rows), columns=FIELDS)


collector = TickCollector()


def finmind_ticks(code, date: str) -> pd.DataFrame | None:
    """付費會員才有。成功回傳 DataFrame(time, price, vol, side, merged)，失敗回 None"""
    if not FINMIND_TOKEN:
        return None
    try:
        df = data.finmind("TaiwanStockPriceTick", data_id=code, start_date=f"{date[:4]}-{date[4:6]}-{date[6:]}")
    except Exception:  # noqa: BLE001
        return None
    if df.empty:
        return None
    side = df.get("TickType", pd.Series(0, index=df.index)).map({1: "S", 2: "B"}).fillna("N")
    return pd.DataFrame({"time": df["Time"], "price": df["deal_price"], "vol": df["volume"], "side": side, "merged": 0})


def ticks(code: str) -> tuple[pd.DataFrame, str, str]:
    """今日成交明細 + 來源說明 + 日期"""
    c = data.resolve(code)["code"]
    collector.add(c)
    date = _now().strftime("%Y%m%d")
    fm = data.cached(f"fmtick:{c}:{date}", 60, lambda: finmind_ticks(c, date))
    if fm is not None:
        return fm, "FinMind 逐筆成交", date
    df = collector.get(c, date)
    if df.empty:  # 今天還沒資料 (盤前/假日) → 用最近一個有收集到的交易日
        days = sorted(p.parent.name for p in TICK_DIR.glob(f"*/{c}.csv"))
        if days:
            date = days[-1]
            df = collector.get(c, date)
    return df, "系統自行收集 (隨報價刷新比對)", date


# ----------------------------------------------------------------- 即時推送
class Streamer(threading.Thread):
    """盤中每 ~2 秒刷新一次：指數 + 網頁正在看的個股 + 持倉 + 成交明細追蹤的股票 (合併成一次請求)
    台指期每 2 秒；網頁用 SSE 接收，有變動立刻更新。盤後放慢到每 30 秒。

    資料源本身的更新頻率：證交所個股與指數約每 5 秒一筆、台指期約每 1 秒；要逐筆即時需改用券商行情 API
    """

    def __init__(self, interval=2.0):
        super().__init__(daemon=True)
        self.interval = interval
        self.watch: dict[str, float] = {}
        self.snap = {"index": {}, "futures": None, "quotes": {}, "ts": None, "health": health()}
        self.version = 0
        self.lock = threading.Lock()

    def add(self, codes):
        now = time.time()
        with self.lock:
            for c in codes:
                c = (c or "").strip()
                if c and not c.startswith("^"):
                    try:
                        self.watch[data.resolve(c)["code"]] = now
                    except Exception:  # noqa: BLE001
                        pass

    def _codes(self):
        from . import broker
        held = {p["code"] for p in broker.open_positions()}
        with self.lock:
            watched = {c for c, t in self.watch.items() if time.time() - t < 600}
        return sorted(held | watched | set(collector._active()))

    def run(self):
        while True:
            t0 = time.time()
            now = _now()
            hm = now.hour * 100 + now.minute
            stock_hours = now.weekday() < 5 and 830 <= hm <= 1345
            fut_hours = now.weekday() < 5 and (845 <= hm <= 1345 or hm >= 1500) or (now.weekday() < 6 and hm < 500)
            try:
                self._step(stock_hours, fut_hours)
            except Exception:  # noqa: BLE001
                pass
            gap = self.interval if (stock_hours or fut_hours) else 30
            time.sleep(max(0.5, gap - (time.time() - t0)))

    def _step(self, stock_hours, fut_hours):
        codes = self._codes()
        try:
            # 盤中：指數 + 個股合併一次抓；盤後：只偶爾刷新
            quotes(list(INDEX_EX) + codes, max_age=self.interval * 0.8 if stock_hours else 25)
        except RuntimeError:
            pass
        fut = futures(max_age=self.interval * 0.8 if fut_hours else 25)
        idx = {k: _cache[k][1] for k in INDEX_EX if k in _cache}
        qs = {c: {k: _cache[c][1][k] for k in ("code", "name", "price", "prev_close", "volume", "bid", "ask",
                                               "high", "low", "limit_up", "limit_down", "time")}
              for c in codes if c in _cache}
        new = {"index": idx, "futures": fut, "quotes": qs, "ts": _now().strftime("%H:%M:%S"), "health": health()}
        sig = (str(idx), str(fut.get("current")), str(qs), str(new["health"]["mis"]["blocked"]), str(new["health"]["taifex"]["blocked"]))
        if sig != getattr(self, "_sig", None):
            self._sig = sig
            with self.lock:
                self.snap = new
                self.version += 1

    def snapshot(self, codes=()):
        with self.lock:
            s = dict(self.snap)
        want = set()
        for c in codes:
            try:
                want.add(data.resolve(c)["code"])
            except Exception:  # noqa: BLE001
                pass
        from . import broker
        want |= {p["code"] for p in broker.open_positions()}
        s["quotes"] = {c: q for c, q in s.get("quotes", {}).items() if c in want}
        return s


streamer = Streamer()
