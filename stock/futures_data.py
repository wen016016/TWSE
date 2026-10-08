"""台指期近月「近全」K 線 (夜盤 15:00~05:00 + 日盤 08:45~13:45 連續)

歷史：期交所「前 30 個交易日期貨每筆成交」壓縮檔 (每天約 1.4MB，含前一晚夜盤)，組成 5分K 後快取在本機
盤中：期交所即時行情 getChartData1M (目前日盤 -F / 最近一次夜盤 -M 的 1分K)，合成 5分K
交易日：一個交易日 = 前一個營業日 15:00 開始的夜盤 + 當天日盤 (和期交所歸屬一致)
"""
import datetime as dt
import io
import time
import zipfile

import numpy as np
import pandas as pd
import requests

from . import data
from .config import DATA_DIR

TZ = "Asia/Taipei"
FUT_DIR = DATA_DIR / "futures"
FUT_DIR.mkdir(exist_ok=True)
TAIFEX_DAILY = "https://www.taifex.com.tw/file/taifex/Dailydownload/DailydownloadCSV/Daily_{y}_{m:02d}_{d:02d}.zip"
CHART_URL = "https://mis.taifex.com.tw/futures/api/getChartData1M"
_session = requests.Session()
_session.headers["User-Agent"] = "Mozilla/5.0"

NIGHT_SLOTS = 168   # 15:00 ~ 04:55
DAY_SLOTS = 60      # 08:45 ~ 13:40
SLOTS = NIGHT_SLOTS + DAY_SLOTS


def _now():
    return dt.datetime.now(dt.timezone(dt.timedelta(hours=8)))


def slot_of(ts: pd.Timestamp) -> int | None:
    """5分K 起始時間 → 交易日內的格號 (0~167 夜盤，168~227 日盤)；不在交易時間回 None"""
    m = ts.hour * 60 + ts.minute
    if m >= 900:                      # 15:00 以後
        return (m - 900) // 5
    if m < 300:                       # 00:00 ~ 04:59
        return (m + 1440 - 900) // 5
    if 525 <= m < 825:                # 08:45 ~ 13:44
        return NIGHT_SLOTS + (m - 525) // 5
    return None


def trade_day_of(ts: pd.Timestamp) -> dt.date:
    """這根 K 棒屬於哪個交易日：15:00 以後 → 下一個營業日；凌晨 → 當天 (遇週末往後推)"""
    d = ts.date()
    m = ts.hour * 60 + ts.minute
    if m >= 900:
        d = d + dt.timedelta(days=1)
    while d.weekday() >= 5:
        d = d + dt.timedelta(days=1)
    return d


# ----------------------------------------------------------------- 歷史 (期交所逐筆成交)
def _ticks_to_5m(txt: str) -> pd.DataFrame | None:
    rows = []
    for line in txt.splitlines()[1:]:
        p = line.split(",")
        if len(p) < 6 or p[1].strip() != "TX":
            continue
        mon = p[2].strip()
        if "/" in mon or "W" in mon:  # 排除價差單、週選
            continue
        rows.append((p[0].strip(), p[3].strip().zfill(6), mon, float(p[4]), float(p[5]) / 2))
    if not rows:
        return None
    df = pd.DataFrame(rows, columns=["date", "time", "mon", "price", "qty"])
    near = min(df["mon"].unique())                     # 近月
    df = df[df["mon"] == near]
    df.index = pd.to_datetime(df["date"] + df["time"], format="%Y%m%d%H%M%S").dt.tz_localize(TZ)
    df = df.sort_index()
    bars = df["price"].resample("5min", label="left", closed="left").ohlc()
    bars["volume"] = df["qty"].resample("5min", label="left", closed="left").sum()
    bars = bars.dropna(subset=["open"])
    bars = bars[[slot_of(t) is not None for t in bars.index]]
    bars["contract"] = near
    return bars


def history_day(day: dt.date) -> pd.DataFrame | None:
    """某交易日 (夜盤 + 日盤) 的近月 5分K；先找本機快取，沒有才下載期交所檔案"""
    f = FUT_DIR / f"TX_{day:%Y%m%d}.csv"
    if f.exists():
        df = pd.read_csv(f, index_col=0, parse_dates=True)
        df.index = pd.to_datetime(df.index, utc=True).tz_convert(TZ)
        return df if len(df) else None
    url = TAIFEX_DAILY.format(y=day.year, m=day.month, d=day.day)
    try:
        r = _session.get(url, timeout=60)
    except Exception:  # noqa: BLE001
        return None
    if r.status_code != 200 or r.content[:2] != b"PK":
        return None
    z = zipfile.ZipFile(io.BytesIO(r.content))
    bars = _ticks_to_5m(z.read(z.namelist()[0]).decode("big5", errors="ignore"))
    if bars is None:
        return None
    bars.to_csv(f)
    time.sleep(0.3)
    return bars


def history(days: list[dt.date]) -> dict:
    out = {}
    for d in days:
        b = history_day(d)
        if b is not None and len(b) > 50:
            out[d] = b
    return out


# ----------------------------------------------------------------- 盤中 (期交所即時 1分K)
def _near_base() -> str | None:
    """目前近月代號 (例 TXFJ6)，從即時行情取得"""
    from .realtime import futures
    f = futures()
    for k in ("day", "night"):
        x = f.get(k)
        if x and x.get("symbol"):
            return x["symbol"].rsplit("-", 1)[0]
    return None


def _chart_1m(symbol: str, base_date: dt.date, night: bool) -> pd.DataFrame | None:
    try:
        j = _session.post(CHART_URL, json={"SymbolID": symbol}, timeout=10).json()
    except Exception:  # noqa: BLE001
        return None
    ticks = (j.get("RtData") or {}).get("Ticks") or []
    rows = []
    for t, o, h, l, c, v in ticks:
        hh, mm = int(t[:2]), int(t[2:4])
        d = base_date
        if night and hh < 15:      # 夜盤過午夜 → 隔天
            d = base_date + dt.timedelta(days=1)
        # 期交所偶爾用 "0460" 表示 05:00 → 用分鐘數相加，不直接塞進時間欄位
        end = pd.Timestamp(dt.datetime(d.year, d.month, d.day), tz=TZ) + pd.Timedelta(minutes=hh * 60 + mm)
        rows.append((end - pd.Timedelta(minutes=1), float(o), float(h), float(l), float(c), float(v)))
    if not rows:
        return None
    df = pd.DataFrame(rows, columns=["t", "open", "high", "low", "close", "volume"]).set_index("t")
    return df[df["close"] > 0]


def live_5m() -> pd.DataFrame | None:
    """最近一次夜盤 + 目前日盤 (若有) 的 5分K"""
    base = _near_base()
    if not base:
        return None
    now = _now()
    hm = now.hour * 100 + now.minute
    night_start = now.date() if hm >= 1500 else now.date() - dt.timedelta(days=1)
    while night_start.weekday() >= 5:
        night_start -= dt.timedelta(days=1)
    parts = [p for p in (_chart_1m(base + "-M", night_start, True),
                         _chart_1m(base + "-F", now.date(), False) if 845 <= hm or hm < 500 else None) if p is not None]
    if not parts:
        return None
    m1 = pd.concat(parts).sort_index()
    m1 = m1[~m1.index.duplicated(keep="last")]
    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    b = m1.resample("5min", label="left", closed="left").agg(agg).dropna(subset=["open"])
    b = b[[slot_of(t) is not None for t in b.index]]
    b["contract"] = base
    return b


def near_full_5m(n_days=30) -> pd.DataFrame:
    """台指期近全 5分K：歷史 (期交所檔案) + 盤中即時"""
    def load():
        idx = data.get_kline(data.INDEX, "1d", live=False).index
        tdays = [t.date() for t in idx[-n_days:]]
        today = _now().date()
        hist = history([d for d in tdays if d < today])
        frames = list(hist.values())
        lv = live_5m()
        if lv is not None:
            have = set(hist)
            lv = lv[[trade_day_of(t) not in have for t in lv.index]]
            frames.append(lv)
        df = pd.concat(frames).sort_index()
        df = df[~df.index.duplicated(keep="last")]
        df["tday"] = [trade_day_of(t) for t in df.index]
        df["slot"] = [slot_of(t) for t in df.index]
        return df
    return data.cached("tx_near_full", 20, load)


def day_paths(df: pd.DataFrame) -> dict:
    """{交易日: (每格收盤 [228], 前一交易日日盤收盤)}，缺的格往前補"""
    out, prev_close = {}, None
    for d, g in df.groupby("tday"):
        arr = np.full(SLOTS, np.nan)
        for s_, c_ in zip(g["slot"].values, g["close"].values):
            if s_ is not None and 0 <= s_ < SLOTS:
                arr[int(s_)] = c_
        first = np.argmax(~np.isnan(arr)) if (~np.isnan(arr)).any() else None
        if first is None:
            continue
        arr = pd.Series(arr).ffill().values
        if prev_close is not None:
            out[d] = (arr, prev_close)
        day_part = g[g["slot"] >= NIGHT_SLOTS]
        if len(day_part):
            prev_close = float(day_part["close"].iloc[-1])
    return out
