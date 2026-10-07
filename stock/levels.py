"""支撐 / 壓力 / 多空關鍵價

判斷方法 (每個價位都會附上依據)：
1. 轉折點：區間內的波段高點、低點 (左右 N 根都比它低/高)，多次在相近價位轉折 = 越強
2. 價位成交量分布 (Volume Profile)：大量換手的價格區 = 很多人的成本 → 支撐/壓力
3. 技術指標價位：均線、布林通道 (動態支撐壓力)
4. 額外價位：昨高/昨低/昨收/VWAP/開盤區間 (當沖)、週線轉折 (中長線)、日線支撐壓力 (當沖)
5. 多方關鍵價：近期「帶量長紅K」的低點，股價守住 = 多方仍掌控
   空方關鍵價：近期「帶量長黑K」的高點，股價站不回去 = 空方仍掌控

週期 / 視角 (PROFILES)：
  1wk      週K
  short    波段短線：日K 近 40 根，5/10/20MA、布林 → 幾天到兩三週的進出場
  long     波段中長線：日K 近 250 根 (約一年) + 週K 轉折，60/120/240MA → 趨勢與大停損
  5m / 1m  當沖：呼叫端只傳「昨天+今天」(5m) 或「今天」(1m) 的資料
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .indicators import round_tick

TF_NAME = {"1wk": "週K", "1d": "日K", "5m": "5分K", "1m": "1分K"}

IND = {"ma5": ("5MA", 0.5), "ma10": ("10MA", 0.6), "ma20": ("20MA", 1.0), "ma60": ("60MA(季線)", 1.2),
       "ma120": ("120MA(半年線)", 1.3), "ma240": ("240MA(年線)", 1.5),
       "bb_up": ("布林上軌", 0.8), "bb_mid": ("布林中軌", 0.6), "bb_dn": ("布林下軌", 0.8)}

PROFILES = {
    "1wk": dict(src="1wk", label="週K", lookback=150, pivot_w=3, rng=0.40, key_window=26,
                ind=["ma5", "ma10", "ma20", "ma60", "bb_up", "bb_mid", "bb_dn"]),
    "1d": dict(src="1d", label="日K", lookback=180, pivot_w=5, rng=0.25, key_window=60,
               ind=["ma5", "ma10", "ma20", "ma60", "ma120", "bb_up", "bb_mid", "bb_dn"]),
    "short": dict(src="1d", label="短線", lookback=40, pivot_w=3, rng=0.15, key_window=30,
                  ind=["ma5", "ma10", "ma20", "bb_up", "bb_mid", "bb_dn"]),
    "long": dict(src="1d", label="中長線", lookback=250, pivot_w=8, rng=0.45, key_window=120,
                 ind=["ma60", "ma120", "ma240"]),
    "5m": dict(src="5m", label="5分K", lookback=None, pivot_w=4, rng=0.06, key_window=60,
               ind=["ma20", "ma60", "bb_up", "bb_dn"]),
    "1m": dict(src="1m", label="1分K", lookback=None, pivot_w=6, rng=0.05, key_window=120,
               ind=["ma20", "ma60"]),
}


@dataclass
class Level:
    price: float
    score: float = 0.0
    touches: int = 0
    last_idx: int = 0
    tags: list = field(default_factory=list)
    kinds: set = field(default_factory=set)

    def to_dict(self, close):
        return {"price": round_tick(self.price), "score": round(self.score, 1), "touches": self.touches,
                "dist_pct": round((self.price / close - 1) * 100, 2), "tags": self.tags}


def find_pivots(df, w):
    h, l = df["high"].values, df["low"].values
    highs, lows = [], []
    for i in range(w, len(df) - w):
        # 左側嚴格、右側不嚴格 → 平頭/平底只取第一根
        if h[i] > h[i - w:i].max() and h[i] >= h[i + 1:i + w + 1].max():
            highs.append(i)
        if l[i] < l[i - w:i].min() and l[i] <= l[i + 1:i + w + 1].min():
            lows.append(i)
    return highs, lows


def volume_profile(df, bins=40):
    lo, hi = df["low"].min(), df["high"].max()
    edges = np.linspace(lo, hi, bins + 1)
    vol = np.zeros(bins)
    for l, h, v in zip(df["low"].values, df["high"].values, df["volume"].values):
        a = np.clip(np.searchsorted(edges, l, side="right") - 1, 0, bins - 1)
        b = np.clip(np.searchsorted(edges, h, side="right") - 1, 0, bins - 1)
        vol[a:b + 1] += v / (b - a + 1)
    return (edges[:-1] + edges[1:]) / 2, vol


def _cluster(points, tol):
    """points: [(price, idx, kind)] -> list[Level]"""
    levels: list[Level] = []
    for price, idx, kind in sorted(points):
        if levels and abs(price - levels[-1].price) <= tol:
            lv = levels[-1]
            lv.price = (lv.price * lv.touches + price) / (lv.touches + 1)
            lv.touches += 1
            lv.last_idx = max(lv.last_idx, idx)
            lv.tags.append(kind)
        else:
            levels.append(Level(price, touches=1, last_idx=idx, tags=[kind], kinds={"轉折"}))
    return levels


def _merge(levels, p, tag, wgt, kind, tol, n):
    near = [lv for lv in levels if abs(lv.price - p) <= tol]
    if near:
        near[0].score += wgt
        near[0].tags.append(tag)
        near[0].kinds.add(kind)
    else:
        levels.append(Level(float(p), score=wgt, last_idx=n - 1, tags=[tag], kinds={kind}))


def _key_candles(d: pd.DataFrame, window: int):
    """找最近 window 根內仍有效的帶量長紅 / 長黑 K (回傳在 d 中的位置)"""
    body = d["close"] - d["open"]
    avg_body = body.abs().rolling(20, min_periods=5).mean()
    vma = d["volume"].rolling(20, min_periods=5).mean()
    big_vol = d["volume"] > 1.3 * vma
    red = (body > 1.5 * avg_body) & big_vol
    black = (-body > 1.5 * avg_body) & big_vol
    closes = d["close"].values
    start = max(0, len(d) - window)
    red.iloc[:start] = False
    black.iloc[:start] = False
    bull = bear = None
    for i in np.where(red.values)[0][::-1]:
        if closes[i:].min() >= d["low"].iloc[i]:
            bull = i
            break
    for i in np.where(black.values)[0][::-1]:
        if closes[i:].max() <= d["high"].iloc[i]:
            bear = i
            break
    return bull, bear


def _fmt_time(ts, tf):
    return ts.strftime("%m/%d %H:%M" if tf in ("5m", "1m") else "%Y/%m/%d")


def compute_levels(df: pd.DataFrame, profile: str, extra: list | None = None, n_each=3) -> dict:
    """df 需已含 add_indicators 欄位。extra = [(price, 標籤)]"""
    cfg = PROFILES[profile]
    tf, label = cfg["src"], cfg["label"]
    d = df.tail(cfg["lookback"]) if cfg["lookback"] else df
    close = float(d["close"].iloc[-1])
    atr = float(d["atr"].iloc[-1])
    n = len(d)
    tol = max(atr * (0.4 if profile == "short" else 0.5), close * 0.004)
    span = f"{d.index[0]:%m/%d}~{d.index[-1]:%m/%d}" if tf in ("5m", "1m") else f"{d.index[0]:%Y/%m/%d}~{d.index[-1]:%Y/%m/%d}"
    why = []

    # 1. 轉折點
    hi, lo = find_pivots(d, cfg["pivot_w"])
    pts = [(float(d["high"].iloc[i]), i, "波段高點") for i in hi] + [(float(d["low"].iloc[i]), i, "波段低點") for i in lo]
    levels = _cluster(pts, tol)
    for lv in levels:
        lv.score = lv.touches * 1.0 + 1.5 * lv.last_idx / n
        if lv.touches >= 2:
            lv.tags.append(f"{lv.touches}次轉折")
    why.append(f"【{label}】範圍 {span} 共 {n} 根{TF_NAME[tf]}，找出波段高低點 {len(pts)} 個，"
               f"價差 {tol:.2f} 內合併為同一價位；轉折次數越多、越接近現在，分數越高。")

    # 2. 價位成交量分布
    centers, vol = volume_profile(d)
    poc = float(centers[vol.argmax()])
    mean_v = vol.mean()
    hvn = [i for i in range(1, len(vol) - 1) if vol[i] >= vol[i - 1] and vol[i] >= vol[i + 1] and vol[i] > 1.3 * mean_v]
    for i in hvn:
        p = float(centers[i])
        is_poc = abs(p - poc) < 1e-9
        _merge(levels, p, "最大量價位(POC)" if is_poc else "大量成交區", 2.0 if is_poc else 1.5, "量", tol, n)
    why.append(f"價位成交量分布：這段期間成交最多的價位 (POC) 約 {round_tick(poc)}，另有 {len(hvn)} 個大量換手區。"
               "大量區 = 很多人的持股成本，股價在它上方時是支撐 (成本區有人護盤)，在下方時是壓力 (解套賣壓)。")

    # 3. 技術指標價位
    used = []
    for col in cfg["ind"]:
        if col in d and pd.notna(d[col].iloc[-1]):
            name, wgt = IND[col]
            p = float(d[col].iloc[-1])
            used.append(f"{name} {p:.2f}")
            _merge(levels, p, name, wgt, "指標", tol, n)
    if used:
        why.append("指標價位：" + "、".join(used) + "。")

    # 4. 額外價位
    for p, name in extra or []:
        if p is None or not np.isfinite(p):
            continue
        _merge(levels, p, name, 1.2, "其他", tol, n)

    rng = cfg["rng"]
    sup = [lv for lv in levels if close * (1 - rng) <= lv.price < close]
    res = [lv for lv in levels if close < lv.price <= close * (1 + rng)]
    # 選價位：分數 × 距離衰減 (越近越重要)；中長線衰減較慢，才挑得到遠一點的大關卡
    decay = 8 if profile == "long" else 3

    def eff(lv):
        return lv.score / (1 + abs(lv.price - close) / (decay * atr))

    def pick(cands):
        """依分數挑，但價位之間至少要隔開一段距離，避免三條線擠在一起"""
        gap = max(2.5 * tol, atr * (1.0 if tf in ("5m", "1m") else 0.6))
        chosen = []
        for lv in sorted(cands, key=lambda x: -eff(x)):
            if all(abs(lv.price - c.price) >= gap for c in chosen):
                chosen.append(lv)
            if len(chosen) == n_each:
                break
        return chosen
    sup = sorted(pick(sup), key=lambda x: -x.price)
    res = sorted(pick(res), key=lambda x: x.price)
    new_high = not res or close >= d["high"].max() * 0.99
    for lv in sup + res:
        lv.tags = list(dict.fromkeys(lv.tags))
        if len(lv.kinds) >= 2:
            lv.tags.append("多重共振")
    why.append("被選出的價位：" + "；".join(
        [f"壓力 {round_tick(lv.price)} ({'、'.join(lv.tags)})" for lv in res[::-1]] +
        [f"支撐 {round_tick(lv.price)} ({'、'.join(lv.tags)})" for lv in sup]) +
        "。轉折、量、指標等不同依據重疊在同一價位 = 多重共振，最難突破/跌破。")

    # 5. 多空關鍵價
    bi, si = _key_candles(d, cfg["key_window"])
    key = {}
    if bi is not None:
        r = d.iloc[bi]
        key["bull"] = {"price": round_tick(float(r["low"])), "half": round_tick(float((r["open"] + r["close"]) / 2)),
                       "time": _fmt_time(d.index[bi], tf), "_ts": d.index[bi],
                       "why": f"{_fmt_time(d.index[bi], tf)} 帶量長紅K (量 {r['volume']:.0f} 張，漲 {r['close'] - r['open']:.2f})，"
                              f"之後收盤都沒跌破它的低點 {r['low']:.2f}，代表多方攻擊K仍有效；長紅K的一半 {(r['open'] + r['close']) / 2:.2f} 是第一道防守。"}
    elif sup:
        key["bull"] = {"price": round_tick(sup[0].price), "half": None, "time": None, "_ts": None,
                       "why": f"近 {cfg['key_window']} 根沒有仍有效的帶量長紅K，改用最近的強支撐 {round_tick(sup[0].price)} ({'、'.join(sup[0].tags)}) 當多方防守價。"}
    if si is not None:
        r = d.iloc[si]
        key["bear"] = {"price": round_tick(float(r["high"])), "half": round_tick(float((r["open"] + r["close"]) / 2)),
                       "time": _fmt_time(d.index[si], tf), "_ts": d.index[si],
                       "why": f"{_fmt_time(d.index[si], tf)} 帶量長黑K (量 {r['volume']:.0f} 張，跌 {r['open'] - r['close']:.2f})，"
                              f"之後收盤都沒站回它的高點 {r['high']:.2f}，代表空方仍壓著；要站回長黑K的一半 {(r['open'] + r['close']) / 2:.2f} 才算止跌。"}
    elif res:
        key["bear"] = {"price": round_tick(res[0].price), "half": None, "time": None, "_ts": None,
                       "why": f"近 {cfg['key_window']} 根沒有仍有效的帶量長黑K，改用最近的強壓力 {round_tick(res[0].price)} ({'、'.join(res[0].tags)}) 當空方關鍵價。"}

    # 盤勢判讀
    bull_p = key.get("bull", {}).get("price")
    bear_p = key.get("bear", {}).get("price")
    trend_ma = "ma60" if profile == "long" else "ma20"
    ma_v = d[trend_ma].iloc[-1] if trend_ma in d else np.nan
    ma_name = IND[trend_ma][0]
    if bear_p and close > bear_p:
        state = "多方強勢：已站上空方關鍵價"
    elif bull_p and close < bull_p:
        state = "空方強勢：已跌破多方關鍵價"
    elif pd.notna(ma_v) and close >= ma_v:
        state = f"區間偏多：在多空關鍵價之間，位於{ma_name}之上"
    else:
        state = f"區間偏空：在多空關鍵價之間，位於{ma_name}之下"
    if new_high:
        why.append("股價在這段期間的最高點附近，上方沒有套牢區，壓力只剩指標價位，目標價改用 ATR 推估。")
    why.append(f"目前價 {close:.2f}，{state}。" +
               (f"守住 {bull_p} 偏多看待，" if bull_p else "") +
               (f"突破 {bear_p} 才轉強攻。" if bear_p else ""))

    return {"profile": profile, "label": label, "tf": tf, "span": span, "close": close, "atr": round(atr, 2),
            "supports": [lv.to_dict(close) for lv in sup],
            "resistances": [lv.to_dict(close) for lv in res],
            "key": key, "state": state, "poc": round_tick(poc), "new_high": bool(new_high),
            "profile_vol": {"price": centers.round(2).tolist(), "vol": vol.round(1).tolist()},
            "_pivots": {"high": list(d.index[hi]), "low": list(d.index[lo])},
            "why": why}


def clean(lv: dict) -> dict:
    """移除畫圖用的內部欄位，給 API 回傳"""
    out = {k: v for k, v in lv.items() if not k.startswith("_") and k != "profile_vol"}
    out["key"] = {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")} for k, v in lv["key"].items()}
    return out


def merge_for_plan(short: dict, long: dict) -> dict:
    """波段計畫：支撐用短線 (停損較近)，壓力 = 短線 + 中長線 (目標可以看比較遠)"""
    res = {r["price"]: r for r in long["resistances"]}
    res.update({r["price"]: r for r in short["resistances"]})
    sup = {s["price"]: s for s in long["supports"]}
    sup.update({s["price"]: s for s in short["supports"]})
    return {"supports": sorted(sup.values(), key=lambda x: -x["price"]),
            "resistances": sorted(res.values(), key=lambda x: x["price"]),
            "key": short["key"]}


def sessions(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """取最近 n 個交易日的盤中資料"""
    days = sorted(set(df.index.date))[-n:]
    return df[np.isin(df.index.date, days)]
