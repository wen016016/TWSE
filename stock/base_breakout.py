"""個股「整理完成、準備啟動」偵測

整理區間：近 20 / 30 / 40 / 60 根日K 中，高低差最窄、且股價沒有明顯單邊走勢的箱子
完成度 (0~100)：
  波動收斂 25  布林通道寬度在近 120 天的低檔 / ATR 比 60 日平均小
  量縮     20  近 5 日均量 / 60 日均量
  越盤越窄 15  箱子切三段，後段振幅比前段小 (VCP)
  均線糾結 15  5 / 10 / 20 日線彼此距離
  位置     15  收盤在箱子上緣
  趨勢     10  季線上揚、股價在季線上 (上漲途中的整理)
階段：已啟動 (帶量突破箱頂) / 整理完成準備突破 / 整理中 / 跌破整理 / 不在整理
價位：觸發價 = 箱頂 + 1 檔；失敗停損 = 箱底與月線取高者 (在現價下方)；目標 = 箱頂 + 箱子高度 (等幅)
歷史驗證 (pooled_stats)：觀察名單 100 檔、近 2 年，完成度 ≥ 70 且尚未突破的日子，之後 20 天內突破箱頂的比例 / 平均最大漲幅 / 最大回檔，和所有日子比較
"""
import datetime as dt
import json
import threading

import numpy as np
import pandas as pd

from . import data
from .config import DATA_DIR
from .indicators import add_indicators, round_tick, tick

WINDOWS = (20, 30, 40, 60)
STATS_FILE = DATA_DIR / "base_stats.json"


def _find_box(d: pd.DataFrame, end: int):
    """回傳 (window, box_high, box_low, height%)：用 end 之前 (不含 end) 的 K 棒找整理箱"""
    best = None
    close = float(d["close"].iloc[end])
    atrp = float(d["atr"].iloc[end] / close)
    for w in WINDOWS:
        if end - w < 0:
            continue
        seg = d.iloc[end - w:end]
        hi, lo = float(seg["high"].max()), float(seg["low"].min())
        height = hi / lo - 1
        # 單邊走勢檢查：前後半段的均價差太大就不算整理
        drift = abs(seg["close"].iloc[w // 2:].mean() / seg["close"].iloc[:w // 2].mean() - 1)
        if height <= max(0.18, 7 * atrp) and drift <= height * 0.45:
            best = (w, hi, lo, height)   # 越長的整理越有份量，取符合條件的最長視窗
    return best


def _score_at(d: pd.DataFrame, end: int):
    """在第 end 根 (含) 的狀態；回傳 dict 或 None"""
    if end < 70:
        return None
    box = _find_box(d, end)
    x = d.iloc[end]
    close = float(x["close"])
    if box is None:
        return {"stage": "不在整理", "score": 0}
    w, hi, lo, height = box
    seg = d.iloc[end - w:end + 1]
    pts, why = {}, []

    bw = (d["bb_up"] - d["bb_dn"]) / d["bb_mid"]
    bw_now = float(bw.iloc[end])
    bw_hist = bw.iloc[max(0, end - 120):end + 1].dropna()
    pct = float((bw_hist < bw_now).mean()) if len(bw_hist) > 20 else 0.5
    atr_ratio = float(x["atr"] / d["atr"].iloc[max(0, end - 60):end].mean())
    pts["收斂"] = 25 * (0.6 * (1 - pct) + 0.4 * np.clip((1.3 - atr_ratio) / 0.6, 0, 1))
    why.append(f"布林通道寬度在近 120 天的 {pct * 100:.0f}% 位置、ATR 是 60 日平均的 {atr_ratio:.2f} 倍")

    v5 = float(d["volume"].iloc[end - 4:end + 1].mean())
    v60 = float(d["volume"].iloc[max(0, end - 60):end].mean()) or 1
    vr = v5 / v60
    pts["量縮"] = 20 * float(np.clip((1.1 - vr) / 0.6, 0, 1))
    why.append(f"近 5 日均量是 60 日均量的 {vr:.2f} 倍" + ("，量縮明顯" if vr < 0.7 else ""))

    body = seg.iloc[:-1]
    cut = [0, len(body) // 3, 2 * len(body) // 3, len(body)]
    thirds = [body.iloc[cut[i]:cut[i + 1]] for i in range(3)]
    rngs = [float(t["high"].max() / t["low"].min() - 1) for t in thirds if len(t)]
    vcp = len(rngs) == 3 and rngs[2] < rngs[1] * 1.05 and rngs[1] < rngs[0] * 1.05
    pts["越盤越窄"] = 15 if vcp else (7 if len(rngs) == 3 and rngs[2] < rngs[0] else 0)
    why.append("箱子三段振幅 " + " → ".join(f"{r_ * 100:.1f}%" for r_ in rngs) + ("，越盤越窄" if vcp else ""))

    mas = [float(x[m]) for m in ("ma5", "ma10", "ma20")]
    conv = (max(mas) - min(mas)) / close
    pts["均線糾結"] = 15 * float(np.clip((0.04 - conv) / 0.03, 0, 1))
    why.append(f"5/10/20 日線相差 {conv * 100:.1f}%" + ("，均線糾結" if conv < 0.02 else ""))

    pos = (close - lo) / (hi - lo) if hi > lo else 0.5
    pts["位置"] = 15 * float(np.clip((pos - 0.4) / 0.5, 0, 1))
    why.append(f"收盤在箱子的 {pos * 100:.0f}% 位置 (0% = 箱底，100% = 箱頂)")

    ma60_up = pd.notna(x["ma60"]) and x["ma60"] > d["ma60"].iloc[end - 10]
    above = pd.notna(x["ma60"]) and close > x["ma60"]
    pts["趨勢"] = 10 * (0.5 * ma60_up + 0.5 * above)
    why.append(("季線上揚" if ma60_up else "季線沒有上揚") + ("、股價在季線上" if above else "、股價在季線下"))

    score = round(sum(pts.values()), 0)
    vma20 = float(d["volume"].iloc[end - 20:end].mean())
    broke_up = close > hi and float(x["volume"]) > 1.5 * vma20
    broke_dn = close < lo and float(x["volume"]) > 1.2 * vma20
    if broke_up:
        stage = "已啟動 (帶量突破箱頂)"
    elif broke_dn:
        stage = "跌破整理 (轉弱)"
    elif score >= 70 and pos >= 0.6:
        stage = "整理完成，準備突破"
    elif score >= 70:
        stage = "整理完成，但還在箱子下半部 (等站回上緣)"
    elif score >= 45:
        stage = "整理中 (接近完成)"
    else:
        stage = "整理中"
    return {"stage": stage, "score": score, "window": w, "box_high": hi, "box_low": lo, "height_pct": round(height * 100, 1),
            "pts": {k: round(v, 1) for k, v in pts.items()}, "why": why, "broke_up": bool(broke_up), "broke_dn": bool(broke_dn),
            "pos": round(pos * 100, 0), "vol_ratio": round(vr, 2)}


def detect(d: pd.DataFrame) -> dict:
    """d：日K (需含 add_indicators)；回傳目前整理狀態 + 價位 + 過去 3 天內是否剛啟動"""
    d = d.copy()
    end = len(d) - 1
    cur = _score_at(d, end)
    if not cur or cur["stage"] == "不在整理":
        # 看最近 3 天有沒有剛突破 (今天不一定還在箱子裡)
        for k in (1, 2, 3):
            prev = _score_at(d, end - k)
            if prev and prev.get("broke_up"):
                prev = dict(prev, stage=f"{k} 天前已啟動 (帶量突破箱頂)")
                cur = prev
                break
    if not cur or cur["stage"] == "不在整理":
        return {"stage": "不在整理", "score": 0, "why": ["近 20~60 天沒有形成明顯的整理箱 (走勢較單邊或振幅太大)"]}
    close = float(d["close"].iloc[-1])
    hi, lo = cur["box_high"], cur["box_low"]
    trigger = round_tick(hi + tick(hi), "up")
    ma20 = float(d["ma20"].iloc[-1])
    stop_cands = [p for p in (lo, ma20) if p < close]
    stop = round_tick(max(stop_cands) - tick(close), "down") if stop_cands else round_tick(lo, "down")
    target = round_tick(hi * (1 + cur["height_pct"] / 100))
    cur.update(trigger=trigger, stop=stop, target=target, box_high=round_tick(hi), box_low=round_tick(lo),
               to_trigger_pct=round((trigger / close - 1) * 100, 2),
               start_date=f"{d.index[-1 - cur['window']]:%Y-%m-%d}")
    stats = pooled_stats(background=True)
    if stats and stats.get("ready"):
        cur["history"] = stats
    return cur


# ----------------------------------------------------------------- 歷史驗證 (觀察名單 100 檔 × 2 年)
_lock = threading.Lock()
_running = {"on": False}


def _compute_stats():
    from .scanner import universe
    daily = data.download_daily(universe(), period="2y")
    rows = []
    for code, df in daily.items():
        d = add_indicators(df)
        c, h, l = d["close"].values, d["high"].values, d["low"].values
        for end in range(80, len(d) - 21, 3):  # 每 3 天取樣一次，加快速度
            r = _score_at(d, end)
            if not r or r["stage"] == "不在整理" or r["broke_up"] or r["broke_dn"]:
                continue
            fut_h, fut_l = h[end + 1:end + 21].max(), l[end + 1:end + 21].min()
            rows.append({"score": r["score"], "breakout": fut_h > r["box_high"] * 1.005,
                         "max_gain": fut_h / c[end] - 1, "max_loss": fut_l / c[end] - 1,
                         "ret20": c[end + 20] / c[end] - 1, "broke_down": fut_l < r["box_low"]})
    df = pd.DataFrame(rows)
    if df.empty:
        return {"ready": False}

    def summ(sub):
        return {"n": int(len(sub)), "p_breakout": round(float(sub["breakout"].mean() * 100), 1),
                "p_breakdown": round(float(sub["broke_down"].mean() * 100), 1),
                "avg_max_gain": round(float(sub["max_gain"].mean() * 100), 1),
                "avg_max_loss": round(float(sub["max_loss"].mean() * 100), 1),
                "avg_ret20": round(float(sub["ret20"].mean() * 100), 1)}
    out = {"ready": True, "date": dt.date.today().isoformat(), "stocks": len(daily),
           "all": summ(df), "high": summ(df[df["score"] >= 70]), "mid": summ(df[(df["score"] >= 45) & (df["score"] < 70)]),
           "low": summ(df[df["score"] < 45])}
    STATS_FILE.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    return out


def pooled_stats(background=False):
    """每天算一次；background=True 時若還沒算好，就在背景算，先回傳 None"""
    try:
        s = json.loads(STATS_FILE.read_text(encoding="utf-8"))
        if s.get("date") == dt.date.today().isoformat():
            return s
    except Exception:  # noqa: BLE001
        s = None
    if background:
        with _lock:
            if not _running["on"]:
                _running["on"] = True

                def job():
                    try:
                        _compute_stats()
                    finally:
                        _running["on"] = False
                threading.Thread(target=job, daemon=True).start()
        return s  # 先用昨天的 (若有)
    return _compute_stats()
