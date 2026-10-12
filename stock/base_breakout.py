"""個股「整理完成、準備啟動」偵測

整理區間：用 box.py 的箱型規則 (近 20~60 根日K、實體為主、箱頂箱底各 2 次觸碰、寬 3~6 倍 ATR20、前後半段均價差 <= 箱寬 45%)
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
STATS_FILE = DATA_DIR / "base_stats_v2.json"   # 箱型規則改版，統計重算


def _find_box(d: pd.DataFrame, end: int):
    """用新箱型規則 (box.py：實體為主、2 次以上觸碰、寬 3~6 倍 ATR20、均價差 <= 箱寬 45%)
    回傳 (window, box_high, box_low, height%, status)；status = range / up / down (第 end 根收盤在箱外)"""
    from .box import atr_mean, find_box
    seg = d.iloc[:end + 1]
    atr = atr_mean(seg, 20)
    bx = find_box(seg, atr, tuple(sorted(WINDOWS, reverse=True)), max_excl=10, max_atr=6)
    if not bx:
        return None
    return bx["window"], bx["hi"], bx["lo"], bx["hi"] / bx["lo"] - 1, bx["status"], bx["after"], bx


def _score_at(d: pd.DataFrame, end: int):
    """在第 end 根 (含) 的狀態；回傳 dict 或 None"""
    if end < 70:
        return None
    box = _find_box(d, end)
    x = d.iloc[end]
    close = float(x["close"])
    if box is None:
        return {"stage": "不在整理", "score": 0}
    w, hi, lo, height, status, after, raw = box
    seg = d.iloc[max(0, end - w - after + 1):end - after + 1]
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
    # 破局：收盤在箱外且隔天未拉回 (after >= 2)，或當天帶量 (> 1.5 倍 20 均量)
    broke_up = status == "up" and (after >= 2 or float(x["volume"]) > 1.5 * vma20)
    broke_dn = status == "down" and after >= 2
    if status == "up" and not broke_up:
        stage = "收盤站上箱頂但量不足 (等隔日確認)"
    elif status == "down" and not broke_dn:
        stage = "收盤跌破箱底 (等隔日確認)"
    elif broke_up:
        stage = "已啟動 (帶量突破箱頂)" if float(x["volume"]) > 1.5 * vma20 else "已啟動 (突破箱頂且隔日未拉回)"
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
            "pos": round(pos * 100, 0), "vol_ratio": round(vr, 2), "after": after, "_box": raw}


def detect(d: pd.DataFrame) -> dict:
    """d：日K (需含 add_indicators)；回傳目前整理狀態 + 價位 + 過去 3 天內是否剛啟動"""
    d = d.copy()
    end = len(d) - 1
    cur = _score_at(d, end)
    # 全域共享箱型：只用「今天」判定的箱型，不再沿用前幾天的舊箱型
    if not cur or cur["stage"] == "不在整理":
        out = {"stage": "趨勢盤，目前無有效箱型", "score": 0,
               "why": ["近 20~60 天沒有形成有效的整理箱 (走勢單邊或振幅太大)，以趨勢盤處理"]}
        # 保留 1~3 天前剛被突破的舊箱型 (只供繪圖 / 等幅預測參考，不當作目前箱型)
        for k in (1, 2, 3):
            prev = _score_at(d, end - k)
            if prev and (prev.get("broke_up") or prev.get("broke_dn")):
                raw = prev["_box"]
                after = int(raw["after"]) + k
                bi = max(0, len(d) - after)
                hi, lo = float(raw["hi"]), float(raw["lo"])
                up = bool(prev.get("broke_up"))
                pl_, ph_ = (hi, hi + (hi - lo)) if up else (lo - (hi - lo), lo)
                out["prev_box"] = {"box_high": round_tick(hi), "box_low": round_tick(lo), "window": int(raw["window"]),
                                   "start_date": f"{pd.Timestamp(raw['start']):%Y-%m-%d}",
                                   "break_date": f"{d.index[bi]:%Y-%m-%d}",
                                   "days_ago": k, "dir": "up" if up else "down",
                                   "proj_low": round_tick(pl_), "proj_high": round_tick(ph_)}
                break
        return out
    raw = cur.pop("_box")
    if "已啟動" in cur["stage"]:
        raw = dict(raw, status="up", confirmed=True)
    cur["shared_box"] = dict(raw, source=cur["stage"])
    close = float(d["close"].iloc[-1])
    hi, lo = cur["box_high"], cur["box_low"]
    # 位置百分比一律用當天收盤 (現價)
    pos = (close - lo) / (hi - lo) * 100 if hi > lo else 50
    cur["pos"] = round(pos, 1)
    cur["why"] = [f"收盤在箱子的 {pos:.1f}% 位置 (0% = 箱底，100% = 箱頂)" if w.startswith("收盤在箱子") else w
                  for w in cur["why"]]
    trigger = round_tick(hi + tick(hi), "up")
    ma20 = float(d["ma20"].iloc[-1])
    stop_cands = [p for p in (lo, ma20) if p < close]
    stop = round_tick(max(stop_cands) - tick(close), "down") if stop_cands else round_tick(lo, "down")
    target = round_tick(hi * (1 + cur["height_pct"] / 100))
    cur.update(trigger=trigger, stop=stop, target=target, box_high=round_tick(hi), box_low=round_tick(lo),
               to_trigger_pct=round((trigger / close - 1) * 100, 2),
               start_date=f"{pd.Timestamp(raw['start']):%Y-%m-%d}")
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
