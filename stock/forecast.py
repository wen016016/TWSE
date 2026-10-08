"""大盤走勢預測 — 以「台指期近全」(近月、夜盤 + 日盤連續) 為主

為什麼用台指期近全：加權指數只有日盤，看不到夜盤；台指期 15:00~05:00 夜盤會反映美股 / 國際盤，隔天日盤延續性高

⚠ 技術指標只能給「偏多 / 偏空的機率」，不是保證。頁面同時顯示這套方法過去的實際命中率，
  命中率不到 55% 時會把顯示的機率往 50% 收斂，並明確警告。

預測走勢線 (相似日法)：
  從近 30 個交易日找「這個交易日從夜盤開盤到現在」走法最像的 6 天，取它們之後走勢的加權平均畫成線，
  再用技術面分數微調；機率 = 相似日中之後真的往該方向走的比例 與 技術面機率 的平均
技術面 (台指期 5分K)：均線、MACD、KD、RSI、布林、量價、本盤開盤 30 分鐘區間、相對本盤開盤 / 昨日日盤收盤、近 30 分鐘動能
即時加分：期現貨價差 (日盤)、前 50 大權值股廣度、台積電 (日盤)
回測：夜盤 18:00 / 21:00、日盤 09:45 / 10:45 / 11:45，只用判斷當天以前的日子，對照到本盤收盤 / 60 分鐘後的實際方向
"""
import base64
import datetime as dt
import io
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from . import data  # noqa: E402
from . import futures_data as FD  # noqa: E402
from .indicators import add_indicators  # noqa: E402
from .levels import compute_levels  # noqa: E402
from .strategy import (Score, score_bbands, score_kd, score_ma, score_macd, score_rsi,  # noqa: E402
                       score_volume_price)

HEAVY = ["2330", "2317", "2454", "2308", "2382", "2412", "2881", "2882", "2891", "2303", "3711", "2886", "2884",
         "1301", "2002", "2357", "3231", "2885", "2892", "2345", "3008", "2327", "2379", "2395", "3034", "3037",
         "2603", "2609", "2615", "3017", "2408", "3443", "3661", "6669", "5269", "2376", "4938", "2301", "1216",
         "2207", "2880", "2883", "2887", "2890", "5880", "2801", "1303", "1326", "3045", "2912"]
K_ANALOG = 6
W_ANALOG = 0.5        # 機率 = W × 相似日 + (1-W) × 技術面
OPEN_BARS = 6         # 開盤 30 分鐘
NS, SLOTS = FD.NIGHT_SLOTS, FD.SLOTS


def _now():
    return dt.datetime.now(dt.timezone(dt.timedelta(hours=8)))


def _sess(slot):
    """格號 → (本盤名稱, 本盤第一格, 本盤最後一格)"""
    return ("夜盤", 0, NS - 1) if slot < NS else ("日盤", NS, SLOTS - 1)


def _slot_label(slot):
    if slot < NS:
        m = (900 + slot * 5) % 1440
    else:
        m = 525 + (slot - NS) * 5
    return f"{m // 60:02d}:{m % 60:02d}"


# ----------------------------------------------------------------- 技術面 (可回測)
def tech_score(df: pd.DataFrame, upto: int, prev_close: float) -> Score:
    """df：台指期 5分K (含指標、tday、slot)；upto：用到第幾根 (位置)；prev_close：前一交易日日盤收盤"""
    s = df.iloc[:upto + 1]
    sc = Score()
    last = float(s["close"].iloc[-1])
    tday, slot = s["tday"].iloc[-1], int(s["slot"].iloc[-1])
    name, first, _ = _sess(slot)
    cur = s[(s["tday"] == tday) & (s["slot"] >= first) & (s["slot"] <= (NS - 1 if first == 0 else SLOTS - 1))]

    score_ma(sc, s, "5分K", "ma5", "ma10", "ma20", "ma60")
    score_macd(sc, s, "5分K", 1.0)
    score_kd(sc, s, "5分K", 0.8)
    score_rsi(sc, s, "5分K", 0.8)
    score_bbands(sc, s, "5分K", 0.8)
    score_volume_price(sc, s, "5分K", look=12, w=0.6)

    if len(cur):
        sopen = float(cur["open"].iloc[0])
        if len(cur) > OPEN_BARS:
            oh, ol = float(cur["high"].iloc[:OPEN_BARS].max()), float(cur["low"].iloc[:OPEN_BARS].min())
            if last > oh:
                sc.add("盤勢", 1.2, f"突破{name}開盤 30 分鐘高點 {oh:,.0f}")
            elif last < ol:
                sc.add("盤勢", -1.2, f"跌破{name}開盤 30 分鐘低點 {ol:,.0f}")
            else:
                sc.add("盤勢", 0, f"在{name}開盤區間 {ol:,.0f} ~ {oh:,.0f} 內整理")
        if last > sopen and last > prev_close:
            sc.add("盤勢", 0.8, f"在{name}開盤 {sopen:,.0f} 與昨日日盤收盤 {prev_close:,.0f} 之上")
        elif last < sopen and last < prev_close:
            sc.add("盤勢", -0.8, f"在{name}開盤 {sopen:,.0f} 與昨日日盤收盤 {prev_close:,.0f} 之下")
        else:
            sc.add("盤勢", 0, f"介於{name}開盤 {sopen:,.0f} 與昨日日盤收盤 {prev_close:,.0f} 之間")
    if len(s) > 7:
        mom = (last / float(s["close"].iloc[-7]) - 1) * 100
        if abs(mom) >= 0.15:
            sc.add("盤勢", 0.8 if mom > 0 else -0.8, f"近 30 分鐘{'上漲' if mom > 0 else '下跌'} {mom:+.2f}%")
    return sc


def _prob(score):
    """技術分數 → 偏多機率 (保守的 logistic，避免過度自信)"""
    return round(100 / (1 + math.exp(-0.28 * score)), 1)


# ----------------------------------------------------------------- 即時加分 (日盤才有：現貨價差、權值股)
def live_extras(sc: Score, last: float, day_session: bool, notes: list):
    from .realtime import index_quote, quotes
    info = {}
    try:
        spot = index_quote("^TWII")
        if spot and spot.get("price"):
            info["spot"] = {"price": spot["price"], "chg_pct": spot.get("chg_pct")}
            if day_session:
                b = last - spot["price"]
                info["basis"] = round(b, 1)
                bp = b / spot["price"] * 100
                if abs(bp) >= 0.1:
                    sc.add("期現貨", 0.8 if b > 0 else -0.8, f"台指期{'正' if b > 0 else '逆'}價差 {b:+.0f} 點 ({bp:+.2f}%)，期貨{'比現貨樂觀' if b > 0 else '比現貨悲觀'}")
                else:
                    sc.add("期現貨", 0, f"期現貨價差 {b:+.0f} 點，中性")
    except Exception as e:  # noqa: BLE001
        notes.append(f"加權指數取得失敗：{e}")
    if not day_session:
        return info
    try:
        q = quotes(HEAVY, max_age=20)
        rows = [x["price"] / x["prev_close"] - 1 for x in q.values() if x.get("price") and x.get("prev_close")]
        if rows:
            ups, downs = sum(r > 0 for r in rows), sum(r < 0 for r in rows)
            avg = float(np.mean(rows)) * 100
            ratio = ups / max(ups + downs, 1)
            info["breadth"] = {"ups": ups, "downs": downs, "avg_pct": round(avg, 2)}
            pts = 1.0 if ratio >= 0.65 else (-1.0 if ratio <= 0.35 else 0)
            sc.add("權值股", pts, f"前 50 大權值股 {ups} 漲 {downs} 跌，平均 {avg:+.2f}%" + ("，買盤擴散" if pts > 0 else "，賣壓擴散" if pts < 0 else "，漲跌互見"))
            t = q.get("2330")
            if t and t.get("prev_close"):
                tc = (t["price"] / t["prev_close"] - 1) * 100
                info["tsmc_pct"] = round(tc, 2)
                if abs(tc) >= 0.5:
                    sc.add("權值股", 0.8 if tc > 0 else -0.8, f"台積電 {t['price']:g} ({tc:+.2f}%)，{'撐盤' if tc > 0 else '拖累指數'}")
    except Exception as e:  # noqa: BLE001
        notes.append(f"權值股報價取得失敗：{e}")
    return info


# ----------------------------------------------------------------- 相似日
def analog(paths: dict, cur: np.ndarray, prev_close: float, n: int, end: int, exclude=None, only_before=None, k=None):
    """找這個交易日前 n 格走法 (相對昨日日盤收盤 %) 最像的 k 天，回傳它們從第 n 格到 end 格的走勢"""
    k = k or K_ANALOG
    me = cur[:n] / prev_close - 1
    w_t = np.linspace(0.5, 1.5, n)
    cands = []
    for day, (arr, pc) in paths.items():
        if day == exclude or (only_before and day >= only_before) or np.isnan(arr[:end + 1]).any():
            continue
        p = arr / pc - 1
        dist = float(np.sqrt(np.average((p[:n] - me) ** 2, weights=w_t)))
        cands.append((dist, day, arr[n - 1:end + 1] / arr[n - 1] - 1))
    if len(cands) < k:
        return None
    cands.sort(key=lambda x: x[0])
    best = cands[:k]
    w = np.array([1 / (d + 1e-4) for d, _, _ in best])
    w = w / w.sum()
    fut = np.vstack([f for _, _, f in best])
    mean = (fut * w[:, None]).sum(axis=0)
    std = np.sqrt(((fut - mean) ** 2 * w[:, None]).sum(axis=0))
    return {"mean": mean, "std": std, "w": w, "fut": fut,
            "days": [{"day": d.strftime("%m/%d"), "dist_pct": round(dist * 100, 3), "end_pct": round(float(f[-1]) * 100, 2)}
                     for (dist, d, f) in best]}


def p_up_at(an, h):
    h = min(h, an["fut"].shape[1] - 1)
    return round(float((an["w"] * (an["fut"][:, h] > 0)).sum() * 100), 1)


def backtest(df: pd.DataFrame, paths: dict):
    """夜盤 18:00 / 21:00、日盤 09:45 / 10:45 / 11:45；只用判斷當天以前的日子 (不偷看未來)"""
    cps = [(36, "夜盤 18:00"), (72, "夜盤 21:00"), (NS + 12, "日盤 09:45"), (NS + 24, "日盤 10:45"), (NS + 36, "日盤 11:45")]
    days = sorted(paths)
    today = FD.trade_day_of(pd.Timestamp(_now()))
    rows = []
    for day in days[10:]:
        if day == today:
            continue
        arr, pc = paths[day]
        pos = np.where(df["tday"].values == day)[0]
        if len(pos) < SLOTS - 5:
            continue
        slot_pos = {int(s_): p for s_, p in zip(df["slot"].values[pos], pos)}
        for n_slot, label in cps:
            if n_slot not in slot_pos:
                continue
            _, _, end = _sess(n_slot)
            an = analog(paths, arr, pc, n_slot + 1, end, only_before=day)
            if an is None:
                continue
            tp = _prob(tech_score(df, slot_pos[n_slot], pc).total)
            for hname, h in (("to_close", end - n_slot), ("next60", 12)):
                actual = arr[min(n_slot + h, end)] - arr[n_slot]
                if actual == 0:
                    continue
                pa = p_up_at(an, h)
                rows.append({"cp": label, "h": hname, "pa": pa, "tp": tp,
                             "pb": W_ANALOG * pa + (1 - W_ANALOG) * tp, "up": actual > 0})
    if not rows:
        return None
    bt = pd.DataFrame(rows)

    def rate(sub, col, strong=False):
        if strong:
            sub = sub[(sub[col] - 50).abs() >= 12]
        sub = sub[sub[col] != 50]
        if sub.empty:
            return None, 0
        return round(float(((sub[col] > 50) == sub["up"]).mean() * 100), 1), int(len(sub))

    out = {"days": len([d for d in days[10:] if d != today]), "table": []}
    for label, col in (("結合 (目前使用)", "pb"), ("只用技術面", "tp"), ("只用相似日", "pa")):
        for strong in (False, True):
            a = rate(bt[bt["h"] == "to_close"], col, strong)
            b = rate(bt[bt["h"] == "next60"], col, strong)
            out["table"].append({"group": label + (" · 強訊號" if strong else ""), "to_close": a[0], "n_close": a[1],
                                 "next60": b[0], "n60": b[1]})
    for _, label in cps:
        sub = bt[bt["cp"] == label]
        a = rate(sub[sub["h"] == "to_close"], "pb")
        b = rate(sub[sub["h"] == "next60"], "pb")
        out["table"].append({"group": f"{label} 判斷", "to_close": a[0], "n_close": a[1], "next60": b[0], "n60": b[1]})
    return out


# ----------------------------------------------------------------- 主流程
def _verdict(p):
    if p >= 65:
        return "偏多"
    if p >= 57:
        return "小幅偏多"
    if p <= 35:
        return "偏空"
    if p <= 43:
        return "小幅偏空"
    return "中性 / 震盪"


def forecast() -> dict:
    notes = []
    raw = FD.near_full_5m()
    df = add_indicators(raw[["open", "high", "low", "close", "volume"]], mas=(5, 10, 20, 60))
    df["tday"], df["slot"], df["contract"] = raw["tday"], raw["slot"].astype(int), raw["contract"]
    paths = FD.day_paths(raw)
    tday = df["tday"].iloc[-1]
    slot = int(df["slot"].iloc[-1])
    sess_name, first, end = _sess(slot)
    last = float(df["close"].iloc[-1])
    if tday not in paths:
        raise RuntimeError("台指期資料不足 (找不到前一交易日收盤)")
    cur, prev_close = paths[tday]

    now = _now()
    last_ts = df.index[-1]
    live = (now - last_ts.to_pydatetime()).total_seconds() < 20 * 60 and slot < end
    sc = tech_score(df, len(df) - 1, prev_close)
    extra = live_extras(sc, last, sess_name == "日盤", notes)
    tech_p = _prob(sc.total)

    bt = backtest(df, paths)
    hit = None
    if bt:
        hit = next((r["to_close"] for r in bt["table"] if r["group"] == "結合 (目前使用)"), None)
    shrink = 1.0 if hit is None else float(np.clip((hit - 50) / 10, 0.15, 1.0))
    weak = hit is not None and hit < 55

    remaining = end - slot if live else 0
    an = analog(paths, cur, prev_close, slot + 1, end, exclude=tday) if remaining > 0 else None
    horizons, path = [], None
    if an is not None:
        ret = W_ANALOG * an["mean"] + (1 - W_ANALOG) * (tech_p - 50) / 50 * an["std"]
        line = last * (1 + ret * (0.4 + 0.6 * shrink))  # 預測力不足時，線也收斂往現價
        path = {"line": line.round(1).tolist(), "lo": (line * (1 - an["std"])).round(1).tolist(),
                "hi": (line * (1 + an["std"])).round(1).tolist()}
        for label, h in (("30 分鐘後", 6), ("60 分鐘後", 12), (f"{sess_name}收盤", remaining)):
            h = min(h, remaining)
            if h <= 0 or any(x["bars"] == h for x in horizons):
                continue
            pa = p_up_at(an, h)
            rawp = W_ANALOG * pa + (1 - W_ANALOG) * tech_p
            p = round(50 + (rawp - 50) * shrink, 1)
            target = float(line[h])
            up = target >= last
            horizons.append({"label": label, "bars": h, "time": _slot_label(slot + h), "target": round(target, 0),
                             "chg_pct": round((target / last - 1) * 100, 2),
                             "lo1": round(float(path["lo"][h]), 0), "hi1": round(float(path["hi"][h]), 0),
                             "p_up": p, "p_raw": round(rawp, 1), "p_line": p if up else round(100 - p, 1),
                             "dir": "上漲" if up else "下跌", "p_analog": pa})
    p_close = horizons[-1]["p_up"] if horizons else round(50 + (tech_p - 50) * shrink, 1)
    today_df = df[df["tday"] == tday]
    lv = compute_levels(today_df if len(today_df) > 30 else df.tail(120), "5m", extra=[(prev_close, "昨日日盤收盤")])

    if weak:
        notes.insert(0, f"⚠ 近 {bt['days']} 個交易日回測命中率只有 {hit}% (接近丟銅板)，目前沒有明顯預測力，"
                        "機率已依實際命中率往 50% 收斂。請勿單獨依此下單")
    if not live:
        notes.append("目前非交易時段 (或資料尚未更新)：預測線只在夜盤 / 日盤交易中產生")
    if an is not None:
        notes.append(f"預測線 = 近 30 個交易日中，從夜盤開盤到現在走法最像的 {K_ANALOG} 天之後的平均走勢，再用技術面微調")
    notes.append("預測為機率性判斷，僅供參考；機率在 40~60% 之間代表方向不明")

    res = {"time": f"{last_ts:%Y-%m-%d %H:%M}", "session": sess_name, "contract": str(df["contract"].iloc[-1]),
           "tday": tday.strftime("%Y-%m-%d"), "live": live, "last": last, "prev_close": prev_close,
           "chg": round(last - prev_close, 1), "chg_pct": round((last / prev_close - 1) * 100, 2),
           "total": sc.total, "tech_p": tech_p, "p_up": p_close, "verdict": _verdict(p_close),
           "hit_rate": hit, "weak": weak, "shrink": round(shrink, 2),
           "by_cat": sc.by_cat(), "items": sc.items, "horizons": horizons, "remaining_bars": remaining,
           "analogs": an["days"] if an else [], "extra": extra, "backtest": bt, "notes": notes,
           "levels": {"supports": lv["supports"], "resistances": lv["resistances"]}}
    res["chart"] = _chart(df, tday, lv, res, an, path, slot, end)
    return res


def _chart(df, tday, lv, res, an, path, slot, end):
    s = df[df["tday"] == tday]
    if len(s) < 20:
        s = df.tail(120)
    x = np.arange(len(s))
    o, h, l, c = (s[k].values for k in ("open", "high", "low", "close"))
    col = np.where(c >= o, "#e53935", "#16a085")
    fig, ax = plt.subplots(figsize=(15, 7), dpi=96)
    slots = s["slot"].values
    night_end = int(np.argmax(slots >= NS)) if (slots >= NS).any() else len(s)
    if night_end > 0:
        ax.axvspan(-0.5, night_end - 0.5, color="#5c6bc0", alpha=0.07, lw=0)
        ax.text(night_end / 2, 1.01, "夜盤 15:00~05:00", transform=ax.get_xaxis_transform(), ha="center", fontsize=9, color="#3949ab")
    if night_end < len(s) or slot >= NS:
        ax.axvline(night_end - 0.5, color="#777", ls=":", lw=1)
        ax.text(night_end + (len(s) - night_end + res["remaining_bars"]) / 2, 1.01, "日盤 08:45~13:45",
                transform=ax.get_xaxis_transform(), ha="center", fontsize=9, color="#555")
    ax.vlines(x, l, h, color=col, lw=0.8)
    ax.bar(x, np.maximum(np.abs(c - o), (h.max() - l.min()) * 0.0015), bottom=np.minimum(o, c), color=col, width=0.62)
    for m, mc in (("ma5", "#f39c12"), ("ma20", "#2980b9")):
        ax.plot(x, s[m].values, color=mc, lw=1, label=m.upper())
    ax.axhline(res["prev_close"], color="#777", ls=":", lw=1)
    ax.text(0, res["prev_close"], f" 昨日日盤收盤 {res['prev_close']:,.0f}", va="bottom", fontsize=8, color="#555")

    last_x = len(s) - 1
    xend = last_x + max(res["remaining_bars"], 0)
    labels = {i: _slot_label(int(sl)) for i, sl in enumerate(slots)}
    for k_ in range(1, res["remaining_bars"] + 1):
        labels[last_x + k_] = _slot_label(slot + k_)
    if path:
        xs = np.arange(last_x, last_x + len(path["line"]))
        up = res["p_up"] >= 50
        cc = "#d32f2f" if up else "#00897b"
        ax.axvspan(last_x, xend + 0.5, color="#fff8e1", alpha=0.75, lw=0, zorder=0)
        for f_ in an["fut"]:
            ax.plot(xs, res["last"] * (1 + f_), color="#9e9e9e", lw=0.7, alpha=0.45, zorder=1)
        ax.fill_between(xs, path["lo"], path["hi"], color=cc, alpha=0.15, lw=0, zorder=2, label="預測區間 (相似日離散程度)")
        ax.plot(xs, path["line"], color=cc, lw=2.6, zorder=4, label="預測走勢線")
        ax.plot([last_x], [res["last"]], "o", color="#111", ms=5, zorder=5)
        for j, hz in enumerate(res["horizons"]):
            xi = last_x + hz["bars"]
            ax.plot([xi], [hz["target"]], "o", color=cc, ms=7, zorder=5)
            top = j % 2 == 0
            ax.annotate(f"{hz['label']} {hz['time']}\n{hz['target']:,.0f}  {hz['dir']}機率 {hz['p_line']}%",
                        (xi, hz["target"]), (0, 28 if top else -32), textcoords="offset points",
                        ha="center", va="bottom" if top else "top", fontsize=9, fontweight="bold", color=cc,
                        bbox=dict(facecolor="white", edgecolor=cc, alpha=0.9, pad=2, lw=0.8),
                        arrowprops=dict(arrowstyle="-", color=cc, lw=0.8), zorder=6)
    for sp in lv["supports"]:
        ax.axhline(sp["price"], color="#1e6fd9", ls="--", lw=0.9)
        ax.text(xend + 1.5, sp["price"], f"支撐 {sp['price']:,.0f}", va="center", fontsize=8.5, color="#1e6fd9")
    for rp in lv["resistances"]:
        ax.axhline(rp["price"], color="#e67e22", ls="--", lw=0.9)
        ax.text(xend + 1.5, rp["price"], f"壓力 {rp['price']:,.0f}", va="center", fontsize=8.5, color="#e67e22")

    # X 軸：夜盤每 2 小時、日盤每小時標一次，不會擠在一起
    ticks = []
    for i in range(xend + 1):
        lab = labels.get(i, "")
        if not lab:
            continue
        hh, mm = int(lab[:2]), int(lab[3:])
        is_night = (i < night_end) if i <= last_x else slot + (i - last_x) < NS
        if mm == 0 and (not is_night or hh % 2 == 1):
            ticks.append(i)
    ax.set_xticks(ticks)
    ax.set_xticklabels([labels[i] for i in ticks], fontsize=9)
    ax.set_xlim(-1, xend + 12)
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left", fontsize=8.5, ncol=4)
    ttl = (f"台指期近全 {res['contract']} 5分K + 預測走勢　{res['session']}收盤偏多機率 {res['p_up']}% ({res['verdict']})"
           if path else f"台指期近全 {res['contract']} 5分K")
    ax.set_title(ttl, loc="left", fontsize=13, fontweight="bold", pad=20)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()
