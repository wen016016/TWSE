"""K 線圖 + 支撐壓力 + 多空關鍵價 + 價位成交量分布，輸出 PNG (base64)

版面：
  左上  K 線、均線、支撐壓力線、多空關鍵價
  左下  每根 K 棒的成交量 (「什麼時候」量大)
  右側  價位成交量分布 (「在什麼價位」成交最多 = 籌碼密集區)
"""
import base64
import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

UP, DN = "#e53935", "#16a085"          # 台股：紅漲綠跌
MA_C = {"ma5": "#f39c12", "ma10": "#8e44ad", "ma20": "#2980b9", "ma60": "#27ae60", "ma120": "#7f8c8d", "ma240": "#5d4037"}

# 每組支撐壓力的畫法：(支撐色, 壓力色, 線型, 線寬, 支撐字, 壓力字, 關鍵價後綴)
STYLE = {
    "main":  ("#1e6fd9", "#e67e22", "--", 1.1, "支撐", "壓力", ""),
    "short": ("#1e6fd9", "#e67e22", "--", 1.1, "短撐", "短壓", "(短)"),
    "long":  ("#0b2e6e", "#8e2c00", "-.", 1.6, "長撐", "長壓", "(長)"),
}
BULL_C, BEAR_C = "#c0392b", "#138d5a"


def _spread(ys, gap):
    """標籤 y 座標錯開，避免重疊"""
    order = np.argsort(ys)
    out = np.array(ys, dtype=float)
    for k in range(1, len(order)):
        a, b = order[k - 1], order[k]
        if out[b] - out[a] < gap:
            out[b] = out[a] + gap
    return out


def plot(df, title, tf, groups, profile_lv, show_vwap=False, bars=None, mas=None, day_split=False, plan=None,
         box=None) -> str:
    """groups: [(lv, kind, show_key)]，kind = main/short/long；plan = 進出場計畫 (畫在右側未來區)；
    box = 整理箱 {start_date, box_high, box_low, trigger, stage, window}"""
    d = df.tail(bars) if bars else df
    x = np.arange(len(d))
    o, h, l, c, v = (d[k].values for k in ("open", "high", "low", "close", "volume"))
    col = np.where(c >= o, UP, DN)
    pos = pd.Series(x, index=d.index)

    fig = plt.figure(figsize=(13.5, 8), dpi=96)
    gs = fig.add_gridspec(4, 7, hspace=0.05, wspace=0.03)
    ax = fig.add_subplot(gs[:3, :6])
    axv = fig.add_subplot(gs[3, :6], sharex=ax)
    axp = fig.add_subplot(gs[:3, 6], sharey=ax)

    # 盤中：依交易日分區，前一天灰底
    if day_split:
        dates = d.index.date
        bounds = [i for i in range(1, len(d)) if dates[i] != dates[i - 1]]
        starts = [0] + bounds
        for j, s in enumerate(starts):
            e = (starts[j + 1] if j + 1 < len(starts) else len(d)) - 0.5
            is_today = j == len(starts) - 1
            if not is_today:
                ax.axvspan(s - 0.5, e, color="#9e9e9e", alpha=0.10, lw=0)
                axv.axvspan(s - 0.5, e, color="#9e9e9e", alpha=0.10, lw=0)
            ax.text((s + e) / 2, 1.005, f"{dates[s]:%m/%d}" + (" 今日" if is_today else " 前一日"),
                    transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=9,
                    color="#333" if is_today else "#888")
        for b in bounds:
            ax.axvline(b - 0.5, color="#777", lw=1, ls=":")
            axv.axvline(b - 0.5, color="#777", lw=1, ls=":")

    if box and box.get("box_high"):
        from matplotlib.patches import Rectangle
        st = pd.Timestamp(box["start_date"]).tz_localize(d.index.tz) if d.index.tz else pd.Timestamp(box["start_date"])
        x0 = int(np.searchsorted(d.index, st))
        if x0 < len(d):
            bc = "#7b1fa2"
            ax.add_patch(Rectangle((x0 - 0.5, box["box_low"]), len(d) - x0, box["box_high"] - box["box_low"],
                                   facecolor=bc, alpha=0.07, edgecolor=bc, lw=1.2, ls="--", zorder=0))
            ax.text(x0, box["box_high"], f" 整理箱 {box['window']}天 · {box['stage']}", va="bottom", fontsize=9,
                    color=bc, fontweight="bold")
    ax.vlines(x, l, h, color=col, lw=0.9)
    body = np.maximum(np.abs(c - o), (h.max() - l.min()) * 0.0015)
    ax.bar(x, body, bottom=np.minimum(o, c), color=col, width=0.62, edgecolor=col)
    for m in mas or MA_C:
        if m in d and d[m].notna().any():
            ax.plot(x, d[m].values, color=MA_C[m], lw=1, label=m.upper())
    if show_vwap and "vwap" in d:
        ax.plot(x, d["vwap"].values, color="#111", lw=1.3, ls=":", label="VWAP")

    lines = []  # (price, text, color, ls, lw)
    for lv, kind, show_key in groups:
        sc, rc, ls, lw, st, rt, suf = STYLE[kind]
        for s in lv["supports"]:
            lines.append((s["price"], f"{st} {s['price']}", sc, ls, lw))
        for r in lv["resistances"]:
            lines.append((r["price"], f"{rt} {r['price']}", rc, ls, lw))
        k = lv["key"]
        if show_key and "bull" in k:
            lines.append((k["bull"]["price"], f"多方關鍵{suf} {k['bull']['price']}", BULL_C, "-", 2.2 if kind != "long" else 1.5))
            ts = k["bull"].get("_ts")
            if ts is not None and ts in pos.index:
                i = pos[ts]
                ax.annotate("關鍵長紅" + suf, (i, l[i]), (i, l[i] - (h.max() - l.min()) * 0.08), color=BULL_C,
                            fontsize=8, ha="center", arrowprops=dict(arrowstyle="->", color=BULL_C))
        if show_key and "bear" in k:
            lines.append((k["bear"]["price"], f"空方關鍵{suf} {k['bear']['price']}", BEAR_C, "-", 2.2 if kind != "long" else 1.5))
            ts = k["bear"].get("_ts")
            if ts is not None and ts in pos.index:
                i = pos[ts]
                ax.annotate("關鍵長黑" + suf, (i, h[i]), (i, h[i] + (h.max() - l.min()) * 0.08), color=BEAR_C,
                            fontsize=8, ha="center", arrowprops=dict(arrowstyle="->", color=BEAR_C))
        if kind != "long":
            for t in lv["_pivots"]["high"]:
                if t in pos.index:
                    ax.plot(pos[t], d["high"].loc[t] + (h.max() - l.min()) * 0.006, "v", color="#999", ms=4)
            for t in lv["_pivots"]["low"]:
                if t in pos.index:
                    ax.plot(pos[t], d["low"].loc[t] - (h.max() - l.min()) * 0.006, "^", color="#999", ms=4)

    lines.append((round(float(c[-1]), 2), f"現價 {c[-1]:g}", "#333", ":", 0.8))

    # 同價位合併標籤 (例：短撐 = 長撐 → 「短撐/長撐 126」；多方關鍵(短)+(長) → 「多方關鍵(短+長)」)
    groups_by_p = {}
    for p, t, cc, ls, lw in lines:
        groups_by_p.setdefault(p, []).append((t.rsplit(" ", 1)[0], cc, ls, lw))
    merged = {}
    for p, items in groups_by_p.items():
        names = list(dict.fromkeys(n for n, *_ in items))
        bases = {n.replace("(短)", "").replace("(長)", "") for n in names}
        if len(names) > 1 and len(bases) == 1:
            names = [bases.pop() + "(短+長)"]
        best = max(items, key=lambda it: it[3])
        merged[p] = ("/".join(names) + f" {p:g}", best[1], best[2], best[3])

    n = len(d)
    plan_items = []
    blocked = plan.get("note") if plan else None   # 漲跌停鎖住 → 不畫進出場價，只顯示提示
    if plan and not blocked:
        long = plan["side"] == "做多"
        e0, e1 = sorted(plan["entry_zone"])
        plan_items = [(plan["stop"], f"停損 {plan['stop']:g} (-{plan['risk_pct']}%)" if long else f"停損 {plan['stop']:g} (+{plan['risk_pct']}%)", "#111"),
                      (plan["target1"], f"目標一 {plan['target1']:g}", "#ad1457"),
                      (plan["target2"], f"目標二 {plan['target2']:g}", "#ad1457")]
        if plan.get("breakout"):
            plan_items.append((plan["breakout"], f"{'突破追價' if long else '跌破放空'} {plan['breakout']:g}", "#6a1b9a"))
    prices = [l.min(), h.max()] + list(merged) + [p for p, *_ in plan_items] + (list(plan["entry_zone"]) if plan_items else [])
    if box and box.get("box_high"):
        prices += [box["box_high"], box["box_low"]]
    lo, hi = min(prices), max(prices)
    pad = (hi - lo) * 0.07
    ax.set_ylim(lo - pad, hi + pad)

    # 未來進出場建議區 (最後一根 K 棒右側)
    F = max(12, n * 0.16) if plan else 0
    fx0, fx1 = n + 0.2, n + F
    if plan:
        ax.axvspan(fx0, fx1, color="#fff8e1", alpha=0.9, lw=0, zorder=0)
        ax.text((fx0 + fx1) / 2, 1.005, "未來進出場建議 →", transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", fontsize=9.5, color="#8d6e00", fontweight="bold")
    if blocked:
        ax.text((fx0 + fx1) / 2, 0.5, f"{blocked}\n\n等打開後\n再重新評估", transform=ax.get_xaxis_transform(),
                ha="center", va="center", fontsize=10, color="#b71c1c", fontweight="bold")
    elif plan:
        zc = "#e53935" if long else "#16a085"
        ax.fill_between([fx0, fx1], e0, e1 if e1 > e0 else e0 + (hi - lo) * 0.004, color=zc, alpha=0.28, zorder=1)
        ax.text(fx0 + 0.4, (e0 + e1) / 2, f"{'拉回買進區' if long else '反彈放空區'}\n{e0:g} ~ {e1:g}",
                fontsize=8.5, va="center", color=zc, fontweight="bold", zorder=5)
        for p, t, cc in plan_items:
            ax.hlines(p, fx0, fx1, color=cc, lw=1.8 if t.startswith("停損") else 1.4,
                      ls="-" if not t.startswith(("突破", "跌破")) else "--", zorder=4)
        py = _spread([p for p, *_ in plan_items], (hi - lo + 2 * pad) * 0.034)
        for (p, t, cc), yy in zip(plan_items, py):
            ax.annotate(t, (fx1, p), (fx1 - 0.3, yy), ha="right", va="bottom", fontsize=8.2, color=cc, fontweight="bold", zorder=6)
        ax.annotate("", (fx0 + F * 0.6, plan["target1"]), (fx0 + F * 0.6, (e0 + e1) / 2),
                    arrowprops=dict(arrowstyle="->", color="#ad1457", lw=1.2, ls="--"), zorder=3)
        ax.axvline(fx0, color="#c9a227", lw=0.8)
    right = fx1 + 0.8
    ys = sorted(merged)
    label_y = _spread(ys, (hi - lo + 2 * pad) * 0.034)
    for p, ly in zip(ys, label_y):
        t, cc, ls, lw = merged[p]
        ax.axhline(p, color=cc, ls=ls, lw=lw, alpha=0.9)
        ax.annotate(t, (len(d) - 0.5, p), (right, ly), color=cc, va="center", fontsize=8.5, fontweight="bold",
                    bbox=dict(facecolor="white", edgecolor=cc, alpha=0.9, pad=1.5, lw=0.6),
                    arrowprops=dict(arrowstyle="-", color=cc, lw=0.6) if abs(ly - p) > 1e-9 else None)

    ax.set_xlim(-1, fx1 + max(22, n * 0.2) if plan else n + max(22, n * 0.2))
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left", fontsize=8, ncol=7, framealpha=0.7)
    ax.set_title(title, fontsize=13, loc="left", fontweight="bold", pad=16 if day_split else 6)
    plt.setp(ax.get_xticklabels(), visible=False)

    axv.bar(x, v, color=col, width=0.62)
    if "vma20" in d:
        axv.plot(x, d["vma20"].values, color="#2980b9", lw=0.9)
    axv.grid(alpha=0.25)
    axv.set_ylabel("每根K棒成交量(張)", fontsize=8)
    fmt = "%H:%M" if tf in ("5m", "1m") else ("%y/%m" if tf == "1wk" else "%y/%m/%d")
    step = max(1, len(d) // 9)
    axv.set_xticks(x[::step])
    axv.set_xticklabels([t.strftime(fmt) for t in d.index[::step]], fontsize=8)

    prof = profile_lv["profile_vol"]
    pp, pv = np.array(prof["price"]), np.array(prof["vol"])
    hgt = (pp[1] - pp[0]) if len(pp) > 1 else 1
    axp.barh(pp, pv, height=hgt * 0.9, color=np.where(pp < c[-1], STYLE["main"][0], STYLE["main"][1]), alpha=0.35)
    axp.axhline(profile_lv["poc"], color="#444", lw=1, ls="-.")
    axp.text(pv.max() * 0.98, profile_lv["poc"], f"POC {profile_lv['poc']}", fontsize=7.5, ha="right", va="bottom", color="#444")
    axp.set_title(f"價位成交量分布\n({profile_lv['span']})", fontsize=8.5)
    axp.set_xticks([])
    plt.setp(axp.get_yticklabels(), visible=False)
    axp.yaxis.tick_right()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()
