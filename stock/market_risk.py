"""大盤回測風險 / 減碼建議 (加權指數，日線級別)

目標：判斷大盤是不是漲多了、快到壓力，接下來幾天回測的機率多高、會回到哪，該不該減碼、減多少。

兩部分：
1. 規則訊號 (每一項附原因)：乖離、過熱 (日 / 週 RSI、KD)、背離、連漲、量價、位置 (壓力 / 前高)、
   跌破短均線、籌碼 (外資現貨、外資台指期留倉、融資)、期貨 (價差、夜盤)
2. 歷史相似情況 (10 年、約 2,400 天)：找乖離 / RSI / 連漲 / 離高點距離 / 近期漲幅 最像現在的 40 天，
   統計它們之後 5 天內回檔 ≥2%、≥3%、碰 10 日線、10 天內碰月線的比例，並和「平常」的比例比較
回測風險分數 = 50% 歷史相似情況回檔機率 + 50% 規則訊號
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
from .indicators import add_indicators, crossed_down, to_weekly  # noqa: E402
from .levels import compute_levels  # noqa: E402

K = 40
FEATS = ["bias5", "bias20", "bias60", "rsi", "rsi_w", "streak", "dist_high", "ret5", "ret20"]


def _now():
    return dt.datetime.now(dt.timezone(dt.timedelta(hours=8)))


def _long_daily():
    import yfinance as yf

    def load():
        df = yf.Ticker(data.INDEX).history(period="10y", auto_adjust=False)
        return data._normalize(df, True)
    df = data.cached("twii10y", 3600, load)
    return data._live(df, data.INDEX, "1d", True)  # 補上今天即時


def _night_implied(d: pd.DataFrame):
    """夜盤 (15:00~隔天 08:45)：用「台指期夜盤 − 收盤價差」推估明天加權指數，當成最新一根 K 棒

    價差 = 台指期日盤收盤 − 加權收盤；推估加權 = 夜盤價 − 價差 (高低點同理)
    回傳 (新的日K, 推估資訊) ；不在夜盤或抓不到期貨時回傳 (原日K, None)
    """
    now = _now()
    hm = now.hour * 100 + now.minute
    if not (hm >= 1500 or hm < 845):
        return d, None
    try:
        from .realtime import futures
        f = futures()
        day, night = f.get("day"), f.get("night")
        if not (day and night and day.get("price") and night.get("price")):
            return d, None
        twii = float(d["close"].iloc[-1])
        basis = float(day["price"]) - twii
        px = float(night["price"]) - basis
        hi = float(night.get("high") or night["price"]) - basis
        lo = float(night.get("low") or night["price"]) - basis
        op = float(night.get("ref") or day["price"]) - basis
        nxt = d.index[-1] + pd.Timedelta(days=1)
        while nxt.weekday() >= 5:
            nxt += pd.Timedelta(days=1)
        d = d.copy()
        d.loc[nxt] = [op, max(hi, px, op), min(lo, px, op), px, 0]
        info = {"tx_night": float(night["price"]), "tx_night_chg_pct": night.get("chg_pct"), "tx_day_close": float(day["price"]),
                "basis": round(basis, 1), "implied": round(px, 1), "implied_chg_pct": round((px / twii - 1) * 100, 2),
                "twii_close": twii, "night_time": night.get("time")}
        return d, info
    except Exception:  # noqa: BLE001
        return d, None


def _features(d: pd.DataFrame) -> pd.DataFrame:
    d = add_indicators(d, mas=(5, 10, 20, 60, 120))
    f = pd.DataFrame(index=d.index)
    f["bias5"] = d["close"] / d["ma5"] - 1
    f["bias20"] = d["close"] / d["ma20"] - 1
    f["bias60"] = d["close"] / d["ma60"] - 1
    f["rsi"] = d["rsi"]
    w = add_indicators(to_weekly(d[["open", "high", "low", "close", "volume"]]), mas=(5, 10, 20))
    f["rsi_w"] = w["rsi"].reindex(d.index, method="ffill")
    up = (d["close"] > d["close"].shift()).astype(int)
    f["streak"] = up.groupby((up != up.shift()).cumsum()).cumsum() * up - (1 - up).groupby((up != up.shift()).cumsum()).cumsum() * (1 - up)
    f["dist_high"] = d["close"] / d["high"].rolling(60).max() - 1
    f["ret5"] = d["close"].pct_change(5)
    f["ret20"] = d["close"].pct_change(20)
    return d, f


def _outcomes(d: pd.DataFrame) -> pd.DataFrame:
    """每一天之後的實際結果 (用來統計機率)"""
    c, lo = d["close"].values, d["low"].values
    ma10, ma20 = d["ma10"].values, d["ma20"].values
    n = len(d)
    o = {"dd5": np.full(n, np.nan), "dd10": np.full(n, np.nan), "touch10": np.full(n, np.nan),
         "touch20": np.full(n, np.nan), "next_down": np.full(n, np.nan)}
    for i in range(n - 10):
        m5, m10 = lo[i + 1:i + 6].min(), lo[i + 1:i + 11].min()
        o["dd5"][i] = m5 / c[i] - 1
        o["dd10"][i] = m10 / c[i] - 1
        o["touch10"][i] = float(m5 <= ma10[i]) if c[i] > ma10[i] else np.nan
        o["touch20"][i] = float(m10 <= ma20[i]) if c[i] > ma20[i] else np.nan
        o["next_down"][i] = float(c[i + 1] < c[i])
    return pd.DataFrame(o, index=d.index)


def _stats(out: pd.DataFrame) -> dict:
    def p(col, cond):
        s = out[col].dropna()
        return round(float(cond(s).mean() * 100), 1) if len(s) else None
    return {"dd2_5d": p("dd5", lambda s: s <= -0.02), "dd3_5d": p("dd5", lambda s: s <= -0.03),
            "dd5_10d": p("dd10", lambda s: s <= -0.05), "touch10_5d": p("touch10", lambda s: s > 0),
            "touch20_10d": p("touch20", lambda s: s > 0), "next_down": p("next_down", lambda s: s > 0),
            "median_dd5": round(float(out["dd5"].dropna().median() * 100), 2) if out["dd5"].notna().any() else None,
            "median_dd10": round(float(out["dd10"].dropna().median() * 100), 2) if out["dd10"].notna().any() else None,
            "n": int(out["dd5"].notna().sum())}


def _rules(d, f, lv_s, lv_l, notes):
    """回傳 [(分數, 分類, 原因)]；正分 = 回測風險升高，負分 = 有撐 / 偏健康"""
    r = []
    x, fx = d.iloc[-1], f.iloc[-1]
    b20, b5, b60 = fx["bias20"] * 100, fx["bias5"] * 100, fx["bias60"] * 100
    if b20 >= 6:
        r.append((2, "乖離", f"指數比月線高 {b20:.1f}%，乖離過大，容易回測月線 {x['ma20']:,.0f}"))
    elif b20 >= 4:
        r.append((1, "乖離", f"指數比月線高 {b20:.1f}%，偏熱"))
    elif b20 <= -6:
        r.append((-2, "乖離", f"指數比月線低 {b20:.1f}%，跌深，反而容易反彈"))
    else:
        r.append((0, "乖離", f"與月線乖離 {b20:+.1f}%，正常範圍"))
    if b5 >= 2.5:
        r.append((1, "乖離", f"比 5 日線高 {b5:.1f}%，短線急漲"))
    if b60 >= 12:
        r.append((1, "乖離", f"比季線高 {b60:.1f}%，中期漲幅已大"))
    rsi, rsi_w = fx["rsi"], fx["rsi_w"]
    if rsi >= 80:
        r.append((2.5, "過熱", f"日 RSI {rsi:.0f}，嚴重過熱"))
    elif rsi >= 72:
        r.append((1.5, "過熱", f"日 RSI {rsi:.0f}，過熱"))
    elif rsi <= 30:
        r.append((-1.5, "過熱", f"日 RSI {rsi:.0f}，超賣"))
    if pd.notna(rsi_w) and rsi_w >= 75:
        r.append((1, "過熱", f"週 RSI {rsi_w:.0f}，週線也過熱"))
    if crossed_down(d["k"], d["d"], 2) and x["k"] > 60:
        r.append((1.5, "過熱", f"日 KD 高檔死亡交叉 (K {x['k']:.0f})"))
    elif x["k"] > 80:
        r.append((0.5, "過熱", f"日 KD 高檔鈍化 (K {x['k']:.0f})，強勢但隨時會轉折"))
    # 背離：價格接近 60 日高點，但 RSI / MACD 柱狀體比前一個高點弱
    if fx["dist_high"] >= -0.01:
        win = d.tail(25)
        prev_peak = win.iloc[:-5]
        if len(prev_peak) and x["rsi"] < prev_peak["rsi"].max() - 8:
            r.append((1.5, "背離", f"指數接近前高，但 RSI ({rsi:.0f}) 比前一波高點 ({prev_peak['rsi'].max():.0f}) 弱 → 頂背離"))
        if len(prev_peak) and x["osc"] < prev_peak["osc"].max() * 0.6 and prev_peak["osc"].max() > 0:
            r.append((1, "背離", "指數接近前高，但 MACD 柱狀體比前一波弱 → 動能背離"))
    st = int(fx["streak"])
    if st >= 7:
        r.append((2, "連漲跌", f"連漲 {st} 天"))
    elif st >= 5:
        r.append((1, "連漲跌", f"連漲 {st} 天"))
    elif st <= -5:
        r.append((-1, "連漲跌", f"連跌 {-st} 天，短線跌多"))
    # 量價
    rng = x["high"] - x["low"]
    if rng > 0 and (x["high"] - max(x["open"], x["close"])) / rng > 0.5:
        r.append((1, "K線", "今天留長上影線，上方賣壓重"))
    if x["close"] < x["open"] and (x["open"] - x["close"]) > 1.5 * d["atr"].iloc[-2]:
        r.append((1.5, "K線", "收長黑 K"))
    # 均線
    if x["close"] < x["ma20"]:
        r.append((2, "均線", f"跌破月線 {x['ma20']:,.0f}，短期趨勢轉弱"))
    elif x["close"] < x["ma5"] and d["close"].iloc[-2] >= d["ma5"].iloc[-2]:
        r.append((1, "均線", f"今天跌破 5 日線 {x['ma5']:,.0f}，短線轉弱的第一個訊號"))
    elif x["close"] > x["ma5"] > x["ma10"] > x["ma20"]:
        r.append((-0.5, "均線", "均線多頭排列，趨勢仍向上"))
    # 位置：離壓力 / 前高
    close = float(x["close"])
    res = [z for z in lv_s["resistances"] + lv_l["resistances"] if z["price"] > close]
    if res:
        near = min(res, key=lambda z: z["price"])
        dist = (near["price"] / close - 1) * 100
        if dist <= 1.5:
            r.append((1, "位置", f"離壓力 {near['price']:,.0f} 只剩 {dist:.1f}% ({'、'.join(near['tags'][:3])})"))
    elif fx["dist_high"] >= -0.005:
        r.append((0, "位置", "指數在區間最高點附近，上方沒有套牢壓力 (強勢，但回測時沒有參考的壓力價)"))
    sup = [z for z in lv_s["supports"] if z["price"] < close]
    if sup:
        ns = max(sup, key=lambda z: z["price"])
        if (1 - ns["price"] / close) * 100 <= 1:
            r.append((-0.5, "位置", f"下方支撐 {ns['price']:,.0f} 很近，跌到有撐"))
    # 籌碼
    try:
        mi = data.market_institutional(days=15)
        if len(mi):
            f3 = float(mi["外資"].tail(3).sum())
            if f3 <= -300:
                r.append((1.5, "籌碼", f"外資近 3 日賣超現貨 {abs(f3):,.0f} 億"))
            elif f3 >= 300:
                r.append((-1, "籌碼", f"外資近 3 日買超現貨 {f3:,.0f} 億"))
            else:
                r.append((0, "籌碼", f"外資近 3 日現貨 {f3:+,.0f} 億"))
    except Exception as e:  # noqa: BLE001
        notes.append(f"外資現貨資料取得失敗：{e}")
    try:
        start = (_now().date() - dt.timedelta(days=20)).isoformat()
        fo = data.finmind("TaiwanFuturesInstitutionalInvestors", data_id="TX", start_date=start)
        fo = fo[fo["institutional_investors"] == "外資"].copy()
        if len(fo) >= 4:
            fo["net"] = fo["long_open_interest_balance_volume"] - fo["short_open_interest_balance_volume"]
            fo = fo.sort_values("date")
            net, chg3 = int(fo["net"].iloc[-1]), int(fo["net"].iloc[-1] - fo["net"].iloc[-4])
            txt = f"外資台指期淨留倉 {net:+,} 口 (近 3 日 {chg3:+,} 口)"
            if chg3 <= -5000:
                r.append((1.5, "籌碼", txt + "，空單大增 / 多單大減"))
            elif chg3 >= 5000:
                r.append((-1, "籌碼", txt + "，多單增加"))
            else:
                r.append((0.5 if net < -20000 else 0, "籌碼", txt))
    except Exception as e:  # noqa: BLE001
        notes.append(f"外資期貨留倉資料取得失敗：{e}")
    try:
        start = (_now().date() - dt.timedelta(days=15)).isoformat()
        mg = data.finmind("TaiwanStockTotalMarginPurchaseShortSale", start_date=start)
        mg = mg[mg["name"] == "MarginPurchase"].sort_values("date")
        if len(mg) >= 6:
            chg = mg["TodayBalance"].iloc[-1] / mg["TodayBalance"].iloc[-6] - 1
            if chg >= 0.015 and fx["ret5"] > 0:
                r.append((1, "籌碼", f"融資 5 日增加 {chg * 100:.1f}%，散戶追高"))
            elif chg <= -0.015:
                r.append((-0.5, "籌碼", f"融資 5 日減少 {abs(chg) * 100:.1f}%，籌碼沉澱"))
    except Exception as e:  # noqa: BLE001
        notes.append(f"融資資料取得失敗：{e}")
    # 期貨
    try:
        from .realtime import futures
        fu = futures()
        cur = fu.get("current")
        if cur and cur.get("price"):
            b = fu.get("basis")
            if fu["session"] == "night" and cur.get("chg_pct") is not None:
                if cur["chg_pct"] <= -0.5:
                    r.append((1, "期貨", f"台指期夜盤 {cur['chg_pct']:+.2f}%，明天開盤偏弱"))
                elif cur["chg_pct"] >= 0.5:
                    r.append((-0.5, "期貨", f"台指期夜盤 {cur['chg_pct']:+.2f}%"))
            elif b is not None and b / close * 100 <= -0.3:
                r.append((1, "期貨", f"台指期逆價差 {b:+.0f} 點，期貨看淡"))
    except Exception:  # noqa: BLE001
        pass
    return r


def _sigmoid(x):
    return 1 / (1 + math.exp(-x))


def market_risk() -> dict:
    notes = []
    raw, night = _night_implied(_long_daily())
    if night:
        notes.insert(0, f"夜盤推估：台指期夜盤 {night['tx_night']:,.0f} − 收盤價差 {night['basis']:+,.0f} 點 "
                        f"= 推估加權 {night['implied']:,.0f} ({night['implied_chg_pct']:+.2f}%)，以下指標用推估值當明天最新一根計算")
    d, f = _features(raw)
    out = _outcomes(d)
    close = float(d["close"].iloc[-1])
    lv_s = compute_levels(d, "short")
    lv_l = compute_levels(d, "long")

    # 歷史相似情況 (排除最近 10 天，結果還沒出來)
    hist = f.iloc[:-10].dropna()
    cur = f.iloc[-1][FEATS]
    mu, sd = hist[FEATS].mean(), hist[FEATS].std().replace(0, 1)
    z = (hist[FEATS] - mu) / sd
    zc = (cur - mu) / sd
    dist = np.sqrt(((z - zc) ** 2).sum(axis=1))
    near_idx = dist.nsmallest(K).index
    analog = _stats(out.loc[near_idx])
    base = _stats(out.loc[hist.index])
    analog_days = [{"date": t.strftime("%Y-%m-%d"), "dd5": round(float(out.loc[t, "dd5"]) * 100, 2)} for t in near_idx[:10]]

    rules = _rules(d, f, lv_s, lv_l, notes)
    rotation = None
    try:  # 外資資金轉向：賣半導體、買金融電信
        from .rotation import recent, signal
        fl = recent(10)
        pts, txt, summ = signal(fl)
        rules.append((pts, "資金轉向", txt))
        rotation = {"flows": [{k: x[k] for k in ("date", "semi", "def", "total")} for x in fl], "summary": summ,
                    "top_semi_sell": fl[-1]["top_semi_sell"] if fl else [], "top_def_buy": fl[-1]["top_def_buy"] if fl else []}
    except Exception as e:  # noqa: BLE001
        notes.append(f"外資產業資金流取得失敗：{e}")
    R = sum(p for p, _, _ in rules)
    p_dd = (analog["dd2_5d"] or 0) / 100
    risk = round(50 * p_dd + 50 * _sigmoid(0.45 * (R - 1)), 0)
    if risk >= 65:
        level, action, cut = "高", "建議減碼一半；跌破月線再出清", 0.5
    elif risk >= 50:
        level, action, cut = "偏高", "建議減碼約 1/3，或把停利拉近到 5 日線", 1 / 3
    elif risk >= 35:
        level, action, cut = "中", "可以續抱，但設好停利 (跌破 5 日線先減碼)", 0.0
    else:
        level, action, cut = "低", "續抱，回測到支撐反而是加碼機會", 0.0

    x = d.iloc[-1]
    targets = [{"name": "5 日線", "price": round(float(x["ma5"]), 0)}, {"name": "10 日線", "price": round(float(x["ma10"]), 0)},
               {"name": "月線 (20MA)", "price": round(float(x["ma20"]), 0)}]
    for s_ in lv_s["supports"][:2]:
        targets.append({"name": "短線支撐 (" + "、".join(s_["tags"][:2]) + ")", "price": s_["price"]})
    for s_ in lv_l["supports"][:1]:
        targets.append({"name": "中長線支撐 (" + "、".join(s_["tags"][:2]) + ")", "price": s_["price"]})
    targets = [dict(t, dist_pct=round((t["price"] / close - 1) * 100, 2)) for t in targets if t["price"] < close]
    targets.sort(key=lambda t: -t["price"])
    typical = round(close * (1 + (analog["median_dd5"] or 0) / 100), 0)

    # 套到國票等券商庫存
    portfolio = None
    try:
        from . import brokers
        mv = 0
        for a in brokers.accounts(False):
            s = brokers.account_summary(a["id"])
            if s and s.get("market_value"):
                mv += s["market_value"]
        if mv:
            portfolio = {"market_value": round(mv), "cut_ratio": round(cut, 2), "cut_amount": round(mv * cut)}
    except Exception:  # noqa: BLE001
        pass

    notes.append(f"歷史相似情況：從 {d.index[0]:%Y/%m} 起約 {base['n']} 個交易日中，找出指標狀態最像現在的 {K} 天統計")
    notes.append("這是風險評估不是保證；強勢多頭時乖離可以維持很久，建議搭配跌破 5 日線 / 月線的實際訊號再動作")
    res = {"time": f"{d.index[-1]:%Y-%m-%d}", "close": close, "risk": risk, "level": level, "action": action,
           "night": night,
           "rules": [{"pts": p, "cat": c, "text": t} for p, c, t in rules], "rule_sum": round(R, 1),
           "analog": analog, "base": base, "analog_days": analog_days, "targets": targets, "typical_pullback": typical,
           "levels": {"short": {"supports": lv_s["supports"], "resistances": lv_s["resistances"]},
                      "long": {"supports": lv_l["supports"], "resistances": lv_l["resistances"]}},
           "portfolio": portfolio, "notes": notes, "rotation": rotation,
           "feats": {k: round(float(v), 4) for k, v in f.iloc[-1][FEATS].items()}}
    res["chart"] = _chart(d, lv_s, lv_l, res)
    return res


def _chart(d, lv_s, lv_l, res):
    s = d.tail(120)
    x = np.arange(len(s))
    o, h, l, c = (s[k].values for k in ("open", "high", "low", "close"))
    col = np.where(c >= o, "#e53935", "#16a085")
    fig, ax = plt.subplots(figsize=(14, 6.6), dpi=96)
    ax.vlines(x, l, h, color=col, lw=0.9)
    ax.bar(x, np.maximum(np.abs(c - o), (h.max() - l.min()) * 0.0015), bottom=np.minimum(o, c), color=col, width=0.62)
    for m, mc, lab in (("ma5", "#f39c12", "5日線"), ("ma10", "#8e44ad", "10日線"), ("ma20", "#2980b9", "月線"), ("ma60", "#27ae60", "季線")):
        ax.plot(x, s[m].values, color=mc, lw=1.1, label=lab)
    right = len(s) + 1
    ys = []
    for z_ in lv_s["resistances"] + lv_l["resistances"]:
        ax.axhline(z_["price"], color="#e67e22", ls="--", lw=0.9)
        ys.append((z_["price"], f"壓力 {z_['price']:,.0f}", "#e67e22"))
    for z_ in lv_s["supports"] + lv_l["supports"][:1]:
        ax.axhline(z_["price"], color="#1e6fd9", ls="--", lw=0.9)
        ys.append((z_["price"], f"支撐 {z_['price']:,.0f}", "#1e6fd9"))
    tp = res["typical_pullback"]
    ax.axhspan(min(tp, res["close"]), res["close"], xmin=0.9, color="#ffcdd2", alpha=0.5)
    ys.append((tp, f"相似情況典型回檔 {tp:,.0f}", "#c62828"))
    ys.sort()
    lo_, hi_ = ax.get_ylim()
    gap = (hi_ - lo_) * 0.035
    last_y = -1e18
    for p, t, cc in ys:
        yy = max(p, last_y + gap)
        last_y = yy
        ax.annotate(t, (len(s) - 1, p), (right, yy), fontsize=8.5, color=cc, va="center", fontweight="bold",
                    arrowprops=dict(arrowstyle="-", color=cc, lw=0.6) if abs(yy - p) > 1e-6 else None)
    step = max(1, len(s) // 8)
    ax.set_xticks(x[::step])
    ax.set_xticklabels([t.strftime("%m/%d") for t in s.index[::step]], fontsize=9)
    ax.set_xlim(-1, len(s) + 22)
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left", fontsize=9, ncol=4)
    color = "#c62828" if res["risk"] >= 50 else ("#ef6c00" if res["risk"] >= 35 else "#2e7d32")
    tag = "  (最後一根 = 夜盤推估)" if res.get("night") else ""
    ax.set_title(f"加權指數日K{tag}　回測風險 {res['risk']:.0f} / 100 ({res['level']})　{res['action']}",
                 loc="left", fontsize=13, fontweight="bold", color=color)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()
