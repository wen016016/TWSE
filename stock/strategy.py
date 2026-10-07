"""進出場評分引擎

每個訊號給 +/- 分並附原因，分類：
  週K / 日K / 5分K / 1分K 技術面 (MA、MACD、KD、RSI、布林、量價)
  籌碼 (三大法人、融資融券、集保大戶)
  大盤 (加權指數趨勢、大盤法人)
總分換成建議：做多 / 偏多 / 觀望 / 偏空，並給進場區、停損、目標價。
"""
import datetime as dt

import numpy as np
import pandas as pd

from . import data
from .chart import plot
from .indicators import add_indicators, add_vwap, crossed_down, crossed_up, round_tick, tick
from .levels import clean, compute_levels, find_pivots, merge_for_plan, sessions


class Score:
    def __init__(self):
        self.items = []

    def add(self, cat, pts, text):
        self.items.append({"cat": cat, "pts": round(float(pts), 1) + 0.0, "text": text})

    @property
    def total(self):
        return round(sum(i["pts"] for i in self.items), 1) + 0.0

    def by_cat(self):
        out = {}
        for i in self.items:
            out[i["cat"]] = round(out.get(i["cat"], 0) + i["pts"], 1) + 0.0
        return out


# =================================================================== 技術面
def score_ma(sc, df, cat, a="ma5", b="ma10", c="ma20", e="ma60"):
    r = df.iloc[-1]
    cols = [x for x in (a, b, c, e) if x in df and pd.notna(r[x])]
    vals = [r[x] for x in cols]
    names = "、".join(x.upper() for x in cols)
    if len(vals) >= 3 and all(vals[i] > vals[i + 1] for i in range(len(vals) - 1)):
        sc.add(cat, 1.5, f"均線多頭排列 ({names} 由上而下)")
    elif len(vals) >= 3 and all(vals[i] < vals[i + 1] for i in range(len(vals) - 1)):
        sc.add(cat, -1.5, f"均線空頭排列 ({names} 由下而上)")
    if c in df and pd.notna(r[c]):
        slope = r[c] - df[c].iloc[-4]
        if r["close"] > r[c]:
            sc.add(cat, 1 if slope > 0 else 0.5, f"收盤 {r['close']:.2f} 站上 {c.upper()} {r[c]:.2f}" + ("，且均線上揚" if slope > 0 else "，但均線仍下彎"))
        else:
            sc.add(cat, -1 if slope < 0 else -0.5, f"收盤 {r['close']:.2f} 跌破 {c.upper()} {r[c]:.2f}" + ("，且均線下彎" if slope < 0 else ""))


def score_macd(sc, df, cat, w=1.0):
    r, p = df.iloc[-1], df.iloc[-2]
    if crossed_up(df["dif"], df["dea"], 3):
        sc.add(cat, 1.2 * w, f"MACD 黃金交叉 (DIF {r['dif']:.2f} 上穿 DEA {r['dea']:.2f})" + ("，且在零軸上" if r["dif"] > 0 else "，零軸下 (反彈性質)"))
    elif crossed_down(df["dif"], df["dea"], 3):
        sc.add(cat, -1.2 * w, f"MACD 死亡交叉 (DIF {r['dif']:.2f} 下穿 DEA {r['dea']:.2f})")
    elif r["osc"] > 0:
        sc.add(cat, (0.6 if r["osc"] > p["osc"] else 0.2) * w, "MACD 柱狀體為紅" + ("且放大" if r["osc"] > p["osc"] else "但縮小，動能減弱"))
    else:
        sc.add(cat, (-0.6 if r["osc"] < p["osc"] else -0.2) * w, "MACD 柱狀體為綠" + ("且擴大" if r["osc"] < p["osc"] else "但收斂，跌勢減緩"))


def score_kd(sc, df, cat, w=1.0):
    r = df.iloc[-1]
    if crossed_up(df["k"], df["d"], 3):
        pts = 1.2 if r["k"] < 50 else 0.6
        sc.add(cat, pts * w, f"KD 黃金交叉 (K {r['k']:.0f} / D {r['d']:.0f})" + ("，低檔交叉較可靠" if r["k"] < 50 else "，位置偏高"))
    elif crossed_down(df["k"], df["d"], 3):
        pts = -1.2 if r["k"] > 50 else -0.6
        sc.add(cat, pts * w, f"KD 死亡交叉 (K {r['k']:.0f} / D {r['d']:.0f})" + ("，高檔交叉轉弱" if r["k"] > 50 else ""))
    elif r["k"] > 80 and df["k"].tail(3).min() > 80:
        sc.add(cat, 0.3 * w, f"KD 高檔鈍化 (K {r['k']:.0f})，強勢但追高風險大")
    elif r["k"] < 20:
        sc.add(cat, -0.3 * w, f"KD 低檔 (K {r['k']:.0f})，弱勢，等交叉再說")
    else:
        sc.add(cat, (0.3 if r["k"] > r["d"] else -0.3) * w, f"KD {'K>D 偏多' if r['k'] > r['d'] else 'K<D 偏空'} (K {r['k']:.0f} / D {r['d']:.0f})")


def score_rsi(sc, df, cat, w=1.0):
    v = df["rsi"].iloc[-1]
    if v >= 80:
        sc.add(cat, -0.8 * w, f"RSI {v:.0f} 過熱，容易拉回")
    elif v >= 55:
        sc.add(cat, 0.5 * w, f"RSI {v:.0f} 多方區")
    elif v <= 20:
        sc.add(cat, 0.3 * w, f"RSI {v:.0f} 超賣，有反彈機會但趨勢仍弱")
    elif v <= 45:
        sc.add(cat, -0.5 * w, f"RSI {v:.0f} 空方區")
    else:
        sc.add(cat, 0, f"RSI {v:.0f} 中性")


def score_bbands(sc, df, cat, w=1.0):
    r = df.iloc[-1]
    if pd.isna(r["bb_up"]):
        return
    width = (r["bb_up"] - r["bb_dn"]) / r["bb_mid"]
    width_prev = ((df["bb_up"] - df["bb_dn"]) / df["bb_mid"]).iloc[-20:-1].mean()
    squeeze_open = width > width_prev * 1.2
    if r["close"] > r["bb_up"]:
        sc.add(cat, (0.8 if squeeze_open else 0.2) * w,
               f"突破布林上軌 {r['bb_up']:.2f}" + ("，通道開口擴大 → 噴出行情" if squeeze_open else "，通道未開，小心假突破"))
    elif r["close"] < r["bb_dn"]:
        sc.add(cat, (-0.8 if squeeze_open else -0.2) * w,
               f"跌破布林下軌 {r['bb_dn']:.2f}" + ("，通道開口擴大 → 殺盤" if squeeze_open else "，有乖離反彈可能"))
    elif r["close"] >= r["bb_mid"]:
        sc.add(cat, 0.4 * w, f"位於布林中軌 {r['bb_mid']:.2f} 之上 (偏多)")
    else:
        sc.add(cat, -0.4 * w, f"位於布林中軌 {r['bb_mid']:.2f} 之下 (偏空)")
    if width < width_prev * 0.7:
        sc.add(cat, 0, f"布林通道收斂 (寬度 {width * 100:.1f}%)，即將變盤，注意突破方向")


def score_volume_price(sc, df, cat, look=20, w=1.0):
    """量價關係"""
    r, p = df.iloc[-1], df.iloc[-2]
    vr = r["volume"] / r["vma20"] if pd.notna(r["vma20"]) and r["vma20"] > 0 else 1
    hi = df["high"].iloc[-look - 1:-1].max()
    lo = df["low"].iloc[-look - 1:-1].min()
    chg = r["close"] - p["close"]
    if r["close"] > hi:
        sc.add(cat, (2 if vr > 1.5 else 0.8) * w, f"突破前 {look} 根高點 {hi:.2f}" + (f"，量增 {vr:.1f} 倍 (有效突破)" if vr > 1.5 else f"，但量只有 {vr:.1f} 倍 (量能不足)"))
    elif r["close"] < lo:
        sc.add(cat, (-2 if vr > 1.5 else -1) * w, f"跌破前 {look} 根低點 {lo:.2f}" + (f"，帶量 {vr:.1f} 倍" if vr > 1.5 else ""))
    elif chg > 0 and vr > 1.5:
        sc.add(cat, 0.8 * w, f"價漲量增 (量 {vr:.1f} 倍)，買盤積極")
    elif chg < 0 and vr > 2:
        sc.add(cat, -1 * w, f"價跌爆量 (量 {vr:.1f} 倍)，有出貨疑慮")
    elif chg > 0 and vr < 0.7:
        sc.add(cat, -0.2 * w, f"價漲量縮 (量 {vr:.1f} 倍)，追價意願不足")
    elif chg < 0 and vr < 0.7:
        sc.add(cat, 0.2 * w, f"價跌量縮 (量 {vr:.1f} 倍)，賣壓不重，屬健康回檔")
    # 長上影線
    rng = r["high"] - r["low"]
    if rng > 0 and (r["high"] - max(r["open"], r["close"])) / rng > 0.5 and vr > 1.3:
        sc.add(cat, -0.6 * w, "帶量長上影線，上方賣壓重")
    if rng > 0 and (min(r["open"], r["close"]) - r["low"]) / rng > 0.5 and vr > 1.3:
        sc.add(cat, 0.6 * w, "帶量長下影線，下檔有人承接")


def score_bias(sc, df, cat, ma="ma20", limit=12):
    r = df.iloc[-1]
    if ma not in df or pd.isna(r[ma]):
        return
    b = (r["close"] / r[ma] - 1) * 100
    if b > limit:
        sc.add(cat, -1, f"與 {ma.upper()} 正乖離 {b:.1f}% 過大，不宜追高，等拉回")
    elif b < -limit:
        sc.add(cat, 0.3, f"與 {ma.upper()} 負乖離 {b:.1f}% 過大，跌深有反彈機會")


def score_tech(sc, df, cat, w=1.0, look=20):
    score_ma(sc, df, cat)
    score_macd(sc, df, cat, w)
    score_kd(sc, df, cat, w)
    score_rsi(sc, df, cat, w)
    score_bbands(sc, df, cat, w)
    score_volume_price(sc, df, cat, look, w)


# =================================================================== 籌碼
def score_chips(sc, code, d, notes):
    avgv = float(d["vma20"].iloc[-1]) if pd.notna(d["vma20"].iloc[-1]) else 0
    chips = {}
    try:
        inst = data.institutional(code)
        if len(inst):
            last5 = inst.tail(5)
            chips["inst"] = [{"date": i.strftime("%m/%d"), **{k: round(float(v), 0) for k, v in r.items()}} for i, r in inst.tail(10).iterrows()]
            th = max(avgv * 5 * 0.02, 50)  # 5 日合計超過均量 2% 才算有意義
            for who, w in (("外資", 1.0), ("投信", 1.2)):
                s = float(last5[who].sum())
                if abs(s) >= th:
                    sc.add("籌碼", w if s > 0 else -w, f"{who}近5日{'買超' if s > 0 else '賣超'} {abs(s):,.0f} 張")
                else:
                    sc.add("籌碼", 0, f"{who}近5日 {s:+,.0f} 張 (中性)")
            streak, sign = 0, np.sign(inst["投信"].iloc[-1])
            for v in inst["投信"].values[::-1]:
                if np.sign(v) == sign and sign != 0:
                    streak += 1
                else:
                    break
            if streak >= 3:
                sc.add("籌碼", 0.8 * sign, f"投信連續{'買' if sign > 0 else '賣'}超 {streak} 天")
            fstreak, fsign = 0, np.sign(inst["外資"].iloc[-1])
            for v in inst["外資"].values[::-1]:
                if np.sign(v) == fsign and fsign != 0:
                    fstreak += 1
                else:
                    break
            if fstreak >= 4:
                sc.add("籌碼", 0.6 * fsign, f"外資連續{'買' if fsign > 0 else '賣'}超 {fstreak} 天")
    except Exception as e:
        notes.append(f"法人資料取得失敗：{e}")

    try:
        mg = data.margin(code)
        if len(mg) > 6:
            m_chg = float(mg["融資餘額"].iloc[-1] - mg["融資餘額"].iloc[-6])
            p_chg = float(d["close"].iloc[-1] / d["close"].iloc[-6] - 1)
            chips["margin"] = {"融資餘額": int(mg["融資餘額"].iloc[-1]), "融資5日增減": int(m_chg),
                               "融券餘額": int(mg["融券餘額"].iloc[-1])}
            if m_chg > 0 and p_chg < 0:
                sc.add("籌碼", -0.6, f"股價5日跌、融資增 {m_chg:,.0f} 張 → 散戶接刀，籌碼凌亂")
            elif m_chg < 0 and p_chg > 0:
                sc.add("籌碼", 0.6, f"股價5日漲、融資減 {abs(m_chg):,.0f} 張 → 籌碼沉澱，主力吃貨")
            elif m_chg > 0 and p_chg > 0.05:
                sc.add("籌碼", -0.3, f"融資跟著追價增 {m_chg:,.0f} 張，散戶過熱")
            ss = mg["融券餘額"]
            if ss.iloc[-1] > ss.iloc[-6] * 1.3 and p_chg > 0:
                sc.add("籌碼", 0.4, "融券增加且股價上漲，有軋空題材")
    except Exception as e:
        notes.append(f"融資券資料取得失敗：{e}")

    try:
        bh = data.big_holders(code)
        if bh:
            chips["big"] = bh
            if "chg_big400" in bh:
                c4 = bh["chg_big400"]
                if abs(c4) >= 0.3:
                    sc.add("籌碼", 1 if c4 > 0 else -1, f"集保400張大戶持股 {bh['big400']:.1f}%，較前週 {c4:+.2f}% → 大戶{'加碼' if c4 > 0 else '減碼'}")
                else:
                    sc.add("籌碼", 0, f"集保400張大戶持股 {bh['big400']:.1f}%，較前週 {c4:+.2f}% (持平)")
                if bh.get("chg_retail", 0) > 0.5 and c4 < 0:
                    sc.add("籌碼", -0.5, "散戶增加、大戶減少 → 籌碼從大戶流向散戶")
            else:
                sc.add("籌碼", 0, f"集保400張大戶持股 {bh['big400']:.1f}%，千張大戶 {bh['big1000']:.1f}% (系統每週累積資料，下週起可比較增減)")
    except Exception as e:
        notes.append(f"集保大戶資料取得失敗：{e}")
    return chips


def score_flow(sc, code, notes, w=1.0):
    """成交明細：大戶單 vs 散戶單 (外盤買 / 內盤賣)"""
    from .flow import analyze_flow
    try:
        f = analyze_flow(code)
    except Exception as e:  # noqa: BLE001
        notes.append(f"成交明細分析失敗：{e}")
        return None
    if not f["ok"]:
        notes.append(f["msg"])
        return f
    sc.add("大戶成交", f["score"] * w, f"{f['verdict']}：{f['text']} ({f['date']})")
    return f


def limit_notes(code, notes):
    try:
        from .realtime import quote
        q = quote(code)
    except Exception:  # noqa: BLE001
        return None
    if q["limit_up"] and q["price"] >= q["limit_up"]:
        notes.append(f"目前漲停 {q['limit_up']}" + ("，委賣已空 (漲停鎖住)，買不到" if not q["ask"] else "，漲停打開中"))
    if q["limit_down"] and q["price"] <= q["limit_down"]:
        notes.append(f"目前跌停 {q['limit_down']}" + ("，委買已空 (跌停鎖住)，賣不掉" if not q["bid"] else ""))
    return q


# =================================================================== 大盤
def market_view(intraday=False) -> tuple[Score, dict]:
    sc = Score()
    m = add_indicators(data.get_kline(data.INDEX, "1d"))
    r = m.iloc[-1]
    info = {"close": round(float(r["close"]), 2), "chg_pct": round(float(r["close"] / m["close"].iloc[-2] - 1) * 100, 2),
            "ma20": round(float(r["ma20"]), 2), "ma60": round(float(r["ma60"]), 2)}
    if r["close"] > r["ma20"]:
        sc.add("大盤", 1, f"加權指數 {r['close']:.0f} 站上月線 {r['ma20']:.0f}")
    else:
        sc.add("大盤", -1, f"加權指數 {r['close']:.0f} 跌破月線 {r['ma20']:.0f}")
    if r["close"] < r["ma60"]:
        sc.add("大盤", -1, f"加權指數跌破季線 {r['ma60']:.0f}，系統性風險高")
    score_macd(sc, m, "大盤", 0.6)
    score_kd(sc, m, "大盤", 0.5)
    try:
        mi = data.market_institutional()
        if len(mi):
            f3 = float(mi["外資"].tail(3).sum())
            info["foreign_3d"] = round(f3, 1)
            if abs(f3) > 50:
                sc.add("大盤", 0.8 if f3 > 0 else -0.8, f"外資近3日{'買超' if f3 > 0 else '賣超'}大盤 {abs(f3):.0f} 億")
    except Exception:
        pass
    try:
        from .realtime import futures
        f = futures()
        cur = f.get("current")
        if cur and cur.get("price"):
            info["futures"] = {"session": f["session"], "price": cur["price"], "chg_pct": cur["chg_pct"], "basis": f.get("basis")}
            if f["session"] == "night" and cur.get("chg_pct") is not None and abs(cur["chg_pct"]) >= 0.5:
                sc.add("大盤", 0.6 if cur["chg_pct"] > 0 else -0.6,
                       f"台指期夜盤 {cur['price']:.0f} ({cur['chg_pct']:+.2f}%)，預告明天開盤{'偏強' if cur['chg_pct'] > 0 else '偏弱'}")
            b = f.get("basis")
            if b is not None and abs(b) >= r["close"] * 0.003:
                sc.add("大盤", 0.3 if b > 0 else -0.3, f"台指期{'正' if b > 0 else '逆'}價差 {b:+.0f} 點，期貨{'看多' if b > 0 else '看空'}")
    except Exception:  # noqa: BLE001
        pass
    if intraday:
        try:
            i5 = add_indicators(data.get_kline(data.INDEX, "5m"), mas=(5, 20))
            today = i5[i5.index.date == i5.index[-1].date()]
            last = today.iloc[-1]
            day_open = today["open"].iloc[0]
            info["intraday_chg"] = round(float(last["close"] / m["close"].iloc[-2 if m.index[-1].date() == today.index[-1].date() else -1] - 1) * 100, 2)
            if last["close"] > day_open and last["close"] > last["ma20"]:
                sc.add("大盤", 1, "盤中指數在開盤價與5分K 20MA之上，大盤偏多")
            elif last["close"] < day_open and last["close"] < last["ma20"]:
                sc.add("大盤", -1, "盤中指數在開盤價與5分K 20MA之下，大盤偏空")
            else:
                sc.add("大盤", 0, "盤中指數震盪")
        except Exception:
            pass
    info["score"] = sc.total
    info["items"] = sc.items
    return sc, info


# =================================================================== 計畫
def plan_long(close, atr, lv, max_stop=0.08, min_stop=0.0, max_t_atr=None, cap=None):
    """max_t_atr：目標超過幾倍 ATR 就改用 ATR 推估 (短線用)；cap：價格上限 (當沖 = 漲停價)"""
    sup = [s["price"] for s in lv["supports"]]
    res = [r["price"] for r in lv["resistances"] if r["price"] - close >= 0.7 * atr]  # 太近的壓力不當目標
    bull = lv["key"].get("bull", {}).get("price")
    base = sup[0] if sup else close - 2 * atr
    if bull and bull < close and bull > base - atr:
        base = min(base, bull)
    stop = base - 0.3 * atr
    stop = max(stop, close * (1 - max_stop))
    if min_stop:
        stop = min(stop, close * (1 - min_stop))
    t1 = res[0] if res else close + 2 * atr
    t2 = res[1] if len(res) > 1 else max(t1 + atr, close + 3.5 * atr)
    if max_t_atr and t1 - close > max_t_atr * atr:
        t1, t2 = close + 2 * atr, min(t1, close + 3.5 * atr)
    note = None
    if cap:
        if close >= cap:
            note = f"已漲停 {cap}，不追價"
        t1, t2 = min(t1, cap), min(t2, cap)
    buy_lo = sup[0] if sup and sup[0] > close - 1.2 * atr else close - 0.5 * atr
    rr = (t1 - close) / (close - stop) if close > stop else 0
    near_res = lv["resistances"][0]["price"] if lv["resistances"] else None
    brk = round_tick(near_res + tick(near_res), "up") if near_res else None
    if brk and cap and brk > cap:
        brk = None
    return {"side": "做多", "entry_zone": [round_tick(buy_lo, "up"), round_tick(close)], "breakout": brk,
            "stop": round_tick(stop, "down"), "target1": round_tick(t1), "target2": round_tick(t2),
            "risk_pct": round((1 - stop / close) * 100, 2), "reward_pct": round((t1 / close - 1) * 100, 2),
            "rr": round(rr, 2), "note": note}


def plan_short(close, atr, lv, max_stop=0.08, min_stop=0.0, max_t_atr=None, cap=None):
    """cap：價格下限 (當沖 = 跌停價)"""
    sup = [s["price"] for s in lv["supports"] if close - s["price"] >= 0.7 * atr]
    res = [r["price"] for r in lv["resistances"]]
    base = res[0] if res else close + 2 * atr
    stop = min(base + 0.3 * atr, close * (1 + max_stop))
    if min_stop:
        stop = max(stop, close * (1 + min_stop))
    t1 = sup[0] if sup else close - 2 * atr
    t2 = sup[1] if len(sup) > 1 else min(t1 - atr, close - 3.5 * atr)
    if max_t_atr and close - t1 > max_t_atr * atr:
        t1, t2 = close - 2 * atr, max(t1, close - 3.5 * atr)
    note = None
    if cap:
        if close <= cap:
            note = f"已跌停 {cap}，不追空"
        t1, t2 = max(t1, cap), max(t2, cap)
    sell_hi = res[0] if res and res[0] < close + 1.2 * atr else close + 0.5 * atr
    rr = (close - t1) / (stop - close) if stop > close else 0
    near_sup = lv["supports"][0]["price"] if lv["supports"] else None
    brk = round_tick(near_sup - tick(near_sup), "down") if near_sup else None
    if brk and cap and brk < cap:
        brk = None
    return {"side": "做空", "entry_zone": [round_tick(close), round_tick(sell_hi, "down")], "breakout": brk,
            "stop": round_tick(stop, "up"), "target1": round_tick(t1), "target2": round_tick(t2),
            "risk_pct": round((stop / close - 1) * 100, 2), "reward_pct": round((1 - t1 / close) * 100, 2),
            "rr": round(rr, 2), "note": note}


def verdict(total, strong=6, mild=3):
    if total >= strong:
        return "強烈偏多", "適合進場做多"
    if total >= mild:
        return "偏多", "可小量分批試單，拉回支撐再加碼"
    if total <= -strong:
        return "強烈偏空", "不宜做多；持股者應出場，可考慮放空"
    if total <= -mild:
        return "偏空", "不宜進場；持股者減碼或嚴守停損"
    return "中性", "觀望，等待方向明確"


def exit_check(close, entry, side, plan, lv, d, mode):
    """已持有部位的出場判斷"""
    sig = []
    pnl = (close / entry - 1) * 100 * (1 if side == "long" else -1)
    bull = lv["key"].get("bull", {}).get("price")
    bear = lv["key"].get("bear", {}).get("price")
    r = d.iloc[-1]
    if side == "long":
        if bull and close < bull:
            sig.append(("出場", f"跌破多方關鍵價 {bull}"))
        if mode == "swing" and pd.notna(r.get("ma20")) and close < r["ma20"] and r["volume"] > r["vma20"]:
            sig.append(("減碼", f"帶量跌破月線 {r['ma20']:.2f}"))
        if crossed_down(d["dif"], d["dea"], 2):
            sig.append(("減碼", "MACD 死亡交叉"))
        if lv["resistances"] and close >= lv["resistances"][0]["price"] * 0.995:
            sig.append(("停利", f"接近壓力 {lv['resistances'][0]['price']}，可先分批停利"))
        if d["rsi"].iloc[-1] > 80:
            sig.append(("停利", "RSI 過熱"))
    else:
        if bear and close > bear:
            sig.append(("出場", f"站上空方關鍵價 {bear}，空單回補"))
        if crossed_up(d["dif"], d["dea"], 2):
            sig.append(("減碼", "MACD 黃金交叉"))
        if lv["supports"] and close <= lv["supports"][0]["price"] * 1.005:
            sig.append(("停利", f"接近支撐 {lv['supports'][0]['price']}，空單可先回補"))
    if mode == "swing" and pnl <= -8:
        sig.append(("出場", f"虧損 {pnl:.1f}% 超過 8% 停損上限"))
    if mode == "daytrade" and pnl <= -1.5:
        sig.append(("出場", f"當沖虧損 {pnl:.1f}% 超過 1.5%"))
    action = "續抱"
    if any(s[0] == "出場" for s in sig):
        action = "建議出場"
    elif any(s[0] in ("減碼", "停利") for s in sig):
        action = "建議減碼 / 分批停利"
    return {"entry": entry, "side": side, "pnl_pct": round(pnl, 2), "action": action,
            "signals": [{"type": a, "text": b} for a, b in sig]}


def _now():
    return dt.datetime.now(dt.timezone(dt.timedelta(hours=8)))


DAILY_MAS = (5, 10, 20, 60, 120, 240)
INTRA_MAS = (5, 10, 20, 60)


def _swing_levels(d, w):
    """波段：短線 (日K近40根) + 中長線 (日K近一年 + 週K轉折)"""
    wk = w.tail(150)
    hi, lo = find_pivots(wk, 3)
    wk_extra = [(float(wk["high"].iloc[i]), "週線高點") for i in hi] + [(float(wk["low"].iloc[i]), "週線低點") for i in lo]
    return compute_levels(d, "short"), compute_levels(d, "long", extra=wk_extra)


def _swing_plans(d, lv_s, lv_l, bear_short=False, bear_long=False):
    """短線計畫：停損看短線支撐、目標可看到中長線壓力；中長線計畫：停損放寬 (2倍ATR、最多15%)"""
    close, atr = float(d["close"].iloc[-1]), float(d["atr"].iloc[-1])
    lv_plan = merge_for_plan(lv_s, lv_l)
    ps = (plan_short if bear_short else plan_long)(close, atr, lv_plan, max_stop=0.08, max_t_atr=4)
    pl = (plan_short if bear_long else plan_long)(close, atr * 2, lv_l, max_stop=0.15)
    ps["horizon"], pl["horizon"] = "短線", "中長線"
    return ps, pl


def _short_chart(d, lv_s, plan, title):
    return plot(d, title, "1d", [(lv_s, "main", True)], lv_s, bars=60, mas=("ma5", "ma10", "ma20"), plan=plan)


def _long_chart(d, lv_l, plan, title):
    return plot(d, title, "1d", [(lv_l, "main", True)], lv_l, bars=250,
                mas=("ma20", "ma60", "ma120", "ma240"), plan=plan)


def _weekly_chart(w, lv_w, title, plan=None):
    return plot(w, title, "1wk", [(lv_w, "main", True)], lv_w, bars=104, mas=("ma5", "ma10", "ma20", "ma60"), plan=plan)


def _intraday_levels(d, m5, m1):
    """當沖：5分K 只看「前一日 + 今日」，1分K 只看今日；再帶入昨高低收、VWAP、開盤區間、日線短線支撐壓力"""
    s5 = sessions(m5, 2)
    s1 = sessions(m1, 1)
    session = s1.index[-1].date()
    prev = d.iloc[-2] if d.index[-1].date() == session else d.iloc[-1]
    c = float(s1["close"].iloc[-1])
    vwap = float(s1["vwap"].iloc[-1])
    extra = [(float(prev["high"]), "昨高"), (float(prev["low"]), "昨低"), (float(prev["close"]), "昨收"), (vwap, "VWAP")]
    or_h = or_l = None
    if len(s1) >= 15:
        or_h, or_l = float(s1["high"].iloc[:15].max()), float(s1["low"].iloc[:15].min())
        extra += [(or_h, "開盤區間高"), (or_l, "開盤區間低")]
    lvd = compute_levels(d, "short")
    extra += [(s["price"], "日線支撐") for s in lvd["supports"] if s["price"] >= c * 0.96]
    extra += [(r["price"], "日線壓力") for r in lvd["resistances"] if r["price"] <= c * 1.04]
    lv5 = compute_levels(s5, "5m", extra=extra)
    lv1 = compute_levels(s1, "1m", extra=extra)
    lv5["why"].insert(0, "當沖只看「前一日 + 今日」兩天的 5分K：更早的盤中轉折對今天影響很小，"
                         "改用日線級別的支撐壓力 (±4% 內) 和昨高、昨低、昨收代表前幾天的關鍵價。")
    lv1["why"].insert(0, "1分K 只看今日，用來抓進場點；昨高低收、VWAP、開盤區間、日線支撐壓力一併列入。")
    return {"s5": s5, "s1": s1, "lv5": lv5, "lv1": lv1, "lvd": lvd, "prev": prev, "or": (or_h, or_l)}


def _intraday_chart(df, lv, tf, title, plan=None):
    return plot(df, title, tf, [(lv, "main", True)], lv, show_vwap=True, mas=("ma5", "ma20", "ma60"),
                day_split=True, plan=plan)


def _intraday_plan(close, atr5, lv, bear=False, code=None):
    """當沖計畫：目標價不超過今日漲跌停"""
    cap = None
    if code:
        try:
            from .realtime import quote
            q = quote(code)
            cap = q["limit_down"] if bear else q["limit_up"]
        except Exception:  # noqa: BLE001
            pass
    p = (plan_short if bear else plan_long)(close, atr5 * 2, lv, max_stop=0.015, min_stop=0.004, cap=cap)
    p["horizon"] = "當沖"
    return p


# =================================================================== 波段
def analyze_swing(code, entry=None, side="long", charts=True, with_chips=True):
    info = data.resolve(code)
    d = add_indicators(data.get_kline(code, "1d"), mas=DAILY_MAS)
    w = add_indicators(data.get_kline(code, "1wk"), mas=(5, 10, 20, 60))
    sc, notes = Score(), []

    score_tech(sc, w, "週K", w=1.2, look=13)
    score_tech(sc, d, "日K", w=1.0, look=20)
    score_bias(sc, d, "日K", "ma20", 12)
    chips = score_chips(sc, info["code"], d, notes) if with_chips else {}
    chips["flow"] = score_flow(sc, info["code"], notes, 0.6)
    limit_notes(info["code"], notes)
    msc, minfo = market_view()
    for i in msc.items:
        sc.add(i["cat"], i["pts"], i["text"])

    lv_s, lv_l = _swing_levels(d, w)
    lv_w = compute_levels(w, "1wk")
    close = float(d["close"].iloc[-1])
    total = sc.total
    tone, advice = verdict(total, 7, 3.5)
    weekly = sc.by_cat().get("週K", 0)
    if weekly < -2 and total > 0:
        advice += "（但週線偏空，屬反彈格局，部位要小、停損要快）"
    plan, plan_l = _swing_plans(d, lv_s, lv_l, bear_short=total <= -3.5, bear_long=weekly < -2)
    if plan["side"] == "做多" and plan["rr"] < 1.5 and total > 0:
        advice += f"；目前風報比 {plan['rr']} 偏低，建議等拉回 {plan['entry_zone'][0]} 附近再進"
    bl = lv_l["key"].get("bull", {}).get("price")
    if bl and plan["side"] == "做多":
        notes.append(f"中長線多方關鍵價 {bl}：跌破代表中期趨勢轉弱，波段部位應全數出場")

    title = f"{info['code']} {info['name']}"
    res = {"code": info["code"], "name": info["name"], "market": info["market"], "mode": "swing", "close": close,
           "time": d.index[-1].strftime("%Y-%m-%d"), "total": total, "tone": tone, "advice": advice,
           "by_cat": sc.by_cat(), "items": sc.items, "plan": plan, "plans": {"short": plan, "long": plan_l},
           "chips": chips, "market_info": minfo,
           "levels": {"short": clean(lv_s), "long": clean(lv_l), "1wk": clean(lv_w)}, "notes": notes}
    if entry:
        res["position"] = exit_check(close, float(entry), side, plan, lv_s, d, "swing")
    if charts:
        res["charts"] = {"short": _short_chart(d, lv_s, plan, f"{title}  短線波段 (日K 近60日)"),
                         "long": _long_chart(d, lv_l, plan_l, f"{title}  中長線波段 (日K 近一年)"),
                         "1wk": _weekly_chart(w, lv_w, f"{title}  週K", plan_l)}
    return res


# =================================================================== 當沖
def analyze_daytrade(code, entry=None, side="long", charts=True, with_chips=True):
    info = data.resolve(code)
    d = add_indicators(data.get_kline(code, "1d"), mas=DAILY_MAS)
    m5 = add_vwap(add_indicators(data.get_kline(code, "5m"), mas=INTRA_MAS))
    m1 = add_vwap(add_indicators(data.get_kline(code, "1m"), mas=INTRA_MAS))
    sc, notes = Score(), []

    session = m1.index[-1].date()
    t1 = m1[m1.index.date == session]
    prev = d.iloc[-2] if d.index[-1].date() == session else d.iloc[-1]
    dprev = d.iloc[:-1] if d.index[-1].date() == session else d

    # 適不適合當沖
    atrp = float(dprev["atr"].iloc[-1] / prev["close"] * 100)
    avgv = float(dprev["vma20"].iloc[-1])
    if avgv < 1000:
        sc.add("當沖條件", -2, f"20日均量 {avgv:,.0f} 張 < 1000 張，流動性不足，不適合當沖")
    elif avgv > 5000:
        sc.add("當沖條件", 0.5, f"20日均量 {avgv:,.0f} 張，流動性佳")
    else:
        sc.add("當沖條件", 0, f"20日均量 {avgv:,.0f} 張")
    if atrp < 1.5:
        sc.add("當沖條件", -1, f"日波動 (ATR) {atrp:.1f}% 太小，扣掉成本 (~0.3%) 獲利空間有限")
    elif atrp >= 2.5:
        sc.add("當沖條件", 0.5, f"日波動 (ATR) {atrp:.1f}%，有足夠空間")
    else:
        sc.add("當沖條件", 0, f"日波動 (ATR) {atrp:.1f}%")

    # 日K：方向
    score_ma(sc, dprev, "日K")
    score_macd(sc, dprev, "日K", 0.6)
    score_kd(sc, dprev, "日K", 0.6)
    pr = prev["high"] - prev["low"]
    if pr > 0:
        pos = (prev["close"] - prev["low"]) / pr
        if pos > 0.75:
            sc.add("日K", 0.5, "昨日收在高檔 (收最高附近)，多方延續機率高")
        elif pos < 0.25:
            sc.add("日K", -0.5, "昨日收在低檔，空方延續機率高")
    day_open = float(t1["open"].iloc[0])
    gap = (day_open / prev["close"] - 1) * 100
    if abs(gap) >= 2:
        sc.add("日K", 0, f"今日跳空{'開高' if gap > 0 else '開低'} {gap:+.1f}%，" + ("留意開高走低" if gap > 0 else "留意開低走高"))

    # 5分K：MA、MACD、KD、RSI、布林、量價、VWAP
    score_tech(sc, m5, "5分K", w=1.0, look=12)
    last5 = m5.iloc[-1]
    if last5["close"] > last5["vwap"]:
        sc.add("5分K", 1, f"價格 {last5['close']:.2f} 在 VWAP {last5['vwap']:.2f} 之上，今日買方成本有撐")
    else:
        sc.add("5分K", -1, f"價格 {last5['close']:.2f} 在 VWAP {last5['vwap']:.2f} 之下，今日進場者多數套牢")

    # 1分K：開盤區間、爆量、VWAP
    c1 = float(t1["close"].iloc[-1])
    if len(t1) >= 15:
        or_h, or_l = float(t1["high"].iloc[:15].max()), float(t1["low"].iloc[:15].min())
        if c1 > or_h:
            sc.add("1分K", 1.5, f"突破開盤15分鐘高點 {or_h:.2f}")
        elif c1 < or_l:
            sc.add("1分K", -1.5, f"跌破開盤15分鐘低點 {or_l:.2f}")
        else:
            sc.add("1分K", 0, f"仍在開盤區間 {or_l:.2f}~{or_h:.2f} 內")
    else:
        notes.append("開盤未滿 15 分鐘，開盤區間尚未成形")
    if len(t1) >= 10:
        rv = t1["volume"].tail(3).mean() / max(t1["volume"].mean(), 1e-9)
        mv = t1["close"].iloc[-1] - t1["open"].iloc[-3]
        if rv > 2:
            sc.add("1分K", 1 if mv > 0 else -1, f"近3分鐘爆量 {rv:.1f} 倍{'上攻' if mv > 0 else '下殺'}")
    score_macd(sc, m1, "1分K", 0.5)
    score_kd(sc, m1, "1分K", 0.4)
    if c1 > prev["high"]:
        sc.add("1分K", 1, f"突破昨高 {prev['high']:.2f}")
    elif c1 < prev["low"]:
        sc.add("1分K", -1, f"跌破昨低 {prev['low']:.2f}")

    chips = score_chips(sc, info["code"], dprev, notes) if with_chips else {}
    for it in sc.items:  # 當沖籌碼權重減半
        if it["cat"] == "籌碼":
            it["pts"] = round(it["pts"] * 0.5, 1) + 0.0
    chips["flow"] = score_flow(sc, info["code"], notes, 1.0)
    limit_notes(info["code"], notes)
    msc, minfo = market_view(intraday=True)
    for i in msc.items:
        sc.add(i["cat"], i["pts"], i["text"])

    it = _intraday_levels(d, m5, m1)
    lv5, lv1, lvd = it["lv5"], it["lv1"], it["lvd"]
    atr5 = float(m5["atr"].iloc[-1])
    total = sc.total
    tone, _ = verdict(total, 7, 4)
    cats = sc.by_cat()
    intraday = cats.get("5分K", 0) + cats.get("1分K", 0)
    if total >= 4 and intraday <= 0:
        total_gate = 0  # 日線偏多但盤中走弱 → 不追
        notes.append(f"總分 {total} 偏多，但盤中 5分K+1分K 合計 {intraday:+.1f} 分 (走弱)，等盤中轉強再做多")
    elif total <= -4 and intraday >= 0:
        total_gate = 0
        notes.append(f"總分 {total} 偏空，但盤中 5分K+1分K 合計 {intraday:+.1f} 分 (轉強)，等盤中轉弱再做空")
    else:
        total_gate = total
    plan = _intraday_plan(c1, atr5, lv5, bear=total_gate <= -4, code=info["code"])
    if total_gate >= 4:
        advice = "可做多當沖：拉回 VWAP 或支撐不破進場，跌破停損價立即出場"
    elif total_gate <= -4:
        advice = "可做空當沖 (需有現股當沖先賣資格)：反彈壓力不過進場，站上停損價立即回補"
    else:
        advice = "多空不明，觀望；等突破開盤區間或站穩 VWAP 再說 (圖上的計畫為條件成立時的參考價位)"
    if total_gate == 0 and total != 0:
        tone = "中性"
    if cats.get("當沖條件", 0) <= -2:
        advice = "這檔流動性 / 波動不適合當沖，建議換標的"

    now = _now()
    hm = now.hour * 100 + now.minute
    data_age = (now - m1.index[-1].to_pydatetime()).total_seconds() / 60
    if now.weekday() >= 5 or hm < 900 or hm > 1330:
        notes.append(f"目前非盤中，分析使用 {m1.index[-1]:%m/%d %H:%M} 的最後資料，開盤後請重新分析")
    else:
        if data_age > 5:
            notes.append(f"免費報價延遲約 {data_age:.0f} 分鐘，下單前務必以券商即時報價為準")
        if hm < 915:
            notes.append("開盤前 15 分鐘波動大，建議等開盤區間成形")
        if hm >= 1300:
            notes.append("13:00 後不建議新倉，所有當沖部位 13:20 前務必平倉")
    notes.append("當沖成本約 0.3% (手續費 0.1425%×2 依折扣、當沖證交稅 0.15%)，目標價要扣掉成本")

    title = f"{info['code']} {info['name']}"
    res = {"code": info["code"], "name": info["name"], "market": info["market"], "mode": "daytrade", "close": c1,
           "time": m1.index[-1].strftime("%Y-%m-%d %H:%M"), "total": total, "tone": tone, "advice": advice,
           "by_cat": sc.by_cat(), "items": sc.items, "plan": plan, "chips": chips, "market_info": minfo,
           "levels": {"5m": clean(lv5), "1m": clean(lv1), "short": clean(lvd)}, "notes": notes,
           "vwap": round(float(last5["vwap"]), 2)}
    if entry:
        res["position"] = exit_check(c1, float(entry), side, plan, lv5, m5, "daytrade")
    if charts:
        res["charts"] = {"5m": _intraday_chart(it["s5"], lv5, "5m", f"{title}  5分K 當沖 (前一日 + 今日)", plan),
                         "1m": _intraday_chart(it["s1"], lv1, "1m", f"{title}  1分K 當沖 (今日)", plan),
                         "1d": _short_chart(d, lvd, None, f"{title}  日K (短線，當沖看大方向)")}
    return res


def analyze(code, mode="swing", **kw):
    r = analyze_daytrade(code, **kw) if mode == "daytrade" else analyze_swing(code, **kw)
    r.setdefault("plans", {"daytrade": r["plan"]})
    for p in list(r["plans"].values()) + [r["plan"]]:
        p["text"] = _plan_text(p)
    return r


def _judge(df, name, lv, look=20):
    """用技術指標判斷支撐/壓力的有效性"""
    sc = Score()
    score_tech(sc, df, name, look=look)
    t = sc.total
    sup = lv["supports"][0]["price"] if lv["supports"] else None
    res = lv["resistances"][0]["price"] if lv["resistances"] else None
    if t >= 3:
        judge = f"指標偏多 ({t:+.1f} 分)：上方壓力 {res} 有機會被突破，回測支撐 {sup} 守住機率高，支撐附近可找買點。"
    elif t <= -3:
        judge = f"指標偏空 ({t:+.1f} 分)：下方支撐 {sup} 有被跌破的風險，反彈到壓力 {res} 容易遇到賣壓，不宜在支撐硬接。"
    else:
        judge = f"指標中性 ({t:+.1f} 分)：預期在 {sup} ~ {res} 區間震盪，靠近支撐偏買、靠近壓力偏賣。"
    return {"total": t, "items": sc.items, "judge": judge}


def _plan_text(p):
    long = p["side"] == "做多"
    e0, e1 = sorted(p["entry_zone"])
    t = [f"【{p['horizon']}{p['side']}】{'拉回買進區' if long else '反彈放空區'} {e0} ~ {e1}"]
    if p.get("breakout"):
        t.append(f"或{'突破' if long else '跌破'} {p['breakout']} {'追價買進' if long else '追價放空'}")
    t.append(f"停損 {p['stop']} ({'-' if long else '+'}{p['risk_pct']}%)")
    t.append(f"目標一 {p['target1']}、目標二 {p['target2']}，風報比 {p['rr']}")
    if p.get("note"):
        t.insert(0, f"⚠ {p['note']}")
    return "；".join(t)


def levels_report(code, tf="1d"):
    """支撐壓力查詢 + 圖 + 未來進出場建議
    1d  → 短線、中長線各一張；1wk → 週K；5m → 前一日+今日；1m → 今日"""
    info = data.resolve(code)
    title = f"{info['code']} {info['name']}"
    if tf == "1d":
        d = add_indicators(data.get_kline(code, "1d"), mas=DAILY_MAS)
        w = add_indicators(data.get_kline(code, "1wk"), mas=(5, 10, 20, 60))
        lv_s, lv_l = _swing_levels(d, w)
        ind = _judge(d, "日K", lv_s)
        ind_w = _judge(w, "週K", lv_l, look=13)
        lv_s["why"].append(ind["judge"])
        lv_l["why"].append("中長線看週K指標 → " + ind_w["judge"])
        ps, pl = _swing_plans(d, lv_s, lv_l, bear_short=ind["total"] <= -3, bear_long=ind_w["total"] <= -3)
        levels = {"short": clean(lv_s), "long": clean(lv_l)}
        plans = {"short": ps, "long": pl}
        charts = {"short": _short_chart(d, lv_s, ps, f"{title}  短線波段支撐壓力 (日K 近60日)"),
                  "long": _long_chart(d, lv_l, pl, f"{title}  中長線波段支撐壓力 (日K 近一年 + 週K轉折)")}
        last = d.index[-1]
    elif tf == "1wk":
        w = add_indicators(data.get_kline(code, "1wk"), mas=(5, 10, 20, 60))
        lv = compute_levels(w, "1wk")
        ind = _judge(w, "週K", lv, look=13)
        lv["why"].append(ind["judge"])
        close, atr = float(w["close"].iloc[-1]), float(w["atr"].iloc[-1])
        p = (plan_short if ind["total"] <= -3 else plan_long)(close, atr, lv, max_stop=0.15)
        p["horizon"] = "週線"
        levels, plans = {"1wk": clean(lv)}, {"1wk": p}
        charts = {"1wk": _weekly_chart(w, lv, f"{title}  週K 支撐壓力", p)}
        last = w.index[-1]
    else:
        d = add_indicators(data.get_kline(code, "1d"), mas=DAILY_MAS)
        m5 = add_vwap(add_indicators(data.get_kline(code, "5m"), mas=INTRA_MAS))
        m1 = add_vwap(add_indicators(data.get_kline(code, "1m"), mas=INTRA_MAS))
        it = _intraday_levels(d, m5, m1)
        src = it["s5"] if tf == "5m" else it["s1"]
        lv = it["lv5" if tf == "5m" else "lv1"]
        ind = _judge(m5 if tf == "5m" else m1, "5分K" if tf == "5m" else "1分K", lv, look=12)
        lv["why"].append(ind["judge"])
        p = _intraday_plan(float(src["close"].iloc[-1]), float(m5["atr"].iloc[-1]), it["lv5"], bear=ind["total"] <= -3, code=info["code"])
        levels, plans = {tf: clean(lv)}, {tf: p}
        charts = {tf: _intraday_chart(src, lv, tf, f"{title}  {'5分K 當沖 (前一日 + 今日)' if tf == '5m' else '1分K 當沖 (今日)'} 支撐壓力", p)}
        last = src.index[-1]
    for p in plans.values():
        p["text"] = _plan_text(p)
    first = next(iter(levels.values()))
    return {"code": info["code"], "name": info["name"], "tf": tf, "close": first["close"],
            "time": last.strftime("%Y-%m-%d %H:%M"), "levels": levels, "plans": plans, "indicators": ind, "charts": charts}
