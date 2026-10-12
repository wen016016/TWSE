"""箱型判定 + 兩種盤勢 (區間震盪 / 強勢趨勢) 的買賣點、停損、多空關鍵價

一、箱型
  1. 主要結構：近 N 根 (日K 20~60) 內，以 K 棒實體上下緣為主、常態影線可包含，
     單一長影線 (影線 > 1 倍 ATR) 剔除；箱頂 / 箱底必須「至少 2 根 K 棒」在 0.3 ATR 內重疊觸碰。
  2. ATR 修正：箱寬需介於 3 ~ 10 倍 ATR (ATR20 均值)，小於 3 倍 = 雜訊、大於 10 倍 = 失去區間意義；
     前後半段均價差不能超過箱寬 45% (排除單邊走勢)。
  3. 破局：收盤站上箱頂 / 跌破箱底，且隔天收盤沒有拉回箱內，才算箱型結束。
二、買賣點
  A 區間震盪 (股價在箱內)：買 箱底 ~ 箱底+0.5ATR、賣 箱頂-0.5ATR ~ 箱頂；停損 收盤跌破 箱底-0.5ATR
  B 強勢趨勢 (突破箱頂 / 均線多頭排列)：買 前箱頂回測 或 近期高點拉回 1~1.5ATR、貼近 5/10MA；
     停損 買進區下緣 -1~1.5ATR (參考近期波段低點)；目標 箱型等幅 1 倍 / 2 倍；收盤跌破 10MA/20MA 移動停利
  空方規則對稱。
"""
import numpy as np
import pandas as pd

from .indicators import round_tick, tick


def atr_mean(d: pd.DataFrame, n=20) -> float:
    pc = d["close"].shift(1)
    tr = pd.concat([d["high"] - d["low"], (d["high"] - pc).abs(), (d["low"] - pc).abs()], axis=1).max(axis=1)
    return float(tr.tail(n).mean())


def _edge(vals, tol, top):
    """最高 (或最低) 且至少 2 根觸碰的價位；最多略過 2 根單獨突出的 K 棒"""
    order = np.argsort(vals)[::-1] if top else np.argsort(vals)
    for i in order[:3]:
        v = vals[i]
        cnt = int(np.sum(np.abs(vals - v) <= tol))
        if cnt >= 2:
            return float(v), cnt
    return None


def _box_in(seg, atr, max_atr=10):
    o, c, h, l = (seg[k].values.astype(float) for k in ("open", "close", "high", "low"))
    btop, bbot = np.maximum(o, c), np.minimum(o, c)
    eh = np.where(h - btop > atr, btop, h)   # 單一長上影線 → 只用實體
    el = np.where(bbot - l > atr, bbot, l)   # 單一長下影線 → 只用實體
    tol = 0.3 * atr
    top, bot = _edge(eh, tol, True), _edge(el, tol, False)
    if not top or not bot:
        return None
    hi, nt = top
    lo, nb = bot
    width = hi - lo
    if width < 3 * atr or width > max_atr * atr:
        return None
    # 均價差：N 根平分前後兩段的收盤均值，|後半段 - 前半段| 必須 <= 箱寬 45%
    half = len(c) // 2
    first_mean, second_mean = float(c[:half].mean()), float(c[len(c) - half:].mean())
    if abs(second_mean - first_mean) > 0.45 * width:
        return None
    return {"hi": hi, "lo": lo, "touch_top": nt, "touch_bot": nb}


def find_box(d: pd.DataFrame, atr: float, windows=(60, 50, 40, 30, 20), max_excl=10, max_atr=10):
    """先找「已破局」的箱子 (箱子之後的收盤全在箱外)，再找目前仍在其中的箱子"""
    n = len(d)
    closes = d["close"].values.astype(float)
    wins = sorted({w for w in windows if w <= n}, reverse=True)

    def pack(b, w, excl, status):
        s, e = n - excl - w, n - excl
        return dict(b, window=w, status=status, confirmed=excl >= 2 or status == "range", after=excl,
                    start=d.index[s], end=d.index[e - 1], height=b["hi"] - b["lo"])

    for excl in range(1, min(max_excl, n) + 1):
        for w in wins:
            if n - excl - w < 0:
                continue
            b = _box_in(d.iloc[n - excl - w:n - excl], atr, max_atr)
            if not b:
                continue
            after = closes[n - excl:]
            if (after > b["hi"]).all():
                return pack(b, w, excl, "up")
            if (after < b["lo"]).all():
                return pack(b, w, excl, "down")
    for w in wins:
        b = _box_in(d.iloc[n - w:], atr, max_atr)
        if b:
            return pack(b, w, 0, "range")
    return None


def _trend(d, atr):
    """1 = 均線多頭排列且月線上揚；-1 = 空頭排列且月線下彎；0 = 扁平 / 糾結"""
    r = d.iloc[-1]
    try:
        ma5, ma10, ma20 = float(r["ma5"]), float(r["ma10"]), float(r["ma20"])
        slope = ma20 - float(d["ma20"].iloc[-6])
    except Exception:  # noqa: BLE001
        return 0
    if any(np.isnan(x) for x in (ma5, ma10, ma20, slope)):
        return 0
    close = float(r["close"])
    if ma5 > ma10 > ma20 and slope > 0.2 * atr and close > ma20:
        return 1
    if ma5 < ma10 < ma20 and slope < -0.2 * atr and close < ma20:
        return -1
    return 0


def _candle_signal(d, bottom=True):
    r = d.iloc[-1]
    o, c, h, l = (float(r[k]) for k in ("open", "close", "high", "low"))
    body = abs(c - o)
    shadow = (min(o, c) - l) if bottom else (h - max(o, c))
    if shadow >= max(body * 1.5, (h - l) * 0.4):
        return "長下影線" if bottom else "長上影線"
    if (c > o) if bottom else (c < o):
        return "紅K" if bottom else "黑K"
    return None


def _key(price, why):
    return {"price": round_tick(price), "half": None, "time": None, "_ts": None, "why": why}


REGIME_TXT = {"A": "情況 A：區間震盪盤", "B多": "情況 B：強勢趨勢盤 (多)", "B空": "情況 B：弱勢趨勢盤 (空)",
              "無箱型": "無有效箱型 (以波段支撐壓力為準)"}
STAGE = {"range": "箱內震盪", "up": "已突破箱頂", "down": "已跌破箱底"}
NO_ROOM = "現價已達區間邊緣壓力，無利潤空間，不開新倉"


def make_plan(d, lv, bear=False, atr=None, atr_n=20, windows=(60, 50, 40, 30, 20),
              cap=None, unit="根", ma_fast="ma10", ma_slow="ma20", set_keys=True, max_atr=10, tech=None,
              box=None, intraday_vol=False):
    """d：該週期 K 線 (含 ma5/ma10/ma20)；lv：compute_levels 結果 (無箱型時用它的支撐壓力，並改寫多空關鍵價)；
    atr：指定 ATR (當沖用 5分K ATR14)，否則用 d 的 ATR(atr_n) 均值；max_atr：箱寬上限 (短線 6、中長線 10)；
    tech：該週期技術指標分數 (確認買賣點可信度)；
    box：共用箱型 (整理偵測 detect() 的 shared_box)，給了就不重新找箱子；
    intraday_vol：當沖 / 台指期 → 突破K量需 > 前 3 根均量 2 倍，否則視為假突破"""
    close = float(d["close"].iloc[-1])
    atr = float(atr) if atr else atr_mean(d, atr_n)
    bx = dict(box) if box else find_box(d, atr, windows, max_atr=max_atr)
    fake_note = None
    if bx and intraday_vol and bx["status"] != "range" and bx["after"] < 2 and "volume" in d:
        i = len(d) - bx["after"]
        prev3 = float(d["volume"].iloc[max(0, i - 3):i].mean()) or 1
        vr3 = float(d["volume"].iloc[i]) / prev3
        if vr3 <= 2:
            fake_note = (f"{'突破' if bx['status'] == 'up' else '跌破'}K量只有前 3 根均量的 {vr3:.1f} 倍 (需 > 2 倍)，"
                         "視為假突破 / 假跌破，仍以箱型區間操作")
            bx = dict(bx, status="range", confirmed=False)
        else:
            fake_note = f"{'突破' if bx['status'] == 'up' else '跌破'}K量是前 3 根均量的 {vr3:.1f} 倍，帶量確認"
    tr = _trend(d, atr)
    r = d.iloc[-1]
    ma = {k: float(r[k]) for k in ("ma5", "ma10", "ma20", "ma60") if k in r and pd.notna(r[k])}
    look = d.tail(20)
    H, L = float(look["high"].max()), float(look["low"].min())
    swing_lo, swing_hi = float(d["low"].tail(10).min()), float(d["high"].tail(10).max())
    why = []
    sup = [s["price"] for s in lv.get("supports", [])]
    res = [x["price"] for x in lv.get("resistances", [])]

    if bx and bx["status"] == "range":
        regime = "A"
    elif bx and bx["status"] == "up":
        regime = "B多"
    elif bx and bx["status"] == "down":
        regime = "B空"
    elif tr > 0:
        regime = "B多"
    elif tr < 0:
        regime = "B空"
    else:
        regime = "無箱型"
    if bx:
        st = STAGE[bx["status"]]
        if bx["status"] != "range" and not bx["confirmed"]:
            st += " (待隔日收盤確認未拉回)"
        why.append(f"箱型 {round_tick(bx['lo'])} ~ {round_tick(bx['hi'])} (近 {bx['window']} {unit}，箱頂觸碰 {bx['touch_top']} 次、"
                   f"箱底 {bx['touch_bot']} 次，寬 {bx['height'] / atr:.1f} 倍 ATR，ATR {atr:.2f})：{st}")
        if box:
            why.append("箱型與「整理完成 / 準備啟動」共用同一個" + (f"：{box['source']}" if box.get("source") else ""))
        if fake_note:
            why.append(fake_note)
        if bx["status"] != "range" and "volume" in d and not intraday_vol:
            i = len(d) - bx["after"]
            vma = float(d["volume"].iloc[max(0, i - 20):i].mean()) or 1
            vr = float(d["volume"].iloc[i]) / vma
            why.append(f"{'突破' if bx['status'] == 'up' else '跌破'}當天量是 20 均量的 {vr:.1f} 倍，"
                       + ("帶量，可信度高" if vr >= 1.5 else "量能不足，留意假突破 / 假跌破"))
    else:
        why.append(f"近 {min(windows)}~{max(windows)} {unit}沒有符合規則的箱型 (箱頂箱底需各 2 次以上觸碰、寬度 3~{max_atr} 倍 ATR {atr:.2f})")
    if tech is not None:
        good = tech >= 3 if not bear else tech <= -3
        bad = tech <= -3 if not bear else tech >= 3
        why.append(f"技術指標 / 量價 {tech:+.1f} 分：" + ("與計畫方向一致，進場區可信度高" if good else
                   "與計畫方向相反，等指標轉向 (KD/MACD 交叉、止跌K) 再進場、部位減半" if bad else
                   "中性，進場區需搭配 K 棒訊號確認"))
    ctx = {"trail": None, "breakout": None}

    def trail_txt(long):
        if ma_fast not in ma:
            return None
        s = f"收盤有效{'跌破' if long else '站回'} {ma_fast.upper()} {ma[ma_fast]:.2f}"
        if ma_slow in ma:
            s += f" 或 {ma_slow.upper()} {ma[ma_slow]:.2f}"
        return s + (" 獲利了結" if long else " 回補")

    def zone_long():
        if regime == "A":
            ctx["breakout"] = bx["hi"] + tick(bx["hi"])
            why.append("情況 A 區間震盪：箱底 ~ 箱底+0.5ATR 低吸、箱頂-0.5ATR ~ 箱頂高拋；收盤跌破 箱底-0.5ATR 停損")
            lo, hi = bx["lo"], bx["lo"] + 0.5 * atr
            if lo <= close <= hi + 0.25 * atr:
                sg = _candle_signal(d, True)
                why.append("目前已在買進區，" + (f"出現止跌訊號 ({sg})，可進場" if sg else "尚未出現止跌訊號 (長下影線或紅K)，等訊號再進"))
            elif close > hi:
                why.append(f"現價 {close} 在箱內中上緣，不追價，等回到箱底區再買")
            return lo, hi, bx["lo"] - 0.5 * atr, bx["hi"] - 0.5 * atr, bx["hi"]
        if regime == "B多":
            if bx and bx["status"] == "up":
                lo, hi = bx["hi"], bx["hi"] + 0.5 * atr
                why.append(f"情況 B 結構轉換：已突破箱頂 {round_tick(bx['hi'])}，拉回測試前箱頂不破買進 (壓力變支撐)")
                t1, t2 = bx["hi"] + bx["height"], bx["hi"] + 2 * bx["height"]
                why.append(f"目標 = 箱型等幅 1 倍 {round_tick(t1)} / 2 倍 {round_tick(t2)}")
            else:
                lo, hi = H - 1.5 * atr, H - 1.0 * atr
                near = []
                for k in ("ma5", "ma10"):
                    if k in ma and lo - 0.5 * atr <= ma[k] <= hi + 0.5 * atr:
                        near.append(f"{k.upper()} {ma[k]:.2f}")
                        lo, hi = min(lo, ma[k]), max(hi, ma[k])
                why.append(f"情況 B 波動拉回：近期高點 {round_tick(H)} 拉回 1~1.5 倍 ATR"
                           + (f"，貼近上升的 {'、'.join(near)}" if near else "") + " 買進")
                above = [p for p in res if p > H + 0.5 * atr]
                t1 = above[0] if above else H + 2 * atr
                t2 = max(t1 + 1.5 * atr, H + 3.5 * atr)
            hi = min(hi, close)
            if lo > hi:
                lo = hi - 0.5 * atr
            stop = min(lo - atr, max(swing_lo - tick(swing_lo), lo - 1.5 * atr))
            ctx["trail"] = trail_txt(True)
            ctx["breakout"] = H + tick(H)
            why.append("停損 = 買進區下緣往下 1~1.5 倍 ATR (參考近期波段低點)；也可用進場當天 K 棒最低點當絕對防守")
            return lo, hi, stop, t1, t2
        base = sup[0] if sup else L
        lo, hi = base, min(close, base + 0.5 * atr)
        if lo > hi:
            lo = hi - 0.5 * atr
        t1 = res[0] if res else close + 2 * atr
        t2 = res[1] if len(res) > 1 else max(t1 + atr, close + 3.5 * atr)
        why.append("空頭趨勢 (均線空頭排列) 中做多屬逆勢，只在波段支撐試單、部位要小" if regime == "B空" else
                   "無有效箱型：以最近波段支撐 ~ 支撐+0.5ATR 為買進區，收盤跌破 支撐-0.5ATR 停損")
        return lo, hi, base - 0.5 * atr, t1, t2

    def zone_short():
        if regime == "A":
            ctx["breakout"] = bx["lo"] - tick(bx["lo"])
            why.append("情況 A 區間震盪：箱頂-0.5ATR ~ 箱頂賣出 / 放空、箱底+0.5ATR ~ 箱底回補；收盤突破 箱頂+0.5ATR 停損")
            lo, hi = bx["hi"] - 0.5 * atr, bx["hi"]
            if lo - 0.25 * atr <= close <= hi:
                sg = _candle_signal(d, False)
                why.append("目前已在賣出區，" + (f"出現止漲訊號 ({sg})" if sg else "尚未出現止漲訊號 (長上影線或黑K)"))
            return lo, hi, bx["hi"] + 0.5 * atr, bx["lo"] + 0.5 * atr, bx["lo"]
        if regime == "B空":
            if bx and bx["status"] == "down":
                lo, hi = bx["lo"] - 0.5 * atr, bx["lo"]
                why.append(f"情況 B 結構轉換：已跌破箱底 {round_tick(bx['lo'])}，反彈測試前箱底不過放空 (支撐變壓力)")
                t1, t2 = bx["lo"] - bx["height"], bx["lo"] - 2 * bx["height"]
                why.append(f"目標 = 箱型等幅 1 倍 {round_tick(t1)} / 2 倍 {round_tick(t2)}")
            else:
                lo, hi = L + 1.0 * atr, L + 1.5 * atr
                for k in ("ma5", "ma10"):
                    if k in ma and lo - 0.5 * atr <= ma[k] <= hi + 0.5 * atr:
                        lo, hi = min(lo, ma[k]), max(hi, ma[k])
                why.append(f"情況 B 波動反彈：近期低點 {round_tick(L)} 反彈 1~1.5 倍 ATR 或貼近下彎的 5/10MA 放空")
                below = [p for p in sup if p < L - 0.5 * atr]
                t1 = below[0] if below else L - 2 * atr
                t2 = min(t1 - 1.5 * atr, L - 3.5 * atr)
            lo = max(lo, close)
            if lo > hi:
                hi = lo + 0.5 * atr
            stop = max(hi + atr, min(swing_hi + tick(swing_hi), hi + 1.5 * atr))
            ctx["trail"] = trail_txt(False)
            ctx["breakout"] = L - tick(L)
            return lo, hi, stop, t1, t2
        base = res[0] if res else H
        lo, hi = max(close, base - 0.5 * atr), base
        if lo > hi:
            hi = lo + 0.5 * atr
        t1 = sup[0] if sup else close - 2 * atr
        t2 = sup[1] if len(sup) > 1 else min(t1 - atr, close - 3.5 * atr)
        why.append("多頭趨勢中放空屬逆勢，部位要小" if regime == "B多" else
                   "無有效箱型：以最近波段壓力-0.5ATR ~ 壓力為放空區，收盤站上 壓力+0.5ATR 停損")
        return lo, hi, base + 0.5 * atr, t1, t2

    lo, hi, stop, t1, t2 = zone_short() if bear else zone_long()
    breakout = ctx["breakout"]
    note = None
    if cap:
        if (not bear and close >= cap) or (bear and close <= cap):
            note = f"已{'跌停' if bear else '漲停'} {cap}，不追價"
        t1, t2 = (max(t1, cap), max(t2, cap)) if bear else (min(t1, cap), min(t2, cap))
        if breakout and ((not bear and breakout > cap) or (bear and breakout < cap)):
            breakout = None
    lo, hi = round_tick(lo, "up" if bear else "down"), round_tick(hi)
    ref = (lo + hi) / 2
    stop = round_tick(stop, "up" if bear else "down")
    risk = abs(ref - stop)
    if regime == "A":
        # 情況 A：預期利潤只算到 箱頂-0.5ATR (做空：箱底+0.5ATR)，不用箱寬或等幅
        t1 = bx["lo"] + 0.5 * atr if bear else bx["hi"] - 0.5 * atr
        reward = (ref - t1) if bear else (t1 - ref)
    else:
        reward = abs(t1 - ref)
    rr = max(reward, 0) / risk if risk > 0 else 0
    if regime == "A":
        # 現價已進入對側邊緣 (做多：>= 箱頂-0.5ATR；做空：<= 箱底+0.5ATR) → 從現價算也沒有利潤空間
        now_reward = (close - t1) if bear else (t1 - close)
        if now_reward <= 0:
            rr = 0
    no_room = regime == "A" and rr <= 0
    if no_room:
        why.append(NO_ROOM)
    box_out = None
    if bx:
        st = pd.Timestamp(bx["start"])
        if st.tzinfo is not None:
            st = st.tz_localize(None)
        box_out = {"box_high": round_tick(bx["hi"]), "box_low": round_tick(bx["lo"]), "window": bx["window"],
                   "start_date": st.strftime("%Y-%m-%d %H:%M"), "status": bx["status"], "confirmed": bx["confirmed"],
                   "stage": STAGE[bx["status"]], "unit": unit, "atr": round(atr, 2),
                   "width_atr": round(bx["height"] / atr, 1)}
    plan = {"side": "做空" if bear else "做多", "entry_zone": [lo, hi],
            "breakout": round_tick(breakout) if breakout else None,
            "stop": stop, "target1": round_tick(t1), "target2": round_tick(t2),
            "risk_pct": round(risk / ref * 100, 2), "reward_pct": round(max(reward, 0) / ref * 100, 2),
            "rr": round(rr, 2), "no_room": no_room, "rr_text": NO_ROOM if no_room else None,
            "note": note, "regime": REGIME_TXT[regime], "trail": ctx["trail"], "why": why,
            "exit_trail": {"long": trail_txt(True), "short": trail_txt(False),
                           "ma_fast": round(ma[ma_fast], 2) if ma_fast in ma else None,
                           "ma_slow": round(ma[ma_slow], 2) if ma_slow in ma else None,
                           "names": [ma_fast.upper(), ma_slow.upper()]},
            "box": box_out, "atr": round(atr, 2)}
    if set_keys:
        _set_keys(lv, bx, regime, atr, H, L, plan, ma.get("ma10"))
    return plan


def _set_keys(lv, bx, regime, atr, H, L, plan, ma10=None):
    """多方關鍵 / 空方關鍵改用箱型規則；箱頂箱底併入支撐壓力"""
    key = lv.setdefault("key", {})
    close = lv.get("close")
    if regime == "A":
        key["bull"] = _key(bx["lo"], f"箱底 {round_tick(bx['lo'])}：收盤跌破且超過 0.5 ATR ({round_tick(bx['lo'] - 0.5 * atr)})、隔天未拉回 = 箱型破局轉空")
        key["bear"] = _key(bx["hi"], f"箱頂 {round_tick(bx['hi'])}：收盤站上且隔天未拉回 = 箱型破局轉強攻")
    elif regime == "B多":
        if bx and bx["status"] == "up":
            key["bull"] = _key(bx["hi"], f"前箱頂 {round_tick(bx['hi'])} (壓力變支撐)：收盤跌回箱內代表假突破")
            key["bear"] = _key(H, f"近期高點 {round_tick(H)}：站上續創波段新高")
        else:
            zl = plan["entry_zone"][0]
            if ma10:
                key["bull"] = _key(ma10, f"10MA {ma10:.2f}：均線多頭排列的趨勢防守，收盤有效跌破代表趨勢轉弱")
            key["bear"] = _key(zl, f"拉回買進區下緣 {zl}：收盤跌破代表拉回過深，空方接手")
    elif regime == "B空":
        if bx and bx["status"] == "down":
            key["bear"] = _key(bx["lo"], f"前箱底 {round_tick(bx['lo'])} (支撐變壓力)：收盤站回箱內代表假跌破")
            key["bull"] = _key(L, f"近期低點 {round_tick(L)}：跌破續創波段新低")
        else:
            zh = plan["entry_zone"][1] if plan["side"] == "做空" else L + 1.5 * atr
            if ma10:
                key["bear"] = _key(ma10, f"10MA {ma10:.2f}：均線空頭排列的趨勢壓力，收盤有效站回代表趨勢轉強")
            key["bull"] = _key(zh, f"反彈放空區上緣 {round_tick(zh)}：收盤站上代表反彈過強，多方接手")
    if bx and close:
        for p, name in ((bx["hi"], "箱頂"), (bx["lo"], "箱底")):
            p = round_tick(p)
            arr = "supports" if p < close else "resistances"
            lst = lv.setdefault(arr, [])
            near = [x for x in lst if abs(x["price"] - p) <= 0.3 * atr]
            if near:
                if name not in near[0]["tags"]:
                    near[0]["tags"].append(name)
            else:
                lst.append({"price": p, "score": 9.9, "touches": 0, "dist_pct": round((p / close - 1) * 100, 2), "tags": [name]})
            lst.sort(key=lambda x: -x["price"] if arr == "supports" else x["price"])
