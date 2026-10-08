"""外資資金轉向偵測：大跌前，外資常「賣半導體、買防禦股 (金融、電信)」

資料：證交所「三大法人買賣超日報 (T86)」× 「每日收盤行情 (MI_INDEX)」→ 外資 (含外資自營商) 每檔買賣超金額
分組：半導體 = 官方分類半導體業；防禦股 = 金融保險業 + 中華電 / 台灣大 / 遠傳
訊號 (依過去一年 236 個交易日回測訂門檻)：
  +3 近 5 天有 ≥4 天「賣半導體、同時買防禦」，或 半導體 5 日賣超 ≥1,300 億且防禦淨買
     → 之後 10 天回檔 ≥3% 約 65~70% (平常 40%)、≥5% 約 40~70% (平常 20%)
  +2 近 5 天有 3 天，或 半導體 5 日賣超 ≥1,000 億且防禦淨買 → 回檔 ≥3% 約 48~59%
  +1 半導體 5 日賣超 ≥1,000 億
  -1 半導體 5 日買超 ≥1,000 億且沒有買防禦
  ⚠ 樣本少 (強訊號約 10~20 天) 且相鄰天數重疊，只能當強烈參考
驗證：validate() 用過去一年，比較「出現轉向訊號後 10 天內大盤回檔 ≥3%」的比例 vs 平常
"""
import datetime as dt
import json

import numpy as np
import pandas as pd

from . import data
from .config import DATA_DIR

ROT_DIR = DATA_DIR / "rotation"
ROT_DIR.mkdir(exist_ok=True)
TELECOM = {"2412", "3045", "4904"}
FIN_CATS = {"金融保險", "金融保險業", "金融業"}


def _groups():
    info = data.stock_info()
    semis = set(info.loc[info["industry_category"] == "半導體業", "stock_id"])
    defensive = set(info.loc[info["industry_category"].isin(FIN_CATS), "stock_id"]) | TELECOM
    return semis, defensive


def _num(s):
    try:
        return float(str(s).replace(",", "").strip())
    except ValueError:
        return 0.0


def _www_json(url, params):
    from .realtime import WWW, _session
    WWW.wait()
    try:
        j = _session.get(url, params=params, timeout=30).json()
        WWW.ok()
        return j
    except Exception:  # noqa: BLE001
        WWW.fail()
        return None


def day_flow(day: dt.date) -> dict | None:
    """某天外資在半導體 / 防禦股的買賣超金額 (億)；結果快取在本機"""
    f = ROT_DIR / f"{day:%Y%m%d}.json"
    if f.exists():
        d = json.loads(f.read_text(encoding="utf-8"))
        return None if d.get("holiday") else d
    ds = day.strftime("%Y%m%d")
    t86 = _www_json("https://www.twse.com.tw/rwd/zh/fund/T86", {"date": ds, "selectType": "ALL", "response": "json"})
    if not t86:
        return None  # 連線失敗：不快取，下次再試
    if t86.get("stat") != "OK":
        f.write_text(json.dumps({"holiday": True}), encoding="utf-8")
        return None
    mi = _www_json("https://www.twse.com.tw/rwd/zh/afterTrading/MI_INDEX", {"date": ds, "type": "ALLBUT0999", "response": "json"})
    if not mi or mi.get("stat") != "OK":
        return None
    close = {}
    for tb in mi.get("tables", []):
        fl = tb.get("fields", [])
        if "證券代號" in fl and "收盤價" in fl:
            ci, pi = fl.index("證券代號"), fl.index("收盤價")
            for r in tb["data"]:
                close[r[ci].strip()] = _num(r[pi])
    semis, defensive = _groups()
    fl = t86["fields"]
    i_code = fl.index("證券代號")
    i_name = fl.index("證券名稱")
    i_f1 = next(i for i, x in enumerate(fl) if x.startswith("外陸資買賣超"))
    i_f2 = next(i for i, x in enumerate(fl) if x.startswith("外資自營商買賣超"))
    rows = []
    for r in t86["data"]:
        code = r[i_code].strip()
        px = close.get(code, 0)
        if not px:
            continue
        amt = (_num(r[i_f1]) + _num(r[i_f2])) * px / 1e8
        grp = "semi" if code in semis else ("def" if code in defensive else "other")
        rows.append((code, r[i_name].strip(), grp, amt))
    df = pd.DataFrame(rows, columns=["code", "name", "grp", "amt"])
    g = df.groupby("grp")["amt"].sum()
    out = {"date": day.isoformat(), "semi": round(float(g.get("semi", 0)), 1), "def": round(float(g.get("def", 0)), 1),
           "total": round(float(df["amt"].sum()), 1),
           "top_semi_sell": df[df["grp"] == "semi"].nsmallest(5, "amt")[["code", "name", "amt"]].round(1).to_dict("records"),
           "top_def_buy": df[df["grp"] == "def"].nlargest(5, "amt")[["code", "name", "amt"]].round(1).to_dict("records")}
    f.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    return out


def recent(n=10) -> list[dict]:
    idx = data.get_kline(data.INDEX, "1d", live=False).index
    days = [t.date() for t in idx[-(n + 3):]]
    now = dt.datetime.now(dt.timezone(dt.timedelta(hours=8)))
    if now.hour < 16:  # 法人買賣超約 15:00~16:00 才公布
        days = [d for d in days if d < now.date()]
    out = [x for x in (day_flow(d) for d in days) if x]
    return out[-n:]


def signal(flows: list[dict]):
    """回傳 (分數, 說明, 摘要)；分數 > 0 = 資金轉向避險、回測風險升高"""
    if len(flows) < 5:
        return 0, "外資產業資金流資料不足", None
    last5 = flows[-5:]
    semi5 = sum(x["semi"] for x in last5)
    def5 = sum(x["def"] for x in last5)
    sell_days = sum(1 for x in last5 if x["semi"] < 0 and x["def"] > 0)
    summ = {"semi5": round(semi5, 1), "def5": round(def5, 1), "rotation_days": sell_days}
    txt = (f"外資近 5 日：半導體 {semi5:+,.0f} 億、防禦股 (金融+電信) {def5:+,.0f} 億，"
           f"{sell_days} 天同時「賣半導體、買防禦股」")
    # 門檻依 2025/10~2026/10 回測 (236 個交易日)：平常 10 天內回檔 ≥3% 40%、≥5% 20%
    if sell_days >= 4 or (semi5 <= -1300 and def5 > 0):
        return 3, txt + " → 強烈資金轉向避險 (歷史上之後 10 天回檔 ≥3% 約 65~70%，≥5% 約 40~70%)", summ
    if sell_days >= 3 or (semi5 <= -1000 and def5 > 0):
        return 2, txt + " → 資金轉向避險 (歷史上之後 10 天回檔 ≥3% 約 48~59%)", summ
    if semi5 <= -1000:
        return 1, txt + " → 外資大賣半導體", summ
    if semi5 >= 1000 and def5 <= 0:
        return -1, txt + " → 外資大買半導體、沒有避險跡象", summ
    return 0, txt + (" (轉向跡象出現，但還沒達到訊號門檻)" if sell_days == 2 else ""), summ


def validate(lookback_days=250) -> dict | None:
    """用已快取的歷史資料驗證：出現轉向訊號後 10 天內，大盤回檔 ≥3% 的比例 vs 平常"""
    import yfinance as yf
    d = data._normalize(yf.Ticker(data.INDEX).history(period="2y", auto_adjust=False), True)
    days = [t.date() for t in d.index[-lookback_days:]]
    flows = {x["date"]: x for x in (day_flow(dd) for dd in days) if x}
    if len(flows) < 60:
        return {"ready": False, "have": len(flows)}
    c, lo = d["close"].values, d["low"].values
    rows = []
    keys = sorted(flows)
    for i, k in enumerate(keys):
        if i < 4:
            continue
        pos = np.where(d.index.date == dt.date.fromisoformat(k))[0]
        if not len(pos) or pos[0] + 10 >= len(d):
            continue
        p = pos[0]
        sc, _, _ = signal([flows[x] for x in keys[i - 4:i + 1]])
        dd10 = lo[p + 1:p + 11].min() / c[p] - 1
        rows.append({"sig": sc, "dd3": dd10 <= -0.03, "dd5": dd10 <= -0.05})
    df = pd.DataFrame(rows)
    on = df[df["sig"] >= 1]
    return {"ready": True, "days": int(len(df)), "signals": int(len(on)),
            "p_dd3_signal": round(float(on["dd3"].mean() * 100), 1) if len(on) else None,
            "p_dd3_base": round(float(df["dd3"].mean() * 100), 1),
            "p_dd5_signal": round(float(on["dd5"].mean() * 100), 1) if len(on) else None,
            "p_dd5_base": round(float(df["dd5"].mean() * 100), 1)}
