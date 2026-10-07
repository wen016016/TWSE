"""大戶 / 散戶 成交分析 (成交明細)

單筆成交 >= 大戶門檻 → 大戶單；<= 散戶門檻 → 散戶單
外盤 (成交在委賣價) = 主動買；內盤 (成交在委買價) = 主動賣

  大戶淨買 > 0 且 散戶淨賣  → 散戶出貨、大戶吸籌  (偏多)
  大戶淨賣 且 散戶淨買 > 0  → 大戶出貨、散戶接刀  (偏空)
  兩者同向                  → 全面買盤 / 全面賣壓
"""
import math

from . import realtime


def thresholds(price: float) -> tuple[int, int]:
    from .broker import get_settings
    f = get_settings()["flow"]
    big = int(f["big_lots"]) if f.get("big_lots") else max(10, math.ceil(f["big_amount"] / (price * 1000)))
    return big, int(f["small_lots"])


def analyze_flow(code: str) -> dict:
    df, source, date = realtime.ticks(code)
    if df.empty:
        return {"ok": False, "source": source,
                "msg": "還沒有成交明細：系統開著時，盤中會自動每 5 秒記錄一次查詢過的股票與持倉"}
    price = float(df["price"].iloc[-1])
    big, small = thresholds(price)
    known = df[df["merged"] == 0]
    total = float(df["vol"].sum())

    def s(mask, side):
        return float(known.loc[mask & (known["side"] == side), "vol"].sum())

    is_big = known["vol"] >= big
    is_small = known["vol"] <= small
    bb, bs = s(is_big, "B"), s(is_big, "S")
    sb, ss = s(is_small, "B"), s(is_small, "S")
    big_net, small_net = bb - bs, sb - ss
    coverage = float(known["vol"].sum()) / total if total else 0
    out = {"ok": True, "source": source, "date": date, "ticks": int(len(df)), "total_vol": round(total),
           "coverage_pct": round(coverage * 100, 1), "big_lots": big, "small_lots": small,
           "big_buy": round(bb), "big_sell": round(bs), "big_net": round(big_net),
           "small_buy": round(sb), "small_sell": round(ss), "small_net": round(small_net),
           "big_count": int(is_big.sum()),
           "big_trades": known[is_big].tail(15).iloc[::-1].to_dict("records")}

    th = max(total * 0.03, big)  # 淨額要超過總量 3% 才算有意義
    if len(df) < 30 or coverage < 0.15:
        out.update(score=0, verdict="資料不足", text=f"成交明細只有 {len(df)} 筆 / 可分類量 {coverage * 100:.0f}%，暫不判斷")
    elif big_net > th and small_net < 0:
        out.update(score=1.5, verdict="散戶出貨、大戶吸籌",
                   text=f"大戶單 (≥{big}張) 淨買 {big_net:,.0f} 張，散戶單 (≤{small}張) 淨賣 {-small_net:,.0f} 張 → 籌碼由散戶流向大戶")
    elif big_net < -th and small_net > 0:
        out.update(score=-1.5, verdict="大戶出貨、散戶接刀",
                   text=f"大戶單 (≥{big}張) 淨賣 {-big_net:,.0f} 張，散戶單 (≤{small}張) 淨買 {small_net:,.0f} 張 → 大戶倒貨給散戶")
    elif big_net > th:
        out.update(score=0.8, verdict="大戶帶頭買進",
                   text=f"大戶單淨買 {big_net:,.0f} 張，散戶也在買 ({small_net:+,.0f} 張)")
    elif big_net < -th:
        out.update(score=-0.8, verdict="大戶帶頭賣出",
                   text=f"大戶單淨賣 {-big_net:,.0f} 張，散戶也在賣 ({small_net:+,.0f} 張)")
    else:
        out.update(score=0, verdict="大戶動向不明顯",
                   text=f"大戶單淨額 {big_net:+,.0f} 張、散戶單淨額 {small_net:+,.0f} 張，未超過門檻 {th:,.0f} 張")
    if source.startswith("系統"):
        out["note"] = (f"資料來源：{source}。只收集到系統開啟後的成交；5 秒內若有多筆成交，只知道最後一筆張數，"
                       f"其餘 {100 - coverage * 100:.0f}% 的量無法分大小。填入 FinMind 付費 token 可取得完整逐筆。")
    return out
