"""選股掃描：先用日K/週K 技術面快篩，再對前幾名做完整分析 (含籌碼、大盤、5分K/1分K)，並依額度算建議張數"""
from concurrent.futures import ThreadPoolExecutor


from . import broker, data
from .config import DATA_DIR
from .indicators import add_indicators, to_weekly
from .strategy import Score, analyze, score_bias, score_tech

# 預設觀察名單：權值股 + 高成交量熱門股。可在 watchlist.txt 自訂 (一行一個代號)
UNIVERSE = """2330 2317 2454 2308 2382 2412 2881 2882 2891 2303 3711 2886 2884 1301 2002 2357 3231 2885 2892 2345
3008 2327 2379 2395 3034 3037 2603 2609 2615 3017 2408 2344 2337 3443 3661 6669 5269 3529 2376 2356
4938 2324 2301 1216 2207 2912 1101 2880 2883 2887 2890 5880 2801 6505 1303 1326 2618 2610 3045 4904
2449 3044 2368 6415 3653 3533 8046 2383 6239 3189 2049 1504 1519 1513 1605 2371 3481 2409 6116 2353
3260 8299 5347 6488 3105 4966 6147 5483 3293 8069 6274 3324 3006 2360 3583 6182 2059 3014 2313 2492""".split()


def universe():
    f = DATA_DIR / "watchlist.txt"
    if f.exists():
        codes = [l.strip().split()[0] for l in f.read_text(encoding="utf-8").splitlines() if l.strip() and not l.startswith("#")]
        if codes:
            return codes
    return UNIVERSE


def _quick(code, df, mode):
    d = add_indicators(df)
    r = d.iloc[-1]
    sc = Score()
    if mode == "swing":
        w = add_indicators(to_weekly(df), mas=(5, 10, 20, 60))
        score_tech(sc, w, "週K", 1.2, 13)
        score_tech(sc, d, "日K", 1.0, 20)
        score_bias(sc, d, "日K")
        ok = r["vma20"] >= 500
    else:
        atrp = r["atr"] / r["close"] * 100
        ok = r["vma20"] >= 2000 and atrp >= 2
        score_tech(sc, d, "日K", 1.0, 20)
        sc.add("條件", min(atrp, 6) / 2, f"ATR {atrp:.1f}%")
        sc.add("條件", 1 if r["vma20"] > 10000 else 0.5, f"均量 {r['vma20']:,.0f} 張")
    return {"code": code, "quick": sc.total, "ok": bool(ok), "close": float(r["close"])}


def scan_base(mode="swing", top=8):
    """找「整理完成，準備突破」或「剛帶量突破」的股票"""
    from .base_breakout import detect, pooled_stats
    from .indicators import tick
    daily = data.download_daily(universe())
    info = data.stock_info().set_index("stock_id")
    found = []
    for code, df in daily.items():
        d = add_indicators(df)
        try:
            b = detect(d)
        except Exception:  # noqa: BLE001
            continue
        st = b["stage"]
        if not (st.startswith("整理完成") or "已啟動" in st or st.startswith("整理中 (接近完成)")):
            continue
        close = float(d["close"].iloc[-1])
        entry = b["trigger"] if close < b["trigger"] else close
        risk = entry - b["stop"]
        rr = round((b["target"] - entry) / risk, 2) if risk > 0 else 0
        size = broker.size_position(mode, entry, b["stop"])
        rank = b["score"] + (25 if "已啟動" in st else 15 if st.startswith("整理完成，準備") else 0)
        found.append({"code": code, "name": info["stock_name"].get(code, code), "close": close, "total": b["score"],
                      "tone": st, "rank": rank,
                      "advice": f"站上 {b['trigger']} 啟動，跌破 {b['stop']} 停損，目標 {b['target']}",
                      "by_cat": {k: v for k, v in (b.get("pts") or {}).items()},
                      "plan": {"side": "做多", "entry_zone": [round(entry - tick(entry), 2), entry], "stop": b["stop"],
                               "target1": b["target"], "target2": round(b["target"] * 1.05, 2), "rr": rr,
                               "breakout": b["trigger"]},
                      "side": "buy", "size": size, "top_reasons": b["why"][:3], "base": b,
                      "actionable": (st.startswith("整理完成，準備") or "已啟動" in st) and size["shares"] > 0 and rr >= 1})
    found.sort(key=lambda r: -r["rank"])
    return {"mode": mode, "scanned": len(daily), "passed": len(found), "results": found[:top],
            "budget": broker.budget_status()[mode], "kind": "base", "stats": pooled_stats(background=True)}


def scan(mode="swing", top=8, direction="long"):
    if direction == "base":
        return scan_base(mode, top)
    codes = universe()
    daily = data.download_daily(codes)
    rows = [_quick(c, df, mode) for c, df in daily.items()]
    rows = [r for r in rows if r["ok"]]
    skipped = 0
    if mode == "daytrade":  # 零股不能當沖 → 額度買不起 1 張的先排除
        avail = broker.budget_status()["daytrade"]["available"]
        skipped = sum(1 for r in rows if r["close"] * 1000 > avail)
        rows = [r for r in rows if r["close"] * 1000 <= avail]
    rows.sort(key=lambda r: r["quick"], reverse=(direction == "long"))
    cand = rows[:top * 2]

    def full(r):
        try:
            a = analyze(r["code"], mode, charts=False)
            return a
        except Exception as e:  # noqa: BLE001
            return {"code": r["code"], "error": str(e)}

    with ThreadPoolExecutor(4) as ex:
        results = list(ex.map(full, cand))
    results = [a for a in results if "error" not in a]
    # 排序 = 總分 + 風報比加權 (風報比差的標的往後排)
    sign = 1 if direction == "long" else -1
    for a in results:
        a["rank"] = a["total"] * sign + 2 * min(a["plan"]["rr"], 2.5)
    results.sort(key=lambda a: a["rank"], reverse=True)
    out = []
    for a in results[:top]:
        p = a["plan"]
        side = "buy" if p["side"] == "做多" else "sell"
        size = broker.size_position(mode, a["close"], p["stop"])
        want = "做多" if direction == "long" else "做空"
        actionable = (("多" in a["tone"] if direction == "long" else "空" in a["tone"]) and p["side"] == want
                      and p["rr"] >= 1 and size["shares"] > 0)
        out.append({"code": a["code"], "name": a["name"], "close": a["close"], "total": a["total"], "tone": a["tone"],
                    "rank": round(a["rank"], 1), "actionable": bool(actionable),
                    "advice": a["advice"], "by_cat": a["by_cat"], "plan": p, "side": side, "size": size,
                    "top_reasons": [i["text"] for i in sorted(a["items"], key=lambda i: -abs(i["pts"]))[:5]]})
    out.sort(key=lambda r: not r["actionable"])  # 可進場的排前面
    return {"mode": mode, "scanned": len(daily), "passed": len(rows), "skipped_budget": skipped, "results": out,
            "budget": broker.budget_status()[mode]}
