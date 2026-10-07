"""下單：資金額度 / 部位計算 / 風控 / 委託 → 模擬帳戶 或 任一券商帳戶 (stock/brokers.py)

流程：
  建議 → preview(產生委託單 + 風控檢查 + 依額度算張數) → confirm (手動按確認，或自動交易直接確認) → 券商帳戶
"""
import datetime as dt
import json
import math
import threading
import uuid

from . import brokers, data
from .config import DATA_DIR, RISK
from .indicators import round_tick

SETTINGS_FILE = DATA_DIR / "settings.json"
LEDGER_FILE = DATA_DIR / "ledger.json"

DEFAULT_SETTINGS = {
    # 可動用資金 (元)：當沖、波段分開
    "budget": {"daytrade": 100_000, "swing": 100_000},
    # 單筆最多可虧損 = 額度 × 這個 %，用停損距離反推張數
    "risk_pct": {"daytrade": 1.0, "swing": 2.0},
    "max_positions": {"daytrade": 3, "swing": 5},
    "allow_odd_lot": True,           # 額度買不起 1 張時，改用零股
    "max_qty_lots": RISK["max_qty_lots"],
    "max_orders_per_day": RISK["max_orders_per_day"],
    "max_price_deviation": RISK["max_price_deviation"],
    # 自動交易
    "auto": {
        "enabled": False,            # 總開關
        "exit": True,                # 自動出場：觸停損、到目標、當沖收盤前強制平倉
        "entry": False,              # 自動進場：定時掃描，符合條件自動買進
        "entry_daytrade": True,      # 自動進場要做的模式
        "entry_swing": False,
        "min_score_daytrade": 6.0,
        "min_score_swing": 8.0,
        "max_new_daytrade": 3,       # 每日最多自動新開幾檔
        "max_new_swing": 2,
        "scan_interval_min": 15,     # 當沖：每幾分鐘掃描一次 (09:15~12:30)
        "swing_entry_time": "13:15", # 波段：每天這個時間掃描進場 (接近收盤確認日K)
        "flatten_time": "13:20",     # 當沖強制平倉時間
        "half_at_target1": True,     # 到目標一先出一半，停損移到成本
        "slippage_ticks": 2,         # 自動單掛價往成交方向多掛幾檔，確保成交
    },
    # 成交明細大戶 / 散戶門檻
    "flow": {"big_amount": 3_000_000, "big_lots": 0, "small_lots": 5},
    # 當沖 / 波段各自預設下到哪個券商帳戶 (paper = 模擬)
    "default_account": {"daytrade": "paper", "swing": "paper"},
}

_lock = threading.RLock()


def _now():
    return dt.datetime.now(dt.timezone(dt.timedelta(hours=8)))


def _load(path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _save(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


# ----------------------------------------------------------------- 設定
def get_settings() -> dict:
    s = _load(SETTINGS_FILE, {})
    if s.get("broker_mode") == "ibf" and "default_account" not in s:  # 舊版設定轉換
        s["default_account"] = {"daytrade": "ibf", "swing": "ibf"}
    out = json.loads(json.dumps(DEFAULT_SETTINGS))
    for k, v in s.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k].update(v)
        else:
            out[k] = v
    return out


def update_settings(patch: dict) -> dict:
    with _lock:
        s = get_settings()
        for k, v in patch.items():
            if k not in DEFAULT_SETTINGS:
                continue
            if isinstance(v, dict) and isinstance(s.get(k), dict):
                s[k].update({kk: type(DEFAULT_SETTINGS[k].get(kk, vv))(vv) for kk, vv in v.items()})
            else:
                s[k] = v
        ids = {a["id"] for a in brokers.accounts()}
        for m in ("daytrade", "swing"):
            if s["budget"][m] < 0:
                raise ValueError("額度不能是負數")
            if s["default_account"][m] not in ids:
                raise ValueError(f"找不到券商帳戶 {s['default_account'][m]}")
        _save(SETTINGS_FILE, s)
        return s


# ----------------------------------------------------------------- 帳本 (委託 + 部位)
def ledger() -> dict:
    return _load(LEDGER_FILE, {"orders": [], "positions": [], "pending": {}})


def _today():
    return _now().strftime("%Y-%m-%d")


def open_positions(mode=None):
    pos = [p for p in ledger()["positions"] if p["qty"] > 0]
    # 當沖部位只算今天的 (隔天視為已了結，請自行確認券商帳務)
    pos = [p for p in pos if p["mode"] != "daytrade" or p["date"] == _today()]
    return [p for p in pos if mode is None or p["mode"] == mode]


def accounts_summary() -> list[dict]:
    """每個券商帳戶的狀態 + 最近一次同步的可用餘額 / 庫存"""
    out = []
    for a in brokers.accounts():
        out.append({**a, "summary": brokers.account_summary(a["id"]) if a["type"] == "web" else None})
    return out


def account_cash(acc_id: str):
    if acc_id == "paper":
        return None
    d = brokers.account_data(acc_id)
    return d.get("cash") if d else None


def budget_status() -> dict:
    s = get_settings()
    out = {}
    for m in ("daytrade", "swing"):
        acc_id = s["default_account"][m]
        cash = account_cash(acc_id)
        used = sum(p["qty"] * p["avg_price"] for p in open_positions(m))
        avail = max(s["budget"][m] - used, 0)
        if cash is not None:  # 不能超過券商帳戶實際可用餘額
            avail = min(avail, cash)
        out[m] = {"budget": s["budget"][m], "used": round(used), "available": round(avail),
                  "positions": len(open_positions(m)), "max_positions": s["max_positions"][m],
                  "account": acc_id, "broker_cash": cash}
    today = [o for o in ledger()["orders"] if o["time"].startswith(_today())]
    out["orders_today"] = len(today)
    out["realized_today"] = round(sum(o.get("pnl", 0) for o in today), 0)
    return out


# ----------------------------------------------------------------- 部位計算
def size_position(mode: str, price: float, stop: float | None) -> dict:
    """依可用額度與停損距離計算股數"""
    s = get_settings()
    b = budget_status()[mode]
    avail = b["available"]
    risk_amt = s["budget"][mode] * s["risk_pct"][mode] / 100
    by_cash = math.floor(avail / price) if price > 0 else 0
    per_share_risk = abs(price - stop) if stop else 0
    by_risk = math.floor(risk_amt / per_share_risk) if per_share_risk > 0 else by_cash
    shares = min(by_cash, by_risk)
    lots = shares // 1000
    note = [f"{'當沖' if mode == 'daytrade' else '波段'}額度 {s['budget'][mode]:,.0f}，可用 {avail:,.0f}",
            f"單筆最大虧損 = 額度 × {s['risk_pct'][mode]}% = {risk_amt:,.0f} 元"]
    if per_share_risk:
        note.append(f"停損距離 {per_share_risk:.2f} 元/股 → 最多 {by_risk:,} 股；資金最多 {by_cash:,} 股")
    if lots >= 1:
        lots = min(lots, s["max_qty_lots"])
        shares = lots * 1000
        note.append(f"→ 建議 {lots} 張")
    elif mode == "daytrade":
        shares = 0
        need = price * 1000
        note.append(f"→ 零股不能當沖：1 張要 {need:,.0f} 元" +
                    (f"，停損距離太大 (1 張停損會虧 {per_share_risk * 1000:,.0f} 元 > 上限 {risk_amt:,.0f})" if need <= avail
                     else "，超過可用額度") + "，不下單")
    elif s["allow_odd_lot"] and shares > 0:
        note.append(f"→ 額度買不起整張，建議零股 {shares} 股 (需用零股交易)")
    else:
        shares = 0
        note.append("→ 額度不足，無法下單")
    return {"shares": int(shares), "lots": int(shares // 1000), "odd": int(shares % 1000), "amount": round(shares * price),
            "max_loss": round(shares * per_share_risk), "note": note}


# ----------------------------------------------------------------- 風控
def risk_check(t: dict) -> list[str]:
    s = get_settings()
    errs = []
    st = budget_status()
    if t["shares"] <= 0:
        errs.append("股數必須大於 0")
    if t["action"] == "open":
        if t["amount"] > st[t["mode"]]["available"] + 1:
            errs.append(f"委託金額 {t['amount']:,.0f} 超過{t['mode_name']}可用額度 {st[t['mode']]['available']:,.0f}")
        if st[t["mode"]]["positions"] >= st[t["mode"]]["max_positions"] and not any(
                p["code"] == t["code"] for p in open_positions(t["mode"])):
            errs.append(f"{t['mode_name']}持倉檔數已達上限 {st[t['mode']]['max_positions']}")
    else:
        pos = [p for p in open_positions(t["mode"]) if p["code"] == t["code"] and p["side"] == t["pos_side"]]
        if not pos or pos[0]["qty"] < t["shares"]:
            errs.append("平倉股數超過系統記錄的持倉")
    if t["mode"] == "daytrade" and t["shares"] % 1000:
        errs.append("零股不能當沖，當沖只能整張")
    if t["shares"] // 1000 > s["max_qty_lots"]:
        errs.append(f"單筆超過 {s['max_qty_lots']} 張上限")
    # 平倉 (含停損) 不受每日筆數限制，避免停損單被擋
    if t["action"] == "open" and st["orders_today"] >= s["max_orders_per_day"]:
        errs.append(f"今日委託已達 {s['max_orders_per_day']} 筆上限")
    try:
        q = current_quote(t["code"])
        last = q["price"]
        dev = abs(t["price"] / last - 1)
        if dev > s["max_price_deviation"]:
            errs.append(f"委託價 {t['price']} 與現價 {last} 偏離 {dev * 100:.1f}%，超過 {s['max_price_deviation'] * 100:.0f}% (防打錯價)")
        if q.get("limit_up") and t["price"] > q["limit_up"] or q.get("limit_down") and t["price"] < q["limit_down"]:
            errs.append(f"委託價超出漲跌停範圍 {q['limit_down']} ~ {q['limit_up']}")
    except Exception:
        errs.append("無法取得現價做價格檢查")
    return errs


def current_quote(code) -> dict:
    """優先用證交所即時報價，失敗退回日K收盤"""
    try:
        from .realtime import quote
        q = quote(code)
        if q.get("price"):
            return q
    except Exception:  # noqa: BLE001
        pass
    return {"price": float(data.get_kline(code, "1d")["close"].iloc[-1])}


def warnings(t: dict) -> list[str]:
    w = []
    now = _now()
    hm = now.hour * 100 + now.minute
    if now.weekday() >= 5 or hm < 830 or hm > 1330:
        w.append("目前非交易時段")
    if t["mode"] == "daytrade" and t["action"] == "open" and hm >= 1300:
        w.append("13:00 後新開當沖倉風險高")
    if t["odd"] and t["shares"] < 1000:
        w.append("零股委託：需使用券商的「盤中零股」下單，成交量較少、可能成交不到")
    if t["mode"] == "daytrade" and t["side"] == "sell" and t["action"] == "open":
        w.append("先賣後買需有現股當沖資格")
    a = get_settings()["auto"]
    if t.get("stop") and t["action"] == "open":
        if a["enabled"] and a["exit"]:
            w.append(f"停損價 {t['stop']}：自動交易已開啟，觸及時系統會自動送出平倉單 (電腦與本系統需保持開啟)")
        else:
            w.append(f"停損價 {t['stop']}：自動出場未開啟，請到「自動交易」開啟，或自行在券商設觸價單")
    return w


# ----------------------------------------------------------------- 委託
def preview(req: dict) -> dict:
    """req: code, mode(daytrade/swing), side(buy/sell), action(open/close), price, shares(可省略=自動), stop, target"""
    info = data.resolve(req["code"])
    mode = req.get("mode", "swing")
    price = round_tick(float(req["price"]))
    action = req.get("action", "open")
    side = req.get("side", "buy")
    stop = round_tick(float(req["stop"]), "down" if side == "buy" else "up") if req.get("stop") else None
    sizing = None
    shares = int(req.get("shares") or 0)
    if not shares and action == "open":
        sizing = size_position(mode, price, stop)
        shares = sizing["shares"]
    pos_side = ("long" if side == "buy" else "short") if action == "open" else ("long" if side == "sell" else "short")
    if not shares and action == "close":
        pos = [p for p in open_positions(mode) if p["code"] == info["code"] and p["side"] == pos_side]
        shares = pos[0]["qty"] if pos else 0
    acc_id = req.get("account")
    if not acc_id and action == "close":  # 平倉 → 送回當初建倉的帳戶
        held = [p for p in open_positions(mode) if p["code"] == info["code"] and p["side"] == pos_side]
        acc_id = held[0].get("account") if held else None
    acc_id = acc_id or get_settings()["default_account"][mode]
    acc = brokers.get_account(acc_id)
    t = {"id": uuid.uuid4().hex[:10], "code": info["code"], "name": info["name"], "mode": mode,
         "account": acc_id, "account_name": acc["name"], "dry_run": bool(acc.get("dry_run")),
         "mode_name": "當沖" if mode == "daytrade" else "波段", "side": side, "action": action, "pos_side": pos_side,
         "price": price, "shares": shares, "lots": shares // 1000, "odd": shares % 1000,
         "amount": round(shares * price), "stop": stop, "target": req.get("target"), "target2": req.get("target2"),
         "auto": bool(req.get("auto")),
         "trade_type": "現股當沖" if mode == "daytrade" else "現股", "reason": req.get("reason", ""),
         "sizing": sizing, "created": _now().isoformat(timespec="seconds")}
    t["errors"] = risk_check(t)
    t["warnings"] = warnings(t)
    if acc["type"] == "web":
        cal = brokers.calibrated(acc_id)
        if not cal["order"]:
            t["errors"].append(f"{acc['name']} 下單頁尚未校正，無法下單 (請擷取下單頁並請 Claude 校正)")
        if acc.get("dry_run"):
            t["warnings"].append(f"{acc['name']} 為「只填單不送出」模式：只會填好委託單並截圖，不會真的送出")
    with _lock:
        lg = ledger()
        lg["pending"] = {k: v for k, v in lg["pending"].items() if v["created"][:10] == _today()}
        lg["pending"][t["id"]] = t
        _save(LEDGER_FILE, lg)
    return t


def confirm(order_id: str) -> dict:
    with _lock:
        lg = ledger()
        t = lg["pending"].pop(order_id, None)
        if not t:
            raise ValueError("找不到這張待確認委託 (可能已送出或過期)")
        errs = risk_check(t)  # 送出前再檢查一次
        if errs:
            _save(LEDGER_FILE, lg)
            raise ValueError("風控未通過：" + "；".join(errs))
        if t["account"] == "paper":
            result = {"status": "模擬成交", "fill_price": t["price"]}
        else:
            result = brokers.broker(t["account"]).place(t)
        t["result"] = result
        t["time"] = _now().isoformat(timespec="seconds")
        filled = result["status"] in ("模擬成交", "已送出")
        if filled:
            t["pnl"] = _apply_fill(lg, t)
        lg["orders"].append(t)
        _save(LEDGER_FILE, lg)
        return t


def _apply_fill(lg, t) -> float:
    """更新系統持倉，回傳已實現損益 (未扣成本)"""
    pos = next((p for p in lg["positions"] if p["code"] == t["code"] and p["mode"] == t["mode"]
                and p["side"] == t["pos_side"] and p["qty"] > 0
                and (t["mode"] != "daytrade" or p["date"] == _today())), None)
    if t["action"] == "open":
        if pos:
            pos["avg_price"] = round((pos["avg_price"] * pos["qty"] + t["price"] * t["shares"]) / (pos["qty"] + t["shares"]), 2)
            pos["qty"] += t["shares"]
        else:
            lg["positions"].append({"code": t["code"], "name": t["name"], "mode": t["mode"], "side": t["pos_side"],
                                    "account": t.get("account", "paper"), "account_name": t.get("account_name", ""),
                                    "qty": t["shares"], "avg_price": t["price"], "stop": t.get("stop"),
                                    "target": t.get("target"), "target2": t.get("target2"), "tp1_done": False,
                                    "date": _today(), "opened": _now().isoformat(timespec="seconds")})
        return 0.0
    if not pos:
        return 0.0
    q = min(pos["qty"], t["shares"])
    pnl = (t["price"] - pos["avg_price"]) * q * (1 if pos["side"] == "long" else -1)
    pos["qty"] -= q
    return round(pnl, 0)


def update_position(code, mode, side, **fields):
    """自動交易用：移動停損、標記已停利一半"""
    with _lock:
        lg = ledger()
        for p in lg["positions"]:
            if p["code"] == code and p["mode"] == mode and p["side"] == side and p["qty"] > 0:
                p.update(fields)
        _save(LEDGER_FILE, lg)


def cancel(order_id: str):
    with _lock:
        lg = ledger()
        lg["pending"].pop(order_id, None)
        _save(LEDGER_FILE, lg)


