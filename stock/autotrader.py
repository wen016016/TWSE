"""自動交易引擎 (背景執行緒，每 2 秒一輪；自動進場掃描在另一條執行緒跑，不會卡住停損監控)

自動出場 (auto.exit)：
  - 多單 現價 <= 停損 → 自動賣出；空單 現價 >= 停損 → 自動買回
  - 到目標一：先出一半 (張數 >= 2 時)，停損移到成本 (保本)，目標換成目標二；張數不足就全出
  - 當沖：flatten_time (預設 13:20) 強制平倉所有當沖部位
自動進場 (auto.entry)：
  - 當沖：09:15~12:30 每 scan_interval_min 分鐘掃描一次
  - 波段：每天 swing_entry_time 掃描一次
  - 只買「可進場 + 分數 >= 門檻 + 額度足夠」的標的，每日最多 max_new_* 檔，同一檔不重複進場
委託價：用即時委買/委賣價，再往成交方向多掛 slippage_ticks 檔 (限價，不超過漲跌停)
所有自動委託一樣經過風控 (額度、張數、價格偏離、漲跌停)
"""
import datetime as dt
import json
import threading
import time
import traceback

from . import broker
from .config import DATA_DIR
from .indicators import tick

LOG_FILE = DATA_DIR / "auto_log.json"


def _now():
    return dt.datetime.now(dt.timezone(dt.timedelta(hours=8)))


def _hm(s: str) -> int:
    h, m = s.split(":")
    return int(h) * 100 + int(m)


class AutoTrader(threading.Thread):
    def __init__(self, interval=2):
        super().__init__(daemon=True)
        self.interval = interval
        self.cooldown: dict = {}       # key -> 下次可再嘗試的時間 (避免重複送單)
        self.last_scan: dict = {}      # mode -> timestamp
        self.swing_done_date = None
        self.status = {"running": True, "last_tick": None, "last_error": None}
        self.lock = threading.Lock()

    # ------------------------------------------------------------ log
    def log(self, kind, msg, **extra):
        try:
            logs = json.loads(LOG_FILE.read_text(encoding="utf-8"))
        except Exception:
            logs = []
        logs.append({"time": _now().isoformat(timespec="seconds"), "kind": kind, "msg": msg, **extra})
        LOG_FILE.write_text(json.dumps(logs[-300:], ensure_ascii=False, indent=1), encoding="utf-8")

    @staticmethod
    def logs(n=100):
        try:
            return json.loads(LOG_FILE.read_text(encoding="utf-8"))[-n:][::-1]
        except Exception:
            return []

    # ------------------------------------------------------------ loop
    def run(self):
        while True:
            try:
                self.step()
                self.status["last_error"] = None
            except Exception as e:  # noqa: BLE001
                self.status["last_error"] = f"{type(e).__name__}: {e}"
                traceback.print_exc()
            self.status["last_tick"] = _now().isoformat(timespec="seconds")
            time.sleep(self.interval)

    def sync_account(self, s, hm):
        """每個「瀏覽器已開 + 帳務已校正」的券商帳戶，盤中每 60 秒同步一次庫存 / 可用餘額"""
        from . import brokers
        if not (830 <= hm <= 1400) or time.time() - getattr(self, "_last_sync", 0) < 60:
            return
        self._last_sync = time.time()
        for b in brokers.open_web_accounts():
            if not brokers.calibrated(b.id)["account"]:
                continue
            try:
                b.sync_account()
            except Exception as e:  # noqa: BLE001
                self.status["last_error"] = f"{b.id} 帳務同步失敗：{e}"

    def step(self):
        s = broker.get_settings()
        a = s["auto"]
        now = _now()
        hm = now.hour * 100 + now.minute
        if now.weekday() < 5:
            self.sync_account(s, hm)
        if not a["enabled"] or now.weekday() >= 5 or not (900 <= hm <= 1330):
            return
        with self.lock:
            if a["exit"]:
                self.check_exits(a, hm)
            if a["entry"] and hm < _hm(a["flatten_time"]):
                self.maybe_enter(a, hm, now)

    # ------------------------------------------------------------ 出場
    def check_exits(self, a, hm):
        from .realtime import quotes
        pos = broker.open_positions()
        if not pos:
            return
        q = quotes([p["code"] for p in pos])
        for p in pos:
            qq = q.get(p["code"])
            if not qq or not qq.get("price"):
                continue
            px = qq["price"]
            long = p["side"] == "long"
            stop, target = p.get("stop"), p.get("target")
            if p["mode"] == "daytrade" and hm >= _hm(a["flatten_time"]):
                self.close(p, p["qty"], qq, a, f"當沖 {a['flatten_time']} 強制平倉 (現價 {px})")
            elif stop and (px <= stop if long else px >= stop):
                self.close(p, p["qty"], qq, a, f"觸及停損 {stop} (現價 {px})")
            elif target and (px >= target if long else px <= target):
                lots = p["qty"] // 1000
                if a["half_at_target1"] and not p.get("tp1_done") and lots >= 2:
                    half = (lots // 2) * 1000
                    if self.close(p, half, qq, a, f"到目標一 {target}，先停利一半 (現價 {px})"):
                        nxt = p.get("target2") or None
                        broker.update_position(p["code"], p["mode"], p["side"], tp1_done=True,
                                               stop=p["avg_price"], target=nxt)
                        self.log("調整", f"{p['code']} 停損移到成本 {p['avg_price']}，目標改為 {nxt or '無 (續抱到停損)'}")
                else:
                    self.close(p, p["qty"], qq, a, f"到達目標 {target}，全數停利 (現價 {px})")

    def _limit_price(self, qq, side, a):
        """往成交方向多掛幾檔，確保成交；不超過漲跌停"""
        px = qq["price"]
        n = a["slippage_ticks"]
        if side == "sell":
            base = qq["bid"][0] if qq["bid"] else px
            p = base - tick(base) * n
            if qq.get("limit_down"):
                p = max(p, qq["limit_down"])
        else:
            base = qq["ask"][0] if qq["ask"] else px
            p = base + tick(base) * n
            if qq.get("limit_up"):
                p = min(p, qq["limit_up"])
        return p

    def close(self, p, shares, qq, a, reason) -> bool:
        key = ("close", p["code"], p["mode"], p["side"], shares)
        if self.cooldown.get(key, 0) > time.time():
            return False
        side = "sell" if p["side"] == "long" else "buy"
        price = self._limit_price(qq, side, a)
        return self._send({"code": p["code"], "mode": p["mode"], "side": side, "action": "close", "price": price,
                           "shares": shares, "account": p.get("account"), "auto": True,
                           "reason": "自動出場：" + reason}, key, 60)

    # ------------------------------------------------------------ 進場
    def maybe_enter(self, a, hm, now):
        modes = []
        if a["entry_daytrade"] and 915 <= hm <= 1230:
            last = self.last_scan.get("daytrade", 0)
            if time.time() - last >= a["scan_interval_min"] * 60:
                modes.append("daytrade")
        if a["entry_swing"] and hm >= _hm(a["swing_entry_time"]) and self.swing_done_date != now.date():
            modes.append("swing")
        if not modes or getattr(self, "_scanning", False):
            return
        for mode in modes:
            self.last_scan[mode] = time.time()
            if mode == "swing":
                self.swing_done_date = now.date()

        def job():
            self._scanning = True
            try:
                for mode in modes:  # 送單時 broker 內部會加鎖，掃描期間停損監控照常運作
                    self.enter(mode, a)
            except Exception as e:  # noqa: BLE001
                self.status["last_error"] = f"自動進場失敗：{e}"
            finally:
                self._scanning = False
        threading.Thread(target=job, daemon=True).start()

    def enter(self, mode, a):
        from . import scanner
        from .realtime import quotes
        name = "當沖" if mode == "daytrade" else "波段"
        today = broker._today()
        opened = [o for o in broker.ledger()["orders"]
                  if o["time"].startswith(today) and o.get("auto") and o["action"] == "open" and o["mode"] == mode]
        room = a[f"max_new_{mode}"] - len({o["code"] for o in opened})
        if room <= 0:
            return
        min_score = a[f"min_score_{mode}"]
        self.log("掃描", f"{name}自動掃描開始 (門檻 {min_score} 分，今日還可開 {room} 檔)")
        r = scanner.scan(mode, top=8)
        held = {(p["code"], p["mode"]) for p in broker.open_positions()}
        cands = [x for x in r["results"] if x["actionable"] and x["total"] >= min_score and (x["code"], mode) not in held]
        if not cands:
            self.log("掃描", f"{name}沒有符合條件的標的")
            return
        q = quotes([x["code"] for x in cands])
        for x in cands:
            if room <= 0:
                break
            qq = q.get(x["code"])
            if not qq or not qq.get("price"):
                continue
            side = x["side"]
            if side == "buy" and not qq["ask"]:
                self.log("略過", f"{x['code']} {x['name']} 漲停鎖住買不到")
                continue
            price = self._limit_price(qq, side, a)
            p = x["plan"]
            size = broker.size_position(mode, price, p["stop"])
            if not size["shares"]:
                self.log("略過", f"{x['code']} {x['name']}：{size['note'][-1]}")
                if broker.budget_status()[mode]["available"] < price * 1000:
                    self.log("掃描", f"{name}可用額度不足，本輪停止自動進場")
                    break
                continue
            key = ("open", x["code"], mode)
            if self._send({"code": x["code"], "mode": mode, "side": side, "action": "open", "price": price,
                           "stop": p["stop"], "target": p["target1"], "target2": p["target2"], "auto": True,
                           "reason": f"自動進場：{x['tone']} {x['total']} 分；" + "；".join(x["top_reasons"][:3])},
                          key, 1800):
                room -= 1

    # ------------------------------------------------------------ 送單
    def _send(self, req, key, cool) -> bool:
        self.cooldown[key] = time.time() + cool
        t = broker.preview(req)
        label = f"{t['code']} {t['name']} {t['mode_name']}{'新倉' if t['action'] == 'open' else '平倉'}{'買進' if t['side'] == 'buy' else '賣出'} {t['shares']:,} 股 @ {t['price']}"
        if t["errors"]:
            broker.cancel(t["id"])
            self.log("擋單", f"{label}：{'；'.join(t['errors'])}", reason=req.get("reason"))
            return False
        try:
            r = broker.confirm(t["id"])
        except Exception as e:  # noqa: BLE001
            self.log("失敗", f"{label}：{e}", reason=req.get("reason"))
            return False
        status = r["result"]["status"]
        ok = status in ("模擬成交", "已送出")
        self.log("委託" if ok else "未送出", f"{label}：{status}", reason=req.get("reason"), pnl=r.get("pnl"))
        if not ok:  # dry_run 只填單 → 5 分鐘內不再重試
            self.cooldown[key] = time.time() + 300
        return ok


trader = AutoTrader()
