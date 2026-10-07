"""多券商帳戶 (網頁下單自動化)

每個帳戶 = 一家券商的一個登入：
  - 自己的瀏覽器設定檔 (登入狀態、憑證分開保存) → 國票、國泰可以同時登入
  - 自己的欄位校正檔 brokers/<帳戶id>.json (下單欄位、庫存表、可用餘額位置)
  - 自己的帳務同步結果 (庫存 / 可用餘額)

任何有「網頁下單」的台灣券商都能接：新增帳戶 → 開啟瀏覽器手動登入 → 擷取頁面 → 請 Claude 校正。
有提供程式交易 API 的券商 (例如永豐 Shioaji、富邦 Neo) 之後可另外加 API 型帳戶。
"""
import datetime as dt
import json
import queue
import re
import shutil
import threading
from concurrent.futures import Future

from .config import BASE_DIR, DATA_DIR

ACCOUNTS_FILE = DATA_DIR / "accounts.json"
SEL_DIR = BASE_DIR / "brokers"
SEL_DIR.mkdir(exist_ok=True)
PROFILE_DIR = DATA_DIR / "profiles"
SNAP_DIR = DATA_DIR / "snapshots"
SCREEN_DIR = DATA_DIR / "screens"
for _d in (PROFILE_DIR, SNAP_DIR, SCREEN_DIR):
    _d.mkdir(exist_ok=True)

# 常見券商 (登入網址請改成你實際用的「網頁下單」頁面)
PRESETS = [
    {"broker": "國票", "login_url": "https://www.ibfs.com.tw/"},
    {"broker": "國泰", "login_url": "https://www.cathaysec.com.tw/"},
    {"broker": "元大", "login_url": "https://www.yuanta.com.tw/"},
    {"broker": "富邦", "login_url": ""},
    {"broker": "凱基", "login_url": "https://www.kgi.com.tw/"},
    {"broker": "永豐金", "login_url": "https://www.sinotrade.com.tw/"},
    {"broker": "群益", "login_url": "https://www.capital.com.tw/"},
    {"broker": "統一", "login_url": ""},
    {"broker": "元富", "login_url": ""},
    {"broker": "兆豐", "login_url": ""},
    {"broker": "台新", "login_url": ""},
    {"broker": "玉山", "login_url": ""},
    {"broker": "華南永昌", "login_url": ""},
    {"broker": "第一金", "login_url": ""},
    {"broker": "康和", "login_url": ""},
    {"broker": "其他", "login_url": ""},
]

SEL_TEMPLATE = {
    "_說明": "這家券商網頁下單頁面的 CSS selector。全部空白 = 尚未校正，系統會拒絕下單 / 同步帳務。"
             "登入後在「券商帳戶」按「擷取目前頁面」，再請 Claude 讀擷取檔幫你填。",
    "order_page_match": "",
    "code": "", "buy": "", "sell": "", "qty": "", "price": "", "odd_lot": "",
    "trade_type": "", "trade_type_options": {"現股": "", "現股當沖": ""},
    "submit": "", "confirm": "",
    "account": {
        "holdings_url": "", "holdings_table": "",
        "holdings_cols": {"code": 0, "name": 1, "qty": 2, "avg_cost": 3, "price": 4, "market_value": 5, "pnl": 6},
        "qty_unit": "股", "cash_url": "", "cash_selector": "",
    },
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


# ----------------------------------------------------------------- 帳戶清單
PAPER = {"id": "paper", "name": "模擬下單", "broker": "模擬", "type": "paper", "dry_run": False}


def _migrate():
    """舊版只有國票：broker_selectors.json → brokers/ibf.json"""
    old = BASE_DIR / "broker_selectors.json"
    if not ACCOUNTS_FILE.exists():
        accs = []
        if old.exists():
            shutil.copy(old, SEL_DIR / "ibf.json")
            accs.append({"id": "ibf", "name": "國票", "broker": "國票", "type": "web",
                         "login_url": "https://www.ibfs.com.tw/", "dry_run": True})
        _save(ACCOUNTS_FILE, accs)


def accounts(include_paper=True) -> list[dict]:
    _migrate()
    accs = _load(ACCOUNTS_FILE, [])
    for a in accs:
        a["calibrated"] = calibrated(a["id"])
        a["synced"] = bool(account_data(a["id"]))
        a["browser_open"] = a["id"] in _brokers and _brokers[a["id"]].ctx is not None
    paper = {**PAPER, "calibrated": {"order": True, "account": False}, "synced": False, "browser_open": False}
    return ([paper] if include_paper else []) + accs


def get_account(acc_id: str) -> dict:
    for a in accounts():
        if a["id"] == acc_id:
            return a
    raise ValueError(f"找不到券商帳戶 {acc_id}")


def add_account(broker: str, name: str = "", login_url: str = "") -> dict:
    with _lock:
        accs = _load(ACCOUNTS_FILE, [])
        base = re.sub(r"[^a-z0-9]", "", {"國票": "ibf", "國泰": "cathay", "元大": "yuanta", "富邦": "fubon", "凱基": "kgi",
                                         "永豐金": "sinopac", "群益": "capital", "統一": "psc", "元富": "masterlink",
                                         "兆豐": "mega", "台新": "taishin", "玉山": "esun", "華南永昌": "hnc",
                                         "第一金": "first", "康和": "concord"}.get(broker, "acct"))
        acc_id, i = base, 2
        while any(a["id"] == acc_id for a in accs):
            acc_id, i = f"{base}{i}", i + 1
        preset = next((p for p in PRESETS if p["broker"] == broker), {})
        a = {"id": acc_id, "name": name or broker, "broker": broker, "type": "web",
             "login_url": login_url or preset.get("login_url", ""), "dry_run": True}
        accs.append(a)
        _save(ACCOUNTS_FILE, accs)
        f = SEL_DIR / f"{acc_id}.json"
        if not f.exists():
            _save(f, SEL_TEMPLATE)
        return a


def update_account(acc_id: str, **fields) -> dict:
    with _lock:
        accs = _load(ACCOUNTS_FILE, [])
        for a in accs:
            if a["id"] == acc_id:
                for k in ("name", "login_url", "dry_run"):
                    if k in fields:
                        a[k] = fields[k]
                _save(ACCOUNTS_FILE, accs)
                return a
        raise ValueError(f"找不到券商帳戶 {acc_id}")


def remove_account(acc_id: str):
    with _lock:
        accs = [a for a in _load(ACCOUNTS_FILE, []) if a["id"] != acc_id]
        _save(ACCOUNTS_FILE, accs)


def selectors(acc_id: str) -> dict:
    return _load(SEL_DIR / f"{acc_id}.json", {})


def calibrated(acc_id: str) -> dict:
    s = selectors(acc_id)
    return {"order": all(s.get(k) for k in ("code", "qty", "price", "buy", "sell")),
            "account": bool(s.get("account", {}).get("holdings_table"))}


# ----------------------------------------------------------------- 帳務資料
def _acc_file(acc_id):
    return DATA_DIR / f"account_{acc_id}.json"


def account_data(acc_id: str) -> dict | None:
    return _load(_acc_file(acc_id), None)


def account_summary(acc_id: str) -> dict | None:
    a = account_data(acc_id)
    if not a:
        return None
    age = (_now() - dt.datetime.fromisoformat(a["synced"])).total_seconds() / 60
    return {"synced": a["synced"], "age_min": round(age, 1), "cash": a.get("cash"),
            "holdings": len(a.get("holdings", [])), "market_value": a.get("market_value"),
            "unrealized": a.get("unrealized")}


# ----------------------------------------------------------------- 瀏覽器
class _BrowserWorker(threading.Thread):
    """Playwright sync API 只能在同一個執行緒使用；所有券商帳戶共用這一條執行緒"""

    def __init__(self):
        super().__init__(daemon=True)
        self.q: queue.Queue = queue.Queue()
        self.start()

    def run(self):
        from playwright.sync_api import sync_playwright
        pw = sync_playwright().start()
        while True:
            fn, fut = self.q.get()
            try:
                fut.set_result(fn(pw))
            except Exception as e:  # noqa: BLE001
                fut.set_exception(e)

    def call(self, fn, timeout=120):
        fut: Future = Future()
        self.q.put((fn, fut))
        return fut.result(timeout=timeout)


_worker = None


def _w():
    global _worker
    if _worker is None:
        _worker = _BrowserWorker()
    return _worker


def _num(s):
    s = str(s).replace(",", "").replace("元", "").replace("股", "").replace("張", "").replace("$", "").strip()
    try:
        return float(s)
    except ValueError:
        return None


def _find(page, sel):
    return next((f for f in page.frames if f.query_selector(sel)), None)


class WebBroker:
    """用網頁下單的券商帳戶"""

    def __init__(self, acc_id):
        self.id = acc_id
        self.ctx = None

    @property
    def acc(self):
        return get_account(self.id)

    def _need_ctx(self):
        if not self.ctx:
            raise RuntimeError(f"{self.acc['name']} 瀏覽器未開啟，請先按「開啟瀏覽器」並登入")

    def open(self):
        acc = self.acc
        if not acc.get("login_url"):
            raise RuntimeError(f"{acc['name']} 尚未設定登入網址")

        def fn(pw):
            if not self.ctx:
                prof = PROFILE_DIR / self.id
                prof.mkdir(parents=True, exist_ok=True)
                self.ctx = pw.chromium.launch_persistent_context(str(prof), channel="chrome", headless=False, no_viewport=True)
            page = self.ctx.pages[0] if self.ctx.pages else self.ctx.new_page()
            page.goto(acc["login_url"])
            page.bring_to_front()
            return {"status": f"已開啟 {acc['name']} 瀏覽器，請手動登入", "url": acc["login_url"]}

        return _w().call(fn)

    def snapshot(self, label="page"):
        """目前頁面 (含所有 iframe) 的 HTML + 截圖存到本機，給 Claude 校正 selector"""
        self._need_ctx()
        ts = _now().strftime("%Y%m%d_%H%M%S")
        safe = "".join(ch for ch in label if ch.isalnum() or ch in "-_") or "page"
        out_dir = SNAP_DIR / self.id
        out_dir.mkdir(exist_ok=True)

        def fn(pw):
            page = self.ctx.pages[-1]
            page.bring_to_front()
            parts = []
            for i, fr in enumerate(page.frames):
                try:
                    parts.append(f"<!-- ===== frame {i} url={fr.url} name={fr.name} ===== -->\n{fr.content()}")
                except Exception as e:  # noqa: BLE001
                    parts.append(f"<!-- frame {i} 讀取失敗 {e} -->")
            html = out_dir / f"{safe}_{ts}.html"
            html.write_text("\n".join(parts), encoding="utf-8")
            png = out_dir / f"{safe}_{ts}.png"
            page.screenshot(path=str(png), full_page=True)
            return {"status": f"已擷取 {self.acc['name']}「{safe}」頁面", "url": page.url, "html": str(html),
                    "png": str(png), "frames": len(page.frames)}

        return _w().call(fn)

    def sync_account(self):
        cfg = selectors(self.id).get("account", {})
        if not cfg.get("holdings_table"):
            raise RuntimeError(f"{self.acc['name']} 帳務頁面尚未校正：登入後到庫存頁、帳務頁各按一次「擷取目前頁面」，再請 Claude 設定")
        self._need_ctx()

        def read(page):
            fr = _find(page, cfg["holdings_table"])
            if not fr:
                raise RuntimeError("找不到庫存表格，請確認已登入並在庫存頁")
            rows = fr.eval_on_selector_all(f"{cfg['holdings_table']} tr",
                                           "rs => rs.map(r => [...r.querySelectorAll('td')].map(td => td.innerText.trim()))")
            cols = cfg.get("holdings_cols", {})
            unit = 1000 if cfg.get("qty_unit") == "張" else 1
            holds = []
            for r in rows:
                if len(r) <= max(cols.values(), default=0):
                    continue
                m = re.search(r"\d{4,6}[A-Z]?", r[cols["code"]])
                qty = _num(r[cols["qty"]])
                if not m or not qty:
                    continue
                h = {"code": m.group(), "name": r[cols["name"]] if "name" in cols else "", "qty": int(qty * unit)}
                for k in ("avg_cost", "price", "pnl", "market_value"):
                    if k in cols:
                        h[k] = _num(r[cols[k]])
                holds.append(h)
            cash = None
            if cfg.get("cash_selector"):
                if cfg.get("cash_url"):
                    page.goto(cfg["cash_url"])
                    page.wait_for_timeout(1500)
                fc = _find(page, cfg["cash_selector"])
                if fc:
                    cash = _num(fc.inner_text(cfg["cash_selector"]))
            acc = {"synced": _now().isoformat(timespec="seconds"), "holdings": holds, "cash": cash,
                   "market_value": sum(h.get("market_value") or 0 for h in holds) or None,
                   "unrealized": sum(h.get("pnl") or 0 for h in holds) or None}
            _save(_acc_file(self.id), acc)
            return acc

        def fn(pw):
            tmp = None  # 有設網址 → 另開分頁讀，不打斷使用者正在看的頁面
            if cfg.get("holdings_url"):
                tmp = page = self.ctx.new_page()
                page.goto(cfg["holdings_url"])
                page.wait_for_timeout(1500)
            else:
                page = self.ctx.pages[-1]
            try:
                return read(page)
            finally:
                if tmp:
                    tmp.close()

        return _w().call(fn)

    def place(self, t: dict) -> dict:
        sel = selectors(self.id)
        miss = [k for k in ("code", "qty", "price", "buy", "sell") if not sel.get(k)]
        acc = self.acc
        if miss:
            raise RuntimeError(f"{acc['name']} 下單頁尚未校正 (缺少 {miss})，請先擷取下單頁並請 Claude 校正")
        self._need_ctx()

        def fn(pw):
            match = sel.get("order_page_match", "")
            page = next((p for p in self.ctx.pages if match and match in p.url), self.ctx.pages[-1])
            page.bring_to_front()
            frame = _find(page, sel["code"])
            if not frame:
                raise RuntimeError(f"在 {acc['name']} 目前頁面找不到下單欄位，請確認已進入下單畫面")
            frame.fill(sel["code"], t["code"])
            frame.press(sel["code"], "Enter")
            frame.wait_for_timeout(500)
            frame.click(sel["buy"] if t["side"] == "buy" else sel["sell"])
            tt = sel.get("trade_type_options", {}).get(t["trade_type"])
            if sel.get("trade_type") and tt:
                frame.select_option(sel["trade_type"], tt)
            if t["shares"] < 1000 and sel.get("odd_lot"):
                frame.click(sel["odd_lot"])
            frame.fill(sel["qty"], str(t["lots"]) if t["shares"] >= 1000 else str(t["shares"]))
            frame.fill(sel["price"], str(t["price"]))
            shot = SCREEN_DIR / f"{t['id']}_filled.png"
            page.screenshot(path=str(shot))
            if acc.get("dry_run", True) or not sel.get("submit"):
                return {"status": "已填單未送出 (只填單模式)", "screenshot": str(shot)}
            frame.click(sel["submit"])
            if sel.get("confirm"):
                frame.wait_for_timeout(800)
                frame.click(sel["confirm"])
            frame.wait_for_timeout(1500)
            shot2 = SCREEN_DIR / f"{t['id']}_submitted.png"
            page.screenshot(path=str(shot2))
            return {"status": "已送出", "screenshot": str(shot2), "fill_price": t["price"]}

        return _w().call(fn)


_brokers: dict[str, WebBroker] = {}


def broker(acc_id: str) -> WebBroker:
    if acc_id == "paper":
        raise ValueError("模擬帳戶不需要瀏覽器")
    get_account(acc_id)
    if acc_id not in _brokers:
        _brokers[acc_id] = WebBroker(acc_id)
    return _brokers[acc_id]


def open_web_accounts() -> list[WebBroker]:
    return [b for b in _brokers.values() if b.ctx is not None]
