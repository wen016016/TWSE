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
import os
import queue
import re
import shutil
import subprocess
import threading
import time
from concurrent.futures import Future

import requests

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
        a["submit_mode"] = submit_mode(a)
        a["synced"] = bool(account_data(a["id"]))
        a["browser_open"] = _port_alive(_port(a["id"]))
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
             "login_url": login_url or preset.get("login_url", ""), "dry_run": True, "submit_mode": "manual"}
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
                for k in ("name", "login_url", "dry_run", "cap_by_cash", "submit_mode"):
                    if k in fields:
                        a[k] = fields[k]
                _save(ACCOUNTS_FILE, accs)
                return a
        raise ValueError(f"找不到券商帳戶 {acc_id}")


def remove_account(acc_id: str):
    with _lock:
        accs = [a for a in _load(ACCOUNTS_FILE, []) if a["id"] != acc_id]
        _save(ACCOUNTS_FILE, accs)


SUBMIT_MODES = {"manual": "你在券商按確定", "auto": "系統自動確定", "test": "測試(填好後取消)"}


def submit_mode(acc: dict) -> str:
    m = acc.get("submit_mode")
    if m in SUBMIT_MODES:
        return m
    return "manual" if acc.get("dry_run", True) else "auto"  # 舊設定相容


def selectors(acc_id: str) -> dict:
    return _load(SEL_DIR / f"{acc_id}.json", {})


def calibrated(acc_id: str) -> dict:
    s = selectors(acc_id)
    return {"order": all(s.get(k) for k in ("code", "qty", "price", "buy", "sell", "confirm")),
            "account": bool(s.get("account_api") or s.get("account", {}).get("holdings_table"))}


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


def _active_page(ctx):
    """使用者目前正在看的分頁 (含彈出視窗)；找不到就用最後開的"""
    for p in reversed(ctx.pages):
        try:
            if p.evaluate("document.visibilityState") == "visible" and p.evaluate("document.hasFocus()"):
                return p
        except Exception:  # noqa: BLE001
            pass
    for p in reversed(ctx.pages):
        try:
            if p.evaluate("document.visibilityState") == "visible":
                return p
        except Exception:  # noqa: BLE001
            pass
    return ctx.pages[-1]


EDGE_PATHS = [r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
              r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"]


def _edge_exe():
    for p in EDGE_PATHS:
        if os.path.exists(p):
            return p
    p = shutil.which("msedge")
    if p:
        return p
    raise RuntimeError("找不到 Microsoft Edge")


def _port(acc_id):
    return 9300 + sum(map(ord, acc_id)) % 500


def _port_alive(port) -> bool:
    try:
        return requests.get(f"http://127.0.0.1:{port}/json/version", timeout=1).ok
    except Exception:  # noqa: BLE001
        return False


class WebBroker:
    """用網頁下單的券商帳戶

    不用自動化工具「啟動」瀏覽器 (券商網站會偵測自動化，安全元件 / 彈窗 / 憑證會失效)，
    而是像平常一樣開一個正常的 Edge (專用設定檔 + 偵錯連接埠)，需要擷取 / 下單時才連上去操作。
    """

    def __init__(self, acc_id):
        self.id = acc_id
        self.ctx = None
        self.browser = None
        self.port = _port(acc_id)

    @property
    def acc(self):
        return get_account(self.id)

    def alive(self) -> bool:
        return _port_alive(self.port)

    def _need_ctx(self):
        if not self.alive():
            self.ctx = self.browser = None
            raise RuntimeError(f"{self.acc['name']} 瀏覽器未開啟，請先按「開啟瀏覽器」並登入")

    def _connect(self, pw):
        """在瀏覽器執行緒裡連上這個帳戶的 Edge (斷線就重連)"""
        if self.browser is not None and self.browser.is_connected() and self.ctx is not None:
            return
        self.browser = pw.chromium.connect_over_cdp(f"http://127.0.0.1:{self.port}")
        self.ctx = self.browser.contexts[0] if self.browser.contexts else self.browser.new_context()

    def open(self):
        acc = self.acc
        url = acc.get("login_url")
        if not url:
            raise RuntimeError(f"{acc['name']} 尚未設定登入網址")
        if not self.alive():
            prof = PROFILE_DIR / self.id
            prof.mkdir(parents=True, exist_ok=True)
            subprocess.Popen([_edge_exe(), f"--remote-debugging-port={self.port}", f"--user-data-dir={prof}",
                              "--no-first-run", "--no-default-browser-check", url])
            for _ in range(40):
                if self.alive():
                    break
                time.sleep(0.5)
            else:
                raise RuntimeError("Edge 沒有啟動成功：如果這個帳戶的 Edge 視窗已經開著，請先把它關掉再按一次")
            return {"status": f"已開啟 {acc['name']} 的 Edge，請手動登入 (第一次需在這個 Edge 申請/匯入憑證)", "url": url}

        def fn(pw):
            self._connect(pw)
            page = self.ctx.new_page()
            page.goto(url)
            page.bring_to_front()
            return {"status": f"{acc['name']} 的 Edge 已經開著，已開新分頁到登入頁", "url": url}

        return _w().call(fn)

    def snapshot(self, label="page"):
        """目前頁面 (含所有 iframe) 的 HTML + 截圖存到本機，給 Claude 校正 selector"""
        self._need_ctx()
        ts = _now().strftime("%Y%m%d_%H%M%S")
        safe = "".join(ch for ch in label if ch.isalnum() or ch in "-_") or "page"
        out_dir = SNAP_DIR / self.id
        out_dir.mkdir(exist_ok=True)

        def fn(pw):
            self._connect(pw)
            page = _active_page(self.ctx)
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

    # ------------------------------------------------------------ 帳務同步
    def _sync_page(self):
        """同步專用分頁 (不碰使用者正在操作的分頁)；關掉了就重開"""
        p = getattr(self, "_sp", None)
        if p is None or p.is_closed():
            p = self._sp = self.ctx.new_page()
        return p

    @staticmethod
    def _rows(page, table_sel):
        fr = _find(page, table_sel)
        if not fr:
            return None
        return fr.eval_on_selector_all(f"{table_sel} tr",
                                       "rs => rs.filter(r => r.offsetParent !== null || r.closest('table'))"
                                       ".map(r => [...r.querySelectorAll('td')].map(td => td.innerText.trim()))")

    def sync_account(self):
        api = selectors(self.id).get("account_api")
        if api:
            return self._sync_api(api)
        cfg = selectors(self.id).get("account", {})
        if not cfg.get("holdings_table"):
            raise RuntimeError(f"{self.acc['name']} 帳務頁面尚未校正：登入後到庫存頁、帳務頁各按一次「擷取」，再請 Claude 設定")
        self._need_ctx()

        def goto(page, url):
            if not url:
                return
            if page.url != url:
                page.goto(url)
                page.wait_for_timeout(2500)
            elif cfg.get("refresh"):
                try:
                    page.click(cfg["refresh"], timeout=2000)
                    page.wait_for_timeout(1500)
                except Exception:  # noqa: BLE001
                    page.reload()
                    page.wait_for_timeout(2500)

        def fn(pw):
            self._connect(pw)
            page = self._sync_page() if cfg.get("holdings_url") else _active_page(self.ctx)
            goto(page, cfg.get("holdings_url"))
            rows = None
            for _ in range(10):  # 表格是非同步載入，等一下
                rows = self._rows(page, cfg["holdings_table"])
                if rows:
                    break
                page.wait_for_timeout(500)
            if rows is None:
                if "login" in page.url.lower() or "default.aspx" not in page.url:
                    raise RuntimeError(f"{self.acc['name']} 似乎已登出，請在 Edge 重新登入")
                raise RuntimeError("找不到庫存表格")
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
                name = r[cols["name"]].replace(m.group(), "", 1).strip() if "name" in cols else ""
                h = {"code": m.group(), "name": name, "qty": int(qty * unit)}
                if "type" in cols:
                    h["type"] = r[cols["type"]]
                for k in ("avg_cost", "price", "pnl", "market_value"):
                    if k in cols:
                        h[k] = _num(r[cols[k]])
                holds.append(h)

            cash = None
            if cfg.get("cash_selector"):
                goto(page, cfg.get("cash_url"))
                fc = _find(page, cfg["cash_selector"])
                if fc:
                    cash = _num(fc.inner_text(cfg["cash_selector"]))
            settles = []
            if cfg.get("settle_table"):
                goto(page, cfg.get("settle_url"))
                sc = cfg.get("settle_cols", {})
                for r in self._rows(page, cfg["settle_table"]) or []:
                    if len(r) <= max(sc.values(), default=0):
                        continue
                    amt = _num(r[sc["amount"]])
                    if amt is None or not re.match(r"\d{4}/\d{2}/\d{2}", r[sc["trade_date"]]):
                        continue
                    settles.append({"trade_date": r[sc["trade_date"]], "settle_date": r[sc["settle_date"]], "amount": amt})
                goto(page, cfg.get("holdings_url"))  # 停回庫存頁，下次只要按更新
            acc = {"synced": _now().isoformat(timespec="seconds"), "holdings": holds, "cash": cash,
                   "settlements": settles,
                   "market_value": sum(h.get("market_value") or 0 for h in holds) or None,
                   "unrealized": sum(h.get("pnl") or 0 for h in holds) or None}
            _save(_acc_file(self.id), acc)
            return acc

        return _w().call(fn, timeout=90)

    _FETCH_JS = """async ([url, params]) => {
        const r = await fetch(url, {method: 'POST', body: new URLSearchParams(params), credentials: 'include',
            headers: {'Content-Type': 'application/x-www-form-urlencoded; charset=UTF-8', 'X-Requested-With': 'XMLHttpRequest'}});
        return await r.text();
    }"""

    def _sync_api(self, api):
        """在使用者已登入的分頁裡，呼叫券商網站自己的資料介面讀庫存 / 餘額 (不切換畫面)"""
        self._need_ctx()
        acc = self.acc

        def call(page, spec):
            txt = page.evaluate(self._FETCH_JS, [spec["url"], spec.get("params", {})])
            try:
                j = json.loads(txt)
            except ValueError:
                raise RuntimeError(f"{acc['name']} 似乎已登出，請在 Edge 重新登入") from None
            if j.get("s") not in (None, "000"):
                raise RuntimeError(f"{acc['name']} 回傳錯誤：{j.get('m')}")
            return j

        def fn(pw):
            self._connect(pw)
            pages = [p for p in self.ctx.pages if api["page_match"] in p.url]
            if not pages:
                raise RuntimeError(f"找不到已登入的 {acc['name']} 分頁，請在 Edge 登入 {acc['name']}")
            page = pages[0]
            ps = api["position"]
            j = call(page, ps)
            holds = []
            for r in j.get(ps["list"], []):
                m = ps["map"]
                qty = _num(r.get(m["qty"], ""))
                if not qty:
                    continue
                h = {"code": r.get(m["code"], "").strip(), "name": r.get(m.get("name", ""), "").strip(), "qty": int(qty)}
                for k in ("type", "pnl_pct"):
                    if k in m:
                        h[k] = r.get(m[k], "")
                for k in ("price", "market_value", "pnl", "avg_cost"):
                    if k in m:
                        h[k] = _num(r.get(m[k], ""))
                holds.append(h)
            cash, cash_name = None, ""
            if api.get("bank"):
                try:
                    b = call(page, api["bank"])
                    lst = b.get(api["bank"]["list"], [])
                    if lst:
                        cash = sum(_num(x.get(api["bank"]["cash"], "")) or 0 for x in lst)
                        cash_name = "、".join(x.get(api["bank"].get("name", ""), "") for x in lst)
                except RuntimeError:
                    pass
            out = {"synced": _now().isoformat(timespec="seconds"), "holdings": holds, "cash": cash, "cash_name": cash_name,
                   "market_value": _num(j.get(ps.get("total", ""), "")) or sum(h.get("market_value") or 0 for h in holds) or None,
                   "unrealized": sum(h.get("pnl") or 0 for h in holds) or None, "broker_update": j.get("update")}
            _save(_acc_file(self.id), out)
            return out

        return _w().call(fn, timeout=60)

    # ------------------------------------------------------------ 下單
    def _orders(self, page, oa):
        """券商委託回報 (用網站自己的資料介面)"""
        try:
            j = json.loads(page.evaluate(self._FETCH_JS, [oa["url"], oa.get("params", {})]))
        except Exception:  # noqa: BLE001
            return None
        f = oa["fields"]
        return [{k: r.get(v, "") for k, v in f.items()} for r in j.get(oa["list"], [])]

    def place(self, t: dict) -> dict:
        """直接在使用者登入的券商網頁上操作：選市場 → 填股號/數量/價格 → 按買進/賣出 → 跳出券商確認視窗
        送單模式 (帳戶 submit_mode)：
          manual = 停在券商確認視窗，由使用者自己按「確定 / 取消」(預設)
          auto   = 系統按「確定」(自動交易用)
          test   = 填好 + 截圖後按「取消」(校正測試)
        manual / auto 送出後查券商委託回報，確認真的有這筆委託才算「已送出」"""
        sel = selectors(self.id)
        miss = [k for k in ("code", "qty", "price", "buy", "sell") if not sel.get(k)]
        acc = self.acc
        if miss:
            raise RuntimeError(f"{acc['name']} 下單頁尚未校正 (缺少 {miss})，請先擷取下單頁並請 Claude 校正")
        self._need_ctx()
        mode = submit_mode(acc)
        oa = sel.get("orders_api")

        def fn(pw):
            self._connect(pw)
            match = sel.get("order_page_match", "")
            sp = getattr(self, "_sp", None)
            cands = [p for p in self.ctx.pages if match and match in p.url and p is not sp]
            page = next((p for p in cands if p.evaluate("document.visibilityState") == "visible"), None) or (cands[0] if cands else None)
            if page is None:
                if not sel.get("order_page_url"):
                    raise RuntimeError(f"找不到 {acc['name']} 下單頁，請在 Edge 打開下單畫面")
                page = self.ctx.new_page()
                page.goto(sel["order_page_url"])
                page.wait_for_timeout(2500)
            page.bring_to_front()
            frame = _find(page, sel["code"])
            if not frame:
                raise RuntimeError(f"在 {acc['name']} 頁面找不到下單欄位 (可能已登出或下單列被收起來)")

            odd = t["shares"] % 1000 != 0
            if sel.get("reset"):
                frame.click(sel["reset"])
            mk = sel.get("market_odd") if odd else sel.get("market_regular")
            if mk:
                frame.click(mk)
            for s in sel.get("pre_click", []):
                if frame.query_selector(s) and frame.is_visible(s):
                    frame.click(s)
            dlg = sel.get("confirm_dialog")

            def dialogs():
                return [d for d in frame.query_selector_all(dlg) if d.is_visible()] if dlg else []

            def close_dialogs():
                """關掉殘留的對話框：下單確認 → 一律按取消 (絕不按確定)；一般提醒 → 按它唯一的按鈕"""
                for d in dialogs():
                    txt = d.inner_text()
                    btns = [b for b in d.query_selector_all("button") if b.is_visible()]
                    if "下單確認" in txt or len(btns) > 1:
                        cancel = next((b for b in btns if "取" in (b.inner_text() or "")), None)
                        if cancel:
                            cancel.click()
                    elif btns:
                        btns[0].click()
                    frame.wait_for_timeout(300)

            def same(a, b):
                a = (a or "").replace(",", "").strip()
                try:
                    return abs(float(a) - float(b)) < 1e-9
                except ValueError:
                    return a == b

            def set_value(s, val):
                """像人一樣點欄位、清空、打字；網站常在查股號後重設欄位，所以填完要核對"""
                for _ in range(3):
                    frame.click(s, click_count=3)
                    frame.fill(s, "")
                    frame.type(s, val, delay=40)
                    frame.dispatch_event(s, "change")
                    frame.dispatch_event(s, "blur")
                    frame.wait_for_timeout(250)
                    if same(frame.input_value(s), val):
                        return
                    frame.wait_for_timeout(400)
                raise RuntimeError(f"{acc['name']} 欄位 {s} 填不進 {val} (網站可能重設了欄位)，已停止，沒有送單")

            close_dialogs()
            set_value(sel["code"], t["code"])
            if sel.get("code_ready"):  # 等網站查到股票名稱
                for _ in range(25):
                    if (frame.input_value(sel["code_ready"]) or "").strip():
                        break
                    frame.wait_for_timeout(200)
                else:
                    raise RuntimeError(f"{acc['name']} 查不到股號 {t['code']}")
                frame.wait_for_timeout(800)  # 網站查完股號還會重設數量/價格，等它做完
            close_dialogs()
            qty = str(t["shares"]) if odd else str(t["shares"] // 1000)
            price = f"{t['price']:g}"
            set_value(sel["qty"], qty)
            set_value(sel["price"], price)
            if sel.get("tif_select") and frame.is_enabled(sel["tif_select"]):  # 盤中零股時選單鎖定 (只能 ROD)
                frame.select_option(sel["tif_select"], sel.get("tif_value", "R"), timeout=3000)
            sell_first = t["mode"] == "daytrade" and t["action"] == "open" and t["side"] == "sell"
            js = sel.get("sell_first_js") if sell_first else sel.get("sell_first_off_js")
            if js:
                frame.evaluate(js)
            # 按下單鍵前最後核對一次三個欄位
            now_vals = {k: (frame.input_value(sel[k]) or "").strip() for k in ("code", "qty", "price")}
            if not (now_vals["code"] == t["code"] and same(now_vals["qty"], qty) and same(now_vals["price"], price)):
                raise RuntimeError(f"{acc['name']} 欄位核對不符 {now_vals}，已停止，沒有送單")
            before_nos = {str(o["no"]) for o in (self._orders(page, oa) or [])} if oa else set()
            frame.click(sel["buy"] if t["side"] == "buy" else sel["sell"])

            shot = SCREEN_DIR / f"{t['id']}_confirm.png"
            text = ""
            if dlg:
                try:
                    frame.wait_for_selector(dlg, state="visible", timeout=5000)
                except Exception:  # noqa: BLE001
                    page.screenshot(path=str(shot))
                    raise RuntimeError(f"{acc['name']} 沒有跳出下單確認視窗 (可能欄位有誤，請看截圖 {shot})")
                frame.wait_for_timeout(300)
                text = "\n".join(d.inner_text() for d in dialogs())
            page.screenshot(path=str(shot))
            if text and "下單確認" not in text:  # 跳的是錯誤提醒，不是確認視窗
                close_dialogs()
                raise RuntimeError(f"{acc['name']} 提醒：{text.strip()[:150]}")
            # 確認視窗內容核對：股號、數量要對得上才按確定
            if text and (t["code"] not in text or qty not in text.replace(",", "")):
                close_dialogs()
                raise RuntimeError(f"確認視窗內容與委託不符，已取消：{text[:120]}")
            if mode == "test":
                close_dialogs()
                return {"status": "已填單未送出 (測試模式，已在確認視窗按取消)", "screenshot": str(shot), "confirm_text": text}
            if mode == "auto":
                frame.click(sel["confirm"])
            else:  # manual：停在確認視窗，等使用者在券商頁面按確定 / 取消 (最多 120 秒)
                for _ in range(240):
                    if not dialogs() or "下單確認" not in " ".join(d.inner_text() for d in dialogs()):
                        break
                    page.wait_for_timeout(500)
                else:
                    close_dialogs()
                    return {"status": "逾時未確認 (已取消)", "screenshot": str(shot), "confirm_text": text}
            # 送出後：等憑證簽章 + 送單，再查委託回報確認
            msg = ""
            new = []
            for _ in range(16):
                page.wait_for_timeout(500)
                try:
                    vis = [d.inner_text() for d in dialogs()]
                    if vis:
                        msg = " / ".join(vis)
                except Exception:  # noqa: BLE001
                    pass
                if oa:
                    after = self._orders(page, oa) or []
                    new = [o for o in after if str(o["no"]) not in before_nos and o["code"] == t["code"]
                           and o["bs"] == ("B" if t["side"] == "buy" else "S")]
                    if new:
                        break
            shot2 = SCREEN_DIR / f"{t['id']}_submitted.png"
            page.screenshot(path=str(shot2))
            if msg and "下單確認" not in msg:
                close_dialogs()
            if oa:
                if not new:
                    st = "未送出 (你在券商取消了)" if mode == "manual" else "送出後查無委託 (請到券商委託回報確認)"
                    return {"status": st, "screenshot": str(shot2), "broker_msg": msg}
                o = new[0]
                failed = bool(o["error"]) or "失敗" in o["status"]
                return {"status": "委託失敗" if failed else "已送出", "screenshot": str(shot2), "fill_price": t["price"],
                        "ordno": o["ordno"], "broker_status": o["status"], "broker_error": o["error"],
                        "matched": o["matched"], "broker_msg": msg,
                        "note": f"券商委託書號 {o['ordno']}：{o['status']} {o['error']}".strip()}
            return {"status": "已送出", "screenshot": str(shot2), "fill_price": t["price"], "broker_msg": msg,
                    "note": "已送出委託；是否成交請以券商委託/成交回報為準"}

        return _w().call(fn, timeout=180)


_brokers: dict[str, WebBroker] = {}


def broker(acc_id: str) -> WebBroker:
    if acc_id == "paper":
        raise ValueError("模擬帳戶不需要瀏覽器")
    get_account(acc_id)
    if acc_id not in _brokers:
        _brokers[acc_id] = WebBroker(acc_id)
    return _brokers[acc_id]


def open_web_accounts() -> list[WebBroker]:
    return [broker(a["id"]) for a in accounts(False) if a["browser_open"]]
