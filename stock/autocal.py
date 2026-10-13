"""券商網頁「帳務同步」自動校正 (不用 Claude)

使用者登入後打開庫存頁按「擷取」→ 系統自動在頁面 (含所有 iframe) 找：
  - 庫存表：表頭含「代號 / 名稱」+「股數 / 庫存」，資料列有股票代號
  - 可用餘額：文字「可用餘額 / 銀行餘額 / 交割帳戶餘額 …」後面的數字
找到就把「網址 + 表頭 + 欄位對應」存進該帳戶校正檔的 account.auto，之後同步帳務自動讀取。

只處理還沒有校正的帳戶；已有 account_api / holdings_table 的帳戶 (例如國票) 完全不經過這裡。
下單欄位 (代號 / 價格 / 數量 / 買賣鍵) 各家差異太大，仍需手動校正。
"""
import re

from . import brokers as B

# 表頭文字 → 欄位 (依序比對，先符合的優先；每個欄位只取一次)
HEAD_RULES = [
    ("_skip", r"損益率|報酬率|%|漲跌|幅"),
    ("pnl", r"損益"),
    ("market_value", r"市值|現值"),
    ("avg_cost", r"均價|平均成本|成本均價|單位成本"),
    ("_skip", r"成本|金額|費|稅"),
    ("price", r"現價|市價|成交價|最新價|收盤價|參考價|即時價"),
    ("qty", r"股數|庫存|數量|今餘"),
    ("code", r"代號|代碼"),
    ("name", r"名稱|股票|商品|證券|標的"),
    ("type", r"類別|種類|交易別|類型"),
]
CODE_RE = re.compile(r"(?<![\d.])(\d{4,6}[A-Z]?)(?![\d.])")
CASH_RE = re.compile(r"(可用餘額|可動用餘額|可用金額|銀行餘額|交割帳戶餘額|交割戶餘額|存款餘額|帳戶餘額)"
                     r"[\s:：]*(?:NT\$|\$|新台幣)?\s*(-?[\d,]+(?:\.\d+)?)")

_TABLES_JS = """() => [...document.querySelectorAll('table')].map(t =>
    [...t.querySelectorAll('tr')].filter(r => r.closest('table') === t)
      .map(r => [...r.querySelectorAll('th,td')].map(c => (c.innerText || '').trim().replace(/\\s+/g, ' '))))"""


def map_header(cells):
    cols = {}
    for i, txt in enumerate(cells):
        if not txt or len(txt) > 20:
            continue
        for field, pat in HEAD_RULES:
            if re.search(pat, txt):
                if field != "_skip" and field not in cols:
                    cols[field] = i
                break
    return cols


def _is_header(cols):
    return "qty" in cols and ("code" in cols or "name" in cols) and len(cols) >= 3


def _parse(frames_tables, want_header=None):
    """frames_tables：[每個 frame 的 [每個 table 的 [每列 cells]]]；回傳 (header, cols, unit, holdings) 或 None"""
    best = None
    for tables in frames_tables:
        flat = [(ti, row) for ti, rows in enumerate(tables) for row in rows]
        for k, (_, head) in enumerate(flat):
            cols = map_header(head)
            if not _is_header(cols):
                continue
            if want_header and head != want_header:
                continue
            code_col = cols.get("code", cols.get("name"))
            unit = 1000 if "張" in head[cols["qty"]] else 1
            holds, data_ti = [], None
            for tj, row in flat[k + 1:]:
                if len(row) != len(head):
                    if data_ti is not None and tj != data_ti:
                        break
                    continue
                if _is_header(map_header(row)):
                    break
                m = CODE_RE.search(row[code_col])
                qty = B._num(row[cols["qty"]])
                if not m or not qty:
                    continue
                data_ti = tj
                name_src = row[cols["name"]] if "name" in cols else ""
                h = {"code": m.group(1), "name": name_src.replace(m.group(1), "", 1).strip(), "qty": int(qty * unit)}
                if "type" in cols:
                    h["type"] = row[cols["type"]]
                for f in ("avg_cost", "price", "pnl", "market_value"):
                    if f in cols:
                        h[f] = B._num(row[cols[f]])
                holds.append(h)
            if want_header or holds:
                cand = (len(holds), head, cols, unit, holds)
                if best is None or cand[0] > best[0]:
                    best = cand
    return best[1:] if best else None


def _find_cash(texts):
    for t in texts:
        m = CASH_RE.search(t or "")
        if m:
            return m.group(1), B._num(m.group(2))
    return None, None


def _scan(page):
    tables, texts = [], []
    for fr in page.frames:
        try:
            tables.append(fr.evaluate(_TABLES_JS))
            texts.append(fr.evaluate("() => document.body ? document.body.innerText : ''"))
        except Exception:  # noqa: BLE001
            pass
    return tables, texts


def learn(br):
    """「擷取」時呼叫：掃使用者目前的頁面，找到庫存表 / 可用餘額就記下來"""
    sel = B.selectors(br.id)
    if sel.get("account_api") or sel.get("account", {}).get("holdings_table"):
        return None  # 已手動校正 (例如國票)，不動

    def fn(pw):
        br._connect(pw)
        page = B._active_page(br.ctx)
        tables, texts = _scan(page)
        return page.url, _parse(tables), _find_cash(texts)

    url, hold, (cash_label, cash) = B._w().call(fn)
    auto = dict(sel.get("account", {}).get("auto") or {})
    msg = []
    if hold:
        head, cols, unit, holds = hold
        auto.update(holdings_url=url, header=head, cols=cols, unit=unit)
        msg.append(f"已自動辨識庫存表 ({len(holds)} 檔，欄位：{'、'.join(cols)})")
    if cash_label:
        auto.update(cash_url=url, cash_label=cash_label)
        msg.append(f"已自動辨識{cash_label} {cash:,.0f}")
    if not msg:
        return None
    sel.setdefault("account", {})["auto"] = auto
    B._save(B.SEL_DIR / f"{br.id}.json", sel)
    return "；".join(msg)


def sync(br):
    """用自動校正結果同步帳務：先掃已開的分頁，找不到再用同步分頁開記下的網址"""
    auto = B.selectors(br.id).get("account", {}).get("auto") or {}
    if not auto.get("header"):
        raise RuntimeError(f"{br.acc['name']} 尚未辨識庫存頁：登入後打開「庫存」頁按一次「擷取」即可自動校正")
    br._need_ctx()
    want_cash = bool(auto.get("cash_label"))

    def find(pages):
        hold = cash = None
        for p in pages:
            try:
                tables, texts = _scan(p)
            except Exception:  # noqa: BLE001
                continue
            hold = hold or _parse(tables, auto["header"])
            if want_cash and cash is None:
                lab, val = _find_cash(texts)
                cash = val if lab else None
        return hold, cash

    def fn(pw):
        br._connect(pw)
        hold, cash = find(list(br.ctx.pages))
        if hold is None and auto.get("holdings_url"):
            sp = br._sync_page()
            sp.goto(auto["holdings_url"])
            for _ in range(10):
                sp.wait_for_timeout(800)
                hold, c2 = find([sp])
                cash = cash if cash is not None else c2
                if hold:
                    break
        if hold is None:
            raise RuntimeError(f"找不到 {br.acc['name']} 的庫存表：可能已登出，或請在 Edge 打開庫存頁再同步一次")
        holds = hold[3]
        out = {"synced": B._now().isoformat(timespec="seconds"), "holdings": holds, "cash": cash,
               "settlements": [], "pending": 0,
               "market_value": sum(h.get("market_value") or 0 for h in holds) or None,
               "unrealized": sum(h.get("pnl") or 0 for h in holds) or None,
               "unrealized_source": "自動辨識庫存頁"}
        B._save(B._acc_file(br.id), out)
        return out

    return B._w().call(fn, timeout=90)
