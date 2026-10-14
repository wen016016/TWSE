"""Auto-fetch Yahoo TW concept-stock / electronics sub-industry lists -> DATA_DIR/themes_auto.json (refresh daily).

On failure keep the previous file; manual themes.json takes priority over same-name auto themes."""
import html
import json
import re
import threading
import time
import urllib.parse as up
from concurrent.futures import ThreadPoolExecutor

import requests

from .config import DATA_DIR

AUTO_FILE = DATA_DIR / "themes_auto.json"
TTL = 86400
LABELS = ("\u6982\u5ff5\u80a1", "\u96fb\u5b50\u7522\u696d")   # concept stocks / electronics sub-industries (Yahoo)
UA = {"User-Agent": "Mozilla/5.0"}
API = "https://tw.stock.yahoo.com/_td-stock/api/resource/StockServices.getClassQuotes"
_lock = threading.Lock()
_mem = {"t": 0, "data": None}


def _categories() -> list:
    r = requests.get("https://tw.stock.yahoo.com/class", headers=UA, timeout=15)
    r.raise_for_status()
    out = []
    for u in re.findall(r'href="([^"]*class-quote\?category=[^"]*)"', r.text):
        q = up.parse_qs(up.urlparse(html.unescape(u)).query)
        name, label = q.get("category", [""])[0], q.get("categoryLabel", [""])[0]
        if name and label in LABELS and (name, label) not in out:
            out.append((name, label))
    return out


def _members(name: str) -> list:
    codes, offset = [], 0
    for _ in range(20):
        url = f"{API};categoryName={up.quote(name)};offset={offset}"
        j = requests.get(url, headers=UA, timeout=15).json()
        for x in j.get("list", []):
            m = re.match(r"^(\d{4})\.TWO?$", x.get("symbol", ""))
            if m and m.group(1) not in codes:
                codes.append(m.group(1))
        nxt = (j.get("pagination") or {}).get("nextOffset")
        if not nxt:
            break
        offset = int(nxt)
    return codes


def fetch() -> dict:
    cats = _categories()
    if not cats:
        raise RuntimeError("Yahoo themes fetch failed")
    with ThreadPoolExecutor(6) as ex:
        res = list(ex.map(lambda c: (c, _safe_members(c[0])), cats))
    out = {name: codes for (name, _), codes in res if codes}
    if len(out) < 10:
        raise RuntimeError("Yahoo themes fetch failed")
    return out


def _safe_members(name):
    try:
        return _members(name)
    except Exception:  # noqa: BLE001
        return []


def _load_file():
    try:
        j = json.loads(AUTO_FILE.read_text(encoding="utf-8"))
        return j.get("time", 0), j.get("themes", {})
    except Exception:  # noqa: BLE001
        return 0, {}


def _refresh():
    try:
        th = fetch()
        AUTO_FILE.write_text(json.dumps({"time": time.time(), "source": "tw.stock.yahoo.com/class", "themes": th},
                                        ensure_ascii=False), encoding="utf-8")
        with _lock:
            _mem.update(t=time.time(), data=th)
    except Exception as e:  # noqa: BLE001
        print("[themes_fetch] update failed, keep old data:", e)
    finally:
        _mem["busy"] = False


def auto_themes(block=False) -> dict:
    """Return auto themes; refresh in background when stale (block=True waits on first run with no cache)."""
    with _lock:
        if _mem["data"] is None:
            t, th = _load_file()
            _mem.update(t=t, data=th)
        stale = time.time() - _mem["t"] > TTL
        if stale and not _mem.get("busy"):
            _mem["busy"] = True
            if block and not _mem["data"]:
                _lock.release()
                try:
                    _refresh()
                finally:
                    _lock.acquire()
            else:
                threading.Thread(target=_refresh, daemon=True).start()
        return _mem["data"] or {}
