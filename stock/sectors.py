"""類股資金流向：各類股成交金額 / 資金占比 / 漲跌幅 / 量最大 & 漲最多的個股

分類：
  題材：themes.json (低軌衛星、CPO、ABF… 可自行編輯)
  產業：證交所 / 櫃買官方產業別 (鋼鐵工業、金融保險業、航運業…)
報價：證交所 MIS 即時行情 (盤中即時、收盤後為當日收盤)；成交金額 = 成交張數 × 1000 × 現價 (估算)
"""
import json
import threading
import time

from . import data
from .config import BASE_DIR
from .realtime import quotes

THEME_FILE = BASE_DIR / "themes.json"
EXCLUDE_IND = ("ETF", "ETN", "指數", "大盤", "受益", "存託", "Index", "創新板")

_lock = threading.Lock()
_cache = {"t": 0, "data": None}


def _themes() -> dict:
    try:
        return {k: v for k, v in json.loads(THEME_FILE.read_text(encoding="utf-8")).items() if not k.startswith("_")}
    except Exception:
        return {}


ALIAS = {"金融業": "金融保險", "金融保險業": "金融保險", "數位雲端類": "數位雲端", "綠能環保類": "綠能環保",
         "運動休閒類": "運動休閒", "觀光餐旅": "觀光事業", "居家生活類": "居家生活", "化學生技醫療": "生技醫療業",
         "電子工業": "其他電子業"}


def _industries(info) -> dict:
    ind = {}
    for _, r in info.iterrows():
        c, cat = r["stock_id"], str(r.get("industry_category", ""))
        cat = ALIAS.get(cat, cat)
        if (len(c) != 4 or not c.isdigit() or not cat or cat in ("nan", "其他")
                or any(x in cat for x in EXCLUDE_IND + ("創新版",))):
            continue
        ind.setdefault(cat, []).append(c)
    return ind


def _stat(name, codes, q, total_value):
    rows = []
    for c in dict.fromkeys(codes):
        x = q.get(c)
        if not x or not x["price"] or not x["prev_close"]:
            continue
        chg = (x["price"] / x["prev_close"] - 1) * 100
        value = x["volume"] * 1000 * x["price"]
        rows.append({"code": c, "name": x["name"], "price": x["price"], "chg_pct": round(chg, 2),
                     "volume": int(x["volume"]), "value_e": round(value / 1e8, 2),
                     "limit_up": bool(x["limit_up"] and x["price"] >= x["limit_up"]),
                     "limit_down": bool(x["limit_down"] and x["price"] <= x["limit_down"])})
    if not rows:
        return None
    val = sum(r["value_e"] for r in rows)
    wchg = sum(r["chg_pct"] * r["value_e"] for r in rows) / val if val else 0
    return {"name": name, "count": len(rows), "value_e": round(val, 1),
            "share_pct": round(val / total_value * 100, 2) if total_value else 0,
            "avg_chg": round(sum(r["chg_pct"] for r in rows) / len(rows), 2), "w_chg": round(wchg, 2),
            "up": sum(r["chg_pct"] > 0 for r in rows), "down": sum(r["chg_pct"] < 0 for r in rows),
            "limit_up": sum(r["limit_up"] for r in rows), "limit_down": sum(r["limit_down"] for r in rows),
            "top_volume": sorted(rows, key=lambda r: -r["volume"])[:5],
            "top_value": sorted(rows, key=lambda r: -r["value_e"])[:5],
            "top_gain": sorted(rows, key=lambda r: -r["chg_pct"])[:5],
            "members": sorted(rows, key=lambda r: -r["value_e"])}


def build() -> dict:
    info = data.stock_info()
    themes = _themes()
    inds = _industries(info)
    all_codes = sorted({c for v in inds.values() for c in v} | {c for v in themes.values() for c in v})
    t0 = time.time()
    q = quotes(all_codes)
    stock_rows = [x for c, x in q.items() if c in set(c for v in inds.values() for c in v)]
    total = sum(x["volume"] * 1000 * x["price"] for x in stock_rows if x["price"]) / 1e8
    theme_stats = [s for s in (_stat(k, v, q, total) for k, v in themes.items()) if s]
    ind_stats = [s for s in (_stat(k, v, q, total) for k, v in inds.items()) if s]
    for lst in (theme_stats, ind_stats):
        lst.sort(key=lambda s: -s["value_e"])
    any_q = next(iter(q.values()), {})
    return {"time": f"{any_q.get('date', '')} {any_q.get('time', '')}", "fetch_sec": round(time.time() - t0, 1),
            "market_value_e": round(total, 1), "stocks": len(stock_rows),
            "themes": theme_stats, "industries": ind_stats,
            "note": "成交金額為估算值 (成交張數 × 現價)；占比 = 該類成交金額 ÷ 全市場個股成交金額。題材股之間會重複 (同一檔可屬多個題材)。"}


def get(force=False, ttl=120) -> dict:
    with _lock:
        if not force and _cache["data"] and time.time() - _cache["t"] < ttl:
            return _cache["data"]
        d = build()
        _cache.update(t=time.time(), data=d)
        return d
