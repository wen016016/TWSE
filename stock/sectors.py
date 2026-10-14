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


def _manual_themes() -> dict:
    try:
        return {k: v for k, v in json.loads(THEME_FILE.read_text(encoding="utf-8")).items() if not k.startswith("_")}
    except Exception:
        return {}


# Yahoo 概念股中偏總經 / 消費品牌 / 過時題材，不當族群標籤 (可在 themes.json 的 "_排除" 自行增加)
NOISE = {
    "蘋果200大供應商", "AI理財機器人", "Sharp", "Google Pixel", "米其林摘星", "夏日飲料", "無人商店", "跨國連鎖餐飲",
    "AirPods", "台幣升值", "中國台商汽車組件", "智慧音箱", "Apple watch", "空污", "空污防治", "Apple Pay",
    "紡織機能運動", "銀髮商機", "雙十一", "日圓貶值", "日圓升值", "互聯網+", "三星", "iPhone", "一帶一路",
    "補教/民辦教育", "新掛牌", "Apple iTV", "小米", "Nike", "Amazon Go應用技術", "iPad", "台資在中國高獲利",
    "人民幣升值", "歐元貶值", "台積電", "聯發科", "網紅經濟", "OPPO", "Intel", "眼球商機", "宅經濟", "HomePod",
    "任天堂Switch", "防疫", "博弈觀光", "Toyota", "丹麥沃旭", "華為", "海外掛牌", "嬰幼兒相關", "通膨", "植物工廠",
    "PlayStation 5", "大數據2.0", "智慧家庭", "智慧城市", "工業4.0", "物聯網", "人臉辨識", "體育/運動產業",
    "醫療呼吸器", "比特幣挖礦", "FinTech", "行動支付", "第三方支付", "電子商務", "VR虛擬實境", "3D列印", "車聯網",
    "穿戴裝置", "Tesla", "折疊手機", "再生循環", "雲端產業", "電競產業", "智慧醫療",
}
MAX_AUTO_TAGS = 2   # 每檔最多顯示幾個 Yahoo 自動題材 (成分股越少 = 越具代表性，優先)
MAX_AUTO_SIZE = 80  # 成分股超過這數量的題材太籠統，不當標籤


def _noise() -> set:
    try:
        extra = json.loads(THEME_FILE.read_text(encoding="utf-8")).get("_排除", [])
    except Exception:  # noqa: BLE001
        extra = []
    return NOISE | set(extra)


def _themes() -> dict:
    """手寫 themes.json 優先，再補 Yahoo 自動抓的概念股 / 電子細分類 (同名以手寫為準)"""
    from .themes_fetch import auto_themes
    man = _manual_themes()
    try:
        auto = auto_themes(block=True)
    except Exception:  # noqa: BLE001
        auto = {}
    noise = _noise()
    return {**man, **{k: v for k, v in auto.items()
                      if k not in man and k not in noise and len(v) <= MAX_AUTO_SIZE}}


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


def code_map() -> dict:
    """代號 → [{"name", "kind"}]：手寫題材 → 自動題材 → 官方產業"""
    man = _manual_themes()
    themes = _themes()
    auto = {k: v for k, v in themes.items() if k not in man}
    inds = data.cached("sector_inds", 3600, lambda: _industries(data.stock_info()))
    out = {}
    for kind, src in (("theme", man), ("industry", inds), ("auto", auto)):
        for name, codes in src.items():
            for c in codes:
                lst = out.setdefault(c, [])
                if not any(x["name"] == name for x in lst):
                    lst.append({"name": name, "kind": kind, "n": len(codes)})
    for c, lst in out.items():
        fixed = [x for x in lst if x["kind"] != "auto"]
        fixed = [x for x in fixed if not (x["kind"] == "industry" and any(
            y is not x and y["kind"] != "industry" and (y["name"] in x["name"] or x["name"] in y["name"]) for y in fixed))]
        names = {x["name"] for x in fixed}
        extra = sorted((x for x in lst if x["kind"] == "auto" and x["name"] not in names), key=lambda x: x["n"])
        out[c] = [{"name": x["name"], "kind": x["kind"]} for x in fixed + extra[:MAX_AUTO_TAGS]]
    return out


def group(name: str) -> dict:
    """單一族群即時漲跌 (依漲幅大 → 跌幅大排序)"""
    codes = _themes().get(name) or data.cached("sector_inds", 3600, lambda: _industries(data.stock_info())).get(name)
    if not codes:
        raise ValueError(f"找不到族群：{name}")
    q = quotes(list(dict.fromkeys(codes)))
    s = _stat(name, codes, q, 0)
    if not s:
        raise ValueError(f"{name} 目前抓不到報價")
    s["members"] = sorted(s["members"], key=lambda r: -r["chg_pct"])
    any_q = next(iter(q.values()), {})
    s["time"] = f"{any_q.get('date', '')} {any_q.get('time', '')}"
    return s


def get(force=False, ttl=120) -> dict:
    with _lock:
        if not force and _cache["data"] and time.time() - _cache["t"] < ttl:
            return _cache["data"]
        d = build()
        _cache.update(t=time.time(), data=d)
        return d
