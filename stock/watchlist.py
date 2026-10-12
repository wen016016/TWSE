"""自選清單：存你正在觀察的股票，並一次評估 (波段分數、整理型態、離支撐 / 壓力多遠)

存在 %LOCALAPPDATA%\\stock-data\\watchlist.json；即時報價由網頁的即時推送更新
"""
import datetime as dt
import json
import threading

from . import data
from .config import DATA_DIR

FILE = DATA_DIR / "watchlist.json"
_lock = threading.Lock()


def _load() -> list[dict]:
    try:
        return json.loads(FILE.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return []


def _save(items):
    FILE.write_text(json.dumps(items, ensure_ascii=False, indent=1), encoding="utf-8")


def items() -> list[dict]:
    return _load()


def add(code: str, note: str = "") -> list[dict]:
    info = data.resolve(code)
    with _lock:
        lst = _load()
        if not any(x["code"] == info["code"] for x in lst):
            lst.append({"code": info["code"], "name": info["name"], "note": note,
                        "added": dt.date.today().isoformat()})
            _save(lst)
        return lst


def remove(code: str) -> list[dict]:
    with _lock:
        lst = [x for x in _load() if x["code"] != code]
        _save(lst)
        return lst


def set_note(code: str, note: str) -> list[dict]:
    with _lock:
        lst = _load()
        for x in lst:
            if x["code"] == code:
                x["note"] = note
        _save(lst)
        return lst


def move(code: str, step: int) -> list[dict]:
    with _lock:
        lst = _load()
        i = next((k for k, x in enumerate(lst) if x["code"] == code), None)
        if i is not None:
            j = max(0, min(len(lst) - 1, i + step))
            lst.insert(j, lst.pop(i))
            _save(lst)
        return lst


def evaluate() -> list[dict]:
    """一次評估全部自選股 (日K 批次下載，約 5~15 秒)"""
    from .base_breakout import detect
    from .indicators import add_indicators, to_weekly
    from .levels import compute_levels
    from .strategy import Score, score_bias, score_tech, verdict

    lst = _load()
    if not lst:
        return []
    daily = data.download_daily([x["code"] for x in lst])
    out = []
    for x in lst:
        row = dict(x)
        df = daily.get(x["code"])
        if df is None:
            row["error"] = "抓不到日K"
            out.append(row)
            continue
        try:
            df = data._live(df, x["code"], "1d", False)  # 補上今天即時
            d = add_indicators(df)
            w = add_indicators(to_weekly(df), mas=(5, 10, 20, 60))
            sc = Score()
            score_tech(sc, w, "週K", 1.2, 13)
            score_tech(sc, d, "日K", 1.0, 20)
            score_bias(sc, d, "日K")
            tone, _ = verdict(sc.total, 7, 3.5)
            close = float(d["close"].iloc[-1])
            lv = compute_levels(d, "short")
            sup = lv["supports"][0] if lv["supports"] else None
            res = lv["resistances"][0] if lv["resistances"] else None
            b = detect(d)
            row.update(score=sc.total, tone=tone, close=close,
                       support=sup and {"price": sup["price"], "dist_pct": sup["dist_pct"]},
                       resistance=res and {"price": res["price"], "dist_pct": res["dist_pct"]},
                       bull_key=lv["key"].get("bull", {}).get("price"), bear_key=lv["key"].get("bear", {}).get("price"),
                       base={"stage": b["stage"], "score": b.get("score"), "trigger": b.get("trigger")},
                       state=lv["state"])
        except Exception as e:  # noqa: BLE001
            row["error"] = str(e)
        out.append(row)
    return out
