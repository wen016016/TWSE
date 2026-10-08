"""股票分析與下單系統 — 本機網頁伺服器

啟動：雙擊 start.bat (會等伺服器好了才用 Edge 開 http://127.0.0.1:8000)
"""
import asyncio
import json
import math
import traceback
from contextlib import asynccontextmanager

import numpy as np
from fastapi import Body, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from stock import broker, brokers, realtime, scanner, strategy
from stock.autotrader import trader
from stock.config import BASE_DIR


@asynccontextmanager
async def lifespan(_app):
    realtime.streamer.start()    # 盤中每 ~2 秒合併抓指數 / 台指期 / 關注個股 (限速避免被封鎖)，SSE 推給網頁；成交明細同時收集
    trader.start()               # 自動交易引擎 (自動交易頁開啟後才會動作)
    yield


app = FastAPI(title="股票分析與下單", lifespan=lifespan)


def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return None if math.isnan(f) or math.isinf(f) else f
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def run(fn, *a, **kw):
    try:
        return JSONResponse(clean(fn(*a, **kw)))
    except (ValueError, RuntimeError) as e:
        raise HTTPException(400, str(e))
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        raise HTTPException(500, f"{type(e).__name__}: {e}")


@app.get("/api/analyze")
def api_analyze(code: str, mode: str = "swing", entry: float | None = None, side: str = "long"):
    return run(strategy.analyze, code, mode, entry=entry, side=side)


@app.get("/api/levels")
def api_levels(code: str, tf: str = "1d"):
    if tf not in ("1wk", "1d", "5m", "1m"):
        raise HTTPException(400, "tf 只能是 1wk / 1d / 5m / 1m")
    return run(strategy.levels_report, code, tf)


@app.get("/api/market")
def api_market():
    return run(lambda: strategy.market_view(intraday=True)[1])


@app.get("/api/ticker")
def api_ticker():
    """頂部即時列：加權、櫃買、台指期 (日/夜盤)、額度、自動交易狀態 (每幾秒輪詢)"""
    def f():
        out = {"index": {}, "futures": None, "health": realtime.health()}
        try:
            out["index"] = realtime.index_quotes()
            out["futures"] = realtime.futures()
        except Exception:  # noqa: BLE001
            pass
        s = broker.get_settings()
        out.update(budget=broker.budget_status(), auto=s["auto"], default_account=s["default_account"],
                   accounts=broker.accounts_summary())
        return out
    return run(f)


@app.get("/api/stream")
async def api_stream(request: Request, codes: str = ""):
    """即時報價推送 (Server-Sent Events)：有變動就送一筆"""
    want = [c for c in codes.split(",") if c]
    realtime.streamer.add(want)

    async def gen():
        last = -1
        n = 0
        while not await request.is_disconnected():
            v = realtime.streamer.version
            if v != last:
                last = v
                payload = json.dumps(clean(realtime.streamer.snapshot(want)), ensure_ascii=False)
                yield "data: " + payload + "\n\n"
            n += 1
            if n % 60 == 0:  # 保持關注 + 心跳
                realtime.streamer.add(want)
                yield ": ping\n\n"
            await asyncio.sleep(0.3)
    return StreamingResponse(gen(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})


@app.get("/api/scan")
def api_scan(mode: str = "swing", top: int = 8, direction: str = "long"):
    return run(scanner.scan, mode, top, direction)


@app.get("/api/settings")
def api_settings():
    return run(lambda: {"settings": broker.get_settings(), "status": broker.budget_status()})


@app.post("/api/settings")
def api_settings_set(patch: dict = Body(...)):
    return run(lambda: {"settings": broker.update_settings(patch), "status": broker.budget_status()})


@app.post("/api/order/preview")
def api_preview(req: dict = Body(...)):
    return run(broker.preview, req)


@app.post("/api/order/confirm")
def api_confirm(req: dict = Body(...)):
    return run(broker.confirm, req["id"])


@app.post("/api/order/cancel")
def api_cancel(req: dict = Body(...)):
    return run(lambda: broker.cancel(req["id"]) or {"ok": True})


@app.get("/api/ledger")
def api_ledger():
    def f():
        lg = broker.ledger()
        pos = broker.open_positions()
        q = {}
        try:
            q = realtime.quotes([p["code"] for p in pos]) if pos else {}
        except Exception:  # noqa: BLE001
            pass
        unreal = 0
        for p in pos:
            px = (q.get(p["code"]) or {}).get("price")
            p["price"] = px
            if px:
                p["pnl"] = round((px - p["avg_price"]) * p["qty"] * (1 if p["side"] == "long" else -1))
                p["pnl_pct"] = round((px / p["avg_price"] - 1) * 100 * (1 if p["side"] == "long" else -1), 2)
                unreal += p["pnl"]
        return {"orders": lg["orders"][-50:][::-1], "positions": pos, "unrealized": unreal,
                "status": broker.budget_status(), "accounts": broker.accounts_summary(),
                "holdings": {a["id"]: brokers.account_data(a["id"]) for a in brokers.accounts(False)}}
    return run(f)


@app.get("/api/market_risk")
def api_market_risk():
    from stock import market_risk
    return run(lambda: data_cached_risk(market_risk.market_risk))


def data_cached_risk(fn):
    from stock import data as D
    return D.cached("market_risk", 60, fn)


@app.get("/api/forecast")
def api_forecast():
    from stock import forecast
    return run(forecast.forecast)


@app.get("/api/sectors")
def api_sectors(force: bool = False):
    from stock import sectors
    return run(sectors.get, force)


@app.get("/api/auto")
def api_auto():
    return run(lambda: {"settings": broker.get_settings()["auto"], "status": trader.status, "logs": trader.logs(),
                        "collector": {"watch": realtime.collector._active(), "error": realtime.collector.last_error},
                        "budget": broker.budget_status()})


# ----------------------------------------------------------------- 券商帳戶
@app.get("/api/accounts")
def api_accounts():
    return run(lambda: {"accounts": broker.accounts_summary(), "presets": brokers.PRESETS,
                        "default_account": broker.get_settings()["default_account"]})


@app.post("/api/accounts/add")
def api_accounts_add(req: dict = Body(...)):
    return run(brokers.add_account, req["broker"], req.get("name", ""), req.get("login_url", ""))


@app.post("/api/accounts/update")
def api_accounts_update(req: dict = Body(...)):
    return run(lambda: brokers.update_account(req["id"], **{k: v for k, v in req.items() if k != "id"}))


@app.post("/api/accounts/remove")
def api_accounts_remove(req: dict = Body(...)):
    def f():
        s = broker.get_settings()
        if req["id"] in s["default_account"].values():
            raise ValueError("這個帳戶是當沖/波段的預設帳戶，請先改預設帳戶再刪除")
        brokers.remove_account(req["id"])
        return {"ok": True}
    return run(f)


@app.post("/api/accounts/{acc_id}/open")
def api_acc_open(acc_id: str):
    return run(lambda: brokers.broker(acc_id).open())


@app.post("/api/accounts/{acc_id}/snapshot")
def api_acc_snapshot(acc_id: str, req: dict = Body(default={})):
    return run(lambda: brokers.broker(acc_id).snapshot(req.get("label", "page")))


@app.post("/api/accounts/{acc_id}/sync")
def api_acc_sync(acc_id: str):
    return run(lambda: brokers.broker(acc_id).sync_account())


app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")


@app.get("/")
def index():
    return FileResponse(BASE_DIR / "static" / "index.html")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
