"""全域設定。資料與快取放在 C 槽 (D 槽空間不足)，可用環境變數 STOCK_DATA_DIR 覆寫。"""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("STOCK_DATA_DIR") or Path(os.environ.get("LOCALAPPDATA", BASE_DIR)) / "stock-data")
DATA_DIR.mkdir(parents=True, exist_ok=True)

# FinMind 免費帳號可不填；註冊後填 token 可提高流量上限
FINMIND_TOKEN = os.environ.get("FINMIND_TOKEN", "")

# 風控：每筆委託都會先經過檢查，超過就拒絕
RISK = {
    "max_order_amount": 300_000,   # 單筆委託金額上限 (元)
    "max_qty_lots": 5,             # 單筆最多張數
    "max_orders_per_day": 10,      # 每日最多委託筆數
    "max_price_deviation": 0.05,   # 委託價與現價偏離上限 (5%)，防止打錯價
    "trading_hours": ("08:30", "13:30"),
}
# 券商帳戶 (登入網址 / 送單模式 / 校正檔) 改在網頁「券商帳戶」頁管理，見 stock/brokers.py
