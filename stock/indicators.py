"""技術指標：均線、MACD、KD、RSI、布林、ATR、量均、VWAP"""
import numpy as np
import pandas as pd


def sma(s, n):
    return s.rolling(n, min_periods=n).mean()


def add_indicators(df: pd.DataFrame, mas=(5, 10, 20, 60, 120)) -> pd.DataFrame:
    df = df.copy()
    c = df["close"]
    for n in mas:
        df[f"ma{n}"] = sma(c, n)

    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    df["dif"] = ema12 - ema26
    df["dea"] = df["dif"].ewm(span=9, adjust=False).mean()
    df["osc"] = df["dif"] - df["dea"]

    # 台灣常用 KD(9,3,3)
    low9 = df["low"].rolling(9, min_periods=1).min()
    high9 = df["high"].rolling(9, min_periods=1).max()
    rsv = ((c - low9) / (high9 - low9).replace(0, np.nan) * 100).fillna(50)
    k, d, kp, dp = [], [], 50.0, 50.0
    for v in rsv:
        kp = kp * 2 / 3 + v / 3
        dp = dp * 2 / 3 + kp / 3
        k.append(kp)
        d.append(dp)
    df["k"], df["d"] = k, d

    delta = c.diff()
    up = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    dn = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    df["rsi"] = 100 - 100 / (1 + up / dn.replace(0, np.nan))
    df["rsi"] = df["rsi"].fillna(100)

    mid = sma(c, 20)
    sd = c.rolling(20).std()
    df["bb_mid"], df["bb_up"], df["bb_dn"] = mid, mid + 2 * sd, mid - 2 * sd

    tr = pd.concat([df["high"] - df["low"], (df["high"] - c.shift()).abs(), (df["low"] - c.shift()).abs()], axis=1).max(axis=1)
    df["atr"] = tr.ewm(alpha=1 / 14, adjust=False).mean()
    df["vma5"] = sma(df["volume"], 5)
    df["vma20"] = sma(df["volume"], 20)
    return df


def add_vwap(df: pd.DataFrame) -> pd.DataFrame:
    """日內成交量加權均價 (每天重算)"""
    df = df.copy()
    tp = (df["high"] + df["low"] + df["close"]) / 3
    day = df.index.date
    pv = (tp * df["volume"]).groupby(day).cumsum()
    v = df["volume"].groupby(day).cumsum()
    df["vwap"] = (pv / v.replace(0, np.nan)).fillna(tp)
    return df


def to_weekly(df: pd.DataFrame) -> pd.DataFrame:
    return df.resample("W-FRI").agg({"open": "first", "high": "max", "low": "min",
                                     "close": "last", "volume": "sum"}).dropna()


def crossed_up(a: pd.Series, b: pd.Series, within=3) -> bool:
    """近 within 根內黃金交叉，且目前仍在上方"""
    diff = (a - b).tail(within + 1).values
    return diff[-1] > 0 and any(diff[i - 1] <= 0 < diff[i] for i in range(1, len(diff)))


def crossed_down(a: pd.Series, b: pd.Series, within=3) -> bool:
    """近 within 根內死亡交叉，且目前仍在下方"""
    diff = (a - b).tail(within + 1).values
    return diff[-1] < 0 and any(diff[i - 1] >= 0 > diff[i] for i in range(1, len(diff)))


def tick(price: float) -> float:
    """台股升降單位"""
    if price < 10:
        return 0.01
    if price < 50:
        return 0.05
    if price < 100:
        return 0.1
    if price < 500:
        return 0.5
    if price < 1000:
        return 1.0
    return 5.0


def round_tick(price: float, how="nearest") -> float:
    t = tick(price)
    q = price / t
    q = {"nearest": round(q), "down": np.floor(q + 1e-9), "up": np.ceil(q - 1e-9)}[how]
    return round(float(q * t), 2)
