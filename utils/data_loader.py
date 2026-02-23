import numpy as np
import pandas as pd


def load_ohlcv(path: str) -> pd.DataFrame:
    """
    Load OHLCV data from a CSV file.

    Args:
        path (str): Path to the OHLCV CSV file.

    Returns:
        pd.DataFrame: DataFrame with OHLCV data.
    """
    df = pd.read_csv(path, parse_dates=True)
    df.columns = df.columns.str.lower()

    # Find datetime column
    dt_col = next(
        (c for c in df.columns if c in ("datetime", "timestamp", "date", "time")),
        df.columns[0],
    )
    df = df.set_index(dt_col)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    # Rename to standard names
    rename = {c: c for c in ("open", "high", "low", "close", "volume") if c in df.columns}
    df = df[list(rename.values())].copy()
    df = df.ffill().dropna()

    print(f"[DATA]  Loaded {len(df):,} bars  {df.index[0]} → {df.index[-1]}")
    return df


def make_synthetic_spy(n: int = 5000, seed: int = 42) -> pd.DataFrame:
    """
    Synthetic 5-minute SPY OHLCV for testing the pipeline end-to-end.
    Geometric Brownian Motion with intraday mean-reversion and volume profile.

    Args:
        n (int, optional): Number of bars. Defaults to 5000.
        seed (int, optional): Random seed. Defaults to 42.

    Returns:
        pd.DataFrame: DataFrame with OHLCV data.
    """
    rng = np.random.default_rng(seed)
    bars_per_day = 78  # 9:30–16:00 in 5-min bars
    n_days = int(np.ceil(n / bars_per_day))

    # Price path (GBM)
    mu = 0.00003  # per-bar drift
    sigma = 0.0008  # per-bar vol  (~1.1 % daily)
    log_r = rng.normal(mu, sigma, n)
    close = 500.0 * np.exp(np.cumsum(log_r))

    # OHLCV
    bar_vol = np.abs(rng.normal(0, sigma * 0.5, n))
    high = close * (1 + bar_vol)
    low = close * (1 - bar_vol)
    open_ = np.roll(close, 1)
    open_[0] = 500.0
    volume = rng.integers(50_000, 500_000, n).astype(float)

    # DatetimeIndex (market hours only)
    from pandas.tseries.offsets import BDay

    start = pd.Timestamp("2022-01-03 09:30:00")
    dates = []
    day = start
    while len(dates) < n:
        for m in range(bars_per_day):
            dates.append(day + pd.Timedelta(minutes=5 * m))
        day = (day + BDay(1)).replace(hour=9, minute=30)
    dates = dates[:n]

    df = pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        index=pd.DatetimeIndex(dates),
    )
    return df
