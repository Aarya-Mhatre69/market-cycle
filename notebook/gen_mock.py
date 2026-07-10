from pathlib import Path

import numpy as np
import pandas as pd
import holidays
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
PLOT_PATH = ROOT / "mock_market_prices.png"

DATA_DIR.mkdir(parents=True, exist_ok=True)

TICKERS = ["TCS.NS", "ACC.NS", "RELIANCE.NS", "INFY.NS", "ITC.NS"]
BASE_PRICES = {
    "TCS.NS": 3500.0,
    "ACC.NS": 2200.0,
    "RELIANCE.NS": 2600.0,
    "INFY.NS": 1500.0,
    "ITC.NS": 420.0,
}
BASE_VOLUMES = {
    "TCS.NS": 3_500_000,
    "ACC.NS": 900_000,
    "RELIANCE.NS": 8_000_000,
    "INFY.NS": 4_000_000,
    "ITC.NS": 12_000_000,
}


def _trading_days(year: int) -> pd.DatetimeIndex:
    dates = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="B")
    india_holidays = holidays.India(years=year)
    return pd.DatetimeIndex([d for d in dates if d.date() not in india_holidays])


def _build_market_factor(n: int, rng: np.random.Generator) -> np.ndarray:
    t = np.arange(n)

    cycle = (
        0.0018 * np.sin(2 * np.pi * t / 220)
        + 0.0012 * np.sin(2 * np.pi * t / 90)
        + 0.0008 * np.sin(2 * np.pi * t / 30)
        + 0.0005 * np.sin(2 * np.pi * t / 11)
    )

    noise = rng.normal(0, 1, n)
    noise = np.convolve(noise, np.ones(21) / 21, mode="same") * 0.0012

    regime = np.zeros(n)
    i = 0
    states = np.array([0.0012, 0.0004, -0.0002, -0.0011])
    probs = np.array([0.28, 0.34, 0.26, 0.12])

    while i < n:
        seg_len = int(rng.integers(18, 70))
        state = float(rng.choice(states, p=probs))
        regime[i : min(i + seg_len, n)] = state
        i += seg_len

    jumps = np.zeros(n)
    for _ in range(max(2, n // 80)):
        idx = int(rng.integers(0, n))
        jumps[idx] += rng.choice([-1, 1]) * rng.uniform(0.01, 0.04)

    return cycle + noise + regime + jumps


def build_year_ohlcv(year: int) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(year)
    dates = _trading_days(year)
    n = len(dates)

    market = _build_market_factor(n, rng)

    vol = np.empty(n)
    vol[0] = 0.010
    for i in range(1, n):
        vol[i] = 0.92 * vol[i - 1] + 0.08 * abs(rng.normal(0, 0.020))
        vol[i] = float(np.clip(vol[i], 0.004, 0.045))

    per_ticker_rows: dict[str, list[dict]] = {t: [] for t in TICKERS}
    last_close = BASE_PRICES.copy()

    for i, day in enumerate(dates):
        mkt = market[i]
        day_vol = vol[i]

        for j, ticker in enumerate(TICKERS):
            beta = 0.75 + 0.15 * j
            idio_vol = day_vol * (0.85 + 0.12 * j)

            drift = (
                0.00018
                + 0.00008 * j
                + 0.00035 * np.sin((i + 10 * j) / (140 + 15 * j))
            )

            prev_close = float(last_close[ticker])

            gap = rng.normal(0, idio_vol * 0.35) + 0.15 * mkt
            open_ = prev_close * np.exp(gap)

            intraday = drift + beta * mkt + rng.normal(0, idio_vol)

            if rng.random() < 0.012:
                intraday += rng.normal(0, 0.04)

            if rng.random() < 0.004:
                intraday += rng.choice([-1, 1]) * rng.uniform(0.05, 0.14)

            close = max(1.0, open_ * np.exp(intraday))

            candle_span = max(abs(close - open_) / open_, 0.002) + rng.uniform(0.002, 0.018)
            wick_up = candle_span * rng.uniform(0.35, 1.20)
            wick_dn = candle_span * rng.uniform(0.35, 1.20)

            high = max(open_, close) * (1 + wick_up)
            low = min(open_, close) * (1 - wick_dn)
            low = max(0.5, low)

            move = abs(np.log(close / open_))
            volume = int(
                BASE_VOLUMES[ticker]
                * rng.lognormal(mean=0.0, sigma=0.32)
                * (1 + 10 * move + 3 * day_vol)
            )

            per_ticker_rows[ticker].append(
                {
                    "date": day.strftime("%Y-%m-%d"),
                    "open": round(float(open_), 2),
                    "high": round(float(high), 2),
                    "low": round(float(low), 2),
                    "close": round(float(close), 2),
                    "volume": volume,
                }
            )

            last_close[ticker] = close

    return {ticker: pd.DataFrame(rows) for ticker, rows in per_ticker_rows.items()}

def save_data_and_plot() -> None:
    years = range(2015, 2026)

    # Collect every year for every ticker
    ticker_data = {ticker: [] for ticker in TICKERS}

    # For plotting
    yearly_frames = []

    for year in years:
        year_frames = build_year_ohlcv(year)

        yearly_frames.append((year, year_frames))

        for ticker, df in year_frames.items():
            ticker_data[ticker].append(df)

    # Save ONE CSV per ticker
    for ticker, dfs in ticker_data.items():
        out_df = (
            pd.concat(dfs, ignore_index=True)
            .sort_values("date")
            .reset_index(drop=True)
        )

        filename = ticker.lower() + ".csv"
        out_path = DATA_DIR / filename

        out_df.to_csv(out_path, index=False)

        print(
            f"saved {out_path} "
            f"({len(out_df):,} rows, "
            f"{out_df.date.iloc[0]} -> {out_df.date.iloc[-1]})"
        )

    # Plot the last two years only (avoids clutter)
    plot_years = yearly_frames[-2:]

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=False)

    for ax, (year, year_frames) in zip(axes, plot_years):
        for ticker, df in year_frames.items():
            plot_df = df.copy()
            plot_df["date"] = pd.to_datetime(plot_df["date"])

            ax.plot(
                plot_df["date"],
                plot_df["close"],
                label=ticker,
                linewidth=1.1,
            )

        ax.set_title(f"Mock Close Prices ({year})")
        ax.set_ylabel("Close")
        ax.grid(alpha=0.25)

    axes[-1].set_xlabel("Date")
    axes[0].legend(
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        frameon=False,
    )

    PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    fig.savefig(PLOT_PATH, dpi=200)

    print(f"saved plot {PLOT_PATH}")

if __name__ == "__main__":
    save_data_and_plot()