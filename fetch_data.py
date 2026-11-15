"""Downloads hourly SOL/USDC OHLCV candles from a public exchange API (no API key needed).

Default source is Bybit Spot SOLUSDC: the same market the live executor feeds into the
model (see readme.md), so training and serving use the same candles and volume units
(volume = base asset, SOL).
Alternative: --source binance (SOLUSDC on Binance, deeper liquidity).

Usage:
    python fetch_data.py [--source bybit|binance] [--years 2] [--out data/sol_usdc_data.csv]

An existing output file is never overwritten silently: it is first moved to <name>.bak-<unixtime>.
"""
import argparse
import os
import shutil
import time

import pandas as pd
import requests

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT = os.path.join(BASE_DIR, "data", "sol_usdc_data.csv")
HOUR_MS = 3_600_000
TIMEOUT = 20


def _get(url: str, params: dict) -> dict | list:
    for attempt in range(5):
        try:
            r = requests.get(url, params=params, timeout=TIMEOUT)
            r.raise_for_status()
            return r.json()
        except requests.RequestException as e:
            if attempt == 4:
                raise
            print(f"  request failed ({e}); retry {attempt + 1}/4")
            time.sleep(2 * (attempt + 1))


def fetch_bybit(start_ms: int, end_ms: int) -> list[list]:
    rows, cursor = [], end_ms
    while cursor > start_ms:
        data = _get(
            "https://api.bybit.com/v5/market/kline",
            {"category": "spot", "symbol": "SOLUSDC", "interval": "60", "limit": 1000, "end": cursor},
        )
        if data.get("retCode") != 0:
            raise RuntimeError(f"Bybit error: {data.get('retMsg')}")
        batch = data["result"]["list"]  # newest first: [start, o, h, l, c, volume, turnover]
        if not batch:
            break
        rows += [[int(b[0]), *map(float, b[1:6])] for b in batch]
        oldest = int(batch[-1][0])
        if oldest >= cursor:
            break
        cursor = oldest - 1
        time.sleep(0.2)
    return rows


def fetch_binance(start_ms: int, end_ms: int) -> list[list]:
    rows, cursor = [], start_ms
    while cursor < end_ms:
        batch = _get(
            "https://data-api.binance.vision/api/v3/klines",
            {"symbol": "SOLUSDC", "interval": "1h", "limit": 1000, "startTime": cursor, "endTime": end_ms},
        )
        if not batch:
            break
        rows += [[int(b[0]), *map(float, b[1:6])] for b in batch]  # [open_time, o, h, l, c, volume, ...]
        cursor = int(batch[-1][0]) + HOUR_MS
        time.sleep(0.2)
    return rows


def download(source: str, years: float) -> pd.DataFrame:
    now_ms = int(time.time() * 1000)
    # Only fully closed candles: the current hour's candle is still forming.
    end_ms = (now_ms // HOUR_MS) * HOUR_MS - 1
    start_ms = end_ms - int(years * 365 * 24) * HOUR_MS
    print(f"Fetching SOLUSDC 1h from {source} ({years} years)...")
    rows = fetch_bybit(start_ms, end_ms) if source == "bybit" else fetch_binance(start_ms, end_ms)
    if not rows:
        raise RuntimeError("No candles returned")

    df = pd.DataFrame(rows, columns=["start_ms", "open", "high", "low", "close", "volume"])
    df = df.drop_duplicates("start_ms").sort_values("start_ms")
    df = df[(df.start_ms >= start_ms) & (df.start_ms + HOUR_MS - 1 <= end_ms)]
    df["timestamp"] = df.start_ms // 1000
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    df = df[["timestamp", "datetime", "open", "high", "low", "close", "volume"]].dropna()
    return df.reset_index(drop=True)


def report(df: pd.DataFrame) -> None:
    gaps = df.timestamp.diff().dropna()
    gaps = gaps[gaps != 3600]
    print(f"Candles: {len(df)} | {df.datetime.iloc[0]} -> {df.datetime.iloc[-1]}")
    print(f"Time gaps: {len(gaps)} | zero-volume candles: {(df.volume == 0).mean():.1%}")
    print(f"Flat candles (high == low): {(df.high == df.low).mean():.1%}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["bybit", "binance"], default="bybit")
    ap.add_argument("--years", type=float, default=2.0)
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()

    candles_df = download(args.source, args.years)
    report(candles_df)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    if os.path.exists(args.out):
        backup = f"{args.out}.bak-{int(time.time())}"
        shutil.move(args.out, backup)
        print(f"Previous file moved to {backup}")
    candles_df.to_csv(args.out, index=False)
    print(f"Saved dataset to: {args.out}")
