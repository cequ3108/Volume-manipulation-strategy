"""Fetch TaiwanStockKBar for multiple symbols (one day per API call)."""

from __future__ import annotations

import os
import time
from pathlib import Path

import pandas as pd
import requests

BASE_URL = "https://api.finmindtrade.com/api/v4/data"
DATA_DIR = Path(__file__).resolve().parent / "data" / "kbar"

# Liquid day-trading candidates + large caps
DEFAULT_SYMBOLS = [
    "2330", "2317", "2454", "2303", "0050", "0056",
    "3481", "6770", "2409", "2344", "3231", "2609",
    "2887", "2324", "1303",
]


def headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {os.environ['FINMIND_TOKEN']}"}


def trading_dates(start: str = "2025-06-01", end: str | None = None) -> list[str]:
    end = end or pd.Timestamp.today().strftime("%Y-%m-%d")
    resp = requests.get(
        BASE_URL,
        headers=headers(),
        params={"dataset": "TaiwanStockTradingDate", "start_date": start, "end_date": end},
        timeout=60,
    )
    resp.raise_for_status()
    df = pd.DataFrame(resp.json()["data"])
    dates = sorted(d for d in df["date"].tolist() if d <= end)
    return dates


def fetch_day(symbol: str, date: str) -> pd.DataFrame:
    resp = requests.get(
        BASE_URL,
        headers=headers(),
        params={
            "dataset": "TaiwanStockKBar",
            "data_id": symbol,
            "start_date": date,
        },
        timeout=60,
    )
    resp.raise_for_status()
    rows = resp.json().get("data") or []
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["datetime"] = pd.to_datetime(df["date"] + " " + df["minute"])
    return df


def main(symbols: list[str] | None = None, n_days: int = 60) -> None:
    symbols = symbols or DEFAULT_SYMBOLS
    dates = trading_dates("2025-06-01")[-n_days:]
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for sym in symbols:
        sym_dir = DATA_DIR / sym
        sym_dir.mkdir(exist_ok=True)
        for i, date in enumerate(dates):
            out = sym_dir / f"{date}.parquet"
            if out.exists():
                continue
            df = fetch_day(sym, date)
            if not df.empty:
                df.to_parquet(out, index=False)
            time.sleep(0.32)
            if (i + 1) % 15 == 0:
                print(f"{sym}: {i + 1}/{len(dates)}")
        print(f"done {sym}")


if __name__ == "__main__":
    main()
