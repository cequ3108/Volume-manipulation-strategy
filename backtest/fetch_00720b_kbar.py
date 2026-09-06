"""Fetch 00720B 1-minute KBar data from FinMind (one day per request)."""

from __future__ import annotations

import os
import time
from pathlib import Path

import pandas as pd
import requests

DATA_DIR = Path(__file__).resolve().parent / "data"
BASE_URL = "https://api.finmindtrade.com/api/v4/data"


def _headers() -> dict[str, str]:
    token = os.environ["FINMIND_TOKEN"]
    return {"Authorization": f"Bearer {token}"}


def fetch_trading_dates(start_date: str = "2025-03-01") -> list[str]:
    resp = requests.get(
        BASE_URL,
        headers=_headers(),
        params={
            "dataset": "TaiwanStockPrice",
            "data_id": "00720B",
            "start_date": start_date,
        },
        timeout=60,
    )
    resp.raise_for_status()
    df = pd.DataFrame(resp.json()["data"])
    return sorted(pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d").tolist())


def fetch_kbar_day(date: str) -> pd.DataFrame:
    resp = requests.get(
        BASE_URL,
        headers=_headers(),
        params={
            "dataset": "TaiwanStockKBar",
            "data_id": "00720B",
            "start_date": date,
        },
        timeout=60,
    )
    resp.raise_for_status()
    payload = resp.json()
    if payload.get("status") != 200:
        raise RuntimeError(f"{date}: {payload}")
    rows = payload.get("data") or []
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["datetime"] = pd.to_datetime(df["date"] + " " + df["minute"])
    return df


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    dates = fetch_trading_dates()
    # Last ~120 trading days for backtest window
    dates = dates[-120:]
    frames: list[pd.DataFrame] = []
    for i, date in enumerate(dates):
        out = DATA_DIR / f"00720B_{date}.parquet"
        if out.exists():
            frames.append(pd.read_parquet(out))
            continue
        df = fetch_kbar_day(date)
        if not df.empty:
            df.to_parquet(out, index=False)
            frames.append(df)
        time.sleep(0.35)
        if (i + 1) % 20 == 0:
            print(f"fetched {i + 1}/{len(dates)}")
    all_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    all_df.to_parquet(DATA_DIR / "00720B_kbar_all.parquet", index=False)
    print(f"saved {len(all_df)} bars across {all_df['date'].nunique() if not all_df.empty else 0} days")


if __name__ == "__main__":
    main()
