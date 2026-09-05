"""Pick top liquid day-trading stocks using sampled daily volume (2-year window)."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import requests

BASE_URL = "https://api.finmindtrade.com/api/v4/data"
OUT = Path(__file__).resolve().parent / "data" / "universe.csv"


def headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {os.environ['FINMIND_TOKEN']}"}


def daytrade_ids(date: str) -> set[str]:
    r = requests.get(
        BASE_URL,
        headers=headers(),
        params={"dataset": "TaiwanStockDayTrading", "start_date": date},
        timeout=60,
    )
    return set(pd.DataFrame(r.json()["data"])["stock_id"].astype(str))


def trading_dates(start: str, end: str) -> list[str]:
    r = requests.get(
        BASE_URL,
        headers=headers(),
        params={"dataset": "TaiwanStockTradingDate", "start_date": start, "end_date": end},
        timeout=60,
    )
    return sorted(pd.DataFrame(r.json()["data"])["date"].tolist())


def main(top_n: int = 25, start: str = "2024-01-01", end: str = "2026-08-31") -> None:
    eligible = daytrade_ids(end)
    dates = trading_dates(start, end)
    # ~40 evenly spaced sample days to estimate average liquidity
    step = max(1, len(dates) // 40)
    sample_dates = dates[::step]
    frames = []
    for d in sample_dates:
        r = requests.get(
            BASE_URL,
            headers=headers(),
            params={"dataset": "TaiwanStockPrice", "start_date": d, "end_date": d},
            timeout=120,
        )
        chunk = pd.DataFrame(r.json()["data"])
        if chunk.empty:
            continue
        chunk = chunk[["date", "stock_id", "Trading_Volume", "close", "max", "min"]]
        frames.append(chunk)
    px = pd.concat(frames, ignore_index=True)
    px = px[px["stock_id"].astype(str).isin(eligible)]
    px = px[px["stock_id"].astype(str).str.match(r"^\d{4}$")]
    for c in ["Trading_Volume", "close", "max", "min"]:
        px[c] = pd.to_numeric(px[c], errors="coerce")
    px["intraday_range_pct"] = (px["max"] - px["min"]) / px["close"]

    agg = (
        px.groupby("stock_id")
        .agg(
            avg_volume=("Trading_Volume", "mean"),
            sample_days=("date", "nunique"),
            avg_range_pct=("intraday_range_pct", "mean"),
        )
        .reset_index()
    )
    agg = agg.sort_values("avg_volume", ascending=False).head(top_n)
    agg["stock_id"] = agg["stock_id"].astype(str).str.zfill(4)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    agg.to_csv(OUT, index=False)
    print(agg.to_string(index=False))
    print(f"\nSaved {len(agg)} symbols -> {OUT}")


if __name__ == "__main__":
    main()
