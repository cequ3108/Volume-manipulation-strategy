"""Fetch TaiwanStockKBar for multiple symbols (one day per API call)."""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import pandas as pd
import requests

BASE_URL = "https://api.finmindtrade.com/api/v4/data"
DATA_DIR = Path(__file__).resolve().parent / "data" / "kbar"
UNIVERSE_CSV = Path(__file__).resolve().parent / "data" / "universe.csv"

DEFAULT_SYMBOLS = [
    "2330", "2317", "2454", "2303", "0050", "0056",
    "3481", "6770", "2409", "2344", "3231", "2609",
    "2887", "2324", "1303",
]


def headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {os.environ['FINMIND_TOKEN']}"}


def trading_dates(start: str, end: str | None = None) -> list[str]:
    end = end or pd.Timestamp.today().strftime("%Y-%m-%d")
    resp = requests.get(
        BASE_URL,
        headers=headers(),
        params={"dataset": "TaiwanStockTradingDate", "start_date": start, "end_date": end},
        timeout=60,
    )
    resp.raise_for_status()
    df = pd.DataFrame(resp.json()["data"])
    return sorted(d for d in df["date"].tolist() if d <= end)


def load_symbols(path: Path | None) -> list[str]:
    if path and path.exists():
        df = pd.read_csv(path, dtype={"stock_id": str})
        return df["stock_id"].str.zfill(4).tolist()
    return DEFAULT_SYMBOLS


def fetch_day(symbol: str, date: str, retries: int = 4) -> pd.DataFrame:
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            resp = requests.get(
                BASE_URL,
                headers=headers(),
                params={"dataset": "TaiwanStockKBar", "data_id": symbol, "start_date": date},
                timeout=60,
            )
            resp.raise_for_status()
            rows = resp.json().get("data") or []
            if not rows:
                return pd.DataFrame()
            df = pd.DataFrame(rows)
            df["datetime"] = pd.to_datetime(df["date"] + " " + df["minute"])
            return df
        except Exception as exc:
            last_err = exc
            time.sleep(2**attempt)
    raise RuntimeError(f"fetch failed {symbol} {date}") from last_err


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2024-01-01", help="First trading date (inclusive)")
    parser.add_argument("--universe", default=str(UNIVERSE_CSV), help="CSV with stock_id column")
    parser.add_argument("--sleep", type=float, default=0.28)
    args = parser.parse_args()

    symbols = load_symbols(Path(args.universe))
    dates = trading_dates(args.start)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    total = len(symbols) * len(dates)
    done = skipped = fetched = empty = 0

    for sym in symbols:
        sym_dir = DATA_DIR / sym
        sym_dir.mkdir(exist_ok=True)
        for date in dates:
            done += 1
            out = sym_dir / f"{date}.parquet"
            if out.exists():
                skipped += 1
                continue
            df = fetch_day(sym, date)
            if not df.empty:
                df.to_parquet(out, index=False)
                fetched += 1
            else:
                empty += 1
            time.sleep(args.sleep)
            if done % 200 == 0:
                print(f"progress {done}/{total} | fetched={fetched} skip={skipped} empty={empty} | last={sym} {date}")
        print(f"done {sym} ({len(list(sym_dir.glob('*.parquet')))} files)")

    print(f"complete: symbols={len(symbols)} dates={len(dates)} fetched={fetched} skipped={skipped} empty={empty}")


if __name__ == "__main__":
    main()
