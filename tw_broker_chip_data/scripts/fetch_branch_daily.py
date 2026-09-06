#!/usr/bin/env python3
"""FinMind broker-branch chip fetcher (rate-limit + checkpoint resume).

Sponsor cannot use whole-day storage_objects (needs SponsorPro).
We fetch per (stock_id, date), aggregate by branch, and resume via
state/checkpoint.json so long backfills survive hourly caps / CI limits.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from io import BytesIO
from pathlib import Path

import pandas as pd
import requests

TAIPEI = timezone(timedelta(hours=8))
BASE = "https://api.finmindtrade.com/api/v4"
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "daily"
STATE_DIR = ROOT / "state"
STATE_PATH = STATE_DIR / "checkpoint.json"
STOCK_CACHE = STATE_DIR / "stock_universe.csv"
LATEST_PATH = STATE_DIR / "latest.json"

MAX_REQUESTS_PER_HOUR = int(os.environ.get("FINMIND_MAX_REQ_PER_HOUR", "520"))
MIN_INTERVAL_SEC = float(os.environ.get("FINMIND_MIN_INTERVAL_SEC", "0.25"))
REQUEST_TIMEOUT = 90

EMPTY_COLS = [
    "date",
    "stock_id",
    "securities_trader_id",
    "securities_trader",
    "buy",
    "sell",
    "buy_amt",
    "sell_amt",
]


@dataclass
class RunBudget:
    max_requests: int
    used: int = 0

    def remain(self) -> int:
        return max(self.max_requests - self.used, 0)

    def consume(self, n: int = 1) -> None:
        self.used += n


class RateLimiter:
    def __init__(
        self,
        max_per_hour: int = MAX_REQUESTS_PER_HOUR,
        min_interval: float = MIN_INTERVAL_SEC,
    ):
        self.max_per_hour = max_per_hour
        self.min_interval = min_interval
        self.timestamps: list[float] = []
        self.last = 0.0

    def wait(self) -> None:
        now = time.time()
        if self.last and now - self.last < self.min_interval:
            time.sleep(self.min_interval - (now - self.last))
            now = time.time()
        cutoff = now - 3600.0
        self.timestamps = [t for t in self.timestamps if t >= cutoff]
        if len(self.timestamps) >= self.max_per_hour:
            sleep_for = self.timestamps[0] + 3600.0 - now + 1.0
            print(f"[rate] hourly cap; sleep {sleep_for:.0f}s", flush=True)
            time.sleep(max(sleep_for, 1.0))
            now = time.time()
            cutoff = now - 3600.0
            self.timestamps = [t for t in self.timestamps if t >= cutoff]
        self.timestamps.append(time.time())
        self.last = self.timestamps[-1]


LIMITER = RateLimiter()


def get_token() -> str:
    tok = os.environ.get("FINMIND_TOKEN") or os.environ.get("FINMIND_API_TOKEN") or ""
    if not tok:
        raise SystemExit("FINMIND_TOKEN is required")
    return tok


def api_json(path: str, params: dict, retries: int = 6) -> dict:
    params = {**params, "token": get_token()}
    last: Exception | None = None
    for i in range(retries):
        LIMITER.wait()
        try:
            r = requests.get(f"{BASE}{path}", params=params, timeout=REQUEST_TIMEOUT)
            if r.status_code == 429:
                wait = 60 * (i + 1)
                print(f"[rate] HTTP 429; sleep {wait}s", flush=True)
                time.sleep(wait)
                continue
            if r.status_code >= 500:
                time.sleep(2 * (i + 1))
                continue
            j = r.json()
            if r.status_code != 200:
                raise RuntimeError(f"{path} -> {r.status_code} {j}")
            return j
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(1.5 * (i + 1))
    raise RuntimeError(last)


def load_state() -> dict:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    return {
        "completed_dates": [],
        "pending_dates": [],
        "in_progress_date": None,
        "in_progress_done_stocks": [],
        "last_run_at": None,
    }


def save_state(state: dict) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    state["last_run_at"] = datetime.now(TAIPEI).isoformat()
    STATE_PATH.write_text(
        json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def stock_universe(force_refresh: bool = False) -> list[str]:
    if STOCK_CACHE.exists() and not force_refresh:
        return pd.read_csv(STOCK_CACHE, dtype=str)["stock_id"].tolist()
    j = api_json("/data", {"dataset": "TaiwanStockInfo"})
    df = pd.DataFrame(j.get("data") or [])
    df = df[df["type"].isin(["twse", "tpex"])].copy()
    df = df[df["stock_id"].astype(str).str.match(r"^\d{4}$")]
    ind = df["industry_category"].astype(str)
    df = df[~ind.str.contains(r"ETF|指數股票型|受益證券", na=False)]
    df = df.drop_duplicates(subset=["stock_id"]).sort_values("stock_id")
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(STOCK_CACHE, index=False)
    return df["stock_id"].astype(str).tolist()


def trading_dates(start: str, end: str) -> list[str]:
    j = api_json(
        "/data",
        {
            "dataset": "TaiwanStockPrice",
            "data_id": "2330",
            "start_date": start,
            "end_date": end,
        },
    )
    return [row["date"] for row in j.get("data") or []]


def day_path(date: str) -> Path:
    return DATA_DIR / f"{date}.parquet"


def persist_day(date: str, frames: list[pd.DataFrame]) -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    frames = [f for f in frames if f is not None and len(f)]
    path = day_path(date)
    if not frames:
        pd.DataFrame(columns=EMPTY_COLS).to_parquet(path, index=False)
        return path
    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(
        subset=["date", "stock_id", "securities_trader_id"], keep="last"
    )
    df.to_parquet(path, index=False)
    print(
        f"[save] {path.name} rows={len(df)} stocks={df['stock_id'].nunique()}",
        flush=True,
    )
    return path


def try_storage_objects(date: str) -> pd.DataFrame | None:
    """Whole-day parquet (SponsorPro). Returns None on current Sponsor plan."""
    LIMITER.wait()
    r = requests.get(
        f"{BASE}/storage_objects",
        params={
            "dataset": "TaiwanStockTradingDailyReport",
            "date": date,
            "token": get_token(),
        },
        timeout=180,
    )
    if r.status_code != 200:
        print(f"[bulk] unavailable HTTP {r.status_code}: {r.text[:160]}", flush=True)
        return None
    if "application/json" in r.headers.get("content-type", ""):
        print(f"[bulk] json: {r.text[:160]}", flush=True)
        return None
    df = pd.read_parquet(BytesIO(r.content))
    if "price" in df.columns:
        df = df.copy()
        df["buy_amt"] = df["buy"] * df["price"]
        df["sell_amt"] = df["sell"] * df["price"]
        df = df.groupby(
            ["date", "stock_id", "securities_trader_id", "securities_trader"],
            as_index=False,
        ).agg(
            buy=("buy", "sum"),
            sell=("sell", "sum"),
            buy_amt=("buy_amt", "sum"),
            sell_amt=("sell_amt", "sum"),
        )
    return df


def fetch_stock_day(stock_id: str, date: str) -> pd.DataFrame:
    j = api_json(
        "/taiwan_stock_trading_daily_report",
        {"data_id": stock_id, "date": date},
    )
    data = j.get("data") or []
    if not data:
        return pd.DataFrame(columns=EMPTY_COLS)
    df = pd.DataFrame(data)
    df["buy_amt"] = df["buy"] * df["price"]
    df["sell_amt"] = df["sell"] * df["price"]
    return df.groupby(
        ["date", "stock_id", "securities_trader_id", "securities_trader"],
        as_index=False,
    ).agg(
        buy=("buy", "sum"),
        sell=("sell", "sum"),
        buy_amt=("buy_amt", "sum"),
        sell_amt=("sell_amt", "sum"),
    )


def fetch_one_date(
    date: str, stocks: list[str], state: dict, budget: RunBudget
) -> bool:
    path = day_path(date)
    completed = set(state.get("completed_dates") or [])
    if date in completed and path.exists():
        print(f"[skip] {date}", flush=True)
        return True

    if state.get("in_progress_date") != date and budget.remain() > 0:
        bulk = try_storage_objects(date)
        budget.consume(1)
        if bulk is not None:
            persist_day(date, [bulk])
            completed.add(date)
            state["completed_dates"] = sorted(completed)
            state["in_progress_date"] = None
            state["in_progress_done_stocks"] = []
            state["pending_dates"] = [
                d for d in state.get("pending_dates", []) if d != date
            ]
            save_state(state)
            return True

    frames: list[pd.DataFrame] = []
    if state.get("in_progress_date") == date:
        done_stocks = set(state.get("in_progress_done_stocks") or [])
    else:
        state["in_progress_date"] = date
        state["in_progress_done_stocks"] = []
        done_stocks = set()
        save_state(state)

    if path.exists():
        try:
            old = pd.read_parquet(path)
            if len(old):
                frames.append(old)
                done_stocks |= set(old["stock_id"].astype(str))
        except Exception as e:  # noqa: BLE001
            print(f"[warn] bad partial {path}: {e}", flush=True)

    pending = [s for s in stocks if s not in done_stocks]
    print(f"[day] {date} pending={len(pending)} budget={budget.remain()}", flush=True)

    for i, sid in enumerate(pending, 1):
        if budget.remain() <= 0:
            if frames:
                persist_day(date, frames)
            state["in_progress_date"] = date
            state["in_progress_done_stocks"] = sorted(done_stocks)
            save_state(state)
            print(f"[budget] pause at {date} stock={sid}", flush=True)
            return False
        try:
            part = fetch_stock_day(sid, date)
            if len(part):
                frames.append(part)
            done_stocks.add(sid)
        except Exception as e:  # noqa: BLE001
            print(f"[error] {date} {sid}: {e}", flush=True)
        finally:
            budget.consume(1)
        if i % 40 == 0:
            persist_day(date, frames)
            state["in_progress_date"] = date
            state["in_progress_done_stocks"] = sorted(done_stocks)
            save_state(state)
            print(f"[progress] {date} {i}/{len(pending)}", flush=True)

    persist_day(date, frames)
    completed.add(date)
    state["completed_dates"] = sorted(completed)
    state["in_progress_date"] = None
    state["in_progress_done_stocks"] = []
    state["pending_dates"] = [d for d in state.get("pending_dates", []) if d != date]
    save_state(state)
    return True


def plan_dates(mode: str, start: str | None, end: str | None) -> list[str]:
    today = datetime.now(TAIPEI).date()
    end_s = end or today.isoformat()
    if mode == "daily":
        start_s = start or (today - timedelta(days=14)).isoformat()
    else:
        start_s = start or (today - timedelta(days=370)).isoformat()
    return trading_dates(start_s, end_s)


def write_latest() -> None:
    latest = sorted(DATA_DIR.glob("*.parquet"))
    pointer = {
        "latest_date": latest[-1].stem if latest else None,
        "n_days": len(latest),
        "updated_at": datetime.now(TAIPEI).isoformat(),
    }
    LATEST_PATH.write_text(
        json.dumps(pointer, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(pointer, ensure_ascii=False), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["auto", "daily", "backfill"], default="auto")
    ap.add_argument("--start", default="")
    ap.add_argument("--end", default="")
    ap.add_argument(
        "--max-requests",
        type=int,
        default=int(os.environ.get("FINMIND_RUN_MAX_REQUESTS", "500")),
    )
    ap.add_argument("--refresh-universe", action="store_true")
    args = ap.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    state = load_state()
    existing = {p.stem for p in DATA_DIR.glob("*.parquet")}

    mode = args.mode
    if mode == "auto":
        mode = "backfill" if len(existing) < 5 else "daily"
    print(f"[run] mode={mode} existing_days={len(existing)}", flush=True)

    stocks = stock_universe(force_refresh=args.refresh_universe)
    print(f"[universe] {len(stocks)} stocks", flush=True)

    dates = plan_dates(mode, args.start or None, args.end or None)
    completed = set(state.get("completed_dates") or []) | existing

    ordered: list[str] = []
    if state.get("in_progress_date"):
        ordered.append(str(state["in_progress_date"]))
    for d in dates:
        if d not in completed and d not in ordered:
            ordered.append(d)
    if mode == "daily":
        ordered = ordered[-3:]

    state["pending_dates"] = ordered
    save_state(state)
    print(f"[queue] {len(ordered)} dates", flush=True)

    budget = RunBudget(max_requests=args.max_requests)
    for d in ordered:
        finished = fetch_one_date(d, stocks, state, budget)
        if not finished:
            print("[stop] resume later from checkpoint", flush=True)
            break
    else:
        print("[done] queue finished for this run", flush=True)

    write_latest()


if __name__ == "__main__":
    main()
