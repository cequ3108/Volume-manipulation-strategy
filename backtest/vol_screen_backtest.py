"""Backtest universal ORB on volatility-screened stock pools."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from scan_daytrade import KBAR_DIR, fee_for, load_symbol, minute_grid, strat_orb, summarize

RESULTS = Path(__file__).resolve().parent / "results"


def daily_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Per-day range% and volume from 1-min bars."""
    rows = []
    for date, grp in raw.groupby("date"):
        g = grp.sort_values("datetime")
        if g.empty:
            continue
        hi, lo = g["high"].max(), g["low"].min()
        close = g["close"].iloc[-1]
        if close <= 0:
            continue
        rows.append(
            {
                "date": date,
                "range_pct": (hi - lo) / close,
                "volume": g["volume"].sum(),
                "open": g["open"].iloc[0],
                "close": close,
            }
        )
    return pd.DataFrame(rows)


def stock_profile(raw: pd.DataFrame) -> dict:
    d = daily_features(raw)
    if d.empty:
        return {}
    return {
        "days": len(d),
        "avg_range_pct": d["range_pct"].mean(),
        "median_range_pct": d["range_pct"].median(),
        "p75_range_pct": d["range_pct"].quantile(0.75),
        "avg_volume": d["volume"].mean(),
    }


def opening_range_pct(day: pd.DataFrame, orb_m: int = 15) -> float:
    cut = (pd.Timestamp("09:00:00") + pd.Timedelta(minutes=orb_m - 1)).time()
    orb = day[day["datetime"].dt.time <= cut]
    if orb.empty:
        return 0.0
    hi, lo = orb["high"].max(), orb["low"].min()
    mid = (hi + lo) / 2
    return (hi - lo) / mid if mid > 0 else 0.0


def orb15_trades(raw: pd.DataFrame, min_opening_range: float = 0.0) -> list[float]:
    sym = raw["stock_id"].iloc[0] if "stock_id" in raw.columns else "0000"
    fee = fee_for(str(sym))
    pnls: list[float] = []
    for _, grp in raw.groupby("date"):
        grid = minute_grid(grp)
        if grid.empty:
            continue
        if opening_range_pct(grid, 15) < min_opening_range:
            continue
        pnls.extend(strat_orb(grid, 15, fee, long_only=False))
    return pnls


def walk_forward_pools(
    profiles: pd.DataFrame, train_end: str, metric: str = "avg_range_pct", top_pct: float = 0.3
) -> tuple[set[str], set[str]]:
    """Pick high-vol names using only data up to train_end."""
    train = profiles[profiles["last_date"] <= train_end].copy()
    if train.empty:
        return set(), set()
    # use in-sample avg range per symbol (computed only on train window)
    ranked = train.sort_values(metric, ascending=False)
    n = max(1, int(len(ranked) * top_pct))
    high = set(ranked.head(n)["symbol"])
    low = set(ranked.tail(n)["symbol"])
    return high, low


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-days", type=int, default=400)
    parser.add_argument("--train-end", default="2025-06-30")
    parser.add_argument("--top-pct", type=float, default=0.3, help="Top fraction by avg range")
    args = parser.parse_args()

    symbols = sorted(p.name for p in KBAR_DIR.iterdir() if p.is_dir())
    profiles = []
    trade_store: dict[str, list[float]] = {}
    trade_store_oos: dict[str, list[float]] = {}

    for sym in symbols:
        raw = load_symbol(sym)
        if raw.empty:
            continue
        n_files = len(list((KBAR_DIR / sym).glob("*.parquet")))
        if n_files < args.min_days:
            continue
        raw = raw.copy()
        raw["stock_id"] = sym
        prof = stock_profile(raw)
        if not prof:
            continue
        prof["symbol"] = sym
        prof["last_date"] = raw["date"].max()
        profiles.append(prof)

        # split IS / OOS by train_end
        is_raw = raw[raw["date"] <= args.train_end]
        oos_raw = raw[raw["date"] > args.train_end]
        trade_store[sym] = orb15_trades(is_raw)
        trade_store_oos[sym] = orb15_trades(oos_raw)

    prof_df = pd.DataFrame(profiles).sort_values("avg_range_pct", ascending=False)
    prof_df.to_csv(RESULTS / "volatility_profiles.csv", index=False)

    # --- static pool: top vol vs bottom vol (full sample) ---
    n_top = max(1, int(len(prof_df) * args.top_pct))
    high_vol = prof_df.head(n_top)["symbol"].tolist()
    low_vol = prof_df.tail(n_top)["symbol"].tolist()

    rows = []
    for label, pool in [("high_vol_static", high_vol), ("low_vol_static", low_vol), ("all", prof_df["symbol"].tolist())]:
        pnls = []
        for s in pool:
            pnls.extend(orb15_trades(load_symbol(s)))
        s = summarize(pnls)
        rows.append({"pool": label, "symbols": ",".join(pool), "n_symbols": len(pool), **s})

    # --- walk-forward: pick pool from IS, test on OOS ---
    is_prof = []
    for sym in prof_df["symbol"]:
        raw = load_symbol(sym)
        is_raw = raw[raw["date"] <= args.train_end]
        p = stock_profile(is_raw)
        p["symbol"] = sym
        is_prof.append(p)
    is_df = pd.DataFrame(is_prof).sort_values("avg_range_pct", ascending=False)
    n_top = max(1, int(len(is_df) * args.top_pct))
    wf_high = is_df.head(n_top)["symbol"].tolist()
    wf_low = is_df.tail(n_top)["symbol"].tolist()

    for label, pool in [("WF_high_vol_OOS", wf_high), ("WF_low_vol_OOS", wf_low)]:
        pnls = []
        for s in pool:
            raw = load_symbol(s)
            oos = raw[raw["date"] > args.train_end]
            pnls.extend(orb15_trades(oos))
        s = summarize(pnls)
        rows.append({"pool": label, "symbols": ",".join(pool), "n_symbols": len(pool), **s})

    # --- high vol pool + daily range filter ---
    for day_filter, label in [(0.0, "high_vol+no_day_filter"), (0.004, "high_vol+day_range0.4%"), (0.006, "high_vol+day_range0.6%")]:
        pnls = []
        for s in high_vol:
            raw = load_symbol(s)
            pnls.extend(orb15_trades(raw, min_opening_range=day_filter))
        s = summarize(pnls)
        rows.append({"pool": label, "symbols": ",".join(high_vol), "n_symbols": len(high_vol), **s})

    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "vol_screen_backtest.csv", index=False)

    print("=== Stock volatility profiles (avg daily range %) ===")
    print(prof_df[["symbol", "days", "avg_range_pct", "median_range_pct"]].to_string(index=False))

    print(f"\n=== High-vol pool (top {args.top_pct:.0%}): {high_vol} ===")
    print(f"=== Low-vol pool (bottom {args.top_pct:.0%}): {low_vol} ===")

    print("\n=== Backtest results (ORB 15min, VIP 1折) ===")
    print(out.to_string(index=False))

    # per-symbol in high vol pool
    print("\n=== Per-symbol ORB15 in HIGH-VOL pool ===")
    for s in high_vol:
        pnls = orb15_trades(load_symbol(s))
        ss = summarize(pnls)
        rng = prof_df.loc[prof_df["symbol"] == s, "avg_range_pct"].iloc[0]
        print(f"{s}: avg_range={rng*100:.2f}% | n={ss['n']} wr={ss['win_rate']:.1%} avg={ss['avg_pnl']*100:.3f}%")


if __name__ == "__main__":
    RESULTS.mkdir(exist_ok=True)
    main()
