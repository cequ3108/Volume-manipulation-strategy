"""Backtest ORB breakout with fixed tick-based take-profit / stop-loss ratios."""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import pandas as pd

from scan_daytrade import FLATTEN, fee_for, load_symbol, minute_grid, rt_cost, strat_orb, summarize

TICK = 0.01
RESULTS = Path(__file__).resolve().parent / "results"
DEFAULT_SYMBOLS = ["2408", "2344", "2337", "6770", "3481"]


def orb_fixed_rr(day: pd.DataFrame, orb_m: int, fee: dict, tp_ticks: int, sl_ticks: int) -> list[float]:
    cut = (pd.Timestamp("09:00:00") + pd.Timedelta(minutes=orb_m - 1)).time()
    orb = day[day["datetime"].dt.time <= cut]
    if len(orb) < 3:
        return []
    hi, lo = orb["high"].max(), orb["low"].min()
    cost = rt_cost(fee)
    pos = entry = None
    stop = tp = None
    pnls: list[float] = []

    for i in range(len(day)):
        t, px = day.iloc[i]["datetime"].time(), day.iloc[i]["close"]
        if t < cut:
            continue
        if t >= FLATTEN:
            if pos == "long":
                pnls.append((px - entry) / entry - cost)
            elif pos == "short":
                pnls.append((entry - px) / entry - cost)
            pos = None
            continue
        if pos is None:
            if px > hi:
                pos, entry = "long", px
                stop, tp = px - sl_ticks * TICK, px + tp_ticks * TICK
            elif px < lo:
                pos, entry = "short", px
                stop, tp = px + sl_ticks * TICK, px - tp_ticks * TICK
        elif pos == "long":
            if px <= stop:
                pnls.append((px - entry) / entry - cost)
                pos = None
            elif px >= tp:
                pnls.append((px - entry) / entry - cost)
                pos = None
        else:
            if px >= stop:
                pnls.append((entry - px) / entry - cost)
                pos = None
            elif px <= tp:
                pnls.append((entry - px) / entry - cost)
                pos = None
    return pnls


def orb_pct_rr(day: pd.DataFrame, orb_m: int, fee: dict, tp_pct: float, sl_pct: float) -> list[float]:
    cut = (pd.Timestamp("09:00:00") + pd.Timedelta(minutes=orb_m - 1)).time()
    orb = day[day["datetime"].dt.time <= cut]
    if len(orb) < 3:
        return []
    hi, lo = orb["high"].max(), orb["low"].min()
    cost = rt_cost(fee)
    pos = entry = None
    stop = tp = None
    pnls: list[float] = []

    for i in range(len(day)):
        t, px = day.iloc[i]["datetime"].time(), day.iloc[i]["close"]
        if t < cut:
            continue
        if t >= FLATTEN:
            if pos == "long":
                pnls.append((px - entry) / entry - cost)
            elif pos == "short":
                pnls.append((entry - px) / entry - cost)
            pos = None
            continue
        if pos is None:
            if px > hi:
                pos, entry = "long", px
                stop, tp = px * (1 - sl_pct), px * (1 + tp_pct)
            elif px < lo:
                pos, entry = "short", px
                stop, tp = px * (1 + sl_pct), px * (1 - tp_pct)
        elif pos == "long":
            if px <= stop:
                pnls.append((px - entry) / entry - cost)
                pos = None
            elif px >= tp:
                pnls.append((px - entry) / entry - cost)
                pos = None
        else:
            if px >= stop:
                pnls.append((entry - px) / entry - cost)
                pos = None
            elif px <= tp:
                pnls.append((entry - px) / entry - cost)
                pos = None
    return pnls


def load_days(symbol: str) -> list[pd.DataFrame]:
    raw = load_symbol(symbol)
    return [minute_grid(g) for _, g in raw.groupby("date") if not minute_grid(g).empty]


def run_scan(symbols: list[str], orb_windows: list[int]) -> pd.DataFrame:
    rows = []
    tick_combos = [
        (4, 2), (5, 2), (5, 3), (6, 3), (6, 4), (8, 4), (8, 5), (10, 5), (10, 6), (12, 6),
    ]
    pct_combos = [
        (0.002, 0.001), (0.003, 0.0015), (0.004, 0.002), (0.005, 0.0025), (0.006, 0.003),
    ]

    for sym in symbols:
        days = load_days(sym)
        if not days:
            continue
        fee = fee_for(sym)
        price = float(load_symbol(sym)["close"].median())
        be_ticks = rt_cost(fee) * price / TICK

        for orb_m in orb_windows:
            base = []
            for d in days:
                base.extend(strat_orb(d, orb_m, fee))
            s = summarize(base)
            rows.append(
                dict(
                    symbol=sym, orb_m=orb_m, exit_type="orb_stop", tp="區間停損", sl="區間停損",
                    breakeven_ticks=be_ticks, **s,
                )
            )

            for tp, sl in tick_combos:
                pnls = []
                for d in days:
                    pnls.extend(orb_fixed_rr(d, orb_m, fee, tp, sl))
                s = summarize(pnls)
                rows.append(
                    dict(
                        symbol=sym, orb_m=orb_m, exit_type="tick", tp=f"+{tp}檔", sl=f"-{sl}檔",
                        breakeven_ticks=be_ticks, **s,
                    )
                )

            for tp, sl in pct_combos:
                pnls = []
                for d in days:
                    pnls.extend(orb_pct_rr(d, orb_m, fee, tp, sl))
                s = summarize(pnls)
                rows.append(
                    dict(
                        symbol=sym, orb_m=orb_m, exit_type="pct",
                        tp=f"+{tp*100:.2f}%", sl=f"-{sl*100:.2f}%",
                        breakeven_ticks=be_ticks, **s,
                    )
                )
        print(f"done {sym}", flush=True)

    return pd.DataFrame(rows)


def pooled_summary(df: pd.DataFrame, symbols: list[str], min_n: int = 100) -> pd.DataFrame:
  """Re-aggregate tick/pct configs across symbols (equal weight per trade)."""
  rows = []
  keys = ["orb_m", "exit_type", "tp", "sl"]
  sub = df[df["symbol"].isin(symbols)]
  for key, grp in sub.groupby(keys):
    total_n = grp["n"].sum()
    if total_n < min_n:
      continue
    weighted_avg = (grp["avg_pnl"] * grp["n"]).sum() / total_n
    weighted_wr = (grp["win_rate"] * grp["n"]).sum() / total_n
    rows.append({**dict(zip(keys, key)), "n": int(total_n), "win_rate": weighted_wr, "avg_pnl": weighted_avg})
  return pd.DataFrame(rows).sort_values("avg_pnl", ascending=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS))
    parser.add_argument("--min-n", type=int, default=100)
    args = parser.parse_args()
    symbols = [s.strip().zfill(4) for s in args.symbols.split(",")]

    df = run_scan(symbols, orb_windows=[5, 10, 15])
    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / "orb_rr_scan.csv"
    df.to_csv(out, index=False)

    pos = df[(df["n"] >= args.min_n) & (df["avg_pnl"] > 0)].sort_values(["avg_pnl", "win_rate"], ascending=False)
    print("\n=== 單檔正期望 (n>=100) TOP 20 ===")
    print(pos.head(20).to_string(index=False))

    for sym in symbols:
        sub = df[(df["symbol"] == sym) & (df["n"] >= args.min_n)].sort_values("avg_pnl", ascending=False).head(5)
        print(f"\n=== {sym} TOP 5 ===")
        print(sub[["orb_m", "exit_type", "tp", "sl", "n", "win_rate", "avg_pnl", "pf"]].to_string(index=False))

    pool = pooled_summary(df, ["2408", "2344", "2337"], min_n=args.min_n)
    pool.to_csv(RESULTS / "orb_rr_pooled_highvol.csv", index=False)
    print("\n=== 高振幅池合併 TOP 15 ===")
    print(pool.head(15).to_string(index=False))

    pool2 = pooled_summary(df, symbols, min_n=args.min_n)
    print("\n=== 全部標的合併 TOP 15 ===")
    print(pool2.head(15).to_string(index=False))


if __name__ == "__main__":
    main()
