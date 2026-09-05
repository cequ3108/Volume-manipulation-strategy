"""Backtest intraday strategies on 00720B using 1-minute KBar data."""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

DATA_PATH = Path(__file__).resolve().parent / "data" / "00720B_kbar_all.parquet"
TICK = 0.01


@dataclass(frozen=True)
class FeeProfile:
    name: str
    buy_rate: float
    sell_rate: float
    sell_tax_rate: float = 0.0  # bond ETF


FEE_PROFILES = [
    FeeProfile("standard_1425", 0.001425, 0.001425),
    FeeProfile("vip_3折", 0.001425 * 0.3, 0.001425 * 0.3),
    FeeProfile("vip_1折", 0.001425 * 0.1, 0.001425 * 0.1),
    FeeProfile("vip_05折", 0.001425 * 0.05, 0.001425 * 0.05),
    FeeProfile("vip_0手續+全退", 0.0, 0.0),
]


def round_trip_cost_pct(price: float, fee: FeeProfile) -> float:
    return fee.buy_rate + fee.sell_rate + fee.sell_tax_rate


def ticks_to_breakeven(price: float, fee: FeeProfile) -> float:
    cost = round_trip_cost_pct(price, fee)
    tick_pct = TICK / price
    return cost / tick_pct


def load_data() -> pd.DataFrame:
    df = pd.read_parquet(DATA_PATH)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values(["date", "datetime"]).reset_index(drop=True)
    return df


def resample_minute_grid(day_df: pd.DataFrame) -> pd.DataFrame:
    """Fill a regular 1-min grid using last close; carry forward for sparse bars."""
    day_df = day_df.copy()
    start = pd.Timestamp(f"{day_df['date'].iloc[0]} 09:00:00")
    end = pd.Timestamp(f"{day_df['date'].iloc[0]} 13:30:00")
    idx = pd.date_range(start, end, freq="1min")
    s = day_df.set_index("datetime")["close"].sort_index()
    s = s.reindex(idx).ffill()
    if s.isna().all():
        return pd.DataFrame()
    s = s.bfill()
    out = pd.DataFrame({"datetime": idx, "close": s.values})
    out["date"] = day_df["date"].iloc[0]
    return out


def apply_costs(entry: float, exit_: float, fee: FeeProfile) -> float:
    gross = (exit_ - entry) / entry
    costs = fee.buy_rate + fee.sell_rate + fee.sell_tax_rate
    return gross - costs


def summarize(trades: pd.DataFrame) -> dict:
    if trades.empty:
        return {"n": 0, "win_rate": np.nan, "avg_pnl": np.nan, "total_pnl": np.nan}
    return {
        "n": len(trades),
        "win_rate": (trades["pnl"] > 0).mean(),
        "avg_pnl": trades["pnl"].mean(),
        "total_pnl": trades["pnl"].sum(),
        "median_pnl": trades["pnl"].median(),
        "profit_factor": trades.loc[trades["pnl"] > 0, "pnl"].sum()
        / max(1e-9, -trades.loc[trades["pnl"] < 0, "pnl"].sum()),
    }


# --- Strategies ---


def strat_mm_spread(day: pd.DataFrame, spread_ticks: int, fee: FeeProfile) -> list[float]:
    """
    Simplified market-making:
    buy when minute close <= prior close - 1 tick after uptick;
    sell when price >= entry + spread_ticks * TICK before 13:20.
    Flatten at 13:20 close if still holding.
    One position at a time.
    """
    pnls: list[float] = []
    pos = None
    entry = None
    for i in range(1, len(day)):
        t = day.iloc[i]["datetime"]
        px = day.iloc[i]["close"]
        prev = day.iloc[i - 1]["close"]
        if t.time() >= pd.Timestamp("13:20:00").time():
            if pos == "long" and entry is not None:
                pnls.append(apply_costs(entry, px, fee))
                pos = None
            continue
        if pos is None and px <= prev - TICK:
            pos = "long"
            entry = px
        elif pos == "long" and entry is not None and px >= entry + spread_ticks * TICK:
            pnls.append(apply_costs(entry, px, fee))
            pos = None
            entry = None
    return pnls


def strat_vwap_revert(day: pd.DataFrame, z_entry: float, fee: FeeProfile) -> list[float]:
    """Fade deviation from running VWAP; exit when back to VWAP or EOD flatten."""
    vol = day.get("volume")
    if vol is None:
        vol = pd.Series(1, index=day.index)
    cum_v = vol.cumsum()
    cum_pv = (day["close"] * vol).cumsum()
    vwap = cum_pv / cum_v.replace(0, np.nan)
    dev = (day["close"] - vwap) / vwap
    pos = None
    entry = None
    pnls: list[float] = []
    for i in range(5, len(day)):
        t = day.iloc[i]["datetime"]
        px = day.iloc[i]["close"]
        if t.time() >= pd.Timestamp("13:20:00").time():
            if pos and entry is not None:
                pnls.append(apply_costs(entry, px, fee) * (1 if pos == "long" else -1))
                pos = None
            continue
        d = dev.iloc[i]
        if pos is None:
            if d <= -z_entry:
                pos, entry = "long", px
            elif d >= z_entry:
                pos, entry = "short", px
        else:
            if pos == "long" and px >= vwap.iloc[i]:
                pnls.append(apply_costs(entry, px, fee))
                pos = None
            elif pos == "short" and px <= vwap.iloc[i]:
                pnls.append(apply_costs(entry, px, fee) * -1 + fee.buy_rate + fee.sell_rate)
                # short PnL: (entry-exit)/entry - costs
                gross = (entry - px) / entry
                pnls[-1] = gross - (fee.buy_rate + fee.sell_rate + fee.sell_tax_rate)
                pos = None
    return pnls


def strat_orb(day: pd.DataFrame, orb_minutes: int, fee: FeeProfile) -> list[float]:
    """Opening range breakout with stop at opposite side of range."""
    orb = day[day["datetime"].dt.time <= pd.Timestamp("09:00:00").time().replace(minute=orb_minutes - 1)]
    if len(orb) < 3:
        return []
    hi = orb["close"].max()
    lo = orb["close"].min()
    pos = None
    entry = None
    stop = None
    pnls: list[float] = []
    for i in range(len(day)):
        t = day.iloc[i]["datetime"]
        px = day.iloc[i]["close"]
        if t.time() < pd.Timestamp(f"09:{orb_minutes:02d}:00").time():
            continue
        if t.time() >= pd.Timestamp("13:20:00").time():
            if pos and entry is not None:
                gross = (px - entry) / entry if pos == "long" else (entry - px) / entry
                pnls.append(gross - (fee.buy_rate + fee.sell_rate + fee.sell_tax_rate))
                pos = None
            continue
        if pos is None:
            if px > hi:
                pos, entry, stop = "long", px, lo
            elif px < lo:
                pos, entry, stop = "short", px, hi
        else:
            hit_stop = (pos == "long" and px <= stop) or (pos == "short" and px >= stop)
            if hit_stop:
                gross = (px - entry) / entry if pos == "long" else (entry - px) / entry
                pnls.append(gross - (fee.buy_rate + fee.sell_rate + fee.sell_tax_rate))
                pos = None
    return pnls


def strat_mean_revert_band(day: pd.DataFrame, lookback: int, band_ticks: int, fee: FeeProfile) -> list[float]:
    """Buy at lower band, sell at upper band around rolling mean."""
    ma = day["close"].rolling(lookback, min_periods=lookback).mean()
    pos = None
    entry = None
    pnls: list[float] = []
    for i in range(lookback, len(day)):
        t = day.iloc[i]["datetime"]
        px = day.iloc[i]["close"]
        center = ma.iloc[i]
        if np.isnan(center):
            continue
        lower = center - band_ticks * TICK
        upper = center + band_ticks * TICK
        if t.time() >= pd.Timestamp("13:20:00").time():
            if pos == "long" and entry is not None:
                pnls.append(apply_costs(entry, px, fee))
            pos = None
            continue
        if pos is None and px <= lower:
            pos, entry = "long", px
        elif pos == "long" and px >= upper:
            pnls.append(apply_costs(entry, px, fee))
            pos = None
    return pnls


def precompute_days(df: pd.DataFrame) -> list[pd.DataFrame]:
    days: list[pd.DataFrame] = []
    for _, raw in df.groupby("date"):
        grid = resample_minute_grid(raw)
        if not grid.empty:
            days.append(grid)
    return days


def run_backtests(df: pd.DataFrame, days: list[pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    breakeven_rows = []
    ref_price = float(df["close"].median())
    for fee in FEE_PROFILES:
        breakeven_rows.append(
            {
                "fee_profile": fee.name,
                "round_trip_cost_pct": round_trip_cost_pct(ref_price, fee) * 100,
                "ticks_to_breakeven@31.73": ticks_to_breakeven(ref_price, fee),
                "min_spread@31.73": ticks_to_breakeven(ref_price, fee) * TICK,
            }
        )
    breakeven_df = pd.DataFrame(breakeven_rows)

    results = []
    for fee in FEE_PROFILES:
        for spread_ticks in [1, 2, 3, 4, 5]:
            all_pnls = []
            for day in days:
                all_pnls.extend(strat_mm_spread(day, spread_ticks, fee))
            s = summarize(pd.DataFrame({"pnl": all_pnls}))
            results.append(
                {
                    "strategy": "mm_spread",
                    "param": f"spread={spread_ticks}tick",
                    "fee_profile": fee.name,
                    **s,
                }
            )

        for z in [0.0005, 0.001, 0.0015, 0.002]:
            all_pnls = []
            for day in days:
                all_pnls.extend(strat_vwap_revert(day, z, fee))
            s = summarize(pd.DataFrame({"pnl": all_pnls}))
            results.append(
                {
                    "strategy": "vwap_revert",
                    "param": f"z={z:.4f}",
                    "fee_profile": fee.name,
                    **s,
                }
            )

        for orb_m in [5, 10, 15, 30]:
            all_pnls = []
            for day in days:
                all_pnls.extend(strat_orb(day, orb_m, fee))
            s = summarize(pd.DataFrame({"pnl": all_pnls}))
            results.append(
                {
                    "strategy": "orb",
                    "param": f"orb={orb_m}m",
                    "fee_profile": fee.name,
                    **s,
                }
            )

        for lookback, band in itertools.product([10, 20, 30], [2, 3, 4]):
            all_pnls = []
            for day in days:
                all_pnls.extend(strat_mean_revert_band(day, lookback, band, fee))
            s = summarize(pd.DataFrame({"pnl": all_pnls}))
            results.append(
                {
                    "strategy": "mean_revert_band",
                    "param": f"lb={lookback},band={band}tick",
                    "fee_profile": fee.name,
                    **s,
                }
            )

    res_df = pd.DataFrame(results)
    return breakeven_df, res_df


def main() -> None:
    df = load_data()
    days = precompute_days(df)
    print(f"Loaded {len(df)} bars, {len(days)} trading days")
    breakeven_df, res_df = run_backtests(df, days)
    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(exist_ok=True)
    breakeven_df.to_csv(out_dir / "breakeven_ticks.csv", index=False)
    res_df.to_csv(out_dir / "strategy_scan.csv", index=False)

    pos = res_df[(res_df["n"] >= 30) & (res_df["avg_pnl"] > 0)].sort_values(
        ["avg_pnl", "win_rate"], ascending=False
    )
    print("\n=== Breakeven ticks (bond ETF 0% tax) ===")
    print(breakeven_df.to_string(index=False))
    print("\n=== Top positive-expectancy configs (n>=30) ===")
    print(pos.head(20).to_string(index=False))
    print("\n=== Best mm_spread per fee profile ===")
    mm = res_df[res_df["strategy"] == "mm_spread"].sort_values(["fee_profile", "avg_pnl"], ascending=[True, False])
    print(mm.groupby("fee_profile").head(1).to_string(index=False))


if __name__ == "__main__":
    main()
