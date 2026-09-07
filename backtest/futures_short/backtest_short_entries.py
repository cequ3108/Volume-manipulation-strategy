#!/usr/bin/env python3
"""Backtest conditional short-entry timings on TX front-month (day session)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path("/workspace/backtest/data/futures/TX_front_day_session.parquet")
OUT_DIR = Path("/workspace/backtest/futures_short")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Round-trip cost in fraction of price (fees + slippage approx).
COST_RT = 0.00015  # ~1.5 bps round trip; conservative for index futures


def load() -> pd.DataFrame:
    df = pd.read_parquet(DATA).copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    for c in ["open", "max", "min", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["ret"] = df["ret_adj"].astype(float)
    df["ma20"] = df["close"].rolling(20).mean()
    df["ma60"] = df["close"].rolling(60).mean()
    df["ma120"] = df["close"].rolling(120).mean()
    df["ma20_slope"] = df["ma20"] - df["ma20"].shift(5)
    df["ma60_slope"] = df["ma60"] - df["ma60"].shift(5)
    df["ll20"] = df["min"].rolling(20).min().shift(1)  # prior 20d low
    df["hh20"] = df["max"].rolling(20).max().shift(1)
    df["ret20"] = df["close"].pct_change(20)
    df["ret60"] = df["close"].pct_change(60)
    df["atr14"] = (df["max"] - df["min"]).rolling(14).mean()
    df["atr_ratio"] = df["atr14"] / df["atr14"].rolling(60).mean()
    # RSI 14
    ch = df["close"].diff()
    up = ch.clip(lower=0).rolling(14).mean()
    down = (-ch.clip(upper=0)).rolling(14).mean()
    rs = up / down.replace(0, np.nan)
    df["rsi14"] = 100 - (100 / (1 + rs))
    df["year"] = df["date"].dt.year
    # market year label by buy&hold
    return df


def apply_cost(position: pd.Series, ret: pd.Series) -> pd.Series:
    """position in {-1,0}; short PnL = position * market_return (pos=-1 → -ret)."""
    pos = position.fillna(0).astype(float)
    turn = pos.diff().abs().fillna(pos.abs())
    # Charge one round-trip cost whenever exposure changes (enter or exit).
    return (pos * ret) - (turn * COST_RT)


def run_strategy(df: pd.DataFrame, want_short: pd.Series, name: str) -> pd.DataFrame:
    """Enter short next open approx by using next-day return after signal at close.
    Signal at t close → position from t+1.
    """
    signal = want_short.fillna(False).astype(bool)
    pos = signal.shift(1).fillna(False).astype(float) * -1.0
    pnl = apply_cost(pos, df["ret"])
    out = df[["date", "year", "close", "ret"]].copy()
    out["position"] = pos
    out["pnl"] = pnl
    out["equity"] = (1 + pnl.fillna(0)).cumprod()
    out["strategy"] = name
    return out


def summarize(out: pd.DataFrame) -> dict:
    pnl = out["pnl"].dropna()
    eq = out["equity"]
    years = max((out["date"].iloc[-1] - out["date"].iloc[0]).days / 365.25, 1e-9)
    total = eq.iloc[-1] / eq.iloc[0] - 1 if len(eq) else 0
    cagr = (eq.iloc[-1]) ** (1 / years) - 1 if eq.iloc[-1] > 0 else -1
    dd = eq / eq.cummax() - 1
    maxdd = dd.min()
    # bull/bear years by buy&hold year return
    bh = out.groupby("year")["ret"].sum()  # rough log-ish; better compound
    by = out.groupby("year").apply(
        lambda g: pd.Series(
            {
                "bh": (1 + g["ret"].fillna(0)).prod() - 1,
                "st": (1 + g["pnl"].fillna(0)).prod() - 1,
                "exposure": (g["position"] < 0).mean(),
            }
        ),
        include_groups=False,
    )
    bull = by[by["bh"] > 0]
    bear = by[by["bh"] <= 0]
    # staircase score: low abs return in bull, high in bear
    return {
        "total_return": float(total),
        "cagr": float(cagr),
        "max_dd": float(maxdd),
        "sharpe": float(pnl.mean() / pnl.std() * np.sqrt(252)) if pnl.std() > 0 else 0.0,
        "time_in_short": float((out["position"] < 0).mean()),
        "bull_years_avg": float(bull["st"].mean()) if len(bull) else np.nan,
        "bear_years_avg": float(bear["st"].mean()) if len(bear) else np.nan,
        "bull_years_median": float(bull["st"].median()) if len(bull) else np.nan,
        "bear_years_median": float(bear["st"].median()) if len(bear) else np.nan,
        "n_bull_years": int(len(bull)),
        "n_bear_years": int(len(bear)),
        "yearly": by.reset_index().to_dict(orient="records"),
    }


def main() -> None:
    df = load()
    # warm-up
    df = df.dropna(subset=["ma120", "ret", "ll20", "rsi14", "atr_ratio"]).copy()

    strategies = {}

    strategies["S0_always_short"] = df["close"] > 0  # always true

    strategies["S1_below_ma20"] = df["close"] < df["ma20"]

    strategies["S2_below_ma60"] = df["close"] < df["ma60"]

    strategies["S3_below_ma60_down"] = (df["close"] < df["ma60"]) & (df["ma60_slope"] < 0)

    strategies["S4_below_ma120_down"] = (df["close"] < df["ma120"]) & (
        df["ma120"] < df["ma120"].shift(5)
    )

    strategies["S5_death_cross"] = df["ma20"] < df["ma60"]

    strategies["S6_break_ll20"] = df["close"] < df["ll20"]

    strategies["S7_ma60_down_and_ll20"] = (df["close"] < df["ma60"]) & (df["ma60_slope"] < 0) & (
        df["close"] < df["ll20"]
    )

    strategies["S8_ret60_neg_ma60"] = (df["ret60"] < 0) & (df["close"] < df["ma60"])

    strategies["S9_rsi_overbought_short"] = df["rsi14"] > 70  # expected poor / for contrast

    strategies["S10_vol_expand_ma60"] = (df["close"] < df["ma60"]) & (df["ma60_slope"] < 0) & (
        df["atr_ratio"] > 1.1
    )

    # Exit overlays: some strategies use same condition as hold; optional tighter exit built into condition.

    rows = []
    equity_map = {}
    yearly_rows = []

    for name, cond in strategies.items():
        out = run_strategy(df, cond, name)
        stats = summarize(out)
        stats["strategy"] = name
        rows.append(stats)
        equity_map[name] = out.set_index("date")["equity"]
        for y in stats["yearly"]:
            yearly_rows.append({"strategy": name, **y})

    summary = pd.DataFrame(rows).drop(columns=["yearly"])
    # Prefer: positive CAGR, bull avg near 0 or > -5%, bear avg strongly positive
    summary["asymmetric_score"] = (
        summary["cagr"].clip(lower=-0.5)
        + summary["bear_years_avg"].fillna(0) * 0.8
        - summary["bull_years_avg"].fillna(0).clip(upper=0).abs() * 1.2  # penalize bull losses
        + summary["bull_years_avg"].fillna(0).clip(lower=0) * 0.3  # small reward if bull positive
        - summary["max_dd"].abs() * 0.3
    )
    summary = summary.sort_values("asymmetric_score", ascending=False)

    yearly = pd.DataFrame(yearly_rows)
    summary.to_csv(OUT_DIR / "short_entry_summary.csv", index=False)
    yearly.to_csv(OUT_DIR / "short_entry_yearly.csv", index=False)

    # equity curves for top strategies
    eq = pd.DataFrame(equity_map)
    eq.to_csv(OUT_DIR / "short_entry_equity.csv")

    # buy & hold for reference
    bh = (1 + df.set_index("date")["ret"].fillna(0)).cumprod()
    bh.to_csv(OUT_DIR / "buyhold_equity.csv", header=["equity"])

    top = summary.head(5).to_dict(orient="records")
    report = {
        "period": [str(df["date"].iloc[0].date()), str(df["date"].iloc[-1].date())],
        "n_days": int(len(df)),
        "cost_rt": COST_RT,
        "session": "TX front-month, day session (position)",
        "signal_timing": "signal at close t, position from t+1 (next-day return)",
        "top_by_asymmetric_score": top,
        "all_ranked": summary.to_dict(orient="records"),
    }
    (OUT_DIR / "short_entry_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("=== RANKED (asymmetric short: flat bull / strong bear) ===")
    cols = [
        "strategy",
        "cagr",
        "total_return",
        "max_dd",
        "sharpe",
        "time_in_short",
        "bull_years_avg",
        "bear_years_avg",
        "asymmetric_score",
    ]
    print(summary[cols].to_string(index=False))
    print("\n=== TOP strategy yearly ===")
    best = summary.iloc[0]["strategy"]
    print(yearly[yearly.strategy == best].to_string(index=False))


if __name__ == "__main__":
    main()
