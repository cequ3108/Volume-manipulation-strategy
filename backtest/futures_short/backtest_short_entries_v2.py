#!/usr/bin/env python3
"""Compare TX short-entry timings (day-session front month)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path("/workspace/backtest/data/futures/TX_front_day_session.parquet")
OUT = Path("/workspace/backtest/futures_short")
OUT.mkdir(parents=True, exist_ok=True)
COST = 0.00015


def load() -> pd.DataFrame:
    df = pd.read_parquet(DATA).copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df["ret"] = df["ret_adj"].astype(float)
    for w in (20, 60, 120):
        df[f"ma{w}"] = df["close"].rolling(w).mean()
    df["ma60_slope"] = df["ma60"] - df["ma60"].shift(5)
    df["ll20"] = df["min"].rolling(20).min().shift(1)
    df["ret20"] = df["close"].pct_change(20)
    df["ret60"] = df["close"].pct_change(60)
    above = (df["close"] > df["ma60"]).fillna(False)
    streak = []
    s = 0
    for a in above:
        s = s + 1 if a else 0
        streak.append(s)
    df["days_above_ma60"] = streak
    return df.dropna(subset=["ma120", "ret", "ll20"]).reset_index(drop=True)


def hold_from_flip(df: pd.DataFrame, min_days_above: int = 20, confirm: int = 1) -> pd.Series:
    """After a sustained uptrend, enter short once price closes below MA60 (optionally confirm N days)."""
    prior_up = df["days_above_ma60"].shift(1).fillna(0)
    below = df["close"] < df["ma60"]
    # streak of consecutive closes below MA60
    below_streak = []
    s = 0
    for b in below.fillna(False):
        s = s + 1 if b else 0
        below_streak.append(s)
    below_streak = pd.Series(below_streak, index=df.index)
    # trigger when below_streak hits confirm AND the move started from a long uptrend
    # approximate: within the current below episode, the day before episode start had prior_up>=min
    episode_start_up = prior_up.where(below_streak == 1).ffill()
    trigger = below & (below_streak >= confirm) & (episode_start_up >= min_days_above)

    hold = np.zeros(len(df), dtype=bool)
    active = False
    for i in range(len(df)):
        if active:
            if not bool(below.iloc[i]):
                active = False
        elif bool(trigger.iloc[i]):
            active = True
        hold[i] = active
    return pd.Series(hold, index=df.index)


def backtest(df: pd.DataFrame, signal: pd.Series) -> pd.DataFrame:
    pos = signal.astype(bool).shift(1).fillna(False).astype(float) * -1.0
    turn = pos.diff().abs().fillna(pos.abs())
    pnl = pos * df["ret"] - turn * COST
    out = pd.DataFrame(
        {
            "date": df["date"],
            "close": df["close"],
            "ret": df["ret"],
            "pos": pos,
            "pnl": pnl,
            "eq": (1 + pnl.fillna(0)).cumprod(),
        }
    )
    out["year"] = out["date"].dt.year
    return out


def summarize(out: pd.DataFrame, name: str) -> dict:
    d0 = out["date"].iloc[0]
    d1 = out["date"].iloc[-1]
    years = max((d1 - d0).days / 365.25, 1e-9)
    cagr = float(out["eq"].iloc[-1] ** (1 / years) - 1)
    maxdd = float((out["eq"] / out["eq"].cummax() - 1).min())
    rows = []
    for y, g in out.groupby("year"):
        rows.append(
            {
                "year": int(y),
                "bh": float((1 + g["ret"].fillna(0)).prod() - 1),
                "st": float((1 + g["pnl"].fillna(0)).prod() - 1),
                "exp": float((g["pos"] < 0).mean()),
            }
        )
    by = pd.DataFrame(rows)
    bull = by[by["bh"] > 0]
    bear = by[by["bh"] <= 0]

    def window_ret(a: str, b: str) -> float:
        w = out[(out["date"] >= a) & (out["date"] <= b)]
        return float((1 + w["pnl"].fillna(0)).prod() - 1)

    return {
        "name": name,
        "cagr": cagr,
        "total": float(out["eq"].iloc[-1] - 1),
        "maxdd": maxdd,
        "exposure": float((out["pos"] < 0).mean()),
        "bull_avg": float(bull["st"].mean()) if len(bull) else np.nan,
        "bear_avg": float(bear["st"].mean()) if len(bear) else np.nan,
        "y2018": window_ret("2018-01-01", "2018-12-31"),
        "y2022": window_ret("2022-01-01", "2022-12-31"),
        "covid": window_ret("2020-02-15", "2020-03-31"),
        "yearly": rows,
    }


def main() -> None:
    df = load()
    conds: dict[str, pd.Series] = {
        "always_short": pd.Series(True, index=df.index),
        "below_ma20": df["close"] < df["ma20"],
        "below_ma60": df["close"] < df["ma60"],
        "ma60_slope_down": (df["close"] < df["ma60"]) & (df["ma60_slope"] < 0),
        "below_ma120_down": (df["close"] < df["ma120"])
        & (df["ma120"] < df["ma120"].shift(5)),
        "death_cross_20_60": df["ma20"] < df["ma60"],
        "break_prior_ll20": df["close"] < df["ll20"],
        "ma60_down_and_ll20": (df["close"] < df["ma60"])
        & (df["ma60_slope"] < 0)
        & (df["close"] < df["ll20"]),
        "ret60_neg_and_ma60": (df["ret60"] < 0) & (df["close"] < df["ma60"]),
        "ma60_and_ret20_lt_-3pct": (df["close"] < df["ma60"])
        & (df["ma60_slope"] < 0)
        & (df["ret20"] < -0.03),
        "flip_ma60_after_20up": hold_from_flip(df, 20, confirm=1),
        "flip_ma60_confirm2": hold_from_flip(df, 20, confirm=2),
        "flip_ma60_after_40up": hold_from_flip(df, 40, confirm=1),
    }

    rows = []
    for name, sig in conds.items():
        out = backtest(df, sig)
        st = summarize(out, name)
        rows.append({k: v for k, v in st.items() if k != "yearly"})
        out.to_csv(OUT / f"eq_{name}.csv", index=False)
        pd.DataFrame(st["yearly"]).to_csv(OUT / f"yearly_{name}.csv", index=False)

    summary = pd.DataFrame(rows)
    summary["score"] = (
        summary["bear_avg"] * 1.0
        - summary["bull_avg"].clip(upper=0).abs() * 1.5
        + summary["bull_avg"].clip(lower=0) * 0.2
        + summary["cagr"] * 0.5
        + summary["y2022"] * 0.35
        + summary["covid"] * 0.2
        + summary["y2018"] * 0.2
    )
    summary = summary.sort_values("score", ascending=False)
    summary.to_csv(OUT / "short_entry_summary_v2.csv", index=False)

    report = {
        "period": [str(df["date"].iloc[0].date()), str(df["date"].iloc[-1].date())],
        "n_days": int(len(df)),
        "cost_rt": COST,
        "rules": "signal close t → short from t+1; flat when signal off; day-session TX front",
        "key_finding": (
            "2016-2026 TX is a strong bull sample; pure short-only strategies rarely have "
            "positive CAGR. Best research targets maximize bear-year capture while keeping "
            "bull-year bleed near zero."
        ),
        "ranked": summary.to_dict(orient="records"),
    }
    (OUT / "short_entry_report_v2.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(summary.to_string(index=False))
    best = summary.iloc[0]["name"]
    print("\nBEST by score:", best)
    print(pd.read_csv(OUT / f"yearly_{best}.csv").to_string(index=False))
    # also print flip confirm2 and ma60_down yearly for comparison
    for alt in ["flip_ma60_confirm2", "ma60_slope_down", "break_prior_ll20", "always_short"]:
        print(f"\nYEARLY {alt}")
        print(pd.read_csv(OUT / f"yearly_{alt}.csv").to_string(index=False))


if __name__ == "__main__":
    main()
