#!/usr/bin/env python3
"""MACD bearish divergence → breakdown short on TX day-session front month.

Compares exit / add-on rules:
  - exit when MACD histogram turns up / MACD line crosses signal up
  - exit on volume spike
  - exit at fixed profit (price decline from entry)
  - optional add after further decline, then shared exits
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path("/workspace/backtest/data/futures/TX_front_day_session.parquet")
OUT = Path("/workspace/backtest/futures_short")
OUT.mkdir(parents=True, exist_ok=True)
COST = 0.00015  # per side flip approx (enter/exit/add charged on |delta|)


def macd_df(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    ema_f = close.ewm(span=fast, adjust=False).mean()
    ema_s = close.ewm(span=slow, adjust=False).mean()
    line = ema_f - ema_s
    sig = line.ewm(span=signal, adjust=False).mean()
    hist = line - sig
    return pd.DataFrame({"macd": line, "signal": sig, "hist": hist})


def swing_points(series: pd.Series, order: int = 5) -> tuple[pd.Series, pd.Series]:
    """Local extrema: True at swing high/low with `order` bars each side."""
    vals = series.values
    n = len(vals)
    sh = np.zeros(n, dtype=bool)
    sl = np.zeros(n, dtype=bool)
    for i in range(order, n - order):
        w = vals[i - order : i + order + 1]
        if vals[i] == np.max(w) and vals[i] > vals[i - 1] and vals[i] >= vals[i + 1]:
            sh[i] = True
        if vals[i] == np.min(w) and vals[i] < vals[i - 1] and vals[i] <= vals[i + 1]:
            sl[i] = True
    return pd.Series(sh, index=series.index), pd.Series(sl, index=series.index)


def find_bearish_div_events(df: pd.DataFrame, order: int = 5) -> pd.DataFrame:
    """Price HH + MACD hist LH within recent swings → bearish divergence event at 2nd swing high."""
    sh, _ = swing_points(df["close"], order=order)
    # also allow swing on high
    sh2, _ = swing_points(df["max"], order=order)
    swing_high = sh | sh2

    events = []
    idxs = list(df.index[swing_high.values])
    for j in range(1, len(idxs)):
        i1, i2 = idxs[j - 1], idxs[j]
        if i2 - i1 < 5 or i2 - i1 > 60:
            continue
        p1, p2 = df.at[i1, "close"], df.at[i2, "close"]
        # use max of local window for price high quality
        p1 = max(p1, df.at[i1, "max"])
        p2 = max(p2, df.at[i2, "max"])
        h1, h2 = df.at[i1, "hist"], df.at[i2, "hist"]
        m1, m2 = df.at[i1, "macd"], df.at[i2, "macd"]
        # classic: price higher high, hist lower high (and preferably both hist > 0 area)
        price_hh = p2 > p1 * 1.001
        hist_lh = h2 < h1
        macd_lh = m2 < m1
        if price_hh and (hist_lh or macd_lh) and h1 > 0:
            events.append(
                {
                    "div_i": i2,
                    "div_date": df.at[i2, "date"],
                    "swing1_i": i1,
                    "swing2_i": i2,
                    "p1": float(p1),
                    "p2": float(p2),
                    "h1": float(h1),
                    "h2": float(h2),
                }
            )
    return pd.DataFrame(events)


@dataclass
class Trade:
    entry_i: int
    entry_date: object
    entry_px: float
    div_date: object
    exit_i: int | None = None
    exit_date: object | None = None
    exit_px: float | None = None
    exit_reason: str = ""
    qty: float = 1.0  # 1 = full, 2 = after add
    add_i: int | None = None
    add_px: float | None = None
    pnl: float | None = None  # fractional on average capital notionally 1


def simulate(
    df: pd.DataFrame,
    events: pd.DataFrame,
    breakdown: str = "swing_low",
    exit_mode: str = "macd_cross_up",
    add_after_dd: float | None = None,
    target_dd: float | None = None,
    max_hold: int = 60,
    vol_spike_mult: float = 2.0,
) -> list[Trade]:
    """
    After divergence, wait for breakdown then short.
    breakdown:
      - swing_low: close < min(low between two swings)
      - ma20: close < ma20
      - both: swing_low and close < ma20
    exit_mode:
      - macd_cross_up: macd crosses above signal
      - hist_turn_up: hist turns from neg/flat to rising across 0 or hist>prev and hist was declining
      - vol_spike: volume > mult * 20d avg while in trade (cover)
      - target: cover after price fell target_dd from entry
      - macd_or_vol: first of macd_cross_up or vol_spike
      - macd_or_target: first of macd_cross_up or target
      - trail_ma20: cover on close > ma20
    add_after_dd: if set (e.g. 0.03), add +1 when unrealized short profit reaches that (price down)
    """
    trades: list[Trade] = []
    vol_ma = df["volume"].rolling(20).mean()

    for _, ev in events.iterrows():
        i0 = int(ev["div_i"])
        # invalidate if another trade still open — skip overlapping for clean stats
        if trades and trades[-1].exit_i is None:
            continue
        if trades and trades[-1].exit_i is not None and trades[-1].exit_i >= i0:
            # allow new setup after previous exit only
            pass

        # swing low between swings
        a, b = int(ev["swing1_i"]), int(ev["swing2_i"])
        swing_low = float(df.loc[a : b + 1, "min"].min())

        entry_i = None
        # search breakdown after divergence (and not too late)
        for i in range(i0 + 1, min(i0 + 1 + 40, len(df))):
            c = df.at[i, "close"]
            ok = False
            if breakdown == "swing_low":
                ok = c < swing_low
            elif breakdown == "ma20":
                ok = c < df.at[i, "ma20"]
            elif breakdown == "both":
                ok = (c < swing_low) and (c < df.at[i, "ma20"])
            if ok:
                entry_i = i
                break
        if entry_i is None:
            continue

        # skip if previous trade not yet exited before this entry
        if trades and trades[-1].exit_i is None:
            continue
        if trades and trades[-1].exit_i is not None and entry_i <= trades[-1].exit_i:
            continue

        tr = Trade(
            entry_i=entry_i,
            entry_date=df.at[entry_i, "date"],
            entry_px=float(df.at[entry_i, "close"]),
            div_date=ev["div_date"],
            qty=1.0,
        )

        added = False
        for i in range(entry_i + 1, min(entry_i + 1 + max_hold, len(df))):
            px = float(df.at[i, "close"])
            # unrealized move from entry (short profit if px < entry)
            move = (tr.entry_px - px) / tr.entry_px

            # optional add
            if add_after_dd is not None and not added and move >= add_after_dd:
                tr.qty = 2.0
                tr.add_i = i
                tr.add_px = px
                added = True

            reason = None
            hist = df.at[i, "hist"]
            hist_prev = df.at[i - 1, "hist"]
            macd = df.at[i, "macd"]
            sig = df.at[i, "signal"]
            macd_prev = df.at[i - 1, "macd"]
            sig_prev = df.at[i - 1, "signal"]
            vol = df.at[i, "volume"]
            vma = vol_ma.at[i]

            macd_up = (macd_prev <= sig_prev) and (macd > sig)
            hist_up = (hist > hist_prev) and (hist_prev < 0) and (hist >= 0)
            # also treat hist turning up while negative as soft signal — stick to cross for clarity
            vol_spike = pd.notna(vma) and vma > 0 and vol >= vol_spike_mult * vma
            hit_target = target_dd is not None and move >= target_dd
            back_ma20 = px > df.at[i, "ma20"]

            if exit_mode == "macd_cross_up" and macd_up:
                reason = "macd_cross_up"
            elif exit_mode == "hist_turn_up" and hist_up:
                reason = "hist_turn_up"
            elif exit_mode == "vol_spike" and vol_spike:
                reason = "vol_spike"
            elif exit_mode == "target" and hit_target:
                reason = "target"
            elif exit_mode == "macd_or_vol" and (macd_up or vol_spike):
                reason = "macd_cross_up" if macd_up else "vol_spike"
            elif exit_mode == "macd_or_target" and (macd_up or hit_target):
                reason = "target" if hit_target and not macd_up else ("macd_cross_up" if macd_up else "target")
            elif exit_mode == "trail_ma20" and back_ma20:
                reason = "trail_ma20"
            elif exit_mode == "macd_or_ma20" and (macd_up or back_ma20):
                reason = "macd_cross_up" if macd_up else "trail_ma20"

            # time stop
            if reason is None and i >= entry_i + max_hold:
                reason = "time_stop"

            if reason:
                # PnL: average entry if added
                if tr.add_px is not None:
                    avg = 0.5 * tr.entry_px + 0.5 * tr.add_px
                    # costs: enter + add + exit ≈ 3 side flips on notional; approximate 1.5 RT on avg capital
                    gross = (avg - px) / avg
                    pnl = gross - 1.5 * COST
                else:
                    gross = (tr.entry_px - px) / tr.entry_px
                    pnl = gross - COST  # ~1 RT
                tr.exit_i = i
                tr.exit_date = df.at[i, "date"]
                tr.exit_px = px
                tr.exit_reason = reason
                tr.pnl = float(pnl)
                break

        if tr.exit_i is None:
            # force close last bar
            i = len(df) - 1
            px = float(df.at[i, "close"])
            if tr.add_px is not None:
                avg = 0.5 * tr.entry_px + 0.5 * tr.add_px
                tr.pnl = float((avg - px) / avg - 1.5 * COST)
            else:
                tr.pnl = float((tr.entry_px - px) / tr.entry_px - COST)
            tr.exit_i = i
            tr.exit_date = df.at[i, "date"]
            tr.exit_px = px
            tr.exit_reason = "eod"
        trades.append(tr)

    return trades


def trade_stats(trades: list[Trade], name: str) -> dict:
    if not trades:
        return {
            "name": name,
            "n": 0,
            "win_rate": np.nan,
            "avg_pnl": np.nan,
            "med_pnl": np.nan,
            "total_pnl_sum": 0.0,
            "profit_factor": np.nan,
            "avg_win": np.nan,
            "avg_loss": np.nan,
            "max_win": np.nan,
            "max_loss": np.nan,
            "avg_hold_days": np.nan,
        }
    pnls = np.array([t.pnl for t in trades], dtype=float)
    wins = pnls[pnls > 0]
    losses = pnls[pnls <= 0]
    holds = np.array([(t.exit_i - t.entry_i) for t in trades], dtype=float)
    gross_win = wins.sum() if len(wins) else 0.0
    gross_loss = -losses.sum() if len(losses) else 0.0
    pf = gross_win / gross_loss if gross_loss > 0 else np.inf
    return {
        "name": name,
        "n": int(len(trades)),
        "win_rate": float((pnls > 0).mean()),
        "avg_pnl": float(pnls.mean()),
        "med_pnl": float(np.median(pnls)),
        "total_pnl_sum": float(pnls.sum()),
        "profit_factor": float(pf) if np.isfinite(pf) else None,
        "avg_win": float(wins.mean()) if len(wins) else 0.0,
        "avg_loss": float(losses.mean()) if len(losses) else 0.0,
        "max_win": float(pnls.max()),
        "max_loss": float(pnls.min()),
        "avg_hold_days": float(holds.mean()),
        "add_rate": float(np.mean([t.add_i is not None for t in trades])),
    }


def main() -> None:
    df = pd.read_parquet(DATA).copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df["ret"] = df["ret_adj"].astype(float)
    df["ma20"] = df["close"].rolling(20).mean()
    df["ma60"] = df["close"].rolling(60).mean()
    m = macd_df(df["close"])
    df = pd.concat([df, m], axis=1)
    df = df.dropna(subset=["ma60", "hist"]).reset_index(drop=True)

    # Bull-market filter for setups: only count divergence while above MA60 (高檔／大多頭結構)
    events = find_bearish_div_events(df, order=5)
    events = events.merge(
        df.reset_index().rename(columns={"index": "div_i"})[["div_i", "close", "ma60"]],
        on="div_i",
        how="left",
    )
    events["in_uptrend"] = events["close"] > events["ma60"]
    bull_events = events[events["in_uptrend"]].copy()

    variants = []
    # breakdown styles x exits
    for br in ["swing_low", "ma20", "both"]:
        for ex in [
            "macd_cross_up",
            "hist_turn_up",
            "vol_spike",
            "trail_ma20",
            "macd_or_vol",
            "macd_or_ma20",
            "target",
            "macd_or_target",
        ]:
            for add in [None, 0.03, 0.05]:
                for tgt in [None, 0.05, 0.08]:
                    # only meaningful combos
                    if ex in ("target", "macd_or_target") and tgt is None:
                        continue
                    if ex not in ("target", "macd_or_target") and tgt is not None:
                        continue
                    name = f"br={br}|ex={ex}|add={add}|tgt={tgt}"
                    variants.append((name, br, ex, add, tgt))

    rows = []
    best_trades = None
    best_score = -1e9
    best_name = ""
    for name, br, ex, add, tgt in variants:
        trs = simulate(
            df,
            bull_events,
            breakdown=br,
            exit_mode=ex,
            add_after_dd=add,
            target_dd=tgt,
            max_hold=60,
            vol_spike_mult=2.0,
        )
        st = trade_stats(trs, name)
        st["breakdown"] = br
        st["exit_mode"] = ex
        st["add_after_dd"] = add
        st["target_dd"] = tgt
        # score: total sum pnl + mild winrate; require enough trades
        score = st["total_pnl_sum"] + 0.1 * (st["win_rate"] if st["n"] else 0)
        if st["n"] < 8:
            score -= 0.5
        st["score"] = float(score)
        rows.append(st)
        if score > best_score and st["n"] >= 8:
            best_score = score
            best_name = name
            best_trades = trs

    summary = pd.DataFrame(rows).sort_values("score", ascending=False)
    summary.to_csv(OUT / "macd_div_short_summary.csv", index=False)

    # Focused comparison table for the user question
    focus = []
    for br in ["both", "swing_low", "ma20"]:
        for ex, add, tgt in [
            ("macd_cross_up", None, None),
            ("vol_spike", None, None),
            ("macd_or_vol", None, None),
            ("trail_ma20", None, None),
            ("macd_or_ma20", None, None),
            ("macd_or_target", None, 0.05),
            ("macd_or_target", None, 0.08),
            ("macd_cross_up", 0.03, None),
            ("macd_cross_up", 0.05, None),
            ("macd_or_target", 0.03, 0.08),
            ("macd_or_vol", 0.03, None),
        ]:
            name = f"br={br}|ex={ex}|add={add}|tgt={tgt}"
            sub = summary[summary["name"] == name]
            if len(sub):
                focus.append(sub.iloc[0].to_dict())
    focus_df = pd.DataFrame(focus)
    if len(focus_df):
        focus_df = focus_df.sort_values("total_pnl_sum", ascending=False)
        focus_df.to_csv(OUT / "macd_div_short_focus.csv", index=False)

    # Save best trade list
    if best_trades:
        td = pd.DataFrame(
            [
                {
                    "div_date": t.div_date,
                    "entry_date": t.entry_date,
                    "entry_px": t.entry_px,
                    "exit_date": t.exit_date,
                    "exit_px": t.exit_px,
                    "exit_reason": t.exit_reason,
                    "pnl": t.pnl,
                    "qty": t.qty,
                    "add_date": df.at[t.add_i, "date"] if t.add_i is not None else None,
                    "add_px": t.add_px,
                    "hold_days": t.exit_i - t.entry_i,
                }
                for t in best_trades
            ]
        )
        td.to_csv(OUT / "macd_div_short_best_trades.csv", index=False)

    # Baseline: both + macd_cross_up trades detail
    base = simulate(df, bull_events, "both", "macd_cross_up", None, None)
    base_df = pd.DataFrame(
        [
            {
                "div_date": t.div_date,
                "entry_date": t.entry_date,
                "exit_date": t.exit_date,
                "exit_reason": t.exit_reason,
                "pnl": t.pnl,
                "hold_days": t.exit_i - t.entry_i,
            }
            for t in base
        ]
    )
    base_df.to_csv(OUT / "macd_div_short_baseline_trades.csv", index=False)

    report = {
        "period": [str(df["date"].iloc[0].date()), str(df["date"].iloc[-1].date())],
        "n_div_events_all": int(len(events)),
        "n_div_events_above_ma60": int(len(bull_events)),
        "definition": {
            "divergence": "price higher high vs prior swing high, MACD hist or line lower high, hist1>0",
            "bull_filter": "divergence forms while price > MA60",
            "entry": "after div, breakdown (swing low / MA20 / both), short next logic at breakdown close",
            "cost": COST,
        },
        "baseline_both_macd_exit": trade_stats(base, "baseline_both_macd"),
        "best_name": best_name,
        "best": summary.iloc[0].to_dict() if len(summary) else None,
        "top10": summary.head(10).to_dict(orient="records"),
        "focus_top": focus_df.head(12).to_dict(orient="records") if len(focus_df) else [],
    }
    (OUT / "macd_div_short_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )

    print("div events", len(events), "above MA60", len(bull_events))
    print("\n=== FOCUS (user question) ===")
    cols = [
        "name",
        "n",
        "win_rate",
        "avg_pnl",
        "total_pnl_sum",
        "profit_factor",
        "avg_hold_days",
        "add_rate",
        "max_win",
        "max_loss",
    ]
    if len(focus_df):
        print(focus_df[cols].head(15).to_string(index=False))
    print("\n=== TOP 10 by score ===")
    print(summary[cols].head(10).to_string(index=False))
    print("\nbaseline both+macd", trade_stats(base, "baseline"))
    print("best", best_name)


if __name__ == "__main__":
    main()
