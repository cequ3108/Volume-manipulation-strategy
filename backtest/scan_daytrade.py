"""Scan directional day-trading strategies across Taiwan stocks (1-min KBar)."""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

KBAR_DIR = Path(__file__).resolve().parent / "data" / "kbar"
RESULTS_DIR = Path(__file__).resolve().parent / "results"

# VIP 1折 + 現股當沖證交稅 0.15%
STOCK_FEE = dict(buy=0.001425 * 0.1, sell=0.001425 * 0.1, tax=0.0015)
ETF_FEE = dict(buy=0.001425 * 0.1, sell=0.001425 * 0.1, tax=0.0010)

FLATTEN = pd.Timestamp("13:20:00").time()


def fee_for(symbol: str) -> dict[str, float]:
    return ETF_FEE if symbol.startswith("00") else STOCK_FEE


def rt_cost(fee: dict[str, float]) -> float:
    return fee["buy"] + fee["sell"] + fee["tax"]


def pnl_long(entry: float, exit_: float, fee: dict[str, float]) -> float:
    return (exit_ - entry) / entry - rt_cost(fee)


def pnl_short(entry: float, exit_: float, fee: dict[str, float]) -> float:
    return (entry - exit_) / entry - rt_cost(fee)


def load_symbol(symbol: str) -> pd.DataFrame:
    sym_dir = KBAR_DIR / symbol
    if not sym_dir.exists():
        return pd.DataFrame()
    frames = [pd.read_parquet(p) for p in sorted(sym_dir.glob("*.parquet"))]
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df.sort_values(["date", "datetime"]).reset_index(drop=True)


def minute_grid(day: pd.DataFrame) -> pd.DataFrame:
    if day.empty:
        return day
    start = pd.Timestamp(f"{day['date'].iloc[0]} 09:00:00")
    end = pd.Timestamp(f"{day['date'].iloc[0]} 13:30:00")
    idx = pd.date_range(start, end, freq="1min")
    g = day.set_index("datetime").sort_index()
    out = pd.DataFrame(index=idx)
    for col in ["open", "high", "low", "close", "volume"]:
        if col in g.columns:
            out[col] = g[col].reindex(idx).ffill().bfill()
    out["date"] = day["date"].iloc[0]
    out = out.reset_index(names="datetime")
    return out


def vwap_series(day: pd.DataFrame) -> pd.Series:
    vol = day["volume"].replace(0, np.nan).fillna(1)
    return (day["close"] * vol).cumsum() / vol.cumsum()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0).rolling(period).mean()
    down = (-delta.clip(upper=0)).rolling(period).mean()
    rs = up / down.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def summarize(trades: list[float]) -> dict:
    if not trades:
        return dict(n=0, win_rate=np.nan, avg_pnl=np.nan, total_pnl=np.nan, pf=np.nan)
    s = pd.Series(trades)
    wins = s[s > 0].sum()
    losses = -s[s < 0].sum()
    return dict(
        n=len(s),
        win_rate=float((s > 0).mean()),
        avg_pnl=float(s.mean()),
        total_pnl=float(s.sum()),
        pf=float(wins / losses) if losses > 0 else float("inf"),
    )


# ---------- strategies ----------


def strat_orb(day: pd.DataFrame, orb_m: int, fee: dict, long_only: bool = False) -> list[float]:
    cut = (pd.Timestamp("09:00:00") + pd.Timedelta(minutes=orb_m - 1)).time()
    orb = day[day["datetime"].dt.time <= cut]
    if len(orb) < max(3, orb_m // 2):
        return []
    hi, lo = orb["high"].max(), orb["low"].min()
    pos = None
    entry = None
    stop = None
    pnls: list[float] = []
    for i in range(len(day)):
        row = day.iloc[i]
        t, px = row["datetime"].time(), row["close"]
        if t < cut:
            continue
        if t >= FLATTEN:
            if pos == "long":
                pnls.append(pnl_long(entry, px, fee))
            elif pos == "short":
                pnls.append(pnl_short(entry, px, fee))
            pos = None
            continue
        if pos is None:
            if px > hi:
                pos, entry, stop = "long", px, lo
            elif not long_only and px < lo:
                pos, entry, stop = "short", px, hi
        else:
            stopped = (pos == "long" and px <= stop) or (pos == "short" and px >= stop)
            if stopped:
                pnls.append(pnl_long(entry, px, fee) if pos == "long" else pnl_short(entry, px, fee))
                pos = None
    return pnls


def strat_orb_vwap(day: pd.DataFrame, orb_m: int, fee: dict) -> list[float]:
    vwap = vwap_series(day)
    cut = (pd.Timestamp("09:00:00") + pd.Timedelta(minutes=orb_m - 1)).time()
    orb = day[day["datetime"].dt.time <= cut]
    if len(orb) < 3:
        return []
    hi, lo = orb["high"].max(), orb["low"].min()
    pos = None
    entry = stop = None
    pnls: list[float] = []
    for i in range(len(day)):
        row = day.iloc[i]
        t, px = row["datetime"].time(), row["close"]
        vw = vwap.iloc[i]
        if t < cut:
            continue
        if t >= FLATTEN:
            if pos == "long":
                pnls.append(pnl_long(entry, px, fee))
            pos = None
            continue
        if pos is None and px > hi and px > vw:
            pos, entry, stop = "long", px, lo
        elif pos == "long" and px <= stop:
            pnls.append(pnl_long(entry, px, fee))
            pos = None
    return pnls


def strat_vwap_pullback(day: pd.DataFrame, fee: dict) -> list[float]:
    vwap = vwap_series(day)
    pos = None
    entry = None
    pnls: list[float] = []
    for i in range(10, len(day)):
        row = day.iloc[i]
        t, px = row["datetime"].time(), row["close"]
        prev = day.iloc[i - 1]["close"]
        vw = vwap.iloc[i]
        if t >= FLATTEN:
            if pos == "long":
                pnls.append(pnl_long(entry, px, fee))
            pos = None
            continue
        if t < pd.Timestamp("09:20:00").time():
            continue
        if pos is None and prev > vw and px <= vw and px > day.iloc[i - 5:i]["low"].min():
            pos, entry = "long", px
        elif pos == "long" and px >= vw * 1.003:
            pnls.append(pnl_long(entry, px, fee))
            pos = None
    return pnls


def strat_momentum_break(day: pd.DataFrame, lookback: int, fee: dict) -> list[float]:
    pos = None
    entry = stop = None
    pnls: list[float] = []
    highs = day["high"].rolling(lookback).max().shift(1)
    lows = day["low"].rolling(lookback).min().shift(1)
    for i in range(lookback + 1, len(day)):
        row = day.iloc[i]
        t, px = row["datetime"].time(), row["close"]
        if t >= FLATTEN:
            if pos == "long":
                pnls.append(pnl_long(entry, px, fee))
            elif pos == "short":
                pnls.append(pnl_short(entry, px, fee))
            pos = None
            continue
        if t < pd.Timestamp("09:15:00").time():
            continue
        hi, lo = highs.iloc[i], lows.iloc[i]
        if pos is None:
            if px > hi:
                pos, entry, stop = "long", px, lo
            elif px < lo:
                pos, entry, stop = "short", px, hi
        else:
            hit = (pos == "long" and px <= stop) or (pos == "short" and px >= stop)
            if hit:
                pnls.append(pnl_long(entry, px, fee) if pos == "long" else pnl_short(entry, px, fee))
                pos = None
    return pnls


def strat_rsi_revert(day: pd.DataFrame, fee: dict, low: int = 25, high: int = 75) -> list[float]:
    r = rsi(day["close"], 14)
    pos = None
    entry = None
    pnls: list[float] = []
    for i in range(20, len(day)):
        t = day.iloc[i]["datetime"].time()
        px = day.iloc[i]["close"]
        rv = r.iloc[i]
        if np.isnan(rv):
            continue
        if t >= FLATTEN:
            if pos == "long":
                pnls.append(pnl_long(entry, px, fee))
            elif pos == "short":
                pnls.append(pnl_short(entry, px, fee))
            pos = None
            continue
        if pos is None:
            if rv <= low:
                pos, entry = "long", px
            elif rv >= high:
                pos, entry = "short", px
        else:
            if pos == "long" and rv >= 55:
                pnls.append(pnl_long(entry, px, fee))
                pos = None
            elif pos == "short" and rv <= 45:
                pnls.append(pnl_short(entry, px, fee))
                pos = None
    return pnls


def strat_gap_go(day: pd.DataFrame, prev_close: float | None, fee: dict, gap_pct: float = 0.005) -> list[float]:
    if prev_close is None or prev_close <= 0:
        return []
    open_px = day.iloc[0]["open"]
    gap = (open_px - prev_close) / prev_close
    if gap < gap_pct:
        return []
    pos = "long"
    entry = open_px
    stop = open_px * (1 - 0.008)
    pnls: list[float] = []
    for i in range(len(day)):
        px = day.iloc[i]["close"]
        t = day.iloc[i]["datetime"].time()
        if px <= stop:
            pnls.append(pnl_long(entry, px, fee))
            return pnls
        if t >= FLATTEN:
            pnls.append(pnl_long(entry, px, fee))
            return pnls
    return pnls


def run_scan(symbols: list[str]) -> pd.DataFrame:
    rows = []
    for sym in symbols:
        raw = load_symbol(sym)
        if raw.empty:
            continue
        fee = fee_for(sym)
        days = []
        prev_close_map: dict[str, float] = {}
        daily_close = raw.groupby("date")["close"].last().to_dict()
        sorted_dates = sorted(raw["date"].unique())
        for j, d in enumerate(sorted_dates):
            if j > 0:
                prev_close_map[d] = daily_close.get(sorted_dates[j - 1])

        for date, grp in raw.groupby("date"):
            grid = minute_grid(grp)
            if grid.empty:
                continue
            days.append((date, grid, prev_close_map.get(date)))

        configs = [
            ("orb", [(m, False) for m in [5, 10, 15, 30]]),
            ("orb_long_only", [(m, True) for m in [5, 10, 15, 30]]),
            ("orb_vwap_long", [(m,) for m in [10, 15, 30]]),
            ("vwap_pullback", [(None,)]),
            ("momentum", [(lb,) for lb in [15, 30, 60]]),
            ("rsi_revert", [(25, 75), (30, 70)]),
            ("gap_go", [(0.005,), (0.008,)]),
        ]

        for strat_name, params in configs:
            for p in params:
                all_pnls: list[float] = []
                for date, grid, prev_c in days:
                    if strat_name == "orb":
                        all_pnls.extend(strat_orb(grid, p[0], fee, long_only=p[1]))
                    elif strat_name == "orb_long_only":
                        all_pnls.extend(strat_orb(grid, p[0], fee, long_only=True))
                    elif strat_name == "orb_vwap_long":
                        all_pnls.extend(strat_orb_vwap(grid, p[0], fee))
                    elif strat_name == "vwap_pullback":
                        all_pnls.extend(strat_vwap_pullback(grid, fee))
                    elif strat_name == "momentum":
                        all_pnls.extend(strat_momentum_break(grid, p[0], fee))
                    elif strat_name == "rsi_revert":
                        all_pnls.extend(strat_rsi_revert(grid, fee, low=p[0], high=p[1]))
                    elif strat_name == "gap_go":
                        all_pnls.extend(strat_gap_go(grid, prev_c, fee, gap_pct=p[0]))

                s = summarize(all_pnls)
                rows.append(
                    {
                        "symbol": sym,
                        "strategy": strat_name,
                        "params": str(p),
                        "round_trip_cost_pct": rt_cost(fee) * 100,
                        **s,
                    }
                )
        print(f"scanned {sym}, {len(sorted_dates)} days")
    return pd.DataFrame(rows)


def main() -> None:
    symbols = sorted(p.name for p in KBAR_DIR.iterdir() if p.is_dir())
    print("symbols:", symbols)
    df = run_scan(symbols)
    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / "daytrade_scan_vip1.csv"
    df.to_csv(out, index=False)

    qualified = df[(df["n"] >= 40) & (df["avg_pnl"] > 0) & (df["win_rate"] >= 0.45)].sort_values(
        ["avg_pnl", "win_rate"], ascending=False
    )
    print("\n=== VIP 1折 | n>=40, win>=45%, avg_pnl>0 ===")
    print(qualified.head(25).to_string(index=False))

    best_per_sym = qualified.sort_values("avg_pnl", ascending=False).groupby("symbol").head(1)
    print("\n=== Best strategy per symbol ===")
    print(best_per_sym.sort_values("avg_pnl", ascending=False).to_string(index=False))

    strat_agg = (
        qualified.groupby(["strategy", "params"])
        .agg(n=("n", "sum"), avg_pnl=("avg_pnl", "mean"), win_rate=("win_rate", "mean"), symbols=("symbol", "count"))
        .reset_index()
        .sort_values("avg_pnl", ascending=False)
    )
    print("\n=== Strategy ranking (avg across symbols) ===")
    print(strat_agg.head(15).to_string(index=False))


if __name__ == "__main__":
    main()
