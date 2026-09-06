#!/usr/bin/env python3
"""Scan Taiwan broker-branch silent accumulation patterns (分點吃貨)."""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests

TOKEN = os.environ.get("FINMIND_TOKEN", "")
BASE = "https://api.finmindtrade.com/api/v4"
OUT_DIR = Path("/workspace/backtest/data/broker_flow")

# Typical electronic / foreign desks — still reported, but tagged separately.
MEGA_KEYWORDS = (
    "台灣匯立",
    "美林",
    "摩根",
    "高盛",
    "大和國泰",
    "港商",
    "瑞銀",
    "花旗",
    "法銀",
    "匯豐",
    "巴克萊",
    "新加坡商",
    "香港商",
    "瑞士商",
    "美商",
    "日商",
    "德商",
    "法商",
    "英商",
    "麥格理",
    "元大證券股份有限公司",  # rare full legal name if appears
)


def api_get(path: str, params: dict, retries: int = 5) -> dict:
    params = {**params, "token": TOKEN}
    last = None
    for i in range(retries):
        try:
            r = requests.get(f"{BASE}{path}", params=params, timeout=90)
            if r.status_code == 429:
                time.sleep(2 ** i)
                continue
            if r.status_code >= 500:
                time.sleep(1.5 * (i + 1))
                continue
            j = r.json()
            if r.status_code != 200:
                raise RuntimeError(f"{path} {params} -> {r.status_code} {j}")
            return j
        except Exception as e:
            last = e
            time.sleep(1.2 * (i + 1))
    raise RuntimeError(last)


def trading_dates(stock_id: str, start: str, end: str) -> list[str]:
    j = api_get(
        "/data",
        {
            "dataset": "TaiwanStockPrice",
            "data_id": stock_id,
            "start_date": start,
            "end_date": end,
        },
    )
    return [row["date"] for row in j.get("data", [])]


def stock_info_map() -> dict[str, str]:
    j = api_get("/data", {"dataset": "TaiwanStockInfo"})
    out = {}
    for row in j.get("data", []):
        sid = str(row.get("stock_id", ""))
        name = row.get("stock_name") or row.get("name") or ""
        if sid:
            out[sid] = name
    return out


def fetch_day_agg(stock_id: str, date: str) -> pd.DataFrame:
    j = api_get(
        "/taiwan_stock_trading_daily_report",
        {"data_id": stock_id, "date": date},
    )
    data = j.get("data") or []
    if not data:
        return pd.DataFrame(
            columns=[
                "date",
                "stock_id",
                "securities_trader_id",
                "securities_trader",
                "buy",
                "sell",
                "buy_amt",
                "sell_amt",
            ]
        )
    df = pd.DataFrame(data)
    df["buy_amt"] = df["buy"] * df["price"]
    df["sell_amt"] = df["sell"] * df["price"]
    g = (
        df.groupby(["date", "stock_id", "securities_trader_id", "securities_trader"], as_index=False)
        .agg(buy=("buy", "sum"), sell=("sell", "sum"), buy_amt=("buy_amt", "sum"), sell_amt=("sell_amt", "sum"))
    )
    return g


def load_or_build_stock_daily(
    stock_id: str, dates: list[str], cache_dir: Path, max_workers: int = 6
) -> pd.DataFrame:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{stock_id}_daily_branch.parquet"
    have = set()
    frames: list[pd.DataFrame] = []
    if cache_path.exists():
        old = pd.read_parquet(cache_path)
        frames.append(old)
        have = set(old["date"].astype(str).unique())
    missing = [d for d in dates if d not in have]
    if missing:
        print(f"  {stock_id}: fetch {len(missing)}/{len(dates)} days", flush=True)

        def one(d: str) -> pd.DataFrame:
            return fetch_day_agg(stock_id, d)

        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = {ex.submit(one, d): d for d in missing}
            for fut in as_completed(futs):
                d = futs[fut]
                try:
                    frames.append(fut.result())
                except Exception as e:
                    print(f"    fail {stock_id} {d}: {e}", flush=True)
        out = pd.concat([f for f in frames if f is not None and len(f)], ignore_index=True)
        out = out.drop_duplicates(
            subset=["date", "stock_id", "securities_trader_id"], keep="last"
        )
        out.to_parquet(cache_path, index=False)
        return out
    return frames[0]


def is_mega(name: str) -> bool:
    return any(k in name for k in MEGA_KEYWORDS)


def score_pairs(df: pd.DataFrame, min_net_shares: int = 80_000) -> pd.DataFrame:
    """min_net_shares in 股; UI 張 = /1000."""
    if df.empty:
        return pd.DataFrame()

    rows = []
    for (sid, bid, bname), g in df.groupby(
        ["stock_id", "securities_trader_id", "securities_trader"]
    ):
        g = g.sort_values("date")
        buy = float(g["buy"].sum())
        sell = float(g["sell"].sum())
        net = buy - sell
        if net < min_net_shares:
            continue
        total = buy + sell
        if total <= 0:
            continue
        buy_ratio = buy / total
        active = g[(g["buy"] > 0) | (g["sell"] > 0)]
        n_days = len(active)
        if n_days < 15:
            continue
        buy_days = int((active["buy"] > 0).sum())
        sell_days = int((active["sell"] > 0).sum())
        net_series = (active["buy"] - active["sell"]).astype(float)
        cum = net_series.cumsum()
        cum_end = float(cum.iloc[-1])
        cum_max = float(cum.max())
        if cum_max <= 0:
            continue
        hold_ratio = cum_end / cum_max  # 1 = never gave back peak inventory
        peak_day_share = float(net_series.max() / net) if net > 0 else 1.0
        avg_buy = float(g["buy_amt"].sum() / buy) if buy else 0.0
        avg_sell = float(g["sell_amt"].sum() / sell) if sell else 0.0
        spread = (avg_sell / avg_buy - 1.0) if avg_buy > 0 and sell > 0 else 0.0

        # Silent accumulation: persistent net buy, little dumping, hold through period.
        score = 0.0
        score += min(net / 1_000_000.0, 5.0) * 20  # size
        score += buy_ratio * 35
        score += min(n_days / 80.0, 1.0) * 20
        score += hold_ratio * 25
        score -= min(max(peak_day_share - 0.25, 0.0), 0.75) * 30  # penalize one-day spike
        if sell > 0 and spread > 0:
            score += min(spread, 0.15) * 40
        mega = is_mega(str(bname))
        if mega:
            score *= 0.55  # still keep but demote vs local branches

        rows.append(
            {
                "stock_id": sid,
                "securities_trader_id": bid,
                "securities_trader": bname,
                "mega_desk": mega,
                "buy_shares": buy,
                "sell_shares": sell,
                "net_shares": net,
                "buy_lots": buy / 1000.0,
                "sell_lots": sell / 1000.0,
                "net_lots": net / 1000.0,
                "buy_ratio": buy_ratio,
                "active_days": n_days,
                "buy_days": buy_days,
                "sell_days": sell_days,
                "hold_ratio": hold_ratio,
                "peak_day_share": peak_day_share,
                "avg_buy": avg_buy,
                "avg_sell": avg_sell,
                "spread": spread,
                "score": score,
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("score", ascending=False)


def default_universe() -> list[str]:
    path = Path("/workspace/backtest/data/yearly_movers/staircase_candidates_2026_v2.csv")
    ids = []
    if path.exists():
        ids = pd.read_csv(path)["stock_id"].astype(str).tolist()[:40]
    extras = ["2330", "2454", "2317", "2382", "2308", "3034", "3711", "6669", "3443", "3653"]
    # keep order unique
    out = []
    for x in ids + extras + ["3008"]:
        if x not in out:
            out.append(x)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2026-04-10")
    ap.add_argument("--end", default="2026-09-04")
    ap.add_argument("--stocks", default="")
    ap.add_argument("--min-net-lots", type=float, default=80.0, help="min net buy in 張")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--top", type=int, default=40)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stocks = [s.strip() for s in args.stocks.split(",") if s.strip()] or default_universe()
    print(f"universe={len(stocks)} {stocks[:10]}...", flush=True)

    dates = trading_dates("2330", args.start, args.end)
    print(f"dates={len(dates)} {dates[0]}..{dates[-1]}", flush=True)

    names = stock_info_map()
    cache_dir = OUT_DIR / "cache_daily"
    all_daily = []
    for i, sid in enumerate(stocks, 1):
        print(f"[{i}/{len(stocks)}] {sid} {names.get(sid,'')}", flush=True)
        try:
            df = load_or_build_stock_daily(sid, dates, cache_dir, max_workers=args.workers)
            all_daily.append(df)
        except Exception as e:
            print(f"  skip {sid}: {e}", flush=True)

    if not all_daily:
        raise SystemExit("no data")
    daily = pd.concat(all_daily, ignore_index=True)
    daily_path = OUT_DIR / f"branch_daily_{args.start}_{args.end}.parquet"
    daily.to_parquet(daily_path, index=False)
    print(f"daily rows={len(daily)} saved {daily_path}", flush=True)

    scored = score_pairs(daily, min_net_shares=int(args.min_net_lots * 1000))
    if scored.empty:
        raise SystemExit("no pairs passed filters")
    scored["stock_name"] = scored["stock_id"].map(names)
    scored = scored.sort_values("score", ascending=False)

    out_csv = OUT_DIR / f"accumulation_scan_{args.start}_{args.end}.csv"
    scored.to_csv(out_csv, index=False)
    local = scored[~scored["mega_desk"]].head(args.top)
    mega = scored[scored["mega_desk"]].head(15)
    local_csv = OUT_DIR / f"accumulation_local_top_{args.start}_{args.end}.csv"
    local.to_csv(local_csv, index=False)

    summary = {
        "period": [args.start, args.end],
        "stocks_scanned": stocks,
        "pairs_total": int(len(scored)),
        "local_top_path": str(local_csv),
        "all_path": str(out_csv),
        "local_top": local.head(25).to_dict(orient="records"),
        "mega_top": mega.head(10).to_dict(orient="records"),
        "kgi_sanduo_3008": scored[
            (scored["securities_trader_id"] == "9275") & (scored["stock_id"] == "3008")
        ].to_dict(orient="records"),
    }
    summary_path = OUT_DIR / f"accumulation_summary_{args.start}_{args.end}.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved {out_csv}", flush=True)
    print(f"saved {local_csv}", flush=True)
    print(f"saved {summary_path}", flush=True)
    print("\n=== LOCAL TOP 20 ===", flush=True)
    cols = [
        "score",
        "stock_id",
        "stock_name",
        "securities_trader",
        "securities_trader_id",
        "net_lots",
        "buy_ratio",
        "active_days",
        "hold_ratio",
        "avg_buy",
        "avg_sell",
    ]
    print(local[cols].head(20).to_string(index=False), flush=True)
    if summary["kgi_sanduo_3008"]:
        print("\n=== 凱基三多/大立光 benchmark ===", flush=True)
        print(pd.DataFrame(summary["kgi_sanduo_3008"])[cols].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
