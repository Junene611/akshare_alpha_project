from __future__ import annotations
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.backtest import build_portfolio_holdings, build_topk_portfolio
from src.config import PRED_FILE, REPORT_ROOT
from src.utils import performance_summary, read_table, write_table


def summarize_by_year(port: pd.DataFrame) -> pd.DataFrame:
    if port.empty:
        return pd.DataFrame()
    periods_per_year = 252 / 5
    rows = []
    x = port.copy()
    x["year"] = pd.to_datetime(x["date"]).dt.year
    for year, g in x.groupby("year"):
        summary = performance_summary(g["net_ret"], periods_per_year=periods_per_year)
        rows.append(
            {
                "year": int(year),
                "periods": len(g),
                "annual_return": summary["annual_return"],
                "annual_vol": summary["annual_vol"],
                "sharpe": summary["sharpe"],
                "max_drawdown": summary["max_drawdown"],
                "avg_turnover": g["turnover"].mean(),
                "avg_trading_cost": g["trading_cost"].mean(),
            }
        )
    return pd.DataFrame(rows)


def top_pick_bucket_summary(holdings: pd.DataFrame, bucket_col: str, q: int = 5) -> pd.DataFrame:
    x = holdings.dropna(subset=["future_ret_5", bucket_col]).copy()
    x["bucket"] = x.groupby("date")[bucket_col].transform(
        lambda s: pd.qcut(s.rank(method="first"), q=q, labels=[f"Q{i}" for i in range(1, q + 1)])
    )
    rows = []
    for bucket, g in x.groupby("bucket"):
        rows.append(
            {
                "bucket": str(bucket),
                "count": len(g),
                "future_ret_5_mean": g["future_ret_5"].mean(),
                "future_ret_5_median": g["future_ret_5"].median(),
                "amount_mean_20_raw_mean": g["amount_mean_20_raw"].mean() if "amount_mean_20_raw" in g.columns else None,
                "turnover_mean_20_raw_mean": g["turnover_mean_20_raw"].mean() if "turnover_mean_20_raw" in g.columns else None,
            }
        )
    return pd.DataFrame(rows).sort_values("bucket")


def write_markdown_report(results: dict[str, dict[str, pd.DataFrame]]) -> None:
    lines = ["# Split Diagnostics", ""]
    for model_name, sections in results.items():
        lines.append(f"## {model_name}")
        lines.append("")
        for title, df in sections.items():
            lines.append(f"### {title}")
            lines.append("")
            if df.empty:
                lines.append("No data.")
            else:
                lines.append(df.to_markdown(index=False))
            lines.append("")
    out = REPORT_ROOT / "split_diagnostics.md"
    out.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    pred = read_table(PRED_FILE, parse_dates=["date"])
    results: dict[str, dict[str, pd.DataFrame]] = {}

    for pred_col in [c for c in ["pred_ridge", "pred_lgbm"] if c in pred.columns]:
        model_name = pred_col.replace("pred_", "")
        port = build_topk_portfolio(pred, pred_col=pred_col, realized_col="future_ret_5")
        holdings = build_portfolio_holdings(pred, pred_col=pred_col, realized_col="future_ret_5")
        yearly = summarize_by_year(port)
        liquidity = top_pick_bucket_summary(holdings, "amount_mean_20_raw")
        price = top_pick_bucket_summary(holdings, "close")

        results[model_name] = {
            "Yearly Net Performance": yearly,
            "Top Picks by Liquidity Bucket": liquidity,
            "Top Picks by Price Bucket": price,
        }

        write_table(yearly, REPORT_ROOT / f"yearly_summary_{model_name}.csv", index=False)
        write_table(liquidity, REPORT_ROOT / f"liquidity_bucket_{model_name}.csv", index=False)
        write_table(price, REPORT_ROOT / f"price_bucket_{model_name}.csv", index=False)

    write_markdown_report(results)
    print(json.dumps({k: list(v.keys()) for k, v in results.items()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
