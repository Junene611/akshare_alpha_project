from __future__ import annotations
import itertools
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.backtest import build_topk_portfolio, summarize_portfolio
from src.config import PRED_FILE, REPORT_ROOT
from src.utils import read_table, write_table


def annualize_period_return(period_ret: pd.Series, periods_per_year: float) -> float:
    x = period_ret.dropna()
    if x.empty:
        return float("nan")
    return float((1 + x).prod() ** (periods_per_year / len(x)) - 1)


def evaluate_grid(pred: pd.DataFrame, pred_col: str) -> pd.DataFrame:
    hold_buffers = [80, 100, 120]
    max_new_names_list = [4, 6, 8]
    min_amount_list = [60_000_000.0, 80_000_000.0, 100_000_000.0]
    min_turnover_list = [1.0, 1.5, 2.0]
    neutralize_options = [True, False]

    rows: list[dict] = []
    for hold_buffer, max_new_names, min_amount, min_turnover, neutralize_industry in itertools.product(
        hold_buffers,
        max_new_names_list,
        min_amount_list,
        min_turnover_list,
        neutralize_options,
    ):
        port = build_topk_portfolio(
            pred,
            pred_col=pred_col,
            realized_col="future_ret_5",
            hold_buffer=hold_buffer,
            max_new_names=max_new_names,
            min_amount_mean_20=min_amount,
            min_turnover_mean_20=min_turnover,
            neutralize_industry=neutralize_industry,
        )
        if port.empty:
            continue

        summary = summarize_portfolio(port)
        periods_per_year = 252 / 5
        rows.append(
            {
                "hold_buffer": hold_buffer,
                "max_new_names": max_new_names,
                "min_amount_mean_20": min_amount,
                "min_turnover_mean_20": min_turnover,
                "neutralize_industry": neutralize_industry,
                "net_annual_return": annualize_period_return(port["net_ret"], periods_per_year=periods_per_year),
                "gross_annual_return": annualize_period_return(port["gross_ret"], periods_per_year=periods_per_year),
                "net_sharpe": summary.get("net", {}).get("sharpe"),
                "net_max_drawdown": summary.get("net", {}).get("max_drawdown"),
                "avg_turnover": summary.get("avg_turnover"),
                "avg_n_new": summary.get("avg_n_new"),
                "avg_illiquid_share": summary.get("avg_illiquid_share"),
                "avg_trading_cost": summary.get("avg_trading_cost"),
            }
        )

    result = pd.DataFrame(rows)
    if result.empty:
        return result

    return result.sort_values(
        ["net_annual_return", "net_sharpe", "avg_trading_cost"],
        ascending=[False, False, True],
    ).reset_index(drop=True)


def main() -> None:
    pred = read_table(PRED_FILE, parse_dates=["date"])
    pred_cols = [c for c in ["pred_lgbm", "pred_ridge"] if c in pred.columns]
    if not pred_cols:
        raise ValueError("No prediction columns found in predictions file")

    REPORT_ROOT.mkdir(parents=True, exist_ok=True)

    best_summary: dict[str, dict] = {}
    for pred_col in pred_cols:
        model_name = pred_col.replace("pred_", "")
        result = evaluate_grid(pred, pred_col=pred_col)
        out_csv = REPORT_ROOT / f"portfolio_grid_search_{model_name}.csv"
        write_table(result, out_csv, index=False)
        if result.empty:
            best_summary[model_name] = {}
            continue
        best_row = result.iloc[0].to_dict()
        best_summary[model_name] = best_row
        print(model_name, json.dumps(best_row, ensure_ascii=False))

    (REPORT_ROOT / "portfolio_grid_search_best.json").write_text(
        json.dumps(best_summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
# 做组合参数搜索