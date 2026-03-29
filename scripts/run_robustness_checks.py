from __future__ import annotations
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.backtest import build_topk_portfolio, summarize_portfolio
from src.config import (
    COST_BPS_ONE_SIDE,
    DEFAULT_NEUTRALIZE_INDUSTRY,
    FEATURE_FILE,
    HOLD_BUFFER,
    ILLIQUID_COST_BPS,
    MAX_NEW_NAMES,
    MIN_AMOUNT_MEAN_20,
    MIN_TURNOVER_MEAN_20,
    PRED_FILE,
    REPORT_ROOT,
)
from src.utils import performance_summary, read_table, write_table


def load_prediction_context():
    pred = read_table(PRED_FILE, parse_dates=["date"])
    feature = read_table(FEATURE_FILE, parse_dates=["date"])
    keep_cols = [
        "date", "instrument", "industry", "index_ret_1", "index_ret_5", "index_vol_20",
        "close", "amount_mean_20_raw", "turnover_mean_20_raw", "future_ret_5",
    ]
    keep_cols = [c for c in keep_cols if c in feature.columns]
    feature = feature[keep_cols].drop_duplicates(subset=["date", "instrument"])
    return pred.merge(feature, on=["date", "instrument"], how="left", suffixes=("", "_feat"))


def summarize_by_year(port: pd.DataFrame) -> pd.DataFrame:
    if port.empty:
        return pd.DataFrame()
    periods_per_year = 252 / 5
    rows = []
    x = port.copy()
    x["year"] = pd.to_datetime(x["date"]).dt.year
    for year, g in x.groupby("year"):
        net_summary = performance_summary(g["net_ret"], periods_per_year=periods_per_year)
        gross_summary = performance_summary(g["gross_ret"], periods_per_year=periods_per_year)
        rows.append(
            {
                "year": int(year),
                "periods": len(g),
                "gross_annual_return": gross_summary["annual_return"],
                "net_annual_return": net_summary["annual_return"],
                "net_sharpe": net_summary["sharpe"],
                "net_max_drawdown": net_summary["max_drawdown"],
                "avg_turnover": g["turnover"].mean(),
                "avg_n_new": g["n_new"].mean() if "n_new" in g.columns else None,
                "avg_trading_cost": g["trading_cost"].mean(),
            }
        )
    return pd.DataFrame(rows).sort_values("year")


def cost_sensitivity(pred: pd.DataFrame, pred_col: str) -> pd.DataFrame:
    cost_grid = [
        (10.0, 10.0),
        (15.0, 15.0),
        (20.0, 20.0),
        (25.0, 25.0),
        (30.0, 30.0),
    ]
    rows = []
    for cost_bps_one_side, illiquid_cost_bps in cost_grid:
        port = build_topk_portfolio(
            pred,
            pred_col=pred_col,
            realized_col="future_ret_5",
            hold_buffer=HOLD_BUFFER,
            max_new_names=MAX_NEW_NAMES,
            min_amount_mean_20=MIN_AMOUNT_MEAN_20,
            min_turnover_mean_20=MIN_TURNOVER_MEAN_20,
            neutralize_industry=DEFAULT_NEUTRALIZE_INDUSTRY,
            cost_bps_one_side=cost_bps_one_side,
            illiquid_cost_bps=illiquid_cost_bps,
        )
        summary = summarize_portfolio(port)
        rows.append(
            {
                "cost_bps_one_side": cost_bps_one_side,
                "illiquid_cost_bps": illiquid_cost_bps,
                "net_annual_return": summary.get("net", {}).get("annual_return"),
                "net_sharpe": summary.get("net", {}).get("sharpe"),
                "net_max_drawdown": summary.get("net", {}).get("max_drawdown"),
                "avg_turnover": summary.get("avg_turnover"),
                "avg_n_new": summary.get("avg_n_new"),
                "avg_trading_cost": summary.get("avg_trading_cost"),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    pred = load_prediction_context()
    if "pred_lgbm" not in pred.columns:
        raise ValueError("pred_lgbm not found in predictions file")

    REPORT_ROOT.mkdir(parents=True, exist_ok=True)

    port = build_topk_portfolio(
        pred,
        pred_col="pred_lgbm",
        realized_col="future_ret_5",
        hold_buffer=HOLD_BUFFER,
        max_new_names=MAX_NEW_NAMES,
        min_amount_mean_20=MIN_AMOUNT_MEAN_20,
        min_turnover_mean_20=MIN_TURNOVER_MEAN_20,
        neutralize_industry=DEFAULT_NEUTRALIZE_INDUSTRY,
        cost_bps_one_side=COST_BPS_ONE_SIDE,
        illiquid_cost_bps=ILLIQUID_COST_BPS,
    )
    summary = summarize_portfolio(port)
    yearly = summarize_by_year(port)
    cost_df = cost_sensitivity(pred, pred_col="pred_lgbm")

    write_table(yearly, REPORT_ROOT / "robustness_yearly_lgbm.csv", index=False)
    write_table(cost_df, REPORT_ROOT / "robustness_cost_sensitivity_lgbm.csv", index=False)
    (REPORT_ROOT / "robustness_summary_lgbm.json").write_text(
        json.dumps(
            {
                "default_config": {
                    "hold_buffer": HOLD_BUFFER,
                    "max_new_names": MAX_NEW_NAMES,
                    "min_amount_mean_20": MIN_AMOUNT_MEAN_20,
                    "min_turnover_mean_20": MIN_TURNOVER_MEAN_20,
                    "neutralize_industry": DEFAULT_NEUTRALIZE_INDUSTRY,
                    "cost_bps_one_side": COST_BPS_ONE_SIDE,
                    "illiquid_cost_bps": ILLIQUID_COST_BPS,
                },
                "summary": summary,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"summary": summary, "years": len(yearly), "cost_scenarios": len(cost_df)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
# 看分年份表现和成本敏感性
