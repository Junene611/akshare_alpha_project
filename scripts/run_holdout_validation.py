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
from src.config import FEATURE_FILE, PRED_FILE, REPORT_ROOT
from src.utils import read_table, write_table


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


GRID = {
    "hold_buffer": [80, 100, 120],
    "max_new_names": [4, 6, 8],
    "min_amount_mean_20": [60_000_000.0, 80_000_000.0, 100_000_000.0],
    "min_turnover_mean_20": [1.0, 1.5, 2.0],
    "neutralize_industry": [True, False],
}


def evaluate_window(df: pd.DataFrame, **kwargs) -> dict | None:
    port = build_topk_portfolio(df, pred_col="pred_lgbm", realized_col="future_ret_5", **kwargs)
    if port.empty:
        return None
    summary = summarize_portfolio(port)
    return {
        "net_annual_return": summary.get("net", {}).get("annual_return"),
        "net_sharpe": summary.get("net", {}).get("sharpe"),
        "net_max_drawdown": summary.get("net", {}).get("max_drawdown"),
        "gross_annual_return": summary.get("gross", {}).get("annual_return"),
        "avg_turnover": summary.get("avg_turnover"),
        "avg_n_new": summary.get("avg_n_new"),
        "avg_trading_cost": summary.get("avg_trading_cost"),
    }


def grid_search(train_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for values in itertools.product(*GRID.values()):
        params = dict(zip(GRID.keys(), values))
        metrics = evaluate_window(train_df, **params)
        if metrics is None:
            continue
        rows.append({**params, **metrics})
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    return result.sort_values(
        ["net_annual_return", "net_sharpe", "avg_trading_cost"],
        ascending=[False, False, True],
    ).reset_index(drop=True)


def validate_split(df: pd.DataFrame, search_end: str, holdout_start: str, holdout_end: str) -> dict:
    search_df = df[(df["date"] <= pd.Timestamp(search_end))].copy()
    holdout_df = df[(df["date"] >= pd.Timestamp(holdout_start)) & (df["date"] <= pd.Timestamp(holdout_end))].copy()

    ranked = grid_search(search_df)
    if ranked.empty:
        return {"search_end": search_end, "holdout_start": holdout_start, "holdout_end": holdout_end, "best_params": {}, "holdout_metrics": {}}

    best_params = ranked.iloc[0][list(GRID.keys())].to_dict()
    holdout_metrics = evaluate_window(holdout_df, **best_params) or {}
    top5_search = ranked.head(5)
    return {
        "search_end": search_end,
        "holdout_start": holdout_start,
        "holdout_end": holdout_end,
        "best_params": best_params,
        "search_top5": top5_search.to_dict(orient="records"),
        "holdout_metrics": holdout_metrics,
    }


def main() -> None:
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    pred = load_prediction_context()
    pred = pred[pred["pred_lgbm"].notna()].copy()

    splits = [
        ("2023-12-31", "2024-01-01", "2024-12-31"),
        ("2024-12-31", "2025-01-01", "2025-12-31"),
    ]

    results = []
    detail: dict[str, dict] = {}
    for search_end, holdout_start, holdout_end in splits:
        out = validate_split(pred, search_end=search_end, holdout_start=holdout_start, holdout_end=holdout_end)
        key = f"{holdout_start[:4]}_holdout"
        detail[key] = out
        holdout_metrics = out.get("holdout_metrics", {})
        results.append(
            {
                "split": key,
                "search_end": search_end,
                "holdout_start": holdout_start,
                "holdout_end": holdout_end,
                **out.get("best_params", {}),
                **holdout_metrics,
            }
        )

    result_df = pd.DataFrame(results)
    write_table(result_df, REPORT_ROOT / "holdout_validation_lgbm.csv", index=False)
    (REPORT_ROOT / "holdout_validation_lgbm.json").write_text(
        json.dumps(detail, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps({"splits": len(results)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
# 做更严格的样本外留出验证