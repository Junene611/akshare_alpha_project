from __future__ import annotations
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.backtest import build_portfolio_holdings, build_topk_portfolio
from src.config import FEATURE_FILE, PRED_FILE, REPORT_ROOT
from src.utils import read_table, write_table


def load_analysis_frame() -> pd.DataFrame:
    pred = read_table(PRED_FILE, parse_dates=["date"])
    feature = read_table(FEATURE_FILE, parse_dates=["date"])
    keep_cols = [
        "date",
        "instrument",
        "industry",
        "index_ret_1",
        "index_ret_5",
        "index_vol_20",
        "amount_mean_20_raw",
        "turnover_mean_20_raw",
        "close",
        "future_ret_5",
    ]
    keep_cols = [c for c in keep_cols if c in feature.columns]
    feature = feature[keep_cols].drop_duplicates(subset=["date", "instrument"])
    merged = pred.merge(feature, on=["date", "instrument"], how="left", suffixes=("", "_feat"))
    for col in ["industry", "amount_mean_20_raw", "turnover_mean_20_raw", "close", "future_ret_5"]:
        feat_col = f"{col}_feat"
        if feat_col in merged.columns:
            merged[col] = merged[col].fillna(merged[feat_col])
            merged = merged.drop(columns=[feat_col])
    return merged


def build_date_state_map(df: pd.DataFrame) -> pd.DataFrame:
    date_state = (
        df[["date", "index_ret_5", "index_vol_20"]]
        .drop_duplicates(subset=["date"])
        .dropna(subset=["index_ret_5", "index_vol_20"])
        .copy()
    )
    if date_state.empty:
        return pd.DataFrame(columns=["date", "market_state"])

    vol_median = date_state["index_vol_20"].median()
    date_state["trend_state"] = date_state["index_ret_5"].apply(lambda x: "up" if x >= 0 else "down")
    date_state["vol_state"] = date_state["index_vol_20"].apply(lambda x: "highvol" if x >= vol_median else "lowvol")
    date_state["market_state"] = date_state["trend_state"] + "_" + date_state["vol_state"]
    return date_state[["date", "market_state"]]


def assign_market_state(df: pd.DataFrame, date_state: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "market_state" in out.columns:
        out = out.drop(columns=["market_state"])
    out = out.merge(date_state, on="date", how="left")
    out["market_state"] = out["market_state"].fillna("unknown")
    return out


def summarize_regime(port: pd.DataFrame) -> pd.DataFrame:
    if port.empty:
        return pd.DataFrame()
    rows = []
    tmp = port.copy()
    tmp["year"] = pd.to_datetime(tmp["date"]).dt.year
    for (year, market_state), g in tmp.groupby(["year", "market_state"]):
        rows.append(
            {
                "year": int(year),
                "market_state": market_state,
                "periods": len(g),
                "gross_ret_mean": g["gross_ret"].mean(),
                "net_ret_mean": g["net_ret"].mean(),
                "avg_turnover": g["turnover"].mean(),
                "avg_n_new": g["n_new"].mean() if "n_new" in g.columns else None,
                "avg_trading_cost": g["trading_cost"].mean(),
            }
        )
    return pd.DataFrame(rows).sort_values(["year", "market_state"]).reset_index(drop=True)


def summarize_holdings_dimension(holdings: pd.DataFrame, group_col: str, year: int) -> pd.DataFrame:
    x = holdings[pd.to_datetime(holdings["date"]).dt.year == year].copy()
    if x.empty or group_col not in x.columns:
        return pd.DataFrame()

    rows = []
    total_count = len(x)
    for key, g in x.groupby(group_col):
        rows.append(
            {
                group_col: str(key),
                "count": len(g),
                "weight_share": len(g) / total_count if total_count else None,
                "future_ret_5_mean": g["future_ret_5"].mean(),
                "future_ret_5_median": g["future_ret_5"].median(),
                "pred_lgbm_mean": g["pred_lgbm"].mean() if "pred_lgbm" in g.columns else None,
                "amount_mean_20_raw_mean": g["amount_mean_20_raw"].mean() if "amount_mean_20_raw" in g.columns else None,
                "turnover_mean_20_raw_mean": g["turnover_mean_20_raw"].mean() if "turnover_mean_20_raw" in g.columns else None,
            }
        )
    return pd.DataFrame(rows).sort_values("count", ascending=False).reset_index(drop=True)


def summarize_liquidity_style(holdings: pd.DataFrame, universe: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for year in sorted(pd.to_datetime(holdings["date"]).dt.year.unique()):
        h = holdings[pd.to_datetime(holdings["date"]).dt.year == year].copy()
        u = universe[pd.to_datetime(universe["date"]).dt.year == year].copy()
        if h.empty or u.empty:
            continue
        rows.append(
            {
                "year": int(year),
                "holdings_amount_median": h["amount_mean_20_raw"].median(),
                "universe_amount_median": u["amount_mean_20_raw"].median(),
                "holdings_turnover_median": h["turnover_mean_20_raw"].median(),
                "universe_turnover_median": u["turnover_mean_20_raw"].median(),
                "holdings_price_median": h["close"].median(),
                "universe_price_median": u["close"].median(),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    analysis = load_analysis_frame()
    date_state = build_date_state_map(analysis)
    analysis = assign_market_state(analysis, date_state)
    holdings = build_portfolio_holdings(analysis, pred_col="pred_lgbm", realized_col="future_ret_5")
    port = build_topk_portfolio(analysis, pred_col="pred_lgbm", realized_col="future_ret_5")
    if holdings.empty or port.empty:
        raise ValueError("No holdings or portfolio built for attribution")

    holdings = assign_market_state(holdings, date_state)
    port = assign_market_state(port, date_state)

    regime_summary = summarize_regime(port)
    industry_2024 = summarize_holdings_dimension(holdings, "industry", year=2024)
    state_2024 = summarize_holdings_dimension(holdings, "market_state", year=2024)
    liquidity_style = summarize_liquidity_style(holdings, analysis[analysis["pred_lgbm"].notna()].copy())

    write_table(regime_summary, REPORT_ROOT / "regime_summary_lgbm.csv", index=False)
    write_table(industry_2024, REPORT_ROOT / "industry_exposure_2024_lgbm.csv", index=False)
    write_table(state_2024, REPORT_ROOT / "market_state_holdings_2024_lgbm.csv", index=False)
    write_table(liquidity_style, REPORT_ROOT / "style_exposure_yearly_lgbm.csv", index=False)

    summary = {
        "worst_year": int(regime_summary.groupby("year")["net_ret_mean"].mean().idxmin()) if not regime_summary.empty else None,
        "top_industries_2024": industry_2024.head(10).to_dict(orient="records"),
        "market_states_2024": state_2024.to_dict(orient="records"),
    }
    (REPORT_ROOT / "regime_attribution_summary_lgbm.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps({"regime_rows": len(regime_summary), "industry_rows": len(industry_2024), "style_rows": len(liquidity_style)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
# 看2024年为什么失效，按市场状态、行业、风格归因