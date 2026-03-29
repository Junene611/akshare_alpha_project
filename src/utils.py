from __future__ import annotations
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def read_table(path: Path, parse_dates: list[str] | None = None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".csv":
        return pd.read_csv(path, parse_dates=parse_dates)
    raise ValueError(f"Unsupported suffix: {path.suffix}")


def write_table(df: pd.DataFrame, path: Path, index: bool = False) -> None:
    ensure_parent(path)
    if path.suffix == ".parquet":
        df.to_parquet(path, index=index)
        return
    if path.suffix == ".csv":
        df.to_csv(path, index=index)
        return
    raise ValueError(f"Unsupported suffix: {path.suffix}")


def robust_clip_by_date(
    df: pd.DataFrame,
    cols: list[str],
    date_col: str,
    lower_q: float = 0.01,
    upper_q: float = 0.99,
) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        # 对每个交易日单独裁尾，避免极端个股把当天截面分布拉坏。
        out[c] = out.groupby(date_col)[c].transform(
            lambda x: x.clip(x.quantile(lower_q), x.quantile(upper_q))
        )
    return out


def zscore_by_date(df: pd.DataFrame, cols: list[str], date_col: str) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        grp = out.groupby(date_col)[c]
        mean = grp.transform("mean")
        std = grp.transform("std").replace(0, np.nan)
        out[c] = (out[c] - mean) / std
    return out


def neutralize_by_date(
    df: pd.DataFrame,
    y_cols: list[str],
    date_col: str,
    industry_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    纯 AKShare 价格量能版通常没有稳定行业字段。
    若无行业列，则本函数直接返回原表。
    """
    if not industry_col or industry_col not in df.columns:
        return df.copy()

    out = df.copy()
    out[industry_col] = out[industry_col].fillna("Unknown").astype(str)
    group_cols = [date_col, industry_col]
    for col in y_cols:
        if col not in out.columns:
            continue
        industry_mean = out.groupby(group_cols)[col].transform("mean")
        out[col] = out[col] - industry_mean
    return out


def rank_ic_by_date(
    df: pd.DataFrame,
    pred_col: str,
    label_col: str,
    date_col: str,
) -> pd.DataFrame:
    rows = []
    for dt, g in df.groupby(date_col):
        x = g[pred_col]
        y = g[label_col]
        mask = x.notna() & y.notna()
        if mask.sum() < 5:
            continue
        # 用秩相关而不是线性相关，主要是评价排序能力而不是点预测精度。
        ic = x[mask].rank().corr(y[mask].rank())
        rows.append({date_col: dt, "rank_ic": ic})
    return pd.DataFrame(rows).sort_values(date_col)


def annualize_return(daily_ret: pd.Series, periods_per_year: int = 252) -> float:
    daily_ret = daily_ret.dropna()
    if daily_ret.empty:
        return np.nan
    cum = (1 + daily_ret).prod()
    n = len(daily_ret)
    return cum ** (periods_per_year / n) - 1


def annualize_vol(daily_ret: pd.Series, periods_per_year: int = 252) -> float:
    daily_ret = daily_ret.dropna()
    if daily_ret.empty:
        return np.nan
    return daily_ret.std() * np.sqrt(periods_per_year)


def max_drawdown(nav: pd.Series) -> float:
    nav = nav.dropna()
    if nav.empty:
        return np.nan
    running_max = nav.cummax()
    dd = nav / running_max - 1
    return dd.min()


def performance_summary(period_ret: pd.Series, periods_per_year: float = 252) -> dict:
    ann_ret = annualize_return(period_ret, periods_per_year=periods_per_year)
    ann_vol = annualize_vol(period_ret, periods_per_year=periods_per_year)
    sharpe = ann_ret / ann_vol if ann_vol and np.isfinite(ann_vol) and ann_vol > 0 else np.nan
    nav = (1 + period_ret.fillna(0)).cumprod()
    mdd = max_drawdown(nav)
    return {
        "annual_return": ann_ret,
        "annual_vol": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": mdd,
    }
