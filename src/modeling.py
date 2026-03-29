from __future__ import annotations
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import (
    DATE_COL, RANDOM_STATE, RIDGE_ALPHA, LGBM_PARAMS,
    TRAIN_START, VALID_START, TEST_START, TEST_END, WALK_FORWARD_FREQ,
    TRAIN_LOOKBACK_YEARS, TRAIN_SAMPLE_EVERY,
)

# 看模型怎么训练，怎么避免未来函数
try:
    from lightgbm import LGBMRegressor
    HAS_LGBM = True
except Exception:
    HAS_LGBM = False
    LGBMRegressor = None


@dataclass
class SplitData:
    train: pd.DataFrame
    valid: pd.DataFrame
    test: pd.DataFrame


def split_by_time(df: pd.DataFrame) -> SplitData:
    # 这里把数据按时间切成：主要是研究期划分，不是随机切分。
    # train
    # valid
    # test
    train = df[(df[DATE_COL] >= pd.Timestamp(TRAIN_START)) & (df[DATE_COL] < pd.Timestamp(VALID_START))].copy()
    valid = df[(df[DATE_COL] >= pd.Timestamp(VALID_START)) & (df[DATE_COL] < pd.Timestamp(TEST_START))].copy()
    test = df[(df[DATE_COL] >= pd.Timestamp(TEST_START)) & (df[DATE_COL] <= pd.Timestamp(TEST_END))].copy()
    return SplitData(train=train, valid=valid, test=test)


def make_ridge_model() -> Pipeline:
    # 定义模型本身
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("model", Ridge(alpha=RIDGE_ALPHA, random_state=RANDOM_STATE)),
    ])


def make_lgbm_model():
    # 定义模型本身
    if not HAS_LGBM:
        raise ImportError("LightGBM is not installed. Please install it first.")
    return LGBMRegressor(**LGBM_PARAMS)


def _select_rebalance_dates(dates: pd.Series | pd.Index, sample_every: int) -> pd.Index:
    unique_dates = pd.Index(sorted(pd.Series(dates).dropna().unique()))
    if sample_every <= 1:
        return unique_dates
    return unique_dates[::sample_every]


def _build_train_mask(date_series: pd.Series, month_start: pd.Timestamp) -> pd.Series:
    # 它的作用是：
    # 对某个月的测试样本，只允许用这个月之前的数据训练
    # 只取最近若干年的历史
    # 再按调仓频率抽取训练截面
    # 这样做有两个好处：
    # 避免未来信息泄露
    # 减少训练样本冗余
    train_mask = date_series < month_start
    if TRAIN_LOOKBACK_YEARS and TRAIN_LOOKBACK_YEARS > 0:
        lookback_start = month_start - pd.DateOffset(years=TRAIN_LOOKBACK_YEARS)
        train_mask = train_mask & (date_series >= lookback_start)
    if not train_mask.any():
        return train_mask

    # 训练阶段只抽取调仓截面，目的是减少冗余样本并让训练频率更贴近组合更新频率。
    rebalance_dates = _select_rebalance_dates(date_series[train_mask], sample_every=TRAIN_SAMPLE_EVERY)
    return train_mask & date_series.isin(rebalance_dates)


def fit_predict(
    # 训练主函数
    # 对测试期按月滚动 walk-forward
    # 每个月都重新训练一次
    # 只预测这个月的样本
    # 最后把所有月份拼起来
    df: pd.DataFrame,
    feature_cols: list[str],
    label_col: str = "label_excess_5",
    model_name: Literal["ridge", "lgbm"] = "ridge",
) -> pd.DataFrame:
    split = split_by_time(df)
    # 这里保留显式切分，主要是为了研究期诊断；真正 walk-forward 时仍然按月滚动扩展训练窗。
    train_valid = pd.concat([split.train, split.valid], axis=0).sort_values(DATE_COL).copy()
    test = split.test.sort_values(DATE_COL).copy()
    if test.empty:
        pred = test.copy()
        pred[f"pred_{model_name}"] = np.nan
        pred.attrs["feature_importance"] = pd.DataFrame({"feature": feature_cols, "importance": np.nan})
        return pred

    predictions = []
    month_starts = pd.date_range(
        start=test[DATE_COL].min().replace(day=1),
        end=test[DATE_COL].max().replace(day=1),
        freq=WALK_FORWARD_FREQ,
    )
    last_model = None
    date_series = df[DATE_COL]
    for month_start in month_starts:
        month_end = month_start + pd.offsets.MonthBegin(1)
        train_mask = _build_train_mask(date_series, month_start)
        test_window = test[(test[DATE_COL] >= month_start) & (test[DATE_COL] < month_end)].copy()
        if not train_mask.any() or test_window.empty:
            continue

        # 这里严格保证每个月只用当月之前的样本训练，避免未来信息泄露。
        X_train = df.loc[train_mask, feature_cols]
        y_train = df.loc[train_mask, label_col]
        X_test = test_window[feature_cols]

        if model_name == "ridge":
            model = make_ridge_model()
        elif model_name == "lgbm":
            model = make_lgbm_model()
        else:
            raise ValueError(model_name)

        model.fit(X_train, y_train)
        test_window[f"pred_{model_name}"] = model.predict(X_test)
        predictions.append(test_window)
        last_model = model

    pred = pd.concat(predictions, axis=0, ignore_index=True).sort_values([DATE_COL, "instrument"]).reset_index(drop=True)

    # 仅输出最后一次重训模型的重要性，目的是做研究说明，不作为严格统计结论。
    if model_name == "lgbm" and last_model is not None and hasattr(last_model, "feature_importances_"):
        fi = pd.DataFrame({"feature": feature_cols, "importance": last_model.feature_importances_}).sort_values(
            "importance", ascending=False
        )
    elif model_name == "ridge" and last_model is not None:
        coef = last_model.named_steps["model"].coef_
        fi = pd.DataFrame({"feature": feature_cols, "importance": coef}).assign(abs_importance=lambda x: x["importance"].abs()).sort_values(
            "abs_importance", ascending=False
        )
    else:
        fi = pd.DataFrame({"feature": feature_cols, "importance": np.nan})

    pred.attrs["feature_importance"] = fi
    return pred
