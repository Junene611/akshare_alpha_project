from __future__ import annotations
import numpy as np
import pandas as pd

from .config import DATE_COL, ID_COL, LABEL_HORIZON, MIN_LISTING_DAYS
from .utils import neutralize_by_date, robust_clip_by_date, zscore_by_date


def apply_basic_filter(df: pd.DataFrame) -> pd.DataFrame:
    # 它先把明显不合格样本去掉：
    # 缺失关键行情字段的去掉
    # close <= 0 的去掉
    # 上市时间太短的去掉
    out = df.copy()
    # 保留有完整价格量的记录
    mask = (
        out["close"].notna() &
        out["open"].notna() &
        out["high"].notna() &
        out["low"].notna() &
        out["volume"].notna() &
        out["amount"].notna()
    )
    out = out.loc[mask].copy()
    out = out[out["close"] > 0].copy()
    out = out.sort_values([ID_COL, DATE_COL]).copy()
    out["listing_age_days"] = out.groupby(ID_COL).cumcount()
    out = out[out["listing_age_days"] >= MIN_LISTING_DAYS].copy()
    return out


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    # 开始造因子，回答“用什么信息来预测未来5日谁更强”
    out = df.copy().sort_values([ID_COL, DATE_COL])
    grp = out.groupby(ID_COL, group_keys=False)

    # 这部分是横截面短周期选股里最基础的价格动量与波动率刻画。
    # 动量因子
    out["ret_1"] = grp["close"].pct_change(1)
    out["ret_5"] = grp["close"].pct_change(5)
    out["ret_10"] = grp["close"].pct_change(10)
    out["ret_20"] = grp["close"].pct_change(20)
    out["ret_60"] = grp["close"].pct_change(60)

    # 波动因子
    out["vol_20"] = grp["ret_1"].rolling(20).std().reset_index(level=0, drop=True)
    out["vol_60"] = grp["ret_1"].rolling(60).std().reset_index(level=0, drop=True)

    # 流动性因子
    out["amount_mean_5"] = grp["amount"].rolling(5).mean().reset_index(level=0, drop=True)
    out["amount_mean_20"] = grp["amount"].rolling(20).mean().reset_index(level=0, drop=True)
    out["volume_mean_20"] = grp["volume"].rolling(20).mean().reset_index(level=0, drop=True)
    out["amount_mean_20_raw"] = out["amount_mean_20"]

    if "turnover" in out.columns:
        out["turnover_mean_20"] = grp["turnover"].rolling(20).mean().reset_index(level=0, drop=True)
        out["turnover_mean_20_raw"] = out["turnover_mean_20"]

    # 结构因子
    out["price_to_ma_5"] = out["close"] / grp["close"].rolling(5).mean().reset_index(level=0, drop=True) - 1
    out["price_to_ma_20"] = out["close"] / grp["close"].rolling(20).mean().reset_index(level=0, drop=True) - 1
    out["reversal_5_20"] = out["ret_5"] - out["ret_20"]

    # 波动因子
    out["hl_range"] = (out["high"] - out["low"]) / out["close"].replace(0, np.nan)

    # Amihud 近似值用来表达“单位成交额对应的价格冲击”，后面既可做特征，也可辅助成本建模。
    out["amihud_20_raw"] = grp["ret_1"].apply(lambda x: x.abs()).reset_index(level=0, drop=True) / out["amount"].replace(0, np.nan)
    out["amihud_20"] = grp["amihud_20_raw"].rolling(20).mean().reset_index(level=0, drop=True)

    # 市场状态因子
    # 市场状态特征直接沿用基准指数时间序列，不做横截面构造。
    out["mkt_ret_1"] = out["index_ret_1"]
    out["mkt_ret_5"] = out["index_ret_5"]
    out["mkt_vol_20"] = out["index_vol_20"]

    return out


def add_label(df: pd.DataFrame) -> pd.DataFrame:
    # 定义标签
    # future_ret_5
    # label_excess_5
    out = df.copy().sort_values([ID_COL, DATE_COL])
    grp = out.groupby(ID_COL, group_keys=False)
    out["future_ret_5"] = grp["close"].shift(-LABEL_HORIZON) / out["close"] - 1
    # 其中真正训练用的是：
    # label_excess_5 = 个股未来5日收益 - 指数未来5日收益
    out["label_excess_5"] = out["future_ret_5"] - out["index_ret_5"]
    return out


def preprocess_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    # 这里做的是特征清洗和标准化：
    # 按日截面裁尾
    # 按日做 z-score
    # 如果有行业字段，就做行业中性化
    # 对市场状态变量做全样本标准化
    out = df.copy()
    cross_sectional_feature_cols = [
        "ret_1", "ret_5", "ret_10", "ret_20", "ret_60",
        "vol_20", "vol_60",
        "amount_mean_5", "amount_mean_20", "volume_mean_20",
        "turnover_mean_20",
        "price_to_ma_5", "price_to_ma_20",
        "reversal_5_20", "hl_range", "amihud_20",
    ]
    market_feature_cols = ["mkt_ret_1", "mkt_ret_5", "mkt_vol_20"]
    cross_sectional_feature_cols = [c for c in cross_sectional_feature_cols if c in out.columns]
    market_feature_cols = [c for c in market_feature_cols if c in out.columns]

    # 先做按日 winsorize，再做按日 z-score，尽量把极端值影响压到截面内部。
    out = robust_clip_by_date(out, cross_sectional_feature_cols, date_col=DATE_COL, lower_q=0.01, upper_q=0.99)
    out = zscore_by_date(out, cross_sectional_feature_cols, date_col=DATE_COL)
    if "industry" in out.columns and cross_sectional_feature_cols:
        out["industry"] = out["industry"].fillna("Unknown").astype(str)
        # 行业内去均值主要是降低模型把行业轮动当成个股 alpha 的风险。
        neutralize_cols = cross_sectional_feature_cols + [c for c in ["label_excess_5"] if c in out.columns]
        out = neutralize_by_date(out, neutralize_cols, date_col=DATE_COL, industry_col="industry")
        out = zscore_by_date(out, cross_sectional_feature_cols, date_col=DATE_COL)
    for c in market_feature_cols:
        # 市场状态列是时间序列变量，这里按全样本裁剪和标准化，而不是按日截面标准化。
        lower = out[c].quantile(0.01)
        upper = out[c].quantile(0.99)
        out[c] = out[c].clip(lower, upper)
        mean = out[c].mean()
        std = out[c].std()
        out[c] = (out[c] - mean) / std if pd.notna(std) and std > 0 else 0.0
    return out, cross_sectional_feature_cols + market_feature_cols


def build_feature_panel(panel: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    # 它只是把上面几步串起来，输出最终训练表。
    x = apply_basic_filter(panel)
    if "industry" in x.columns:
        x["industry"] = x["industry"].fillna("Unknown").astype(str)
    x = add_features(x)
    x = add_label(x)
    x, feature_cols = preprocess_features(x)
    x = x.dropna(subset=["label_excess_5"]).copy()
    return x, feature_cols

# 从行情数据里提炼模型能学的信号，并构造训练目标。