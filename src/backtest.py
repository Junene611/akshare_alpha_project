from __future__ import annotations
import pandas as pd

from .config import (
    DATE_COL, ID_COL, REBALANCE_EVERY, TOP_K, HOLD_BUFFER, PRICE_BUCKETS,
    MAX_NEW_NAMES, DEFAULT_NEUTRALIZE_INDUSTRY, COST_BPS_ONE_SIDE, ILLIQUID_COST_BPS, MIN_CLOSE_PRICE,
    MIN_AMOUNT_MEAN_20, MIN_TURNOVER_MEAN_20, MAX_ABS_PCT_CHG, MAX_NAMES_PER_INDUSTRY,
    PRICE_TILT_STRENGTH, REGIME_AWARE_ENABLED, REGIME_LOOKBACK_VOL_COL, REGIME_TREND_COL,
    ADVERSE_MARKET_STATES, ADVERSE_STATE_MAX_NEW_NAMES, ADVERSE_STATE_HOLD_BUFFER, ADVERSE_STATE_TOP_K,
    ADVERSE_STATE_FORCE_NEUTRALIZE,
)
from .utils import neutralize_by_date, performance_summary


def _filter_tradeable(g: pd.DataFrame) -> pd.DataFrame:
    # 定义哪些股票允许进组合
    # 过滤条件包括：
    # 价格太低不要
    # 成交额不够不要
    # 换手率不够不要
    # 接近涨跌停不要
    return _filter_tradeable_with_thresholds(
        g,
        min_close_price=MIN_CLOSE_PRICE,
        min_amount_mean_20=MIN_AMOUNT_MEAN_20,
        min_turnover_mean_20=MIN_TURNOVER_MEAN_20,
        max_abs_pct_chg=MAX_ABS_PCT_CHG,
    )


def _filter_tradeable_with_thresholds(
    g: pd.DataFrame,
    min_close_price: float,
    min_amount_mean_20: float,
    min_turnover_mean_20: float,
    max_abs_pct_chg: float,
) -> pd.DataFrame:
    if {"close", "amount_mean_20_raw", "turnover_mean_20_raw", "pct_chg"}.issubset(g.columns):
        # 这一步把“极端难交易”样本提前挡掉，避免模型排序信号被不可实现样本污染。
        g = g[
            (g["close"] >= min_close_price) &
            (g["amount_mean_20_raw"] >= min_amount_mean_20) &
            (g["turnover_mean_20_raw"] >= min_turnover_mean_20) &
            (g["pct_chg"].abs() < max_abs_pct_chg * 100)
        ].copy()
    return g


def _neutralize_prediction_by_industry(g: pd.DataFrame, pred_col: str) -> pd.DataFrame:
    # 如果开启行业中性化，就在行业内去均值，降低模型把行业轮动误当 alpha 的风险。
    if "industry" not in g.columns or pred_col not in g.columns:
        return g
    out = g.copy()
    out["industry"] = out["industry"].fillna("Unknown").astype(str)
    out = neutralize_by_date(out, [pred_col], date_col=DATE_COL, industry_col="industry")
    return out


def _assign_price_bucket(g: pd.DataFrame, price_buckets: int) -> pd.DataFrame:
    out = g.copy()
    bucket_count = min(price_buckets, len(out))
    out["price_bucket"] = pd.qcut(
        out["close"].rank(method="first"),
        q=bucket_count,
        labels=[f"Q{i}" for i in range(1, bucket_count + 1)],
    )
    return out


def _build_market_state_by_date(df: pd.DataFrame) -> dict:
    # 把每个日期打上市场状态标签，后面感知换仓的基础
    # up_highvol
    # up_lowvol
    # down_highvol
    # down_lowvol
    if REGIME_TREND_COL not in df.columns or REGIME_LOOKBACK_VOL_COL not in df.columns:
        return {}
    date_state = (
        df[[DATE_COL, REGIME_TREND_COL, REGIME_LOOKBACK_VOL_COL]]
        .drop_duplicates(subset=[DATE_COL])
        .dropna(subset=[REGIME_TREND_COL, REGIME_LOOKBACK_VOL_COL])
        .copy()
    )
    if date_state.empty:
        return {}
    # 用全样本的指数波动中位数把每个调仓日划分成高波/低波状态。
    vol_threshold = float(date_state[REGIME_LOOKBACK_VOL_COL].median())
    date_state["market_state"] = date_state.apply(
        lambda row: (
            ("up" if float(row[REGIME_TREND_COL]) >= 0 else "down") +
            "_" +
            ("highvol" if float(row[REGIME_LOOKBACK_VOL_COL]) >= vol_threshold else "lowvol")
        ),
        axis=1,
    )
    return dict(zip(date_state[DATE_COL], date_state["market_state"]))


def _apply_price_tilt(g: pd.DataFrame, pred_col: str, price_tilt_strength: float) -> pd.DataFrame:
    out = g.copy()
    out[pred_col] = pd.to_numeric(out[pred_col], errors="coerce")
    if "close" not in out.columns or price_tilt_strength == 0:
        out["_selection_score"] = out[pred_col]
        return out
    out["close"] = pd.to_numeric(out["close"], errors="coerce")
    price_rank = out["close"].rank(pct=True, method="average").fillna(0.5) - 0.5
    out["_selection_score"] = out[pred_col] + price_tilt_strength * price_rank
    out["_selection_score"] = pd.to_numeric(out["_selection_score"], errors="coerce")
    return out


def _industry_key(df: pd.DataFrame) -> pd.Series:
    if "industry" not in df.columns:
        return pd.Series("Unknown", index=df.index)
    return df["industry"].fillna("Unknown").astype(str)


def _respect_industry_cap(
    # 这个函数控制单一行业最多持有多少只股票，避免组合被一个热门行业挤满。
    picks: pd.DataFrame,
    candidates: pd.DataFrame,
    industry_cap: int,
    score_col: str,
) -> pd.DataFrame:
    if picks.empty or industry_cap <= 0:
        return picks
    # 先保留分数高的名字，再在候选池里补位，避免组合被少数热门行业挤满。
    out = picks.sort_values(score_col, ascending=False).copy()
    kept_rows = []
    industry_counts: dict[str, int] = {}
    for _, row in out.iterrows():
        industry = str(row.get("industry", "Unknown"))
        if industry_counts.get(industry, 0) >= industry_cap:
            continue
        kept_rows.append(row)
        industry_counts[industry] = industry_counts.get(industry, 0) + 1
    out = pd.DataFrame(kept_rows, columns=out.columns)
    if len(out) >= len(picks):
        return out

    refill = candidates[
        ~candidates[ID_COL].isin(set(out[ID_COL].tolist()))
    ].sort_values(score_col, ascending=False)
    for _, row in refill.iterrows():
        industry = str(row.get("industry", "Unknown"))
        if industry_counts.get(industry, 0) >= industry_cap:
            continue
        out = pd.concat([out, row.to_frame().T], axis=0, ignore_index=True)
        industry_counts[industry] = industry_counts.get(industry, 0) + 1
        if len(out) >= len(picks):
            break
    if score_col in out.columns:
        out[score_col] = pd.to_numeric(out[score_col], errors="coerce")
    return out


def _select_price_neutral_picks(
    # 是组合选股核心。
        # 它不是简单 top20，而是：
        # 先对价格分桶
        # 每个桶分配配额
        # 优先保留旧仓
        # 再从新股票里补足
        # 再做行业集中度约束
    # 这一步是在解决：
        # “怎么从模型分数，变成一个更稳定、可实现、风格不过度偏斜的组合。”
    ranked: pd.DataFrame,
    pred_col: str,
    prev_names: set,
    top_k: int,
    hold_buffer: int,
    price_buckets: int,
    max_new_names: int,
    industry_cap: int,
    price_tilt_strength: float,
) -> pd.DataFrame:
    # 组合构建分三步：旧仓优先续持、分桶补新仓、最后再做行业集中度约束。
    ranked = _apply_price_tilt(ranked, pred_col=pred_col, price_tilt_strength=price_tilt_strength)
    score_col = "_selection_score"
    ranked = ranked.sort_values(score_col, ascending=False).copy()
    ranked = _assign_price_bucket(ranked, price_buckets)
    bucket_labels = list(pd.Series(ranked["price_bucket"]).dropna().astype(str).unique())
    bucket_labels.sort()
    bucket_quota_base = top_k // max(len(bucket_labels), 1)
    bucket_quota_extra = top_k % max(len(bucket_labels), 1)
    buffer_per_bucket = max(2, hold_buffer // max(len(bucket_labels), 1))
    remaining_new_budget = top_k if not prev_names else max_new_names

    picks_by_bucket: list[pd.DataFrame] = []
    picked_names: set[str] = set()
    for idx, bucket in enumerate(bucket_labels):
        quota = bucket_quota_base + (1 if idx < bucket_quota_extra else 0)
        bucket_ranked = ranked[ranked["price_bucket"].astype(str) == bucket].copy().sort_values(score_col, ascending=False)
        bucket_ranked["bucket_rank"] = range(1, len(bucket_ranked) + 1)
        keep = bucket_ranked[
            bucket_ranked[ID_COL].isin(prev_names) &
            (bucket_ranked["bucket_rank"] <= quota + buffer_per_bucket)
        ].copy()
        keep = keep.nlargest(min(quota, len(keep)), score_col)

        need = max(quota - len(keep), 0)
        new_candidates = bucket_ranked[
            ~bucket_ranked[ID_COL].isin(set(keep[ID_COL])) &
            ~bucket_ranked[ID_COL].isin(prev_names)
        ].copy()
        new_picks = new_candidates.head(min(need, remaining_new_budget)).copy()
        remaining_new_budget -= len(new_picks)

        picked = pd.concat([keep, new_picks], axis=0, ignore_index=True)
        picks_by_bucket.append(picked)
        picked_names |= set(picked[ID_COL].tolist())

    picks = pd.concat(picks_by_bucket, axis=0, ignore_index=True) if picks_by_bucket else ranked.head(top_k).copy()
    if len(picks) < top_k:
        keep_fill = ranked[
            ranked[ID_COL].isin(prev_names) &
            ~ranked[ID_COL].isin(picked_names)
        ].head(top_k - len(picks)).copy()
        picks = pd.concat([picks, keep_fill], axis=0, ignore_index=True)
        picked_names |= set(keep_fill[ID_COL].tolist())

    if len(picks) < top_k and remaining_new_budget > 0:
        new_fill = ranked[
            ~ranked[ID_COL].isin(prev_names) &
            ~ranked[ID_COL].isin(picked_names)
        ].head(min(top_k - len(picks), remaining_new_budget)).copy()
        picks = pd.concat([picks, new_fill], axis=0, ignore_index=True)
        picked_names |= set(new_fill[ID_COL].tolist())

    if len(picks) < top_k:
        any_fill = ranked[~ranked[ID_COL].isin(picked_names)].head(top_k - len(picks)).copy()
        picks = pd.concat([picks, any_fill], axis=0, ignore_index=True)
    picks = _respect_industry_cap(
        picks.nlargest(top_k, score_col).copy(),
        candidates=ranked,
        industry_cap=industry_cap,
        score_col=score_col,
    )
    return picks.nlargest(top_k, score_col).copy()


def build_portfolio_holdings(
    # 这是调仓主逻辑。
    # 每个调仓日会做：
        # 过滤可交易股票
        # 必要时行业中性化
        # 如果市场状态不好，就更保守换仓
        # 选出当天持仓列表
    pred_df: pd.DataFrame,
    pred_col: str,
    realized_col: str = "future_ret_5",
    top_k: int = TOP_K,
    hold_buffer: int = HOLD_BUFFER,
    rebalance_every: int = REBALANCE_EVERY,
    price_buckets: int = PRICE_BUCKETS,
    max_new_names: int = MAX_NEW_NAMES,
    neutralize_industry: bool = DEFAULT_NEUTRALIZE_INDUSTRY,
    min_close_price: float = MIN_CLOSE_PRICE,
    min_amount_mean_20: float = MIN_AMOUNT_MEAN_20,
    min_turnover_mean_20: float = MIN_TURNOVER_MEAN_20,
    max_abs_pct_chg: float = MAX_ABS_PCT_CHG,
    industry_cap: int = MAX_NAMES_PER_INDUSTRY,
    price_tilt_strength: float = PRICE_TILT_STRENGTH,
    regime_aware: bool = REGIME_AWARE_ENABLED,
) -> pd.DataFrame:
    df = pred_df.sort_values([DATE_COL, pred_col], ascending=[True, False]).copy()
    all_dates = sorted(df[DATE_COL].dropna().unique())
    rebalance_dates = all_dates[::rebalance_every]
    market_state_by_date = _build_market_state_by_date(df) if regime_aware else {}
    holdings = []
    for dt in rebalance_dates:
        g = df[df[DATE_COL] == dt].dropna(subset=[pred_col, realized_col]).copy()
        effective_top_k = top_k
        effective_hold_buffer = hold_buffer
        effective_max_new_names = max_new_names
        effective_neutralize_industry = neutralize_industry
        if regime_aware:
            market_state = market_state_by_date.get(dt)
            if market_state in ADVERSE_MARKET_STATES:
                # 不利状态下不直接停掉策略，而是通过更保守的换仓和行业约束降风险。
                effective_top_k = min(top_k, ADVERSE_STATE_TOP_K)
                effective_hold_buffer = max(hold_buffer, ADVERSE_STATE_HOLD_BUFFER)
                effective_max_new_names = min(max_new_names, ADVERSE_STATE_MAX_NEW_NAMES)
                effective_neutralize_industry = effective_neutralize_industry or ADVERSE_STATE_FORCE_NEUTRALIZE
        g = _filter_tradeable_with_thresholds(
            g,
            min_close_price=min_close_price,
            min_amount_mean_20=min_amount_mean_20,
            min_turnover_mean_20=min_turnover_mean_20,
            max_abs_pct_chg=max_abs_pct_chg,
        )
        if effective_neutralize_industry:
            g = _neutralize_prediction_by_industry(g, pred_col)
        g = g.dropna(subset=[pred_col]).copy()
        if len(g) < effective_top_k:
            continue
        ranked = g.nlargest(len(g), pred_col).copy()
        picks = _select_price_neutral_picks(
            ranked,
            pred_col,
            prev_names=set() if not holdings else set(holdings[-1][ID_COL].tolist()),
            top_k=effective_top_k,
            hold_buffer=effective_hold_buffer,
            price_buckets=price_buckets,
            max_new_names=effective_max_new_names,
            industry_cap=industry_cap,
            price_tilt_strength=price_tilt_strength,
        )
        picks[DATE_COL] = dt
        holdings.append(picks)
    if not holdings:
        return pd.DataFrame()
    return pd.concat(holdings, axis=0, ignore_index=True)


def build_topk_portfolio(
    # 有了持仓以后，再去算：
        # 毛收益
        # 换手率
        # 低流动性占比
        # 交易成本
        # 净收益
        # 净值曲线
    pred_df: pd.DataFrame,
    pred_col: str,
    realized_col: str = "future_ret_5",
    top_k: int = TOP_K,
    hold_buffer: int = HOLD_BUFFER,
    rebalance_every: int = REBALANCE_EVERY,
    cost_bps_one_side: float = COST_BPS_ONE_SIDE,
    illiquid_cost_bps: float = ILLIQUID_COST_BPS,
    price_buckets: int = PRICE_BUCKETS,
    max_new_names: int = MAX_NEW_NAMES,
    neutralize_industry: bool = DEFAULT_NEUTRALIZE_INDUSTRY,
    min_close_price: float = MIN_CLOSE_PRICE,
    min_amount_mean_20: float = MIN_AMOUNT_MEAN_20,
    min_turnover_mean_20: float = MIN_TURNOVER_MEAN_20,
    max_abs_pct_chg: float = MAX_ABS_PCT_CHG,
    industry_cap: int = MAX_NAMES_PER_INDUSTRY,
    price_tilt_strength: float = PRICE_TILT_STRENGTH,
    regime_aware: bool = REGIME_AWARE_ENABLED,
) -> pd.DataFrame:
    holdings = build_portfolio_holdings(
        pred_df,
        pred_col=pred_col,
        realized_col=realized_col,
        top_k=top_k,
        hold_buffer=hold_buffer,
        rebalance_every=rebalance_every,
        price_buckets=price_buckets,
        max_new_names=max_new_names,
        neutralize_industry=neutralize_industry,
        min_close_price=min_close_price,
        min_amount_mean_20=min_amount_mean_20,
        min_turnover_mean_20=min_turnover_mean_20,
        max_abs_pct_chg=max_abs_pct_chg,
        industry_cap=industry_cap,
        price_tilt_strength=price_tilt_strength,
        regime_aware=regime_aware,
    )
    if holdings.empty:
        return holdings

    rows = []
    prev_names = set()
    cost_rate = cost_bps_one_side / 10000.0
    illiquid_cost_rate = illiquid_cost_bps / 10000.0
    for dt, picks in holdings.groupby(DATE_COL):
        names = set(picks[ID_COL].tolist())
        # 这里用新旧持仓交集近似换手率，目标是让回测成本和持仓稳定性直接挂钩。
        denom = max(len(prev_names), len(names), 1)
        turnover = 1.0 if not prev_names else 1 - len(prev_names & names) / denom
        gross_ret = picks[realized_col].mean()
        illiquid_share = (
            (picks["amount_mean_20_raw"] < (min_amount_mean_20 * 2)).mean()
            if "amount_mean_20_raw" in picks.columns else 0.0
        )
        # 成本分成固定佣金/冲击部分，低流动性占比越高，附加冲击惩罚越重。
        trading_cost = 2 * (cost_rate + illiquid_cost_rate * illiquid_share) * turnover
        net_ret = gross_ret - trading_cost # 真正关心的是净收益，不是模型分数
        rows.append({
            DATE_COL: dt,
            "gross_ret": gross_ret,
            "net_ret": net_ret,
            "turnover": turnover,
            "n_holdings": len(picks),
            "n_kept": 0 if not prev_names else len(prev_names & names),
            "n_new": len(names - prev_names) if prev_names else len(names),
            "illiquid_share": illiquid_share,
            "trading_cost": trading_cost,
        })
        prev_names = names

    port = pd.DataFrame(rows)
    if port.empty:
        return port
    port = port.sort_values(DATE_COL)
    if not port.empty:
        port["gross_nav"] = (1 + port["gross_ret"].fillna(0)).cumprod()
        port["net_nav"] = (1 + port["net_ret"].fillna(0)).cumprod()
    return port


def summarize_portfolio(port: pd.DataFrame) -> dict:
    # 最后算：
    # 年化收益
    # 年化波动
    # Sharpe
    # 最大回撤
    # 平均换手
    # 平均新增持仓
    # 平均交易成本
    if port.empty:
        return {}
    periods_per_year = 252 / REBALANCE_EVERY
    gross = performance_summary(port["gross_ret"], periods_per_year=periods_per_year)
    net = performance_summary(port["net_ret"], periods_per_year=periods_per_year)
    return {
        "gross": gross,
        "net": net,
        "avg_turnover": port["turnover"].mean(),
        "avg_n_new": port["n_new"].mean() if "n_new" in port.columns else None,
        "avg_illiquid_share": port["illiquid_share"].mean(),
        "avg_trading_cost": port["trading_cost"].mean(),
    }
