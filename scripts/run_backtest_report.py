from __future__ import annotations
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import FEATURE_FILE, PRED_FILE, PRED_ROOT, FIG_ROOT, REPORT_ROOT
from src.backtest import build_topk_portfolio, summarize_portfolio
from src.report import plot_nav, plot_turnover, plot_rank_ic
from src.utils import rank_ic_by_date, read_table, write_table


def load_prediction_context():
    # 1. 读取预测结果（模型给出的分数）
    pred = read_table(PRED_FILE, parse_dates=["date"])
    # 2. 读取特征数据（包含原始行情，如收盘价、成交量等）
    feature = read_table(FEATURE_FILE, parse_dates=["date"])
    # 3. 挑选需要的列
    # 回测不仅要知道预测分，还需要知道当天的收盘价（用来算买卖）、行业（用来分析）、真实涨跌幅（用来算收益）
    keep_cols = [
        "date", "instrument", "industry", "index_ret_1", "index_ret_5", "index_vol_20",
        "close", "amount_mean_20_raw", "turnover_mean_20_raw", "future_ret_5",
    ]
    keep_cols = [c for c in keep_cols if c in feature.columns]
    # 4. 去重并合并
    # 把预测分和行情数据拼在一起，形成一张“全知全能”的大表
    feature = feature[keep_cols].drop_duplicates(subset=["date", "instrument"])
    return pred.merge(feature, on=["date", "instrument"], how="left", suffixes=("", "_feat"))

#生成研报模板
def write_report_template(summary: dict, model_name: str):
    # 这是一个简单的文本生成器
    # 它把回测的结果（summary）填进一个 Markdown 模板里
    text = f"""# 项目报告模板（AKShare 版）

## 1. 研究目标
基于 AKShare 提供的 A 股历史日频行情与指数日频数据，构建价格量能和市场状态特征，预测未来 5 日超额收益，并检验在成本约束下是否能够形成稳定可交易组合。

## 2. 数据来源
- AKShare: stock_info_a_code_name
- AKShare: stock_zh_a_hist
- AKShare: stock_zh_index_daily_em

## 3. 标签定义
- 个股未来 5 日收益
- 减去基准指数未来 5 日收益得到超额收益标签

## 4. 模型
- 当前结果对应模型：{model_name}

## 5. 核心结果
```json
{json.dumps(summary, indent=2, ensure_ascii=False)}
```

## 6. 图表
请插入：
- rank_ic_{model_name}.png
- nav_{model_name}.png
- turnover_{model_name}.png

## 7. 可扩展方向
- 加入行业信息与行业中性化
- 加入估值/财务因子
- 引入更严格的 walk-forward 训练
- 做分年份稳健性分析
"""
    out = REPORT_ROOT / f"project_report_{model_name}.md"
    out.write_text(text, encoding="utf-8")
    print(f"Saved report template to: {out}")

#核心回测循环，遍历每个模型，并给他们打分
def main():
    # 1. 加载“全知全能”的大表
    pred = load_prediction_context()
    # 2. 看看表里有哪些模型的预测列（pred_ridge 或 pred_lgbm）
    model_candidates = [c for c in ["pred_ridge", "pred_lgbm"] if c in pred.columns]
    if not model_candidates:
        raise ValueError("No prediction columns found in predictions file")
    # 3. 创建输出文件夹
    FIG_ROOT.mkdir(parents=True, exist_ok=True)
    PRED_ROOT.mkdir(parents=True, exist_ok=True)
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)

    summary_by_model: dict[str, dict] = {}
    # --- 循环评估每个模型 ---
    for pred_col in model_candidates:
        # 从列名提取模型名字，比如 "pred_ridge" -> "ridge"
        model_name = pred_col.replace("pred_", "")
        # 【步骤 A】计算 RankIC（预测能力打分）
        # 看看模型预测的分数，和真实的涨跌幅相关性高不高
        rank_ic = rank_ic_by_date(pred, pred_col=pred_col, label_col="label_excess_5", date_col="date")
        # 【步骤 B】构建投资组合（模拟炒股）
        # build_topk_portfolio：这是回测的核心！
        # 它的逻辑通常是：每天选出预测分最高的 Top N 只股票，假设买入，持有一定时间后卖出
        port = build_topk_portfolio(pred, pred_col=pred_col, realized_col="future_ret_5")
        # 【步骤 C】汇总绩效指标
        # 计算总收益率、年化收益、最大回撤、夏普比率等
        summary = summarize_portfolio(port)

        # 【步骤 D】画图（可视化）
        # 1. 画 RankIC 图：看预测稳定性
        plot_rank_ic(rank_ic, FIG_ROOT / f"rank_ic_{model_name}.png")
        # 2. 画净值曲线：看赚了多少钱（这是老板最爱看的图）最重要的一张图
        plot_nav(port, FIG_ROOT / f"nav_{model_name}.png")
        # 3. 画换手率图：看交易频繁程度（手续费杀手）
        plot_turnover(port, FIG_ROOT / f"turnover_{model_name}.png")

        # 【步骤 E】保存结果
        # 保存每日持仓和收益明细，买入卖出操作、当日的盈亏
        write_table(port, PRED_ROOT / f"portfolio_returns_{model_name}.csv", index=False)
        # 打印核心指标
        print(model_name, summary)
        # 生成 Markdown 研报
        write_report_template(summary, model_name)
        summary_by_model[model_name] = summary

    # 最后把所有模型的总结存成一个 JSON 文件
    (REPORT_ROOT / "backtest_summary.json").write_text(
        json.dumps(summary_by_model, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )# 包含了夏普比率、最大回撤等关键数字

if __name__ == "__main__":
    main()
# Step 5 回测与评估，把模型预测的结果，放到真实历史行情中去模拟炒股。