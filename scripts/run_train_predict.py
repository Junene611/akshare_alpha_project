from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
# --- 路径配置 ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
# --- 导入配置 ---
# FEATURE_FILE: 上一步生成的特征数据（食材）
# FIG_ROOT: 存放生成的图片（比如特征重要性图）
# PRED_FILE: 存放预测结果的表格
# REPORT_ROOT: 存放评估报告
from src.config import FEATURE_FILE, FIG_ROOT, PRED_FILE, REPORT_ROOT
# --- 导入功能模块 ---
# fit_predict: 核心函数，负责“训练”和“预测”
from src.modeling import fit_predict
# save_feature_importance: 画图工具，画出哪些因子最重要
from src.report import save_feature_importance
# rank_ic_by_date: 评估工具，计算 RankIC（衡量预测准不准的核心指标）
from src.utils import rank_ic_by_date
from src.utils import read_table, write_table


def main():
    # 1. 参数解析
    parser = argparse.ArgumentParser()
    # 允许选择模型：ridge(线性回归的一种), lgbm(LightGBM, 树模型), 或者 both(两个都跑)
    parser.add_argument("--model", type=str, default="both", choices=["ridge", "lgbm", "both"])
    args = parser.parse_args()

    # 2. 读取特征数据
    # 这就是上一步生成的那个大宽表
    df = read_table(FEATURE_FILE, parse_dates=["date"])

    # 3. 筛选特征列 (关键步骤！)
    # 模型不能吃“日期”、“股票名”这些非数字信息，也不能吃“收盘价”这种原始价格（因为价格高低不代表涨跌）
    # 我们要把用来做输入的“特征”和用来验证结果的“标签”区分开
    drop_cols = {
        "date", "instrument", "stock_name",  # 基础信息，不是特征
        "label_excess_5", "future_ret_5",  # 这是答案（标签），不能作为题目（特征）输入
        "open", "high", "low", "close", "volume", "amount",  # 原始行情数据，通常不直接用
        "index_close", "index_ret_1", ...,  # 指数数据（有时作为特征，这里被排除了）
        "amihud_20_raw",  # 其他被排除的列
    }

    # 第一层过滤：剔除掉上面的黑名单，且只保留数字类型的列 (biufc 代表布尔、整型、无符号、浮点、复数)
    feature_cols = [c for c in df.columns if c not in drop_cols and df[c].dtype.kind in "biufc"]

    # 第二层过滤：剔除掉全是空值的列
    feature_cols = [c for c in feature_cols if df[c].notna().any()]

    # 4. 创建输出文件夹
    FIG_ROOT.mkdir(parents=True, exist_ok=True)
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)

    outputs = []  # 用来存两个模型的预测结果
    metrics = {}  # 用来存两个模型的评分

#Ridge 模型（线性模型）
    # 定义基础列，用于最后合并结果
    base_cols = ["date", "instrument", "stock_name", "label_excess_5", "future_ret_5", ...]

    # --- 如果选择了 ridge 或 both ---
    if args.model in ("ridge", "both"):
        # 1. 训练并预测
        # fit_predict 会自动切分训练集/测试集，训练模型，然后返回预测结果
        pred_r = fit_predict(df, feature_cols=feature_cols, model_name="ridge")

        # 2. 保存预测结果（只保留基础列 + 预测列）
        outputs.append(pred_r[base_cols + ["pred_ridge"]])

        # 3. 评估效果
        # 计算 RankIC：相关系数。简单说，就是看预测值和真实涨跌幅的相关性。
        # 值越接近1越好，0.05以上通常就算有效了。
        ric_r = rank_ic_by_date(pred_r, pred_col="pred_ridge", label_col="label_excess_5", date_col="date")

        # 4. 分析特征重要性
        # 看看模型觉得哪些因子最重要（比如是动量重要，还是成交量重要）
        fi_r = pred_r.attrs.get("feature_importance")
        if fi_r is not None and not fi_r.empty:
            write_table(fi_r, REPORT_ROOT / "feature_importance_ridge.csv", index=False)
            # 画一张柱状图
            save_feature_importance(fi_r, FIG_ROOT / "feature_importance_ridge.png")

        # 5. 记录指标
        metrics["ridge"] = {
            "rank_ic_mean": float(ric_r["rank_ic"].mean()) if not ric_r.empty else None,
            "prediction_rows": int(len(pred_r)),
            "feature_count": int(len(feature_cols)),
        }
        print("Ridge RankIC mean:", metrics["ridge"]["rank_ic_mean"])

#LightGBM 模型（树模型）
    # --- 如果选择了 lgbm 或 both ---
    # 逻辑和上面完全一样，只是换了个模型名字
    if args.model in ("lgbm", "both"):
        pred_l = fit_predict(df, feature_cols=feature_cols, model_name="lgbm")
        outputs.append(pred_l[base_cols + ["pred_lgbm"]])

        ric_l = rank_ic_by_date(pred_l, pred_col="pred_lgbm", label_col="label_excess_5", date_col="date")

        fi_l = pred_l.attrs.get("feature_importance")
        if fi_l is not None and not fi_l.empty:
            write_table(fi_l, REPORT_ROOT / "feature_importance_lgbm.csv", index=False)
            save_feature_importance(fi_l, FIG_ROOT / "feature_importance_lgbm.png")

        metrics["lgbm"] = {
            "rank_ic_mean": float(ric_l["rank_ic"].mean()) if not ric_l.empty else None,
            "prediction_rows": int(len(pred_l)),
            "feature_count": int(len(feature_cols)),
        }
        print("LightGBM RankIC mean:", metrics["lgbm"]["rank_ic_mean"])

#合并结果并归档
    # 1. 合并结果
    # 如果只跑了一个模型，outputs[0] 就是结果
    # 如果跑了两个，就把它们按 ["date", "instrument"...] 拼起来
    pred = outputs[0]
    for x in outputs[1:]:
        pred = pred.merge(x, on=base_cols, how="outer")

    # 2. 保存预测表
    # 这张表里会有：日期、股票代码、真实涨跌幅、Ridge预测值、LGBM预测值
    write_table(pred, PRED_FILE, index=False)

    # 3. 保存评估报告
    # 把 RankIC 等指标存成 JSON 文件，方便以后查看
    (REPORT_ROOT / "model_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    ) #模型的RankIC是多少，判断模型是否有效

    print(f"Saved predictions to: {PRED_FILE}") #记录了每一天每一支股票的预测涨跌幅
    print(pred.head())

if __name__ == "__main__":
    main()
# Step 4 训练模型，对未来预测，最后进行打分。机器学习建模流程
# 结果是：
# 每只股票每个调仓日都有一个 pred_lgbm 或 pred_ridge
# 但注意，这还不是策略收益，只是“模型分数”。