from __future__ import annotations
import sys
from pathlib import Path

# --- 路径配置 ---
# 还是老规矩，把项目根目录加到系统路径里，这样 Python 才能找到 src 文件夹
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# --- 导入配置 ---
# 这里定义了三个关键文件的保存路径
from src.config import PANEL_FILE, FEATURE_FILE, MODEL_FILE
# --- 导入功能模块 ---
# load_bundle: 读取之前下载好的原始数据（像把菜篮子提进厨房）
# build_clean_panel: 清洗数据（洗菜、切菜）
from src.data_loader import load_bundle, build_clean_panel
# build_feature_panel: 计算因子/特征（给菜调味、摆盘）
from src.features import build_feature_panel
# write_table: 保存表格的工具
from src.utils import write_table


def main():
    # ================= 步骤 1：读取原始数据 =================
    # 把之前下载的股票行情数据（Open, High, Low, Close, Volume）读进内存
    # bundle 通常是一个字典或者对象，装着原始数据
    bundle = load_bundle()

    # ================= 步骤 2：数据清洗与对齐 =================
    # 这一步非常关键！
    # 原始数据里可能有停牌、退市、或者不同股票的数据长度不一样。
    # build_clean_panel 会把这些数据整理成一张整齐的“长表”
    # 格式通常是：[日期, 股票代码, 收盘价, 成交量...]
    panel = build_clean_panel(bundle)

    # 把清洗好的基础行情表保存下来
    write_table(panel, PANEL_FILE, index=False)
    print(f"Saved panel to: {PANEL_FILE}")

    # ================= 步骤 3：特征工程 =================
    # 这是量化的核心！
    # 基于清洗好的行情数据，计算各种指标（比如：5日均线、动量、波动率等）
    # 这些指标就是机器学习的“特征”
    # 返回值：feature_panel 是包含特征的数据表，feature_cols 是特征的名字列表
    feature_panel, feature_cols = build_feature_panel(panel)

    # 把包含特征的数据表保存两份：
    # 1. FEATURE_FILE: 用于后续分析的特征文件
    # 2. MODEL_FILE: 直接用于模型训练的文件（有时候这两者是一样的）
    write_table(feature_panel, FEATURE_FILE, index=False)
    write_table(feature_panel, MODEL_FILE, index=False)

    # ================= 步骤 4：打印报告 =================
    print(f"Saved feature panel to: {FEATURE_FILE}")
    print(f"Rows: {len(feature_panel):,}")  # 告诉你是多少行数据（样本量）
    print(f"Features ({len(feature_cols)}): {feature_cols}")  # 告诉你算出了哪些特征（比如 ['ret_5d', 'volatility', ...]）


if __name__ == "__main__":
    main()
# Step 3 数据清洗与特征工程，把原始数据变成了可用数据