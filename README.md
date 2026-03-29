# A股截面收益预测与组合回测研究（纯 AKShare 版）

本项目完全基于 **AKShare** 直连数据源，不依赖 BigQuant、Tushare 或手动平台导出。

## 项目目标

- 获取 A 股股票列表
- 批量下载股票历史日频行情
- 下载基准指数历史日频行情
- 构建价格量能与市场状态特征
- 预测未来 5 日超额收益
- 构建 Top-k 等权组合并完成回测
- 导出 RankIC、净值曲线、换手率图和项目报告模板

## 核心接口

- `stock_info_a_code_name`：A 股股票代码和简称
- `stock_zh_a_hist`：A 股历史日频行情
- `stock_zh_index_daily_em`：指数历史日频行情

## 目录结构

```text
akshare_alpha_project/
├── data/
│   ├── raw/
│   │   ├── stock_list.csv
│   │   ├── a_share_daily.csv
│   │   └── index_000300.csv
│   ├── processed/
│   │   ├── panel.csv
│   │   ├── feature_panel.csv
│   │   └── model_dataset.csv
│   └── predictions/
│       ├── predictions.csv
│       └── portfolio_returns_*.csv
├── reports/
│   ├── figures/
│   └── project_report_template.md
├── scripts/
│   ├── run_fetch_stock_list.py
│   ├── run_download_market_data.py
│   ├── run_build_panel.py
│   ├── run_train_predict.py
│   └── run_backtest_report.py
└── src/
    ├── config.py
    ├── utils.py
    ├── data_fetcher.py
    ├── data_loader.py
    ├── features.py
    ├── modeling.py
    ├── backtest.py
    └── report.py
```

## 依赖安装

当前默认输出格式为 `csv`，因此不强制依赖 `pyarrow`。

```bash
pip install pandas numpy scikit-learn lightgbm matplotlib akshare scipy tqdm
```

如果后续要恢复 `parquet` 输出，再额外安装 `pyarrow` 并切回相应配置即可。

## 运行顺序

### 1) 拉取 A 股股票列表

```bash
python scripts/run_fetch_stock_list.py
```

### 2) 批量下载股票日频和指数日频

先做一个小样本测试：

```bash
python scripts/run_download_market_data.py --limit 100 --start 20180101 --end 20251231
```

如果前 100 只跑通，再全量跑。全量下载建议开启续跑和分批落盘：

```bash
python scripts/run_download_market_data.py --start 20180101 --end 20251231 --resume --batch-size 200
```

### 3) 构建清洗面板与特征

```bash
python scripts/run_build_panel.py
```

### 4) 训练并生成预测

```bash
python scripts/run_train_predict.py --model both
```

### 5) 回测并导出图表

```bash
python scripts/run_backtest_report.py
```

## 第一版项目设定

- 数据源：AKShare
- 股票池：下载到的全部 A 股，默认过滤 `4/8/9` 前缀证券、ST、名称含退市风险股票，并剔除上市未满 250 个交易日样本
- 标签：未来 5 日收益减去沪深 300 未来 5 日收益
- 特征：价格动量、波动率、流动性、市场状态
- 模型：Ridge / LightGBM
- 组合：每 5 个交易日调仓，买入预测值最高的前 20 只股票，使用更宽的持仓缓冲区降低换手，并过滤低价、低流动性、接近涨跌停样本
- 成本：单边 20 bps，并对低流动性持仓追加冲击成本惩罚

## 输出结果

- `data/processed/feature_panel.csv`
- `data/predictions/predictions.csv`
- `data/predictions/portfolio_returns_ridge.csv`
- `data/predictions/portfolio_returns_lgbm.csv`
- `reports/figures/` 下的 RankIC、净值、换手率图
- `reports/project_report_template.md`

## 说明

这套代码的第一版 **不依赖估值 / 财务因子**。原因很直接：免费、稳定、全市场、长历史的逐日估值字段，不如价格量能接口更容易拿到。第一版先把价格量能版本跑通，再逐步增强因子模块，更适合作为完整研究闭环的起点。
