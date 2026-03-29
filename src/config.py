from pathlib import Path

# 配置路径
PROJECT_ROOT = Path(__file__).resolve().parents[1]

#数据文件放哪里
DATA_ROOT = PROJECT_ROOT / "data"
RAW_ROOT = DATA_ROOT / "raw" #原始数据
PROCESSED_ROOT = DATA_ROOT / "processed" #加工数据
PRED_ROOT = DATA_ROOT / "predictions" #预测结果

REPORT_ROOT = PROJECT_ROOT / "reports"
FIG_ROOT = REPORT_ROOT / "figures"

#数据文件
STOCK_LIST_FILE = RAW_ROOT / "stock_list.csv"
STOCK_DAILY_FILE = RAW_ROOT / "a_share_daily.csv"
INDEX_FILE = RAW_ROOT / "index_000300.csv"

#文件路径配置
PANEL_FILE = PROCESSED_ROOT / "panel.csv" #清洗后的原始行情面板
FEATURE_FILE = PROCESSED_ROOT / "feature_panel.csv" #算好因子的数据表
MODEL_FILE = PROCESSED_ROOT / "model_dataset.csv" #最终喂给模型训练的数据集（可能包含了一些特殊的筛选）
PRED_FILE = PRED_ROOT / "predictions.csv" #模型跑出来的预测结果

#列名标准化定义
DATE_COL = "date"
ID_COL = "instrument"
NAME_COL = "stock_name"

#基准与数据清洗规则，对标沪深300指数
BENCHMARK_SYMBOL = "sh000300"

#标签周期是多少
LABEL_HORIZON = 5 #未来5天的收益率 label

#调仓频率是多少
REBALANCE_EVERY = 5 #每5天调仓一次

#组合持多少只股票
TOP_K = 20

#持仓缓冲池大小。
#程序可能会先选出 100只 高分股票放入缓冲池，然后再从里面挑。这通常用于后续的价格分层或行业分层逻辑。
HOLD_BUFFER = 100

#把股票按价格分成 5个档位（比如0-10元，10-20元...）。
#防止选出来的20只股票全是低价股或全是高价股，强制让组合在价格维度上分散。
PRICE_BUCKETS = 5

#每次调仓，最多换6只股票。
MAX_NEW_NAMES = 6

#默认不做行业中性化
DEFAULT_NEUTRALIZE_INDUSTRY = False #策略允许押注某个行业（比如全买半导体）。如果设为 True，策略会强制在每个行业里选股，保证不偏科。

#同一个行业，最多买3只。为了防止风险过于集中。
MAX_NAMES_PER_INDUSTRY = 3

#价格倾斜强度。
#如果设为 0：完全听模型的预测分。
#如果设为正数：会额外照顾那些近期跌幅大的股票（博反弹）。
#这里设为 0，说明策略纯粹依赖模型预测，不人为干预。
PRICE_TILT_STRENGTH = 0.0

#开启市场状态感知。
REGIME_AWARE_ENABLED = True

#用“指数20日波动率”和“指数5日收益率”来判断当前市场是好是坏。
REGIME_LOOKBACK_VOL_COL = "index_vol_20"
REGIME_TREND_COL = "index_ret_5"

#定义什么是“坏市场”。这里定义了两种：
#up_lowvol：上涨但低波动（可能是诱多，涨不动了）。
#down_lowvol：下跌且低波动（阴跌，最磨人）。
ADVERSE_MARKET_STATES = ("up_lowvol", "down_lowvol")

#坏市场时，换股更谨慎，最多只换4只（平时是6只）。
ADVERSE_STATE_MAX_NEW_NAMES = 4

#坏市场时，缓冲池扩大到 140只。这意味着筛选标准变严了，要在更多股票里挑优等生。
ADVERSE_STATE_HOLD_BUFFER = 140
ADVERSE_STATE_TOP_K = TOP_K

#坏市场时，强制开启行业中性化。
ADVERSE_STATE_FORCE_NEUTRALIZE = True

#成本怎么设
COST_BPS_ONE_SIDE = 20.0 #单边交易成本是20个基点
ILLIQUID_COST_BPS = 20.0 #流动性差的股票（很难买卖），额外再加 0.2% 的冲击成本。

#流动性门槛怎么设
MIN_CLOSE_PRICE = 3.0 #股价必须大于3
MIN_AMOUNT_MEAN_20 = 60_000_000.0 #成交额，过去20天平均每天成交额大于6000万
MIN_TURNOVER_MEAN_20 = 2.0 #换手率，过去20天平均换手率大于2%

#剔除当日涨跌幅超过 9.5% 的数据。主要是为了剔除涨停或跌停的数据。涨停时往往买不进去，跌停时卖不出来，这些数据对模型来说是“无效信号”，甚至会误导模型。
MAX_ABS_PCT_CHG = 0.095

#训练/验证/测试时间怎么切
TRAIN_START = "2018-01-01" #训练集 (2018-2021)
VALID_START = "2022-01-01" #验证集 (2022)
TEST_START = "2023-01-01" #测试集 (2023-2025)
TEST_END = "2025-12-31"


WALK_FORWARD_FREQ = "MS"
TRAIN_LOOKBACK_YEARS = 5

#训练样本的采样频率。
#既然是5天调仓一次，那训练数据也每隔5天取一个样本，避免数据过于密集导致模型偏向短期噪音。
TRAIN_SAMPLE_EVERY = REBALANCE_EVERY

#随机种子。
RANDOM_STATE = 42

#Ridge回归的正则化强度。
#数值越大，模型越简单（限制越死），防止过拟合。
RIDGE_ALPHA = 2.0

#下载每只股票数据后，休眠 0.2 秒。
DOWNLOAD_SLEEP_SEC = 0.2

# 剔除ST股
BAD_NAME_KEYWORDS = ["ST", "*ST", "退", "退市"]

#剔除代码以 8、4、9 开头的股票。8/4开头：通常是北交所股票（波动大、门槛高）或老三板。
#9开头：可能是B股（用外币交易）。
#这里为了策略稳健，只玩A股主板、创业板和科创板。
EXCLUDE_CODE_PREFIX = ("8", "4", "9")

#上市至少 250天（约1年）才买入。避开次新股。次新股波动极其剧烈，且财务数据不稳定，很难用模型预测。
MIN_LISTING_DAYS = 250

#LGBM参数
LGBM_PARAMS = {
    "objective": "regression",
    "n_estimators": 200,# 200棵树
    "learning_rate": 0.03,# 学习率很低，防止过拟合
    "num_leaves": 15, # 叶子节点少，模型简单，泛化能力强
    "max_depth": 5,# 树的最大深度：5层，同样是防止模型太复杂
    "subsample": 0.7,# 每次只用70%的数据，增加随机性
    "colsample_bytree": 0.7,# 列采样：每次只用70%的特征训练
    "min_child_samples": 300,# 叶子节点最少样本数：必须凑够300个样本才分裂，防止学偏
    "reg_alpha": 1.0, # L1正则化：让不重要的特征权重变0
    "reg_lambda": 2.0,# L2正则化：让特征权重变小
    "random_state": RANDOM_STATE,# 随机种子
    "n_jobs": -1, # 并行计算：用满所有CPU核心
}
