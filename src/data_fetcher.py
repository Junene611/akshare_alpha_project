from __future__ import annotations
# 这是一个“未来特性”导入。
# 作用：允许你在代码中使用字符串来表示类型（比如 def func() -> "DataFrame":），
# 这在 Python 旧版本中是为了避免循环导入报错，现在是一种良好的编程习惯。

import time
# 导入时间模块，主要用于让程序“暂停”一会儿（防止下载太快被封IP）。

from typing import Optional
# 导入类型提示工具。Optional[int] 意思是一个变量可以是 int，也可以是 None。

import akshare as ak
# 导入 AKShare 库，并重命名为 ak。这是我们的数据源，相当于“原材料供应商”。

import pandas as pd
# 导入 Pandas 库，并重命名为 pd。这是 Python 处理表格数据的神器。

from tqdm import tqdm
# 导入 tqdm 库。它的作用是在循环时显示一个进度条，让你知道下载进度。

from .config import DOWNLOAD_SLEEP_SEC, BAD_NAME_KEYWORDS, EXCLUDE_CODE_PREFIX
# 从同目录下的 config.py 文件导入配置参数。
# .config 表示当前包下的 config 模块。

#智能列名适配工具（防崩溃机制）
def _normalize_col_name(name: object) -> str:
    # 定义一个私有函数（以下划线开头），用于标准化列名。
    # 参数 name: object，表示可以是任何类型，但通常是字符串。
    # 返回值 -> str：返回一个字符串。

    return "".join(str(name).strip().lower().replace("_", "").replace(" ", ""))
    # 这一行做了很多事：
    # 1. str(name): 强制把列名转成字符串。
    # 2. .strip(): 去掉首尾空格。
    # 3. .lower(): 全部转成小写（比如 "Code" 变 "code"）。
    # 4. .replace("_", ""): 去掉下划线（"stock_code" 变 "stockcode"）。
    # 5. .replace(" ", ""): 去掉中间空格（"stock code" 变 "stockcode"）。
    # 6. "".join(...): 把处理后的字符拼回去。
    # 目的：让 "Stock_Code", "stock code", "STOCKCODE" 都变成一样的 "stockcode"，方便比对。


def _pick_column(df: pd.DataFrame, candidates: list[str], fallback_idx: int = 0) -> str:
    # 定义一个函数，用于在表格中“猜”哪一列是我们想要的。
    # 参数 df: 数据表。
    # 参数 candidates: 候选列名列表，比如 ["股票代码", "code", "symbol"]。
    # 参数 fallback_idx: 兜底索引，如果名字都对不上，就默认取第几列。
    # 返回值 -> str: 返回找到的真实列名。

    # 创建一个字典，key是标准化后的列名，value是原始列名。
    # 比如：{"stockcode": "股票代码", "price": "收盘价"}
    normalized = {_normalize_col_name(col): col for col in df.columns}

    # 遍历我们想要的候选名字
    for candidate in candidates:
        # 把候选名字也标准化，去字典里查
        match = normalized.get(_normalize_col_name(candidate))
        if match is not None:
            # 如果找到了，直接返回原始的列名（比如 "股票代码"）
            return match

    # 如果上面循环结束都没找到，说明名字全变了，启用兜底方案
    if fallback_idx >= len(df.columns):
        # 如果兜底索引超出了表格列数，报错
        raise KeyError(f"Unable to locate columns {candidates} from {list(df.columns)}")

    # 否则返回表格中指定位置的列名
    return df.columns[fallback_idx]


def _rename_with_aliases(
        df: pd.DataFrame,
        alias_map: dict[str, list[str]],
        fallback_map: dict[str, int] | None = None,
) -> pd.DataFrame:
    # 这是一个批量重命名列的函数。
    # 参数 alias_map: 别名映射表。比如 {"date": ["日期", "time"]}，意思是我们想要 "date"，但源数据可能是 "日期" 或 "time"。
    # 参数 fallback_map: 兜底位置表。比如 {"date": 0}，意思是如果名字找不到，就默认第0列是日期。
    # 返回值 -> pd.DataFrame: 返回改名后的新表格。

    fallback_map = fallback_map or {}
    # 如果没传兜底表，就设为空字典，防止报错。

    rename_map: dict[str, str] = {}
    # 准备一个空字典，用来存 {旧列名: 新列名}。

    for target_col, aliases in alias_map.items():
        # 遍历映射表。target_col是我们想要的标准名（如"date"），aliases是可能的旧名列表。

        # 调用上面的 _pick_column 函数，找到真正的源列名
        source_col = _pick_column(df, aliases, fallback_idx=fallback_map.get(target_col, 0))

        # 记录：源列名 -> 目标列名
        rename_map[source_col] = target_col

    # 使用 Pandas 的 rename 函数，一次性替换所有列名
    return df.rename(columns=rename_map).copy()
    # .copy() 是为了创建一个新对象，避免修改原数据（SettingWithCopyWarning）。

#获取行业数据
def fetch_stock_industry_map() -> pd.DataFrame:
    # 获取股票和行业对应关系的函数。
    # 返回值 -> pd.DataFrame: 返回一个表格，包含股票和它的行业。

    board_df = ak.stock_board_industry_name_em()
    # 调用 AKShare 接口，获取所有“行业板块”的名字（比如“白酒”、“新能源”）。

    if board_df is None or board_df.empty:
        return pd.DataFrame(columns=["instrument", "industry", "industry_count", "industry_all"])
        # 如果接口挂了或没数据，返回一个空表格，但列名先定义好，防止后面代码报错。

    # 找到板块名称那一列。候选名有“板块名称”、“板块”等，找不到就取第1列。
    board_col = _pick_column(board_df, ["板块名称", "板块", "name", "industry"], fallback_idx=1)

    # 提取所有板块名字，去重，排序，转成列表。
    board_names = sorted(board_df[board_col].dropna().astype(str).unique().tolist())

    rows: list[pd.DataFrame] = []
    # 准备一个空列表，用来存每个板块的股票数据。

    for board_name in tqdm(board_names, desc="Downloading industry map"):
        # 循环每个板块名字。tqdm 会显示进度条。
        try:
            # 调用 AKShare 接口，获取这个板块里包含哪些股票。
            cons = ak.stock_board_industry_cons_em(symbol=board_name)
        except Exception as e:
            print(f"[WARN] industry {board_name}: {e}")
            time.sleep(DOWNLOAD_SLEEP_SEC)
            continue
            # 如果下载出错（比如网络波动），打印警告，睡一会儿，跳过这个板块，继续下一个。

        if cons is None or cons.empty:
            time.sleep(DOWNLOAD_SLEEP_SEC)
            continue
            # 如果下载下来是空的，也跳过。

        # 找到股票代码列。
        code_col = _pick_column(cons, ["代码", "股票代码", "证券代码", "code"], fallback_idx=1)

        # 只保留代码列，并改名为 "instrument"。
        tmp = cons[[code_col]].rename(columns={code_col: "instrument"}).copy()

        # 确保代码是6位字符串（比如 1 -> 000001）。
        tmp["instrument"] = tmp["instrument"].astype(str).str.zfill(6)

        # 给这一批股票打上当前板块的标签。
        tmp["industry"] = str(board_name)

        # 把处理好的这一小块数据加到列表里。
        rows.append(tmp[["instrument", "industry"]])

        time.sleep(DOWNLOAD_SLEEP_SEC)
        # 礼貌性休眠，防止请求太快被封IP。

    if not rows:
        return pd.DataFrame(columns=["instrument", "industry", "industry_count", "industry_all"])
        # 如果所有板块都下载失败，返回空表。

    # 把所有板块的数据拼成一个大表。
    merged = pd.concat(rows, axis=0, ignore_index=True).drop_duplicates()

    # 聚合操作：按股票代码分组。
    agg = (
        merged.groupby("instrument")["industry"]
        .agg(
            # 1. 主行业：把所有行业排序，取第一个作为主行业。
            industry=lambda x: sorted(set(x.astype(str)))[0],

            # 2. 行业数量：统计这只股票涉及几个行业。
            industry_count=lambda x: len(set(x.astype(str))),

            # 3. 所有行业：把所有行业用竖线 "|" 拼起来，比如 "白酒|食品饮料"。
            industry_all=lambda x: "|".join(sorted(set(x.astype(str)))),
        )
        .reset_index()
        # 重置索引，让 instrument 变回普通列。
    )
    return agg

#获取股票池
def fetch_stock_list() -> pd.DataFrame:
    # 获取全A股列表，并清洗。

    df = ak.stock_info_a_code_name()
    # 调用 AKShare 获取所有 A 股的代码和名字。

    # 把列名转成小写，方便查找。
    cols = {c.lower(): c for c in df.columns}

    # 找代码列，找不到就取第0列。
    code_col = cols.get("code", df.columns[0])

    # 找名字列，找不到就取第1列。
    name_col = cols.get("name", df.columns[1] if len(df.columns) > 1 else df.columns[0])

    # 改名：代码 -> instrument, 名字 -> stock_name
    out = df.rename(columns={code_col: "instrument", name_col: "stock_name"}).copy()

    # 格式化：代码补零成6位。
    out["instrument"] = out["instrument"].astype(str).str.zfill(6)

    # 格式化：名字转字符串。
    out["stock_name"] = out["stock_name"].astype(str)

    # --- 黑名单清洗 ---

    # 检查名字里有没有 ST, *ST, 退 等字眼。
    bad_name_mask = out["stock_name"].apply(lambda s: any(k in s for k in BAD_NAME_KEYWORDS))
    # any() 函数：只要列表里有一个关键词在股票名里，就返回 True。

    # 检查代码是不是以 8, 4, 9 开头（北交所、B股等）。
    bad_code_mask = out["instrument"].str.startswith(EXCLUDE_CODE_PREFIX)

    # 取反：保留那些 名字没问题 且 代码没问题的股票。
    out = out.loc[~bad_name_mask & ~bad_code_mask].reset_index(drop=True)

    # --- 关联行业数据 ---

    industry_map = fetch_stock_industry_map()
    # 调用刚才写的函数，获取行业表。

    if not industry_map.empty:
        # 如果有行业数据，就合并到主表里。
        # on="instrument": 按股票代码合并。
        # how="left": 左连接，保留左边所有股票，右边匹配不到的填 NaN。
        out = out.merge(industry_map, on="instrument", how="left")
    else:
        # 如果行业数据获取失败，手动填默认值。
        out["industry"] = "Unknown"
        out["industry_count"] = 0
        out["industry_all"] = "Unknown"

    # 填补缺失值：如果合并后有股票没匹配到行业，填 "Unknown"。
    out["industry"] = out["industry"].fillna("Unknown").astype(str)
    out["industry_count"] = out["industry_count"].fillna(0).astype(int)
    out["industry_all"] = out["industry_all"].fillna(out["industry"]).astype(str)

    return out

#获取个股行情
def fetch_stock_daily(symbol: str, start_date: str, end_date: str, adjust: str = "") -> pd.DataFrame:
    # 获取单只股票的日线数据。
    # symbol: 股票代码，如 "000001"。
    # start_date/end_date: 起止日期，如 "20230101"。
    # adjust: 复权类型，""为空，"qfq"为前复权，"hfq"为后复权。

    df = ak.stock_zh_a_hist(
        symbol=symbol,
        period="daily",
        start_date=start_date,
        end_date=end_date,
        adjust=adjust,
    )
    # 调用 AKShare 接口下载数据。

    if df is None or df.empty:
        return pd.DataFrame()
        # 没数据就返回空表。

    # --- 列名标准化 ---
    out = _rename_with_aliases(
        df,
        alias_map={
            # 我们想要 "date"，源数据可能是 "日期" 或 "date"
            "date": ["日期", "date"],
            # 我们想要 "instrument"，源数据可能是 "股票代码" 等
            "instrument": ["股票代码", "代码", "证券代码", "instrument", "symbol"],
            # 其他行情数据同理
            "open": ["开盘", "open"],
            "close": ["收盘", "close"],
            "high": ["最高", "high"],
            "low": ["最低", "low"],
            "volume": ["成交量", "volume"],
            "amount": ["成交额", "amount"],
            "amplitude": ["振幅", "amplitude"],
            "pct_chg": ["涨跌幅", "pct_chg", "pctchg"],
            "chg": ["涨跌额", "chg", "change"],
            "turnover": ["换手率", "turnover"],
        },
        fallback_map={
            # 如果名字全变了，就按位置硬猜。比如第0列是日期，第1列是代码...
            "date": 0,
            "instrument": 1,
            "open": 2,
            "close": 3,
            "high": 4,
            "low": 5,
            "volume": 6,
            "amount": 7,
            "amplitude": 8,
            "pct_chg": 9,
            "chg": 10,
            "turnover": 11,
        },
    )

    # 如果改名后还是没有 "instrument" 列（比如源数据里根本没这列），手动补上。
    if "instrument" not in out.columns:
        out["instrument"] = symbol

    # 格式化日期列
    out["date"] = pd.to_datetime(out["date"])

    # 格式化代码列
    out["instrument"] = out["instrument"].astype(str).str.zfill(6)

    return out

#获取指数与批量下载
def fetch_index_daily(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    # 获取指数数据（如沪深300）。逻辑和上面获取个股类似，只是接口不同。

    df = ak.stock_zh_index_daily_em(symbol=symbol, start_date=start_date, end_date=end_date)
    # 调用指数接口。

    if df is None or df.empty:
        return pd.DataFrame()

    # 改名，指数数据通常没有涨跌幅列，所以 alias_map 少了一些字段。
    out = _rename_with_aliases(
        df,
        alias_map={
            "date": ["日期", "date"],
            "open": ["开盘", "open"],
            "close": ["收盘", "close"],
            "high": ["最高", "high"],
            "low": ["最低", "low"],
            "volume": ["成交量", "volume"],
            "amount": ["成交额", "amount"],
        },
        fallback_map={
            "date": 0,
            "open": 1,
            "close": 2,
            "high": 3,
            "low": 4,
            "volume": 5,
            "amount": 6,
        },
    )
    out["date"] = pd.to_datetime(out["date"])
    return out


def batch_fetch_stock_daily(
        stock_list: pd.DataFrame,
        start_date: str,
        end_date: str,
        limit: Optional[int] = None,
        adjust: str = "",
) -> pd.DataFrame:
    # 批量下载函数。
    # stock_list: 股票列表表格。
    # limit: 限制下载数量（用于测试）。

    rows = []
    # 准备一个列表存数据。

    # 提取股票代码列，转成列表。
    symbols = stock_list["instrument"].astype(str).str.zfill(6).tolist()

    if limit is not None:
        symbols = symbols[:limit]
        # 如果限制了数量，只取前 limit 个。

    for symbol in tqdm(symbols, desc="Downloading A-share daily data"):
        # 循环下载，带进度条。
        try:
            # 调用单只股票下载函数
            df = fetch_stock_daily(symbol, start_date=start_date, end_date=end_date, adjust=adjust)
            if not df.empty:
                rows.append(df)
                # 有数据就加到列表里。
        except Exception as e:
            print(f"[WARN] {symbol}: {e}")
            # 报错打印警告，但不中断程序。

        time.sleep(DOWNLOAD_SLEEP_SEC)
        # 每次下载完睡一会儿。

    if not rows:
        return pd.DataFrame()
        # 如果全失败了，返回空表。

    # 把所有股票的数据拼起来
    out = pd.concat(rows, axis=0, ignore_index=True)

    # 排序：先按代码排，再按日期排。
    out = out.sort_values(["instrument", "date"]).reset_index(drop=True)

    return out
