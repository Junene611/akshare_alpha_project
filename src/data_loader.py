from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import pandas as pd

from .config import STOCK_LIST_FILE, STOCK_DAILY_FILE, INDEX_FILE
from .utils import read_table


@dataclass
class DataBundle:
    stock_list: pd.DataFrame
    stock_daily: pd.DataFrame
    index_daily: pd.DataFrame


def _read(path: Path) -> pd.DataFrame:
    return read_table(path)


def load_bundle() -> DataBundle:
    # 这里负责把原始文件统一读成标准 dtypes，避免后续 merge 时出现代码和日期不一致。
    # 把股票列表、个股日线、指数日线都读进来
    # 确保 instrument 和 date 类型统一
    stock_list = _read(STOCK_LIST_FILE)
    stock_daily = _read(STOCK_DAILY_FILE)
    index_daily = _read(INDEX_FILE)

    stock_list["instrument"] = stock_list["instrument"].astype(str).str.zfill(6)
    stock_daily["instrument"] = stock_daily["instrument"].astype(str).str.zfill(6)
    stock_daily["date"] = pd.to_datetime(stock_daily["date"])
    index_daily["date"] = pd.to_datetime(index_daily["date"])

    return DataBundle(stock_list=stock_list, stock_daily=stock_daily, index_daily=index_daily)


def build_clean_panel(bundle: DataBundle) -> pd.DataFrame:
    # 先给指数算出：
    # index_ret_1
    # index_ret_5
    # index_vol_20
    # 然后把股票日线和股票列表 merge
    # 再和指数状态 merge
    # 最后得到一个“研究面板”：
    # 每一行是某只股票某一天的观测，里面既有股票自己的行情，也有行业信息和市场状态。
    stock = bundle.stock_daily.copy()
    stock_list = bundle.stock_list.copy()
    idx = bundle.index_daily.copy()

    idx = idx.sort_values("date")
    # 基准指数收益和波动率在这里先展开，后续既用于标签也用于市场状态判断。
    idx["index_ret_1"] = idx["close"].pct_change()
    idx["index_ret_5"] = idx["close"].pct_change(5)
    idx["index_vol_20"] = idx["index_ret_1"].rolling(20).std()

    panel = stock.merge(stock_list, on="instrument", how="inner")
    panel = panel.merge(
        idx[["date", "close", "index_ret_1", "index_ret_5", "index_vol_20"]].rename(columns={"close": "index_close"}),
        on="date",
        how="left",
    )
    panel = panel.sort_values(["instrument", "date"]).reset_index(drop=True)
    return panel
