from __future__ import annotations
import argparse # 【新增】用于处理命令行参数，比如 --start 20200101
import sys
import time # 【新增】用于让程序“睡”一会儿，防止下载太快被封IP
from pathlib import Path

import pandas as pd

# --- 路径配置 ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# --- 导入配置和工具 ---
# 导入了更多的配置项，比如下载间隔时间、基准指数符号等
from src.config import STOCK_LIST_FILE, STOCK_DAILY_FILE, INDEX_FILE, BENCHMARK_SYMBOL, DOWNLOAD_SLEEP_SEC
# 导入了两个下载函数：一个下个股，一个下指数
from src.data_fetcher import fetch_index_daily, fetch_stock_daily
# 导入工具：确保文件夹存在、读写表格
from src.utils import ensure_parent
from src.utils import read_table, write_table

#自定义工具函数 append_csv 追加写入，能把新数据接在旧数据后面
def append_csv(df: pd.DataFrame, path: Path) -> None:
    """
        将数据追加写入 CSV 文件。
        如果文件不存在，则创建并写入表头；如果存在，则直接追加数据，不写表头。
    """
    if df.empty:
        return
    # 确保文件所在的文件夹存在
    ensure_parent(path)
    # 判断文件是否存在
    # 如果文件不存在，说明是第一次写入，需要写入表头 (header=True)
    # 如果文件已存在，说明是追加模式，不需要表头 (header=False)
    header = not path.exists()
    # mode="a" 表示 append 模式（追加），而不是默认的 write 模式（覆盖）
    df.to_csv(path, mode="a", header=header, index=False)

#参数解析（让脚本变灵活）可以在命令行控制程序的行为，不需要修改代码
def main():
    # 1. 创建参数解析器
    parser = argparse.ArgumentParser()
    # 2. 定义允许的参数
    # 比如：python main.py --start 20230101 --limit 10
    parser.add_argument("--start", type=str, default="20180101")
    parser.add_argument("--end", type=str, default="20251231")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--adjust", type=str, default="", help="'' / qfq / hfq")
    parser.add_argument("--offset", type=int, default=0, help="start from the Nth stock in the filtered stock list")
    parser.add_argument("--batch-size", type=int, default=200, help="flush stock daily data to disk every N stocks")
    parser.add_argument("--resume", action="store_true", help="skip instruments already written to the output csv")
    # 3. 解析参数
    args = parser.parse_args()

#准备股票列表（切片与过滤）
    # 1. 读取股票列表
    stock_list = read_table(STOCK_LIST_FILE)
    # 2. 数据清洗：确保股票代码是 6 位字符串（比如 '1' 变成 '000001'）
    # .str.zfill(6) 是 pandas 的字符串操作，不足6位前面补0
    stock_list["instrument"] = stock_list["instrument"].astype(str).str.zfill(6)
    # 3. 提取代码列变成列表
    symbols = stock_list["instrument"].tolist()
    # 4. 根据命令行参数进行切片
    if args.limit is not None:
        symbols = symbols[:args.limit]# 只取前 limit 个
    if args.offset:
        symbols = symbols[args.offset:]# 跳过前 offset 个

#断点续传逻辑（避免重复劳动）
    completed = set()# 用一个集合记录已经下载好的股票
    # 如果开启了 resume 模式，且文件已经存在
    if args.resume and STOCK_DAILY_FILE.exists():
        # 读取已存在文件中的股票代码列
        existing = pd.read_csv(STOCK_DAILY_FILE, usecols=["instrument"])
        # 整理成集合，方便快速查找
        completed = set(existing["instrument"].astype(str).str.zfill(6).unique().tolist())
        # 从待下载列表中剔除已存在的股票
        symbols = [s for s in symbols if s not in completed]
        print(f"Resume enabled, skipping {len(completed):,} instruments already in {STOCK_DAILY_FILE}")
    # 如果没开 resume 模式但文件存在，说明是全新下载，删掉旧文件
    elif STOCK_DAILY_FILE.exists() and not args.resume:
        STOCK_DAILY_FILE.unlink()# 删除文件

#主循环（下载与分批保存）先攒一批数据在内存里，攒够了再写硬盘
    print(f"Downloading {len(symbols):,} instruments with batch size {args.batch_size}")
    buffer = []# 【内存缓冲区】用来暂存下载的数据
    done = 0# 计数器，记录下载了多少只
    for symbol in symbols:
        try:
            # 1. 调用接口下载单只股票数据
            df = fetch_stock_daily(symbol, start_date=args.start, end_date=args.end, adjust=args.adjust)
            # 2. 如果数据不为空，放入缓冲区
            if not df.empty:
                buffer.append(df)
        except Exception as e:
            # 3. 捕获异常，防止一只股票失败导致整个程序崩溃
            print(f"[WARN] {symbol}: {e}")

        done += 1
        # 4. 【关键逻辑】每下载 batch_size 只股票，就强制保存一次
        if done % args.batch_size == 0:
            # 把缓冲区里的 DataFrame 拼成一个大表
            chunk = pd.concat(buffer, axis=0, ignore_index=True) if buffer else pd.DataFrame()
            # 写入硬盘（追加模式）每200次写入一次硬盘
            append_csv(chunk, STOCK_DAILY_FILE)
            print(f"Flushed {done:,} / {len(symbols):,} instruments to {STOCK_DAILY_FILE}")
            buffer = []# 【清空缓冲区】释放内存
        # 5. 礼貌性休眠，防止被封 IP
        time.sleep(DOWNLOAD_SLEEP_SEC)

    # 循环结束后，如果缓冲区里还有剩余数据（不足 batch_size 的部分），也要保存
    if buffer:
        chunk = pd.concat(buffer, axis=0, ignore_index=True)
        append_csv(chunk, STOCK_DAILY_FILE)

#收尾工作（下载指数）
    # 确认文件保存成功
    if STOCK_DAILY_FILE.exists():
        print(f"Saved stock daily to: {STOCK_DAILY_FILE}")
        print(pd.read_csv(STOCK_DAILY_FILE, nrows=5))# 预览前5行
    # 最后下载大盘指数（基准数据）
    index_df = fetch_index_daily(
        symbol=BENCHMARK_SYMBOL,
        start_date=args.start,
        end_date=args.end,
    )
    # 指数数据量小，直接一次性写入
    write_table(index_df, INDEX_FILE, index=False)
    print(f"Saved index daily to: {INDEX_FILE}")
    print(index_df.head())

if __name__ == "__main__":
    main()
# Step 2 数据下载与保存逻辑