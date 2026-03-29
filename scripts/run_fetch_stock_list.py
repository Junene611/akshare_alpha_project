from __future__ import annotations
import sys
from pathlib import Path

# 【核心逻辑】获取项目根目录的绝对路径
# __file__ 是当前这个脚本文件的路径
# .resolve() 把它变成绝对路径（比如 D:/Project/src/main.py）
# .parents[1] 获取它的上一级目录（即项目根目录 D:/Project）
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 【核心逻辑】把项目根目录加入到 Python 的搜索路径中
# 为什么要这样做？
# 如果不加这一行，Python 只能找到当前文件夹里的文件。
# 加了这一行，Python 就能识别 "src" 这个文件夹，从而允许你写 "from src.config import ..."
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 从 src/config.py 文件中，引入 STOCK_LIST_FILE 这个变量（可能是文件保存路径）
from src.config import STOCK_LIST_FILE
# 从 src/data_fetcher.py 文件中，引入 fetch_stock_list 这个函数（负责干活：下载数据）
from src.data_fetcher import fetch_stock_list
# 从 src/utils.py 文件中，引入 write_table 这个函数（负责干活：保存数据）
from src.utils import write_table

# 主程序逻辑
def main():
    # 1. 调用下载功能
    # 运行 fetch_stock_list() 函数，把返回的结果（股票列表表格）赋值给变量 df
    df = fetch_stock_list()
    # 2. 调用保存功能
    # 把 df 这个表格，保存到 STOCK_LIST_FILE 指定的路径去
    # index=False 意思是保存时不要把行号（0, 1, 2...）也存进去
    write_table(df, STOCK_LIST_FILE, index=False)
    # 3. 打印反馈信息（给用户看的）
    print(f"Saved stock list to: {STOCK_LIST_FILE}")# 告诉你在哪保存的
    print(df.head())# 打印表格的前5行，让你检查一下
    print("Total stocks:", len(df))# 打印一共有多少只股票

# 【标准写法】
# 这句话的意思是：只有当你直接运行这个脚本时（比如点击运行按钮），
# 才会执行 main() 函数。
# 如果是别的文件 import 了这个文件，main() 不会自动运行。
if __name__ == "__main__":
    main()
# Step 1 下载数据
