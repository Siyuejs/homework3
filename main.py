import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- 中文字体设置 (防止后续画图出现框框乱码) ---
# Windows 用户默认使用 SimHei，Mac 用户请注释掉第一行，取消注释第二行
plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


def task1_preprocessing(file_path):
    print("=== 任务1：数据预处理 ===")

    # 1. 读取数据 (使用制表符分隔)
    df = pd.read_csv(file_path, sep='\t')
    print("--- 前5行数据 ---")
    print(df.head())
    print("\n--- 数据集基本信息 ---")
    df.info()

    # 2. 时间解析
    # 将交易时间转换为 datetime 类型，并提取小时
    df['交易时间'] = pd.to_datetime(df['交易时间'])
    df['hour'] = df['交易时间'].dt.hour

    # 3. 构造衍生字段
    # 计算搭乘站点数 (取下车站点与上车站点的绝对差值)
    df['ride_stops'] = (df['下车站点'] - df['上车站点']).abs()

    # 删除 ride_stops 为 0 的异常记录
    initial_len = len(df)
    df = df[df['ride_stops'] != 0]
    deleted_stops_count = initial_len - len(df)
    print(f"\n--- 异常记录处理 ---")
    print(f"删除了 {deleted_stops_count} 条搭乘站点数为0的异常记录。")

    # 4. 缺失值检查
    print("\n--- 缺失值检查 ---")
    missing_values = df.isnull().sum()
    print(missing_values[missing_values > 0])

    # 缺失值处理策略：若存在缺失值，直接删除含有缺失值的整行记录
    initial_len_before_na = len(df)
    df = df.dropna()
    deleted_na_count = initial_len_before_na - len(df)
    if deleted_na_count > 0:
        print(f"执行缺失值处理：删除了 {deleted_na_count} 条含有缺失值的记录。")
    else:
        print("数据集中不存在缺失值，无需处理。")

    return df


if __name__ == "__main__":
    # 确保 ICData.csv 在当前目录下
    df_clean = task1_preprocessing('ICData.csv')