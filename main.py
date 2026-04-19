import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- 中文字体设置 (防止画图出现框框乱码) ---
plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows 请用 SimHei, Mac 请改用 Arial Unicode MS
plt.rcParams['axes.unicode_minus'] = False


def task1_preprocessing(file_path):
    print("=== 任务1：数据预处理 ===")

    # 1. 读取数据
    # 【核心修复】：作业说明里写的是 \t 分隔，但实际文件内容是逗号分隔！
    # 所以直接使用默认的逗号分隔读取即可，不需要传 sep='\t'
    df = pd.read_csv(file_path, encoding='utf-8')

    print("--- 前5行数据 ---")
    print(df.head())
    print("\n--- 数据集基本信息 ---")
    df.info()

    # 2. 时间解析
    df['交易时间'] = pd.to_datetime(df['交易时间'])
    df['hour'] = df['交易时间'].dt.hour

    # 3. 构造衍生字段
    # 计算搭乘站点数 (取下车站点与上车站点的绝对差值)
    df['ride_stops'] = (df['下车站点'] - df['上车站点']).abs()

    # 过滤：删除 ride_stops 为 0 的异常记录
    initial_len = len(df)
    df = df[df['ride_stops'] != 0]
    deleted_stops_count = initial_len - len(df)
    print(f"\n--- 异常记录处理 ---")
    print(f"删除了 {deleted_stops_count} 条搭乘站点数为0的异常记录。")

    # 4. 缺失值检查
    print("\n--- 缺失值检查 ---")
    missing_values = df.isnull().sum()
    print(missing_values[missing_values > 0])

    # 缺失值处理：若存在缺失值，直接删除含有缺失值的整行记录
    initial_len_before_na = len(df)
    df = df.dropna()
    deleted_na_count = initial_len_before_na - len(df)
    if deleted_na_count > 0:
        print(f"执行缺失值处理：删除了 {deleted_na_count} 条含有缺失值的记录。")
    else:
        print("数据集中不存在缺失值，无需处理。")

    return df


# ==========================================
# 任务 2：时间分布分析
# ==========================================
def task2_time_analysis(df):
    print("\n=== 任务2：时间分布分析 ===")

    # 题目要求 (a)：仅统计刷卡类型=0（即上车刷卡）的记录
    boarding_df = df[df['刷卡类型'] == 0].copy()

    # 【必须使用 numpy】提取 hour 列为 numpy 数组进行高效统计
    hours_arr = boarding_df['hour'].values

    # 使用 numpy 条件过滤统计
    early_morning_count = np.sum(hours_arr < 7)  # 早峰前 (<7)
    late_night_count = np.sum(hours_arr >= 22)  # 深夜时段 (>=22)
    total_boarding = len(hours_arr)  # 全天上车总刷卡量

    print(f"早峰前时段 (<7点) 刷卡量: {early_morning_count} 次，占比: {early_morning_count / total_boarding:.2%}")
    print(f"深夜时段 (>=22点) 刷卡量: {late_night_count} 次，占比: {late_night_count / total_boarding:.2%}")

    # 题目要求 (b)：24小时分布柱状图可视化
    # 统计各小时刷卡量，并补齐可能缺失的小时(0-23)
    hour_counts = boarding_df['hour'].value_counts().reindex(range(24), fill_value=0)

    plt.figure(figsize=(10, 6))

    # 生成颜色列表：<7或>=22高亮显示（橙色），其余为天蓝色
    colors = ['orange' if (h < 7 or h >= 22) else 'skyblue' for h in range(24)]

    # 使用 matplotlib 画柱状图
    bars = plt.bar(hour_counts.index, hour_counts.values, color=colors)

    # 图表格式设置
    plt.title('24小时公交刷卡量分布图', fontsize=15)
    plt.xlabel('小时 (0-23)', fontsize=12)
    plt.ylabel('刷卡量 (次)', fontsize=12)
    plt.xticks(range(0, 24, 2))  # x轴步长为2
    plt.grid(axis='y', linestyle='--', alpha=0.7)  # 水平网格线

    # 添加图例 (借助自定义的图例句柄)
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='orange', label='早峰前及深夜时段'),
                       Patch(facecolor='skyblue', label='常规时段')]
    plt.legend(handles=legend_elements)

    plt.savefig('hour_distribution.png', dpi=150)
    print("图像已保存为 hour_distribution.png")


# ==========================================
# 任务 3：线路站点分析
# ==========================================
def analyze_route_stops(df, route_col='线路号', stops_col='ride_stops'):
    """
    计算各线路乘客的平均搭乘站点数及其标准差。
    Parameters
    ----------
    df : pd.DataFrame  预处理后的数据集
    route_col : str    线路号列名
    stops_col : str    搭乘站点数列名
    Returns
    -------
    pd.DataFrame  包含列：线路号、mean_stops、std_stops，按 mean_stops 降序排列
    """
    # 聚合计算均值和标准差
    agg_df = df.groupby(route_col)[stops_col].agg(['mean', 'std']).reset_index()
    agg_df.columns = [route_col, 'mean_stops', 'std_stops']
    agg_df = agg_df.sort_values(by='mean_stops', ascending=False)
    return agg_df


def task3_visualization(df):
    print("\n=== 任务3：线路站点分析 ===")

    # 1. 调用自定义函数并打印前10行
    route_stats = analyze_route_stops(df)
    print("--- 各线路平均搭乘站点数（Top 10） ---")
    print(route_stats.head(10))

    # 2. 提取均值最高的前15条线路名单
    top15_routes = route_stats.head(15)['线路号'].tolist()

    # 3. 筛选这15条线路的【原始记录】交给 seaborn 画图
    # 技巧：seaborn 只有接收原始数据，才能通过 errorbar='sd' 自动算出误差棒
    plot_data = df[df['线路号'].isin(top15_routes)].copy()
    plot_data['线路号'] = plot_data['线路号'].astype(str)  # 转为字符串，避免y轴当成数字刻度
    order_list = [str(x) for x in top15_routes]

    plt.figure(figsize=(10, 8))

    # 使用 seaborn 水平条形图
    sns.barplot(data=plot_data,
                x='ride_stops',
                y='线路号',
                order=order_list,
                errorbar='sd',  # 误差棒显示标准差
                capsize=0.3,  # 横帽宽度
                palette="Blues_d")

    plt.title('均值最高的前15条线路搭乘站点数分析', fontsize=15)
    plt.xlabel('平均搭乘站点数', fontsize=12)
    plt.ylabel('线路号', fontsize=12)
    plt.xlim(left=0)  # x轴从0开始

    plt.tight_layout()
    plt.savefig('route_stops.png', dpi=150)
    print("图像已保存为 route_stops.png")


# ==========================================
# 主程序入口
# ==========================================
if __name__ == "__main__":
    # 任务1
    df_clean = task1_preprocessing('ICData.csv')

    # 任务2
    task2_time_analysis(df_clean)

    # 任务3
    task3_visualization(df_clean)