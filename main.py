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
# ==========================================
# 任务 4：高峰小时系数计算 (PHF)
# ==========================================
def task4_phf_calculation(df):
    print("\n=== 任务4：高峰小时系数计算 ===")
    # 仅统计上车记录
    boarding_df = df[df['刷卡类型'] == 0].copy()

    # --- 1. 高峰小时自动识别 ---
    hour_counts = boarding_df['hour'].value_counts()
    peak_hour = hour_counts.idxmax()  # 找出刷卡量最大的小时
    peak_hour_vol = hour_counts.max()  # 获取该小时的总刷卡量

    peak_start = f"{peak_hour:02d}:00"
    peak_end = f"{peak_hour + 1:02d}:00"
    print(f"高峰小时：{peak_start} ~ {peak_end}，刷卡量：{peak_hour_vol} 次")

    # --- 2. 分钟粒度聚合与 PHF 计算 (含核心手写注释) ---

    # 提取高峰小时内的所有记录
    peak_df = boarding_df[boarding_df['hour'] == peak_hour].copy()

    # 【必须将时间列设为索引，才能使用 pandas 的 resample 时间重采样功能】
    peak_df.set_index('交易时间', inplace=True)

    # ---------------------------------------------------------
    # 以下为需要在 README 报告中提交的核心代码及逐行中文注释
    # ---------------------------------------------------------
    # 使用 resample 按 '5min' (5分钟) 为时间窗口进行重采样，并用 size() 统计每个窗口内的刷卡记录数
    resampled_5min = peak_df.resample('5min').size()
    # 找出聚合后客流量最大的那个 5 分钟区间的刷卡量
    max_5min_vol = resampled_5min.max()
    # 找出该最大刷卡量对应的起始时间点
    max_5min_time = resampled_5min.idxmax()

    # 使用 resample 按 '15min' (15分钟) 为时间窗口进行重采样并统计记录数
    resampled_15min = peak_df.resample('15min').size()
    # 找出客流量最大的 15 分钟区间的刷卡量
    max_15min_vol = resampled_15min.max()
    # 找出该最大刷卡量对应的起始时间点
    max_15min_time = resampled_15min.idxmax()

    # 根据公式计算 PHF5：高峰小时总刷卡量 ÷ (12 × 最大5分钟刷卡量)
    phf5 = peak_hour_vol / (12 * max_5min_vol)
    # 根据公式计算 PHF15：高峰小时总刷卡量 ÷ (4 × 最大15分钟刷卡量)
    phf15 = peak_hour_vol / (4 * max_15min_vol)
    # ---------------------------------------------------------

    # 格式化输出时间段字符串 (如 08:15~08:20)
    time_str_5m = f"{max_5min_time.strftime('%H:%M')}~{(max_5min_time + pd.Timedelta(minutes=5)).strftime('%H:%M')}"
    time_str_15m = f"{max_15min_time.strftime('%H:%M')}~{(max_15min_time + pd.Timedelta(minutes=15)).strftime('%H:%M')}"

    print(f"最大5分钟刷卡量（{time_str_5m}）：{max_5min_vol} 次")
    print(f"PHF5  = {peak_hour_vol} / (12 × {max_5min_vol}) = {phf5:.4f}")
    print(f"最大15分钟刷卡量（{time_str_15m}）：{max_15min_vol} 次")
    print(f"PHF15 = {peak_hour_vol} / ( 4 × {max_15min_vol}) = {phf15:.4f}")


# ==========================================
# 任务 5：线路驾驶员信息批量导出
# ==========================================
def task5_export_drivers(df):
    print("\n=== 任务5：线路驾驶员信息批量导出 ===")

    folder_name = "线路驾驶员信息"
    # 如果文件夹不存在，则在根目录下创建
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    # 强转线路号为数字，过滤出 1101 至 1120 的记录
    df['线路号_num'] = pd.to_numeric(df['线路号'], errors='coerce')
    target_routes = list(range(1101, 1121))
    sub_df = df[df['线路号_num'].isin(target_routes)]

    generated_files = []

    for route in target_routes:
        # 筛选单条线路
        route_df = sub_df[sub_df['线路号_num'] == route]
        # 提取并去重
        driver_mapping = route_df[['车辆编号', '驾驶员编号']].drop_duplicates()

        # 写入 txt
        file_path = os.path.join(folder_name, f"{route}.txt")
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(f"线路号: {route}\n")
            f.write("车辆编号\t驾驶员编号\n")
            for _, row in driver_mapping.iterrows():
                f.write(f"{row['车辆编号']}\t{row['驾驶员编号']}\n")

        generated_files.append(file_path)

    print(f"成功生成了 {len(generated_files)} 个 txt 文件。路径示例：")
    for p in generated_files[:3]:
        print(" ->", p)
    print(" -> ...")


# ==========================================
# 任务 6：服务绩效排名与热力图
# ==========================================
def task6_performance_heatmap(df):
    print("\n=== 任务6：服务绩效排名与热力图 ===")

    # 仅统计有效上车刷卡记录作为乘客人次
    boarding_df = df[df['刷卡类型'] == 0]

    # 分别统计各维度的 Top 10 (value_counts 默认降序排列)
    top10_drivers = boarding_df['驾驶员编号'].value_counts().head(10)
    top10_routes = boarding_df['线路号'].value_counts().head(10)
    top10_stops = boarding_df['上车站点'].value_counts().head(10)
    top10_vehicles = boarding_df['车辆编号'].value_counts().head(10)

    print("Top 10 司机:", top10_drivers.index.tolist())
    print("Top 10 线路:", top10_routes.index.tolist())
    print("Top 10 站点:", top10_stops.index.tolist())
    print("Top 10 车辆:", top10_vehicles.index.tolist())

    # 构造 4 x 10 的 DataFrame 用于画热力图
    heatmap_data = pd.DataFrame([
        top10_drivers.values,
        top10_routes.values,
        top10_stops.values,
        top10_vehicles.values
    ], index=['司机', '线路', '上车站点', '车辆'], columns=[f"Top{i}" for i in range(1, 11)])

    plt.figure(figsize=(12, 4))

    # 绘制热力图 (annot=True 标注数值，fmt="d" 确保是整数，cmap="YlOrRd" 黄橙红渐变)
    sns.heatmap(heatmap_data, annot=True, fmt="d", cmap="YlOrRd")

    plt.title('公交服务绩效排名热力图 (按乘客人次)', fontsize=15)
    plt.suptitle('反映最繁忙的司机、线路、站点及车辆Top10客流分布', fontsize=10, color='gray')
    plt.xlabel('排名标签', fontsize=12)
    plt.ylabel('服务维度', fontsize=12)

    # x 轴标签旋转 0 度
    plt.xticks(rotation=0)

    # bbox_inches='tight' 保证边缘完整不被裁剪
    plt.tight_layout()
    plt.savefig('performance_heatmap.png', dpi=150, bbox_inches='tight')
    print("图像已保存为 performance_heatmap.png")

    # --- 结论说明 (>= 50字) ---
    conclusion = """
    【结论说明】
    从上述服务绩效热力图可以清晰观察到客流在不同维度的极度不均衡分布规律：
    1. 在“线路”和“上车站点”维度，Top 1 与 Top 2 的客流量呈现断层式领先（深红色），远超第3名及以后，这说明城市公交客流高度集中在极少数核心干线和关键枢纽站点。
    2. 相比之下，“司机”和“车辆”的客流递减梯度相对平缓，说明公交公司内部的运力排班较为均衡，没有出现个别司机或车辆过度超负荷运转的极端情况。
    """
    print(conclusion)


# ==========================================
# 完整主程序入口
# ==========================================
if __name__ == "__main__":
    # 任务1
    df_clean = task1_preprocessing('ICData.csv')
    # 任务2
    task2_time_analysis(df_clean)
    # 任务3
    task3_visualization(df_clean)

    # 任务4
    task4_phf_calculation(df_clean)
    # 任务5
    task5_export_drivers(df_clean)
    # 任务6
    task6_performance_heatmap(df_clean)
    print("\n🎉 全部作业代码运行完毕！")