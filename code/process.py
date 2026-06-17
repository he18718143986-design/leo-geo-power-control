import pandas as pd
import numpy as np
from typing import Dict, List
import os
from pathlib import Path

def data_process(T: int, geo_path: str = None, interference_file: str = None, satellite_dir: str = None) -> Dict:

    """加载并处理STK仿真数据"""

    R = 6371.0  # 地球半径 (km)
    lat = np.deg2rad(5)  # 转换为弧度
    lon = np.deg2rad(10)  # 转换为弧度
    h = 0.0  # 高度 (km)
    
    # 计算地面站的ECEF坐标
    x_gs = (R + h) * np.cos(lat) * np.cos(lon)
    y_gs = (R + h) * np.cos(lat) * np.sin(lon)
    z_gs = (R + h) * np.sin(lat)

    # 列定义
    geo_columns = [
        "Time (UTCG)", "Xmtr Power (dBW)", "Xmtr Gain (dB)", "C/I (dB)",
        "Range (km)", "Rcvr Gain (dB)", "Carrier Power at Rcvr Input (dBW)",
        "C/(N+I) (dB)", "Bandwidth (MHz)", "Rcvd. Frequency (GHz)", "Tequiv (K)"
    ]
    
    # 加载GEO数据
    geo_data = pd.read_csv(geo_path, skiprows=6)
    geo_data.columns = geo_data.columns.str.strip()
    
    # 验证GEO列
    missing = set(geo_columns) - set(geo_data.columns)
    if missing:
        raise ValueError(f"GEO数据缺少必要列: {missing}")

    interference = pd.read_csv(interference_file, skiprows=7, on_bad_lines='skip')
    interference['Time'] = pd.to_datetime(interference['Time (UTCG)'], format="%d %b %Y %H:%M:%S.%f")
    interference['Time'] = interference['Time'].apply(lambda x: x.replace(second=0, microsecond=0).strftime("%d/%m/%Y %H:%M:%S"))
    interference['Time (UTCG)'] = pd.to_datetime(interference['Time (UTCG)'])

    interference_groups = interference.groupby('IF Source ID')
    interference_dict = {name: group for name, group in interference_groups}
    interference_data = {i: interference_dict.get(i) for i in range(6)}
    
    # 处理每个干扰源的数据
    interference_processed = {}
    for i in range(6):
        df = interference_data[i]

        # 拼接完整路径，只传一个参数给 read_csv
        sat_path = os.path.join(satellite_dir, f"LEOSAT{i+1}_Fixed_Position_Velocity.csv")
        satellite_data = pd.read_csv(sat_path)
        satellite_data = satellite_data[['Time (UTCG)', 'x (km)', 'y (km)', 'z (km)']]
        
        result = []
        for index, row in satellite_data.iterrows():
            time = row['Time (UTCG)']
            x_s = row['x (km)']
            y_s = row['y (km)']
            z_s = row['z (km)']
            
            # 计算相对位置向量
            dx = x_s - x_gs
            dy = y_s - y_gs
            dz = z_s - z_gs
            
            # 将相对位置向量转换为ENU坐标系
            t = np.array([
                [-np.sin(lon), np.cos(lon), 0],
                [-np.sin(lat)*np.cos(lon), -np.sin(lat)*np.sin(lon), np.cos(lat)],
                [np.cos(lat)*np.cos(lon), np.cos(lat)*np.sin(lon), np.sin(lat)]
            ])
            
            enu = np.dot(t, np.array([dx, dy, dz]))
            e, n, u = enu[0], enu[1], enu[2]
            range_km = np.sqrt(e**2 + n**2 + u**2)
            
            # 计算方位角
            azimuth = np.arctan2(e, n)
            azimuth_deg = np.rad2deg(azimuth)
            
            # 计算仰角
            elevation = np.arcsin(u / np.sqrt(e**2 + n**2 + u**2))
            elevation_deg = np.rad2deg(elevation) + np.random.normal(0, 0.1)

            # 计算自由空间损耗
            c = 3e8  # 光速 (m/s)
            f = 14 * 1e9  # 频率 (Hz)
            free_space_loss = 20 * np.log10((4 * np.pi * range_km * 1e3 * f) / c)

            def itu_r_s1428_rx_gain(phi_deg: float, theta_deg: float, theta_3db: float = 1.0) -> float:
                """改进版接收增益模型"""
                theta_off = np.arccos(np.cos(np.deg2rad(phi_deg)) * np.cos(np.deg2rad(theta_deg)))
                theta_off_deg = np.rad2deg(theta_off)
                if theta_off_deg <= theta_3db:
                    return 45 - 12 * (theta_off_deg/theta_3db)**2
                else:
                    return 45 - 12 - 30 * np.log10(theta_off_deg/theta_3db)

            def itu_r_s1528_tx_gain(theta_deg: float, D: float = 0.1) -> float:
                """改进版发射增益模型"""
                c = 3e8  # 光速
                wavelength = c / (15 * 1e9)
                max_gain = 10 * np.log10(4.5 * (D/wavelength)**2) 
                if theta_deg <= 2.0:
                    return max_gain
                else:
                    return max_gain - 35 * np.log10(theta_deg/2.0)

            # === 自动融合STK增益数据 ===
            # 优先从df（STK干扰表）查找当前时间的增益
            leo_tx_gain_stk = None
            geo_rx_gain_stk = None
            if df is not None and "Time (UTCG)" in df.columns:
                df_row = df[df["Time (UTCG)"] == time]
                if not df_row.empty:
                    if "Xmtr Gain (dB)" in df_row.columns:
                        leo_tx_gain_stk = df_row["Xmtr Gain (dB)"].values[0]
                    if "Rcvr Gain (dB)" in df_row.columns:
                        geo_rx_gain_stk = df_row["Rcvr Gain (dB)"].values[0]

            # 计算自定义模型增益
            leo_tx_gain_model = itu_r_s1528_tx_gain(90 - abs(elevation_deg))
            geo_rx_gain_model = itu_r_s1428_rx_gain(azimuth_deg, elevation_deg)

            # 自动选择增益：优先STK，其次主瓣修正，再用模型
            # 主瓣方向判据（更宽松，适合实际轨道）
            is_main_lobe = abs(elevation_deg) > 60

            # LEO发射增益
            if leo_tx_gain_stk is not None and not np.isnan(leo_tx_gain_stk):
                leo_tx_gain = leo_tx_gain_stk
            elif is_main_lobe:
                leo_tx_gain = 15.0
            else:
                leo_tx_gain = leo_tx_gain_model

            # GEO接收增益
            # STK干扰表通常没有Rcvr Gain (dB)，主瓣方向直接设为40dB
            if geo_rx_gain_stk is not None and not np.isnan(geo_rx_gain_stk):
                geo_rx_gain = geo_rx_gain_stk
            elif is_main_lobe:
                geo_rx_gain = 40.0
            else:
                geo_rx_gain = geo_rx_gain_model

            # === 强制主瓣修正：仅主瓣方向才赋值主瓣增益，其余为模型 ===
            # 仅在主瓣方向才赋值主瓣增益，旁瓣方向必须用模型输出
            # 这样可保证只有主瓣时增益为正值，其余为负值或小值

            # 干扰功率计算（dBW）
            interference_power = 15 + leo_tx_gain - free_space_loss + geo_rx_gain

            result.append({
                'Time (UTCG)': time,
                'Xmtr Power (dBW)': 15,
                'Xmtr Gain (dB)': leo_tx_gain,
                'Rcvr Gain (dB)': geo_rx_gain,
                'Range (km)': range_km,
                'Power At Rcvr Input (dBW)': interference_power,
                'Xmtr Azimuth - Phi (deg)': azimuth_deg,
                'Xmtr Elevation - Theta (deg)': elevation_deg,
                'Free Space Loss (dB)': free_space_loss,
                'C/I (dB)': 25.0
            })
        result_df = pd.DataFrame(result)

        result_df['Time (UTCG)'] = pd.to_datetime(result_df['Time (UTCG)'])
        full_time_range = pd.date_range(start='2024-08-10 00:00:00', end='2024-08-11 00:00:00', freq='T')
        full_time_df = pd.DataFrame(full_time_range, columns=['Time (UTCG)'])
        #合并卫星和干扰数据
        merged_df = pd.merge(full_time_df, df, on='Time (UTCG)', how='left')
        merged_df = pd.merge(merged_df, result_df, on='Time (UTCG)', how='left', suffixes=('', '_results'))

        merged_df['Xmtr Azimuth - Phi (deg)'].fillna(merged_df['Xmtr Azimuth - Phi (deg)_results'], inplace=True)
        merged_df['Xmtr Elevation - Theta (deg)'].fillna(merged_df['Xmtr Elevation - Theta (deg)_results'], inplace=True)
        merged_df['Range (km)'].fillna(merged_df['Range (km)_results'], inplace=True)
        merged_df['Free Space Loss (dB)'].fillna(merged_df['Free Space Loss (dB)_results'], inplace=True)
        merged_df['Power At Rcvr Input (dBW)'].fillna(merged_df['Power At Rcvr Input (dBW)_results'], inplace=True)
        merged_df['Xmtr Gain (dB)'].fillna(merged_df['Xmtr Gain (dB)_results'], inplace=True)
        merged_df['C/I (dB)'].fillna(merged_df['C/I (dB)_results'], inplace=True)
        merged_df['Xmtr Power (dBW)'].fillna(merged_df['Xmtr Power (dBW)_results'], inplace=True)
        merged_df['Rcvr Gain (dB)'].fillna(merged_df['Rcvr Gain (dB)_results'], inplace=True)
 
        selected_columns = ["Time (UTCG)", "Xmtr Power (dBW)", "Xmtr Gain (dB)", "Range (km)", "Power At Rcvr Input (dBW)", "C/I (dB)",
                            'Xmtr Azimuth - Phi (deg)', 'Xmtr Elevation - Theta (deg)', 'Free Space Loss (dB)', 'Rcvr Gain (dB)']

        interference_processed[i] = merged_df[selected_columns]

    # 横向合并所有干扰源数据
    leo_data = interference_processed[0]
    for i in range(1, 6):
        leo_data = pd.merge(leo_data, interference_processed[i], on='Time (UTCG)', how='left', suffixes=(f'_src{i-1}', f'_src{i}'))

    geo_data['Time (UTCG)'] = pd.to_datetime(geo_data['Time (UTCG)'], format="%d %b %Y %H:%M:%S.%f")

    # 合并GEO数据
    geo_merged = pd.merge(geo_data[geo_columns], leo_data, on='Time (UTCG)', how='left')

    # 将数据限制在前100个时间点
    geo_merged = geo_merged[:T]
    geo_merged['Noise Power (dB)'] = 10 * np.log10(1.38e-23 * geo_merged['Tequiv (K)'] * (geo_merged['Bandwidth (MHz)'] * 1e6) + 1e-30)

    # 构建参数矩阵
    leo_params = ["Xmtr Power (dBW)", "Xmtr Gain (dB)", "Range (km)", "Power At Rcvr Input (dBW)", "Rcvr Gain (dB)",
                  "C/I (dB)", 'Xmtr Azimuth - Phi (deg)', 'Xmtr Elevation - Theta (deg)', 'Free Space Loss (dB)']
    
    param_data = {}
    for param in leo_params:
        matrix = np.zeros((len(geo_merged), 6))
        for src in range(6):
            col_name = f"{param}_src{src}"
            if col_name in geo_merged.columns:
                matrix[:, src] = geo_merged[col_name].values
            else:
                matrix[:, src] = np.nan  # 处理可能的列缺失
        param_name = param.split('(')[0].strip().replace(' ', '_')
        param_data[f"LEO_{param_name}"] = matrix

    # 计算总干扰
    linear_interf = 10 ** (param_data["LEO_Power_At_Rcvr_Input"] / 10)
    total_interf = 10 * np.log10(linear_interf.sum(axis=1) + 1e-30)

    # 计算干扰功率 I (dBW)：
    C_dBW = geo_merged['Carrier Power at Rcvr Input (dBW)']
    CI_dB  = geo_merged['C/I (dB)']

    C_lin = 10 ** (C_dBW / 10)

    I_lin = C_lin / (10 ** (CI_dB / 10))
    geo_merged['Interference_Power_dBW'] = 10 * np.log10(I_lin + 1e-30)

    # 构建最终数据
    processed = {
        "timestamps": geo_merged['Time (UTCG)'].values.astype('datetime64[s]').astype(int), 
        "GEO_TxPower": geo_merged['Xmtr Power (dBW)'].values, # GEO发射机的功率，单位是dBW
        "GEO_TxGain": geo_merged['Xmtr Gain (dB)'].values, # GEO发射天线的增益，单位dB
        "GEO_RxGain": geo_merged['Rcvr Gain (dB)'].values, # GEO接收天线的增益，单位dB
        "GEO_Range": geo_merged['Range (km)'].values, # GEO卫星与地面站之间的距离，单位公里
        "GEO_Signal_Power": geo_merged['Carrier Power at Rcvr Input (dBW)'].values, # GEO信号在接收机输入端的载波功率，单位dBW
        "GEO_C/I_dB": geo_merged['C/I (dB)'].values, # GEO链路的载波与干扰比，单位dB
        "GEO_Interference_Power_dBW": geo_merged['Interference_Power_dBW'].values, # GEO链路的干扰功率，单位dBW
        "GEO_SINR": geo_merged['C/(N+I) (dB)'].values, # GEO链路的信号与干扰加噪声比，单位dB
        "GEO_Bandwidth": geo_merged['Bandwidth (MHz)'].values, # 带宽，单位MHz
        "GEO_Freq": geo_merged['Rcvd. Frequency (GHz)'].values, # 接收频率，单位GHz
        "GEO_Tequiv": geo_merged['Tequiv (K)'].values,
        "Noise_Power_dB": geo_merged['Noise Power (dB)'].values, # 噪声功率的dBW表示，考虑了等效噪声温度和带宽。
        **param_data,
        #"LEO_Xmtr_Power​": # 每个LEO的发射功率，单位dBW
        #"LEO_Xmtr_Gain": # 每个LEO的发射天线增益，单位dB
        #"​LEO_Range": # 每个LEO到GEO地面站的距离，单位公里
        #"​LEO_Power_At_Rcvr_Input": # 每个LEO在GEO接收机输入端的干扰功率，单位dBW，由发射功率、增益、路径损耗等计算得到。
        #"​LEO_C/I​": # 每个LEO的载波与干扰比
        #"​LEO_Xmtr_Azimuth_-_Phi": # LEO相对于地面站的方位角，单位度
        #"LEO_Xmtr_Elevation_-_Theta": # LEO相对于地面站的俯仰角，单位度
        #"LEO_Free_Space_Loss": # LEO的自由空间损耗，单位dB
        "Total_Interference": total_interf # 所有LEO干扰功率的总和，转换为dBW后的总干扰
    }
    
    return processed

def calculate_each_leo_interference(tx_powers: np.ndarray, reference_interfs: np.ndarray) -> np.ndarray:
    """
    输入:
        tx_powers: 新的 LEO 发射功率 (dBW), shape=(6,)
        reference_interfs: 原始干扰功率 (dBW)，对应 15 dBW 时每颗 LEO 产生的干扰, shape=(6,)
    输出:
        每个 LEO 的干扰功率 (dBW), shape=(6,)
    关系: 干扰功率随发射功率线性变化，基于已知 15 dBW 的参考值平移
    """
    # vectorized: 对应元素相加
    return reference_interfs + (tx_powers - 15.0)

def calculate_geo_sinr_dbw(
    geo_power_dbw: float,
    noise_power_dbw: float,
    reference_interfs: np.ndarray,
    tx_powers: np.ndarray,
    geo_sinr_ref: float
) -> float:
    """
    GEO SINR随tx_powers动态变化，基于仿真基准修正
    """
    # 理论SINR（15dBW基准）
    C_lin = 10 ** (geo_power_dbw / 10)
    N_lin = 10 ** (noise_power_dbw / 10)
    I_lin_15 = np.sum(10 ** (reference_interfs / 10))
    sinr_15_theory = 10 * np.log10(C_lin / (N_lin + I_lin_15 + 1e-30))

    # 理论SINR（当前功率）
    interfs_new = reference_interfs + (tx_powers - 15.0)
    I_lin_new = np.sum(10 ** (interfs_new / 10))
    sinr_new_theory = 10 * np.log10(C_lin / (N_lin + I_lin_new + 1e-30))

    # 修正：用仿真基准+理论变化量
    return geo_sinr_ref + (sinr_new_theory - sinr_15_theory)


def calculate_each_leo_sinr(
    tx_powers: np.ndarray,
    reference_ci: np.ndarray
) -> np.ndarray:
    """
    输入:
        tx_powers: 新的 LEO 发射功率 (dBW), shape=(6,)
        reference_ci: 对应于 reference_tx dBW 时测得的每颗 LEO C/N (dB), shape=(6,)
        reference_tx: 参考发射功率 (dBW)，默认 15 dBW
    输出:
        每颗 LEO 的新 C/N (dB), shape=(6,)
    """
    return reference_ci + (tx_powers - 15.0)


# 修改后的主程序部分
if __name__ == "__main__":
    # 加载数据
    T = 1441
    BASE_DIR = Path(__file__).resolve().parent.parent
    FREQ = 15

    # 1) 载入并处理 STK 数据
    processed_data = data_process(
        T=T,
        geo_path=BASE_DIR/"link"/f"Link_{FREQ}GHz_BPSK.csv",
        interference_file=BASE_DIR/"Interference"/f"Interference_{FREQ}GHz_BPSK.csv",
        satellite_dir=BASE_DIR/"Satellite"/f"{FREQ}GHz_BPSK"
    )

    tx_powers = np.full(6, 15.0, dtype=float)

    raw_geo_sinr_list = []
    new_geo_sinr_list = []
    time_idx_list = []
    diff_list = []
    raw_infer_list = []
    new_infer_list = []

    for time_idx in range(T):  # 24个时间步
        #tx = np.full(6, (15.0-time_idx), dtype=float)
        tx = np.full(6, 10.0, dtype=float)
        # 获取基础参数

        raw_geo_sinr = processed_data["GEO_SINR"][time_idx]  # 原始GEO SINR
        raw_leo_interf = processed_data["LEO_Power_At_Rcvr_Input"][time_idx]  # 原始LEO干扰功率
        raw_leo_ci = processed_data["LEO_C/I"][time_idx]  # 原始LEO C/I值
        geo_carrier = processed_data["GEO_Signal_Power"][time_idx] # -152.735
        noise_power = processed_data["Noise_Power_dB"][time_idx] # -173.5497093527876
        geo_bandwidth = processed_data['GEO_Bandwidth'][time_idx] # 32.0
        tx_gain_db = processed_data["LEO_Xmtr_Gain"][time_idx]  # 发射增益 #[1.1251 -27.54207625 -23.07956898 -37.2920074  -27.99932795 -19.13452101]
        rx_gain_db = processed_data["GEO_RxGain"][time_idx]  # GEO接收增益 数值为0
        path_loss_db = processed_data["LEO_Free_Space_Loss"][time_idx]#[178.7427 194.79093062 196.20428396 183.72598046 194.60913683 196.86529079]
        geo_total_infer = processed_data["GEO_Interference_Power_dBW"][time_idx]
        #print(geo_total_infer)
        # 1. 计算新干扰功率
        new_interfs = calculate_each_leo_interference(tx, processed_data["LEO_Power_At_Rcvr_Input"][time_idx])

        # 2. GEO SINR
        geo_sinr = calculate_geo_sinr_dbw(
            processed_data["GEO_Signal_Power"][time_idx], 
            processed_data["Noise_Power_dB"][time_idx], 
            processed_data["LEO_Power_At_Rcvr_Input"][time_idx],
            tx,
            processed_data["GEO_SINR"][time_idx]  # 使用原始GEO SINR作为参考
            )
        
        thrpt_geo_bps = 32 * 1e6 * np.log2(1 + 10**(geo_sinr/10))
        # 3. 每颗 LEO 的 SINR
        leo_sinrs = calculate_each_leo_sinr(tx, processed_data["LEO_C/I"][time_idx])
        

        # 输出结果
        #print(f"\n=== 时间步 {time_idx} ===")
        #print(f"原始GEO SINR: {raw_geo_sinr:.2f} dB → 新GEO SINR: {geo_sinr:.2f} dB")
        #print(f"新GEO吞吐量: {thrpt_geo_bps / 1e6:.2f} Mbps")
        #print(f"原始LEO→GEO干扰功率: {raw_leo_interf} dBW → 新LEO→GEO干扰功率: {new_interfs} dBW")
        #linear_total = np.sum(10**(raw_leo_interf / 10))
        #reverse_total = 10**(processed_data["GEO_Interference_Power_dBW"][time_idx] / 10)
        #print(f"linear sum of per-LEO I: {10*np.log10(linear_total):.2f} dBW")
        #print("总干扰功率 I (dBW) :", processed_data["GEO_Interference_Power_dBW"][time_idx])
        linear_total = np.sum(10**(raw_leo_interf / 10))
        geo_interf_dbw = processed_data["GEO_Interference_Power_dBW"][time_idx]
        infer_total = np.sum(10**(new_interfs / 10))

        #print(f"原始LEO C/I: {raw_leo_ci} dB → 新LEO SINR: {leo_sinrs} dB")
        throughputs = 32 * 1e6 * np.log2(1 + 10**(np.array(leo_sinrs) / 10))
        avg_throughput = np.mean(throughputs)
        #print(f"LEO平均吞吐量: {avg_throughput / 1e6:.2f} Mbps")
        #print("时间步", time_idx, "发射功率", tx, "计算GEO SINR", geo_sinr, "原始GEO SINR", raw_geo_sinr)

        raw_geo_sinr_list.append(raw_geo_sinr)
        new_geo_sinr_list.append(geo_sinr)
        time_idx_list.append(time_idx)
        raw_infer_list.append(linear_total)
        new_infer_list.append(infer_total)
        

    import matplotlib.pyplot as plt
    import matplotlib as mpl
    mpl.rcParams['font.sans-serif'] = ['Songti SC']
    mpl.rcParams['axes.unicode_minus'] = False
    # 可视化
    plt.figure(figsize=(18, 12))
    
    #plt.plot(time_idx_list, new_geo_sinr_list, label='5dbw/GEO SINR', marker='s')
    #plt.plot(time_idx_list, raw_geo_sinr_list, label='15dbw/GEO SINR')
    plt.plot(time_idx_list, new_infer_list, label='5dbw/GEO SINR', marker='s')
    plt.plot(time_idx_list, raw_infer_list, label='15dbw/GEO SINR')
    plt.xlabel('时间步')
    #plt.ylabel('GEO SINR (dB)')
    plt.ylabel('GEO干扰功率 (dBW)')
    plt.title('每个时间步的GEO SINR变化')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
