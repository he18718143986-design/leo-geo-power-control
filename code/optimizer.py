# optimizer.py

import numpy as np
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.operators.crossover.sbx import SBX  # 交叉操作符
from pymoo.operators.mutation.pm import PM 
from pymoo.optimize import minimize
from constants import (
    NUM_LEO, P_MIN_DBW, P_MAX_DBW, MAX_INTERFERENCE_DBM,
    PENALTY_LAMBDA_GEO, PENALTY_LAMBDA_LEO, PENALTY_LAMBDA_IF,
    SINR_LEO_THRESH_DB, SINR_GEO_THRESH_DB
)
from process import (
    data_process,
    calculate_each_leo_interference,
    calculate_geo_sinr_dbw,
    calculate_each_leo_sinr
)
class Evaluator:
    """
    说明:
        - current_data 必须包含 LEO_Xmtr_Gain, LEO_Rcvr_Gain, LEO_Range, LEO_Power_At_Rcvr_Input, GEO_Bandwidth, GEO_Signal_Power, Noise_Power_dB
    """
    def __init__(self):
        pass

    def calculate_constraints(self, geo_sinr, leo_sinrs, leo_interf):
        """计算约束违例量（硬约束）"""

        # GEO SINR 约束违例量（需 >= 阈值 dB）
        geo_violation = max(0, SINR_GEO_THRESH_DB - geo_sinr)
        
        # LEO SINR 约束违例量（需 >= 阈值 dB）
        leo_violation = np.sum(np.clip(SINR_LEO_THRESH_DB - leo_sinrs, 0, None))

        # 总干扰约束违例量（需 <= max_interference dBm）
        total_interf_lin = np.sum(10**(leo_interf/10))
        total_interf_dbw = 10 * np.log10(total_interf_lin + 1e-30)
        total_interf_dbm = total_interf_dbw + 30  # dBW → dBm
        interf_violation = max(0, total_interf_dbm - MAX_INTERFERENCE_DBM)
        
        return np.array([geo_violation, leo_violation, interf_violation])

    def evaluate(self, power_vec, current_data, with_penalty=False):
        """
        输入:
            - power_vec: shape=(NUM_LEO,)，每个LEO的发射功率(dBW)
            - current_data: dict，包含所有LEO和GEO参数
            - with_penalty: 是否返回减去 penalty 的吞吐量（优化用），默认 False
        输出:
            - total_thrpt_bps: GEO+LEO总吞吐量(Mbps)
            - total_power_linear: 总发射功率(W)
            - constraints: 约束违例数组
            - penalty: 罚分（仅用于调试/分析）
        """
        P = np.asarray(power_vec)
        reference_power_dbw = np.asarray(current_data["LEO_Power_At_Rcvr_Input"])
        bw = float(current_data["GEO_Bandwidth"]) * 1e6

        # 干扰功率
        LeoI = np.array([reference_power_dbw[i] + (P[i] - 15.0) for i in range(len(P))])  # 单颗线性缩放

        # GEO SINR
        geo_sinr = calculate_geo_sinr_dbw(
            current_data["GEO_Signal_Power"], 
            current_data["Noise_Power_dB"], 
            reference_power_dbw, 
            P,
            current_data["GEO_SINR"])

        # LEO SINR
        leo_sinrs = calculate_each_leo_sinr(P, current_data["LEO_C/I"])

        # 吞吐量
        #thrpt_geo_bps = bw * np.log2(1 + 10**(geo_sinr/10))
        throughputs = bw * np.log2(1 + 10**(np.array(leo_sinrs) / 10))
        avg_throughput = np.mean(throughputs)
        constraints = self.calculate_constraints(geo_sinr, leo_sinrs, LeoI)

        # 罚分
        penalty = (
            PENALTY_LAMBDA_GEO * constraints[0] +
            PENALTY_LAMBDA_LEO * constraints[1] +
            PENALTY_LAMBDA_IF  * constraints[2]
        )

        #total_thrpt_bps = (thrpt_geo_bps + avg_throughput) / 1e6
        total_thrpt_bps = avg_throughput / 1e6
        if with_penalty:
            total_thrpt_bps = total_thrpt_bps - penalty

        total_power_linear = np.sum(10**(P/10))
        #total_power_linear = 10 * np.log10(np.sum(10**(P/10)) + 1e-30)
        
        return total_thrpt_bps, total_power_linear, constraints, penalty

class LEO3GEOProblem(Problem):
    """
    输入:
        - current_data: dict，当前时刻的所有参数
        - evaluator: Evaluator实例
        - pmin, pmax: 功率上下限
    输出:
        - F: shape=(pop_size, 2)，目标值
        - G: shape=(pop_size, 3)，约束违例
    """
    def __init__(self, current_data, evaluator, pmin, pmax):
        super().__init__(
            n_var=NUM_LEO,
            n_obj=2,
            n_constr=3,
            xl=pmin * np.ones(NUM_LEO),  # 每个变量的下限（如pmin=-10，则每颗LEO的下限都是-10dBW）
            xu=pmax * np.ones(NUM_LEO)   # 每个变量的上限（如pmax=20，则每颗LEO的上限都是20dBW）
        )
        self.current_data = current_data
        self.evaluator = evaluator

    def _evaluate(self, X, out, *args, **kwargs):
        pop_size = X.shape[0]
        F = np.zeros((pop_size, 2))  # 目标函数矩阵
        G = np.zeros((pop_size, 3))  # 约束违例矩阵
        
        for i in range(pop_size):
            # 计算目标和约束
            total_thrpt_bps, total_power, constraints, _ = self.evaluator.evaluate(X[i], self.current_data, with_penalty=True)
            
            # 目标函数设置
            F[i, 0] = -total_thrpt_bps  # 最大化转换为最小化
            F[i, 1] = total_power           # 直接最小化总功率
            
            # 约束违例量（pymoo要求<=0为满足约束）
            G[i, 0] = -constraints[0]  # GEO SINR违例转为 <=0形式
            G[i, 1] = -constraints[1]  # LEO SINR违例
            G[i, 2] = -constraints[2]  # 总干扰违例
        
        out["F"] = F
        out["G"] = G

class ThroughputOnly(Problem):
    """
    输入:
        - current_data: dict，当前时刻的所有参数
        - evaluator: Evaluator实例
        - pmin, pmax: 功率上下限
    输出:
        - F: shape=(pop_size, 1)，目标值
        - G: shape=(pop_size, 3)，约束违例
    """
    def __init__(self, current_data, evaluator, pmin, pmax):
        super().__init__(
            n_var=NUM_LEO,
            n_obj=1,
            n_constr=3,
            xl=pmin * np.ones(NUM_LEO),
            xu=pmax * np.ones(NUM_LEO)
        )
        self.current_data = current_data
        self.evaluator = evaluator

    def _evaluate(self, X, out, *args, **kwargs):
        pop_size = X.shape[0]
        F = np.zeros((pop_size, 1))
        G = np.zeros((pop_size, 3))
        
        for i in range(pop_size):
            total_thrpt_bps, _, constraints, _ = self.evaluator.evaluate(X[i], self.current_data, with_penalty=True)
            F[i, 0] = -total_thrpt_bps  # 最大化转最小化
            
            G[i, 0] = -constraints[0]
            G[i, 1] = -constraints[1]
            G[i, 2] = -constraints[2]
        
        out["F"] = F
        out["G"] = G

if __name__ == "__main__":

    current_data = {
        "LEO_Power_At_Rcvr_Input": [-164.571, -240.17556484855987, -238.2903207175171, -241.4804534851404, -246.99248730249894, -242.73950600905755],  # 新增：参考干扰功率
        "LEO_C/I": [8.5229, 25.0, 25.0, 17.7343, 25.0, 25.0],
        "GEO_Bandwidth":   32.0,         # GEO链路带宽（MHz）
        "GEO_Signal_Power": -152.735,      # GEO信号功率（dBW）
        "GEO_SINR": 11.3179,  # GEO SINR（dB）
        "Noise_Power_dB":  -173.5497093527876         # 噪声功率（dB）
    }

    # ================== 配置评估器 ==================
    evaluator = Evaluator()

    # ================== 运行双目标优化 ==================
    problem_multi = LEO3GEOProblem(
        current_data,
        evaluator,
        pmin=P_MIN_DBW,
        pmax=P_MAX_DBW
    )
    algorithm_multi = NSGA2(
        pop_size=100,
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(prob=0.1, eta=20),
        eliminate_duplicates=True
    )
    res_multi = minimize(problem_multi, algorithm_multi, ('n_gen', 50), verbose=False)

    # 输出双目标结果
    print("\n双目标优化结果 (LEO3GEOProblem):")
    print(f"找到的Pareto解数量: {len(res_multi.X)}")
    print("最优功率配置示例 (dBW):")
    print(res_multi.X[0])  # 应为长度6的数组
    print("对应目标值 [吞吐量(bps), 总功率(W)]:")
    print(np.column_stack((-res_multi.F[:, 0], res_multi.F[:, 1]))[0])

    # 新增：输出真实吞吐量（不含 penalty）
    real_thrpt, _, _, _ = evaluator.evaluate(res_multi.X[0], current_data, with_penalty=False)
    print(f"真实吞吐量 (Mbps): {real_thrpt}")

    # ================== 运行单目标优化 ==================
    problem_single = ThroughputOnly(
        current_data,
        evaluator,
        pmin=P_MIN_DBW,
        pmax=P_MAX_DBW
    )
    algorithm_single = GA(
        pop_size=200,
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(prob=0.2, eta=20),
        eliminate_duplicates=True
    )
    res_single = minimize(problem_single, algorithm_single, ('n_gen', 100), verbose=False)

    # ================== 输出单目标结果 ==================
    print("\n单目标优化结果 (ThroughputOnly):")

    if res_single.X.ndim == 2:
        optimal_power = res_single.X[0]
    else:
        optimal_power = res_single.X

    print("最优功率配置 (dBW):", optimal_power)

    if res_single.F.ndim == 2:
        throughput = -res_single.F[0][0]
    else:
        throughput = -res_single.F[0]
    # 新增：输出真实吞吐量
    real_thrpt_single, _, _, _ = evaluator.evaluate(optimal_power, current_data, with_penalty=False)
    print(f"平均吞吐量 (优化目标): {throughput:.1f} bps")
    print(f"真实吞吐量 (Mbps): {real_thrpt_single}")