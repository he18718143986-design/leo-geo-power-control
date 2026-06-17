# main.py

import numpy as np
from pathlib import Path
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.optimize import minimize

from constants import (
    OFF_SPRINGS, POP_SIZE, MAX_GEN, P_MIN_DBW, P_MAX_DBW,
    CROSSOVER_PROB, ETA_C, MUTATION_PROB, ETA_M,
    THROUGHPUT_WEIGHT, POWER_WEIGHT
    
)
from process import (
    data_process,
    calculate_each_leo_interference,
    calculate_geo_sinr_dbw,
    calculate_each_leo_sinr
)
from optimizer import Evaluator, LEO3GEOProblem, ThroughputOnly
from visualization import ResultVisualizer

def processed(T):
    """
    处理 STK 数据，返回包含所有必要参数的字典。
    """
    BASE_DIR = Path(__file__).resolve().parent.parent
    FREQ = 15  # GHz

    processed_data = data_process(
        T=T,
        geo_path=BASE_DIR/"link"/f"Link_{FREQ}GHz_BPSK.csv",
        interference_file=BASE_DIR/"Interference"/f"Interference_{FREQ}GHz_BPSK.csv",
        satellite_dir=BASE_DIR/"Satellite"/f"{FREQ}GHz_BPSK"
    )
    
    return processed_data

def run_optimizer(processed_data):
    T = len(processed_data["timestamps"])
    multi_results, single_results = [], []
    
    for idx in range(T):
        # 修正：LEO_C/I 直接传list，不做float转换
        current = {
            "LEO_Power_At_Rcvr_Input": processed_data["LEO_Power_At_Rcvr_Input"][idx].tolist(),
            "GEO_Bandwidth":    float(processed_data["GEO_Bandwidth"][idx]),
            "GEO_Signal_Power": float(processed_data["GEO_Signal_Power"][idx]),
            "Noise_Power_dB":   float(processed_data["Noise_Power_dB"][idx]),
            "GEO_SINR": processed_data["GEO_SINR"][idx],
            "LEO_C/I": processed_data["LEO_C/I"][idx].tolist()
        }
        evaluator = Evaluator()

        # ——— 多目标(NSGA-II) ———
        prob_m = LEO3GEOProblem(current, evaluator, P_MIN_DBW, P_MAX_DBW)
        alg_m = NSGA2(
            pop_size=POP_SIZE, n_offsprings=OFF_SPRINGS,
            crossover=SBX(prob=CROSSOVER_PROB, eta=ETA_C),
            mutation=PM(prob=MUTATION_PROB, eta=ETA_M),
            eliminate_duplicates=True,
            archive_type="eps",
            save_history=False        
        )
        res_m = minimize(prob_m, alg_m, ('n_gen', MAX_GEN), verbose=False, seed=42)

        # ——— 单目标(GA) ———
        prob_s = ThroughputOnly(current, evaluator, P_MIN_DBW, P_MAX_DBW)
        alg_s = GA(
            pop_size=POP_SIZE, n_offsprings=OFF_SPRINGS,
            crossover=SBX(prob=CROSSOVER_PROB, eta=ETA_C),
            mutation=PM(prob=MUTATION_PROB, eta=ETA_M),
            eliminate_duplicates=True,
            save_history=False       
        )
        res_s = minimize(prob_s, alg_s, ('n_gen', MAX_GEN), verbose=False, seed=42)

        multi_results.append(res_m)
        single_results.append(res_s)

    return multi_results, single_results

def plot_time_idx(idx, path_file=None):
    try:
        processed_data = processed(idx+1)

        current = {
            "LEO_Power_At_Rcvr_Input": processed_data["LEO_Power_At_Rcvr_Input"][idx].tolist(),
            "GEO_Bandwidth":    float(processed_data["GEO_Bandwidth"][idx]),
            "GEO_Signal_Power": float(processed_data["GEO_Signal_Power"][idx]),
            "Noise_Power_dB":   float(processed_data["Noise_Power_dB"][idx]),
            "GEO_SINR": processed_data["GEO_SINR"][idx],
            "LEO_C/I": processed_data["LEO_C/I"][idx].tolist()
        }
        evaluator = Evaluator()

        # ——— 多目标(NSGA-II) ———
        prob_m = LEO3GEOProblem(current, evaluator, P_MIN_DBW, P_MAX_DBW)
        alg_m = NSGA2(
            pop_size=POP_SIZE, n_offsprings=OFF_SPRINGS,
            crossover=SBX(prob=CROSSOVER_PROB, eta=ETA_C),
            mutation=PM(prob=MUTATION_PROB, eta=ETA_M),
            eliminate_duplicates=True,
            archive_type="eps",
            save_history=True        
        )
        res_multi = minimize(prob_m, alg_m, ('n_gen', MAX_GEN), verbose=False, seed=42)

        # ——— 单目标(GA) ———
        prob_s = ThroughputOnly(current, evaluator, P_MIN_DBW, P_MAX_DBW)
        alg_s = GA(
            pop_size=POP_SIZE, n_offsprings=OFF_SPRINGS,
            crossover=SBX(prob=CROSSOVER_PROB, eta=ETA_C),
            mutation=PM(prob=MUTATION_PROB, eta=ETA_M),
            eliminate_duplicates=True,
            save_history=True       
        )
        res_single = minimize(prob_s, alg_s, ('n_gen', MAX_GEN), verbose=False, seed=42)

        # （1）收敛历史（用真实吞吐量）
        histories = {}
        for label, res, eval_current in [
            ('多目标优化', res_multi, prob_m.evaluator),
            ('单目标优化', res_single, prob_s.evaluator)
        ]:
            gens = []
            avgf = []
            bestf= []
            for gen, alg in enumerate(res.history):
                if hasattr(alg, "pop") and alg.pop is not None:
                    X = alg.pop.get("X")
                    if X is not None and len(X) > 0:
                        # 重新计算真实吞吐量
                        thrputs = []
                        for x in X:
                            thrpt, _, _, _ = eval_current.evaluate(x, current, with_penalty=False)
                            thrputs.append(thrpt)
                        thrputs = np.array(thrputs)
                        gens.append(gen)
                        avgf.append(np.mean(thrputs))
                        bestf.append(np.max(thrputs))
            histories[label] = {
                "generation": gens,
                "avg_fitness": avgf,
                "best_fitness": bestf
            }

        # （2）帕累托前沿集（用真实吞吐量和总功率W）
        class Ind:
            def __init__(self, thrpt, power_w): self.fitness=type("F",(),{"values": (thrpt, power_w)})
        pareto_sets = {'多目标优化': [
            Ind(
                evaluator.evaluate(x, current, with_penalty=False)[0],
                -np.sum(10**(np.array(x, dtype=float)/10))  # 总发射功率(W)
            ) for x in res_multi.X
        ]}

        viz = ResultVisualizer(path_file)
        #print(histories)
        viz.plot_convergence(histories)
        #print(pareto_sets)
        viz.plot_enhanced_pareto(pareto_sets)

    except Exception as e:
        print(f"程序运行出错: {str(e)}")  # 移除exc_info参数

def plot_time_all(T, path_file=None):

    processed_data = processed(T+1)

    # 2) 优化结果缓存
    results_file = Path(__file__).resolve().parent.parent/"results_all.npz"
    if not results_file.exists():
        print("开始运行优化器...")
        multi_results, single_results = run_optimizer(processed_data)
        # 只存必要的 X 和 F
        multi_X = np.array([r.X for r in multi_results], dtype=object)
        multi_F = np.array([r.F for r in multi_results], dtype=object)
        single_X = np.array([r.X for r in single_results], dtype=object)
        single_F = np.array([r.F for r in single_results], dtype=object)
        np.savez(results_file, mX=multi_X, mF=multi_F, sX=single_X, sF=single_F)
    else:
        print("加载已存在的优化结果...")
        data = np.load(results_file, allow_pickle=True)
        #multi_F = data['mF']; multi_X = data['mX']
        #single_F = data['sF']; single_X = data['sX']
        #T = len(processed["timestamps"])
        multi_F = data['mF'][:T]; multi_X = data['mX'][:T]
        single_F = data['sF'][:T]; single_X = data['sX'][:T]
        # 重建简易结果对象，只保留 X、F
        class R: pass
        multi_results = []; single_results = []
        for X,F in zip(multi_X, multi_F):
            r = R(); r.X = X; r.F = F
            multi_results.append(r)
        for X,F in zip(single_X, single_F):
            r = R(); r.X = X; r.F = F
            single_results.append(r)

    viz = ResultVisualizer(path_file)

    # (4) 多目标方案：计算每步加权权后的最优解
    best_solutions = []
    best_throughputs = []
    for r in multi_results:
        pareto_F = r.F  # shape: (n_pareto, 2)
        throughputs = -pareto_F[:, 0]
        powers = pareto_F[:, 1]
        thr_norm = (throughputs - throughputs.min()) / (throughputs.max() - throughputs.min() + 1e-12)
        pow_norm = (powers - powers.min()) / (powers.max() - powers.min() + 1e-12)
        score = THROUGHPUT_WEIGHT * thr_norm + POWER_WEIGHT * (1 - pow_norm)
        idx_best = np.argmax(score)
        best_solutions.append(r.X[idx_best])
        best_throughputs.append(throughputs[idx_best])

    # (4) 功率动态：每步最优解
    power_history = {
        '多目标优化': np.vstack([np.array(x, dtype=float) for x in best_solutions]),
        '单目标优化': np.vstack([np.array(r.X, dtype=float) for r in single_results])
    }
    #print(f"多目标优化功率历史: {power_history['多目标优化']}")
    #print(f"单目标优化功率历史: {power_history['单目标优化']}")
    viz.plot_total_power_comparison(power_history)
    viz.plot_2_satellite_power_dynamics(power_history)
    viz.plot_6_satellite_power_dynamics(power_history)

    # (4) 吞吐对比：每步 best throughput（用真实吞吐量）
    def get_best_real_thrpt(r, evaluator, processed_data, idx, x_override=None):
        # 取最优解
        if x_override is not None:
            x = np.array(x_override, dtype=float)
        elif hasattr(r, "X"):
            if isinstance(r.X, np.ndarray) and r.X.ndim == 2:
                x = np.array(r.X[0], dtype=float)
            else:
                x = np.array(r.X, dtype=float)
        else:
            return np.nan
        current = {
            "LEO_Power_At_Rcvr_Input": processed_data["LEO_Power_At_Rcvr_Input"][idx].tolist(),
            "GEO_Bandwidth":    float(processed_data["GEO_Bandwidth"][idx]),
            "GEO_Signal_Power": float(processed_data["GEO_Signal_Power"][idx]),
            "Noise_Power_dB":   float(processed_data["Noise_Power_dB"][idx]),
            "GEO_SINR": processed_data["GEO_SINR"][idx],
            "LEO_C/I": processed_data["LEO_C/I"][idx].tolist()
        }
        thrpt, _, _, _ = evaluator.evaluate(x, current, with_penalty=False)
        return float(thrpt)

    evaluator = Evaluator()
    throughput_data = {
        '多目标优化': [ get_best_real_thrpt(r, evaluator, processed_data, idx, x_override=best_solutions[idx]) for idx, r in enumerate(multi_results) ],
        '单目标优化': [ get_best_real_thrpt(r, evaluator, processed_data, idx) for idx, r in enumerate(single_results) ]
    }

    #print(f"多目标优化吞吐量历史: {throughput_data['多目标优化']}")
    #print(f"单目标优化吞吐量历史: {throughput_data['单目标优化']}")
    viz.plot_throughput_comparison(throughput_data)

    # (5) 性能表：平均指标（用真实吞吐量）
    thr_m_list = throughput_data['多目标优化']
    pow_m_list = power_history['多目标优化']
    thr_s_list = throughput_data['单目标优化']
    pow_s_list = power_history['单目标优化']

    performance = {
        '多目标优化': {
            'throughput': np.mean(thr_m_list),
            'power':       np.mean(np.sum(10 ** (pow_m_list / 10), axis=1)),
            'efficiency':  np.mean(thr_m_list)/np.mean(np.sum(10 ** (pow_m_list / 10), axis=1))
        },
        '单目标优化': {
            'throughput': np.mean(thr_s_list),
            'power':       np.mean(np.sum(10 ** (pow_s_list / 10), axis=1)),
            'efficiency':  np.mean(thr_s_list)/np.mean(np.sum(10 ** (pow_s_list / 10), axis=1))
        }
    }

    viz.generate_performance_table(performance, save_path=True)

    # 4) 对比三种方案：原始、单目标、双目标
    scenarios = ["原始LEO功率", "单目标优化", "多目标优化"]
    results = {name: {"interference": [], "geo_sinr": [], "leo_sinrs_each": []}
               for name in scenarios}

    for t in range(T):
        raw_tx   = np.full(6, 15.0, dtype=float) # 原始 LEO 发射功率
        #geo_sinr = processed["GEO_SINR"][t]
        raw_intf = processed_data["LEO_Power_At_Rcvr_Input"][t]
        noise_pw = processed_data["Noise_Power_dB"][t]
        geo_pw   = processed_data["GEO_Signal_Power"][t]
        leo_ci = processed_data["LEO_C/I"][t]
        geo_ci = processed_data["GEO_SINR"][t]

        tx_raw = raw_tx
        tx_s = np.array(pow_s_list[t], dtype=float).flatten()
        tx_m = np.array(pow_m_list[t], dtype=float).flatten()

        for name, tx in zip(scenarios, [tx_raw, tx_s, tx_m]):
            #tx = np.array(tx, dtype=float) 
            leo_intfs= calculate_each_leo_interference(tx, raw_intf)
            total_leo_intf = np.array(leo_intfs).flatten()[0]
            geo_sinr = calculate_geo_sinr_dbw(geo_pw, noise_pw, raw_intf, tx, geo_ci)
            leo_sinrs = calculate_each_leo_sinr(tx, leo_ci)

            results[name]["interference"].append(total_leo_intf)
            results[name]["geo_sinr"].append(geo_sinr)
            results[name]["leo_sinrs_each"].append(leo_sinrs)

    #print("原始LEO功率 GEO SINR：", results["原始LEO功率"]["geo_sinr"][70:80])
    #print("单目标优化 GEO SINR：", results["单目标优化"]["geo_sinr"][70:80])
    #print("多目标优化 GEO SINR：", results["多目标优化"]["geo_sinr"][70:80])
   
    # 5) 将结果转换为 NumPy 数组，便于后续处理
    # 转为数组
    for name in scenarios:
        results[name]["interference"]   = np.array(results[name]["interference"])
        results[name]["geo_sinr"]       = np.array(results[name]["geo_sinr"])
        results[name]["leo_sinrs_each"] = np.vstack(results[name]["leo_sinrs_each"])

    time = np.arange(T)

    # 6) 绘制时序对比
    viz.plot_interference_time(time, results)
    viz.plot_geo_sinr_time(time, results)
    viz.plot_leo_sinr_time(time, results)

if __name__ == "__main__":
    try:

        # 获得第idx个时间步(idx<1441)的优化结果和历史收敛曲线和帕累托分布
        plot_time_idx(idx=10, path_file = {'paths': {'results': Path(__file__).resolve().parent.parent }}) 

        # 获得全部时间步优化结果和T个时间步长的其他所有图片(T<1441)
        plot_time_all(T=1441, path_file = {'paths': {'results': Path(__file__).resolve().parent.parent }})  

    except Exception as e:
        
        print(f"程序运行出错: {str(e)}")
