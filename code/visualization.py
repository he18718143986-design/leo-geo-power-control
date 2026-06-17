import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, List
from pathlib import Path
import logging
import matplotlib as mpl
mpl.rcParams['font.sans-serif'] = ['Songti SC']
mpl.rcParams['axes.unicode_minus'] = False

class ResultVisualizer:
    def __init__(self, config: Dict):
        if "paths" not in config or "results" not in config["paths"]:
            raise KeyError("配置缺少 'paths' 或 'results'")
        results_path = Path(config["paths"]["results"])
        self.output_dir = results_path / "figures"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.plot_config = {'figsize': (12, 8), 'dpi': 150, 'save_format': 'png'}
        self.color_palette = sns.color_palette("husl", 8)

    def plot_interference_time(self, time, results):
        """
        绘制 GEO 接收总干扰功率随时间变化曲线，
        并确保所有方案都能对齐到同一个 time 长度。
        """
        fig, ax = plt.subplots(
            figsize=self.plot_config['figsize'],
            dpi=self.plot_config['dpi']
        )

        linestyles = {
            "原始LEO功率": "-",
            "单目标优化": "--",
            "多目标优化": "-."
        }
        markers = {
            "原始LEO功率": "o",
            "单目标优化": "s",
            "多目标优化": "^"
        }

        T = len(time)
        x = np.asarray(time)

        for name, data in results.items():
            y = np.asarray(data["interference"], dtype=float)

            # 如果 y 长度不足，用 nan 填充到 T；如果过长，截断
            if len(y) < T:
                y = np.concatenate([y, np.full(T - len(y), np.nan)])
            elif len(y) > T:
                y = y[:T]

            ax.plot(
                x, y,
                label=name,
                linestyle=linestyles.get(name, "-"),
                marker=markers.get(name, None),
                markevery=max(1, T // 20),
                linewidth=1,
                alpha=0.8
            )

        # 阈值线
        ax.axhline(-160.0, color='r', ls='--', label='干扰功率阈值')

        ax.set_xlabel("时间步")
        ax.set_ylabel("GEO 接收总干扰功率 (dBW)")
        ax.set_title("GEO 接收总干扰功率随时间变化")
        ax.legend()
        ax.grid(True)

        plt.tight_layout()
        self._save_or_show(self.output_dir / "interference_time.png")
        plt.close(fig)


    def plot_geo_sinr_time(self, time: np.ndarray, results: Dict[str, Dict]):
        """
        绘制 GEO SINR 随时间变化
        results[name]['geo_sinr'] 为 shape=(T,) 的数组，单位 dB
        """
        fig, ax = plt.subplots(figsize=self.plot_config['figsize'], dpi=self.plot_config['dpi'])
        # 使用不同类型的虚线
        style_map = {
            "原始LEO功率": dict(color="#1f77b4", linestyle="--", linewidth=1.5, dashes=(5, 2), alpha=0.9),
            "单目标优化": dict(color="#ff7f0e", linestyle=":", linewidth=1.5, dashes=(2, 2), alpha=0.9),
            "多目标优化": dict(color="#2ca02c", linestyle="dashdot", linewidth=1.5, dashes=(3, 5, 1, 5), alpha=0.9),
        }
        for name, data in results.items():
            style = style_map.get(name, dict(linestyle="--", linewidth=1.5))
            ax.plot(
                time, data["geo_sinr"], 
                label=name, 
                **style
            )
        ax.axhline(15.0, color='r', ls='--', linewidth=2, label='GEO SINR 阈值')
        ax.set_title("GEO SINR 随时间变化 (dB)")
        ax.set_xlabel("时间步")
        ax.set_ylabel("GEO SINR (dB)")
        ax.legend(fontsize='medium')
        ax.grid(True, linestyle=':', alpha=0.7)
        self._save_or_show(self.output_dir / f"geo_sinr_time.{self.plot_config['save_format']}")
        plt.close(fig)

    def plot_leo_sinr_time(self, time: np.ndarray, results: Dict[str, Dict]):
        """
        绘制每颗 LEO 的 SINR 随时间变化，6 个子图按两列三行排列
        results[name]['leo_sinrs_each'] 为 shape=(T,6) 的数组
        """
        n_leos = 6
        # 两列三行
        rows, cols = 3, 2
        fig, axes = plt.subplots(
            rows, cols,
            figsize=(self.plot_config['figsize'][0], self.plot_config['figsize'][1] * rows / 2),
            dpi=self.plot_config['dpi'],
            sharex=True
        )
        axes = axes.flatten()
        markers = ['o','v','s','^','<','>']

        for name, data in results.items():
            leo_sinrs = data["leo_sinrs_each"]  # (T,6)
            for i in range(n_leos):
                ax = axes[i]
                ax.plot(
                    time, leo_sinrs[:, i],
                    label=name,
                    linestyle='-' if name != "恒定 5 dBW" else '--',
                    linewidth =1,
                    marker=markers[i],
                    markevery=max(1, len(time)//20)
                )
        # 在每个子图上加阈值线和网格、标签
        for i, ax in enumerate(axes[:n_leos]):
            ax.axhline(5.0, color='r', ls='--', label='阈值 5 dB')
            ax.set_ylabel(f"LEO{i+1} SINR (dB)")
            ax.grid(True)
            # 只在第一个子图显示图例
            if i == 0:
                ax.legend(fontsize='small', loc='upper right')

        # 多余的空子图隐藏
        for j in range(n_leos, len(axes)):
            fig.delaxes(axes[j])

        # 统一 X 轴标签
        axes[-1].set_xlabel("时间步")

        plt.tight_layout()
        self._save_or_show(self.output_dir / f"leo_sinr_time.{self.plot_config['save_format']}")
        plt.close(fig)

    def plot_convergence(self, histories: Dict):
        """多方案收敛曲线对比"""
        # Validate input structure
        if not isinstance(histories, dict) or not all(isinstance(h, dict) for h in histories.values()):
            raise TypeError("`histories` must be a dictionary where each value is a dictionary containing 'generation', 'avg_fitness', and 'best_fitness'.")

        fig, ax = plt.subplots(figsize=self.plot_config['figsize'], dpi=self.plot_config['dpi'])
        
        colors = {'多目标优化': "#0a0db5ff", '单目标优化': "#f29303"}
        
        for label, history in histories.items():
            generations = history['generation']
            avg_fitness = history['avg_fitness']
            best_fitness = history['best_fitness']
            
            ax.plot(generations, best_fitness, label=f'{label}-最佳', 
                    color=colors[label], linestyle='-', linewidth=2)
            ax.plot(generations, avg_fitness, label=f'{label}-平均',
                    color=colors[label], linestyle='--', alpha=0.7)
        
        ax.set_title("多方案收敛特性对比")
        ax.set_xlabel("迭代次数")
        ax.set_ylabel("适应度值")
        ax.legend(loc='lower right')
        self._save_or_show(self.output_dir / f"convergence_comparison.png")
        plt.close(fig)  # 确保关闭
        return fig

    def plot_enhanced_pareto(self, pareto_sets: Dict):
        """增强型帕累托前沿对比"""
        if not pareto_sets:
            logging.warning("帕累托前沿数据为空")
            return
        for label, front in pareto_sets.items():
            if not all(hasattr(ind, 'fitness') for ind in front):
                invalid_inds = [ind for ind in front if not hasattr(ind, 'fitness')]
                logging.error(f"帕累托前沿 '{label}' 中的解缺少 'fitness' 属性: {invalid_inds}")
                raise ValueError(f"帕累托前沿 '{label}' 中的解缺少 'fitness' 属性: {invalid_inds}")

        fig, ax = plt.subplots(figsize=self.plot_config['figsize'], dpi=self.plot_config['dpi'])
        
        markers = {'多目标优化': 'o', '单目标优化': 's'}
        colors = {'多目标优化': "#0a0db5ff", '单目标优化': "#f29303"}
        
        for label, front in pareto_sets.items():
            throughputs = [ind.fitness.values[0] for ind in front]
            powers = [-ind.fitness.values[1] for ind in front]
            
            ax.scatter(throughputs, powers, 
                       marker=markers.get(label, 'o'), 
                       edgecolors=colors.get(label, "#333"),
                       facecolors='none',
                       s=80,
                       label=label)
        
        ax.set_title("多方案帕累托前沿对比")
        ax.set_xlabel("系统吞吐量 (Mbps)")
        ax.set_ylabel("总发射功率 (W)")
        ax.legend()
        self._save_or_show(self.output_dir / "enhanced_pareto_comparison.png")
        plt.close(fig)  # 确保关闭
        return fig

    def plot_total_power_comparison(self, power_dict: Dict[str, np.ndarray]) -> plt.Figure:
        """
        绘制不同优化方案下的总功率对比曲线
        参数：
            power_dict: 例如 {'多目标功率': ndarray, '单目标功率': ndarray}
        """
        fig, ax = plt.subplots(figsize=self.plot_config['figsize'], dpi=self.plot_config['dpi'])

        for label, power_history in power_dict.items():
            if not isinstance(power_history, np.ndarray):
                raise TypeError(f"{label} 的功率数据不是 numpy.ndarray 类型")

            total_power = np.sum(10 ** (power_history / 10), axis=1)
            linestyle = '--' if '单目标' in label else '-'
            ax.plot(np.arange(len(total_power)), total_power, label=label, linestyle=linestyle, linewidth=1)

        ax.set_title("系统总发射功率对比")
        ax.set_xlabel("时间步")
        ax.set_ylabel("总发射功率 (W)")
        ax.legend()
        self._save_or_show(self.output_dir / f"total_power_comparison.{self.plot_config['save_format']}")
        return fig

    def plot_6_satellite_power_dynamics(self, power_dict: Dict[str, np.ndarray]) -> plt.Figure:
        """
        绘制每个 LEO 卫星在不同优化方案下的功率动态变化
        参数：
            power_dict: 例如 {'多目标功率': ndarray, '单目标功率': ndarray}
        """
        NUM_LEO = next(iter(power_dict.values())).shape[1] if power_dict else 0
        # 两列三行布局
        ncols = 2
        nrows = int(np.ceil(NUM_LEO / ncols))
        fig, axs = plt.subplots(nrows, ncols, figsize=(12, 3 * nrows), dpi=self.plot_config['dpi'], sharex=True)
        axs = axs.flatten() if NUM_LEO > 1 else [axs]

        for sat_idx in range(NUM_LEO):
            ax = axs[sat_idx]
            for label, power_history in power_dict.items():
                linestyle = '--' if '单目标' in label else '-'
                ax.plot(np.arange(power_history.shape[0]), power_history[:, sat_idx], label=label, linestyle=linestyle, linewidth=1)
            ax.set_ylabel(f"LEO-{sat_idx+1} 发射功率 (dBW)")
            ax.legend()
        # 隐藏多余子图
        for i in range(NUM_LEO, len(axs)):
            fig.delaxes(axs[i])
        axs[-1].set_xlabel("时间步")
        fig.suptitle("各LEO卫星功率动态对比")
        self._save_or_show(self.output_dir / f"satellite_power_dynamics.{self.plot_config['save_format']}")
        return fig

    def plot_2_satellite_power_dynamics(self, power_dict: Dict[str, np.ndarray]) -> List[plt.Figure]:
        """
        分别绘制多目标和单目标下6颗LEO卫星功率动态（每种方案一张图，每张图6条曲线）
        参数：
            power_dict: 例如 {'多目标优化': ndarray, '单目标优化': ndarray}
        返回：
            figures: [多目标图, 单目标图]
        """
        figures = []
        for label in ['多目标优化', '单目标优化']:
            if label not in power_dict:
                continue
            power_history = power_dict[label]  # shape: (T, 6)
            NUM_LEO = power_history.shape[1]
            fig, ax = plt.subplots(figsize=(12, 6), dpi=self.plot_config['dpi'])
            for sat_idx in range(NUM_LEO):
                ax.plot(
                    np.arange(power_history.shape[0]),
                    power_history[:, sat_idx],
                    label=f"LEO-{sat_idx+1}",
                    linewidth=1
                )
            ax.set_xlabel("时间步")
            ax.set_ylabel("发射功率 (dBW)")
            ax.set_title(f"{label}下6颗LEO卫星功率动态")
            ax.legend()
            self._save_or_show(self.output_dir / f"satellite_power_dynamics_{'multi' if label == '多目标优化' else 'single'}.{self.plot_config['save_format']}")
            figures.append(fig)
        return figures

    def plot_throughput_comparison(self, throughput_data: Dict) -> plt.Figure:

        fig, ax = plt.subplots(figsize=self.plot_config['figsize'], dpi=self.plot_config['dpi'])
        
        # 绘制每个方案的吞吐量曲线
        for label, data in throughput_data.items():
            linestyle = '--' if '单目标' in label else '-'
            sns.lineplot(x=np.arange(len(data)), y=data, label=label, linestyle=linestyle, ax=ax,)
        
        ax.set_title("系统吞吐量性能对比")
        ax.set_xlabel("时间步")
        ax.set_ylabel("平均吞吐量 (Mbps)")
        ax.legend()
        
        self._save_or_show(self.output_dir / f"throughput_comparison.{self.plot_config['save_format']}")
        return fig

    def generate_performance_table(self, data: Dict, save_path: str = None):

        # 确保每个方案的键值正确
        required_keys = ['throughput', 'power', 'efficiency']
        for scheme in data.values():
            for k in required_keys:
                if k not in scheme:
                    raise KeyError(f"性能数据缺少必要字段 '{k}'")

        df = pd.DataFrame.from_dict(data, orient='index')
        df.columns = ['平均吞吐量 (Mbps)', '总功率 (W)', '功率效率 (Mbps/W)']
        
        # 设置显示格式
        styler = df.style.format({
            '平均吞吐量 (Mbps)': '{:.1f}',
            '总功率 (W)': '{:.1f}',
            '功率效率 (Mbps/W)': '{:.2f}'
        }).highlight_max(axis=0, color='#FFE08C')
        
        # 保存为图片
        if save_path:
            fig, ax = plt.subplots(figsize=(8, 3), dpi=150)
            ax.axis('off')
            ax.table(cellText=df.values.round(2),
                    rowLabels=df.index,
                    colLabels=df.columns,
                    cellLoc='center',
                    loc='center',
                    bbox=[0, 0, 1, 1])
            self._save_or_show(self.output_dir / "performance_table.png")
        
        return styler

    def _save_or_show(self, save_path: str = None):
        """统一处理图片保存或显示"""
        try:
            plt.tight_layout(pad=3.0, h_pad=2.0, w_pad=2.0)  # 增加水平和垂直边距
        except Exception as e:
            print(f"警告: {e}")
            print("尝试调整图形大小以适应布局...")
            fig = plt.gcf()
            fig.set_size_inches(fig.get_size_inches()[0] + 2, fig.get_size_inches()[1] + 1)  # 增加图形宽度和高度
            plt.tight_layout(pad=3.0, h_pad=2.0, w_pad=2.0)  # 再次尝试应用布局调整

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)  # 确保目录存在
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

# 修改示例用法部分 -----------------------------------------------------
if __name__ == "__main__":
    # 初始化可视化器
    visualizer = ResultVisualizer()
    
    # 生成符合多目标结构的模拟数据
    convergence_data = {
        '多目标优化': {
            'generation': list(range(100)),
            'avg_fitness': [80 + 10 * np.log(i + 1) for i in range(100)],
            'best_fitness': [90 + 5 * np.sqrt(i) for i in range(100)]
        },
        '单目标优化': {
            'generation': list(range(100)),
            'avg_fitness': [75 + 8 * np.log(i + 1) for i in range(100)],
            'best_fitness': [85 + 4 * np.sqrt(i) for i in range(100)]
        }
    }
    
    # 绘制收敛曲线
    visualizer.plot_convergence(convergence_data)
    
    # 生成帕累托前沿数据
    class MockIndividual:
        def __init__(self, values):
            self.fitness = type('Fitness', (), {'values': values})
    
    pareto_front = [
        MockIndividual([380 + np.random.rand() * 20, 80 + np.random.rand() * 20]) 
        for _ in range(30)
    ]
    visualizer.plot_enhanced_pareto({'多目标优化': pareto_front}, ref_point=(400, 100))
    
    # 生成功率动态数据
    power_history = np.array([
        15 + 5 * np.sin(np.linspace(0, 4 * np.pi, 100)) + np.random.normal(0, 1, 100)
        for _ in range(6)
    ]).T
    visualizer.plot_power_dynamics(power_history)
    
    # 生成性能对比数据
    performance_data = {
        '多目标优化': {'throughput': 386.2, 'power': 87.3, 'efficiency': 4.42},
        '单目标优化': {'throughput': 402.1, 'power': 123.5, 'efficiency': 3.26}
    }
    visualizer.generate_performance_table(performance_data, save_path=True)
