# LEO-GEO Power Control / LEO-GEO 频谱共享功率控制

> Multi-objective genetic-algorithm optimizer for 6-LEO satellite transmit power
> under LEO-GEO spectrum sharing: maximize throughput while minimizing total power,
> subject to GEO/LEO SINR and interference constraints. Built on STK simulation
> exports and pymoo (NSGA-II + GA).
>
> 基于遗传算法的 LEO-GEO 频谱共享多目标功率控制：在 GEO/LEO SINR 与干扰约束下，
> 优化 6 颗 LEO 卫星发射功率，最大化吞吐量并最小化总功耗。使用 STK 仿真数据与
> pymoo（NSGA-II + GA）实现。

---

## Highlights / 项目亮点

- **Dual optimization modes / 双优化模式**：NSGA-II multi-objective (throughput ↑,
  power ↓) vs. single-objective GA (throughput only) — side-by-side comparison.
- **STK data pipeline / STK 数据处理**：ingest GEO link, per-LEO interference, and
  satellite ephemeris CSVs; compute ENU geometry, ITU-R antenna gains, noise power.
- **Constraint-aware evaluation / 约束感知评估**：GEO SINR ≥ 15 dB, LEO SINR ≥ 5 dB,
  total interference ≤ −130 dBm — violated constraints incur penalty weights.
- **Rich visualization / 丰富可视化**：Pareto front, convergence curves, per-satellite
  power dynamics, interference/SINR time series, performance summary table.

## Architecture / 模块结构

| File | Role |
|---|---|
| `code/process.py` | STK CSV ingestion, interference geometry, parameter matrices. STK 数据处理。 |
| `code/optimizer.py` | `Evaluator`, `LEO3GEOProblem` (NSGA-II), `ThroughputOnly` (GA). 优化器。 |
| `code/constants.py` | Physical & GA hyper-parameters, SINR thresholds, penalty weights. 常量配置。 |
| `code/visualization.py` | Result plots (Pareto, convergence, time series, performance table). 可视化。 |
| `code/main.py` | End-to-end driver: process → optimize → visualize. 主程序。 |

```
STK CSVs (link / interference / satellite ephemeris)
    ↓  process.py
Per-timestep channel parameters (GEO SINR, LEO C/I, noise, interference)
    ↓  optimizer.py  (NSGA-II or GA, 6 decision vars = LEO tx power dBW)
Optimal power vectors per timestep
    ↓  visualization.py
figures/ — Pareto, convergence, power dynamics, SINR/interference curves
```

## Data layout / 数据目录

```
leo-geo-power-control/
├── code/           # Python source
├── link/           # GEO link CSV (STK export)
├── Interference/   # Per-LEO interference CSV
└── Satellite/      # LEO ephemeris CSVs (6 satellites)
    └── 15GHz_BPSK/ # Default frequency band (included)
```

The repo ships **15 GHz BPSK** sample data (~1.4 MB) so you can run immediately.
Other bands (12–18 GHz) are excluded — add your own STK exports following the same
naming convention.

## Quick start / 快速开始

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cd code
python main.py
```

By default `main.py` runs:
1. `plot_time_idx(idx=10)` — single-timestep Pareto + convergence plots.
2. `plot_time_all(T=1441)` — full 24-hour optimization + all comparison charts.

> First full run optimizes 1441 timesteps and caches results to `results_all.npz`
> (~20 MB, gitignored). Subsequent runs load the cache.

To test the optimizer alone (no STK data needed):

```bash
cd code && python optimizer.py
```

## Not included / 未包含项

| Excluded | Reason |
|---|---|
| `results_all.npz` (20 MB) | Optimization cache — regenerated on first run |
| `figures/` (7 MB) | Generated plot outputs |
| Other frequency bands (12–18 GHz) | Duplicate STK exports; add locally if needed |
| `3.7特征数据/` | Separate feature-engineering notebooks & `.npy` arrays |
| PDFs / Word docs | Client deliverables & reference papers |
| `LEO.zip` (22 MB) | Archive duplicate of full project folder |

## Tech stack / 技术栈

`Python 3.8+` · `NumPy` · `pandas` · `matplotlib` · `seaborn` · `pymoo` (NSGA-II / GA)

## License

[MIT](LICENSE)
