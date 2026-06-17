# constants.py

# ———————————————— 物理常数 ————————————————

k_BOLTZMANN = 1.38e-23  # 玻尔兹曼常数 [J/K]
T_SYS       = 290       # 默认系统噪声温度 [K] （若 CSV 中无 Tequiv，可回退到此值）

# ———————————————— 星座常数 ————————————————

NUM_LEO     = 6         # LEO 卫星颗数（固定）
P_MIN_DBW   = 5.0       # LEO 发射功率下限 [dBW]
P_MAX_DBW   = 15.0      # LEO 发射功率上限 [dBW]

# ———————————————— 优化器参数 ————————————————

MAX_GEN        = 100    # 最大迭代代数
POP_SIZE       = 200    # 种群规模
OFF_SPRINGS    = 150    # 每代生成的子代数量
CROSSOVER_PROB = 0.9    # SBX 交叉概率
MUTATION_PROB  = 0.3    # 每基因变异概率
ETA_C          = 20.0   # SBX 分布指数
ETA_M          = 20.0   # 多项式变异分布指数

# ———————————————— SINR 与干扰阈值 ————————————————
SINR_LEO_THRESH_DB  = 5.0         # LEO 端 SINR 最低容忍 [dB]
SINR_GEO_THRESH_DB  = 15.0        # GEO 端 SINR 最低容忍 [dB]
MAX_INTERFERENCE_DBM = -130        # GEO 端最大可容忍总干扰 [dBm]

# ———————————————— 罚分权重 ————————————————
# 当 SINR < 阈值 时，每 1dB 的差额所施加的罚分权重
PENALTY_LAMBDA_GEO = 1e6           # GEO SINR 违例罚分
PENALTY_LAMBDA_LEO = 1            # LEO SINR 违例罚分
PENALTY_LAMBDA_IF  = 1e5            # 干扰超限罚分

# ———————————————— 帕累托加权参数 ————————————————
THROUGHPUT_WEIGHT = 0.9   # 吞吐量权重
POWER_WEIGHT      = 0.1   # 功率权重
