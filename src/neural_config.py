# neural_config.py —— 所有脚本 import 的配置

from paths import DATA_PROCESSED

DATA_PATH = str(DATA_PROCESSED / "model_ready_dataset.csv")   # 你的 pipeline 输出

SEED = 42
TARGET_ANNUAL_VOL = 0.15               # 论文 sigma_tgt = 15%
TRADING_DAYS = 252

# 论文 8 个特征（与 pipeline_data_processing 输出列名一致）
FEATURES = [
    "norm_ret_1d", "norm_ret_1m", "norm_ret_3m", "norm_ret_6m", "norm_ret_1y",
    "macd_8_24", "macd_16_48", "macd_32_96",
]
MACD_PAIRS = [(8, 24), (16, 48), (32, 96)]

# 回看窗口
MLP_LOOKBACK = 5                       # 论文 tau = 5
LSTM_LOOKBACK = 63                     # 论文 LSTM 展开 63 步

# ---------------------------------------------------------------- 防过拟合 / 搜索设置 ---
# True : 冒烟测试（只为确认流程不报错，结果没有意义，几分钟）
# False: 正式训练（很慢，尤其是 LSTM）
QUICK_TEST = True

VAL_FRACTION = 0.10                    # 论文: 训练块最后 10% 作验证集（fold 1 的验证集很小、很噪）

# 随机搜索空间（取自论文 Exhibit 9；去掉了 lr=1/10/100、梯度范数 1e-4 这类不合理的极端值；
# weight_decay 是额外增加的正则项，论文没有）
SEARCH_SPACE = {
    "hidden_size":   [5, 10, 20, 40, 80],
    "dropout":       [0.1, 0.2, 0.3, 0.4, 0.5],
    "batch_size":    [512, 1024, 2048],
    "learning_rate": [1e-4, 1e-3, 1e-2],
    "max_grad_norm": [0.1, 1.0, 10.0],
    "weight_decay":  [0.0, 1e-5, 1e-4, 1e-3],
}

if QUICK_TEST:
    SEARCH_ITER = 2                    # MLP 每个 fold 随机搜索几组
    SEARCH_ITER_LSTM = 2               # LSTM 每个 fold 随机搜索几组（LSTM 慢，可以比 MLP 少）
    SEARCH_MAX_EPOCHS = 2
    SEARCH_PATIENCE = 2
    MAX_EPOCHS = 3                     # 最终模型最大轮数
    PATIENCE = 2
    TOP_K = 1                          # 取搜索中验证 Sharpe 最高的几组配置
    N_SEEDS = 2                        # 每组配置用几个随机种子
else:
    SEARCH_ITER = 20
    SEARCH_ITER_LSTM = 10
    SEARCH_MAX_EPOCHS = 30             # 搜索阶段缩短训练，省时间
    SEARCH_PATIENCE = 8
    MAX_EPOCHS = 100                   # 论文
    PATIENCE = 25                      # 论文
    TOP_K = 3
    N_SEEDS = 3
