# paths.py —— 项目里所有文件夹的位置，统一在这里定义。
# 其他模块和 notebook 都从这里 import，不要再手写相对路径。
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]          # 项目根目录（src/ 的上一级）

# ---- 输入 ----
DATA_RAW = ROOT / "data" / "raw"                    # 原始数据（价格、成本表），不手动修改
DATA_PROCESSED = ROOT / "data" / "processed"        # 特征工程的输出
DATA_CACHE = ROOT / "data" / "cache"                # 缓存，不上传 git

# ---- 输出 ----
RESULTS = ROOT / "results"
RES_BENCH_REPLICATION = RESULTS / "benchmark" / "replication"   # benchmark_replication.ipynb
RES_BENCH_BAZ = RESULTS / "benchmark" / "baz"                   # benchmark_baz.ipynb
RES_TRAINING = RESULTS / "training"                 # train_mlp / train_lstm
RES_EVALUATION = RESULTS / "evaluation"             # model_compare / cost_eval / smooth_eval
RES_BACKTEST = RESULTS / "backtest"                 # 测试期回测

# ---- 实盘流程 ----
LIVE = ROOT / "live"

# 文件夹不存在时自动创建
for _d in (DATA_PROCESSED, DATA_CACHE, RES_BENCH_REPLICATION, RES_BENCH_BAZ,
           RES_TRAINING, RES_EVALUATION, RES_BACKTEST):
    _d.mkdir(parents=True, exist_ok=True)
