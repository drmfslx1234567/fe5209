# FE5209 小组项目：基于深度动量网络的 ETF 交易策略

本仓库在 21 只 ETF 上复现 Lim, Zohren & Roberts (2019) *Enhancing Time Series Momentum Strategies Using Deep Neural Networks* 的核心方法，并扩展了交易成本分析、仓位平滑，以及课程要求的自动更新流程。

- **模型**：MLP 和 LSTM 直接输出每个资产的仓位（[-1, 1]），以最大化 Sharpe 比率为训练目标。
- **框架**：时间序列动量的波动率缩放，每个资产的仓位乘以 `sigma_tgt / sigma_t`，组合在资产间等权。
- **基准**：Long-Only、Sgn(Returns)（Moskowitz et al. 2012）、MACD（Baz et al. 2015）。
- **评估**：walk-forward，5 年扩展窗口，所有结果均为样本外。

> 本项目所有数字都是历史回测结果，不代表未来表现。

### 资产池（2026-10-10 确定）

共 21 只，定义在 `src/universe.py`，全项目统一从这里读取：

| 类别 | 数量 | 标的 |
|---|---|---|
| 美股大盘和风格 | 5 | SPY、QQQ、DIA、MDY、IWM |
| 美股行业 | 11 | XLK、XLV、XLF、XLE、XLY、XLP、XLI、XLB、XLU、XLRE、XLC |
| 贵金属 | 2 | GLD、SLV |
| 美国长债（3 倍杠杆） | 1 | TMF |
| 香港 | 1 | 2800.HK |
| 新加坡 | 1 | ES3.SI |

筛选标准：前 20 只是成本表中流动性最高的一层（近 3 年日成交额中位数不低于 1 亿美元）；ES3.SI 是课程要求必须包含的新加坡标的。筛选只依据流动性，与任何策略的回测表现无关。

项目早期使用过约 50 只 ETF 的资产池，原始价格和成本表中仍保留这些标的的数据。

---

## 1. 当前进度

| 部分 | 状态 | 说明 |
|---|---|---|
| 原始数据 | 已完成，**需更新到测试期** | 价格和交易成本表在 `data/raw/`，目前只到 2026-09-02 |
| 特征工程 | 已完成 | `src/pipeline_data_processing.py`，输出需在本地运行生成 |
| 基准策略 | 已完成，待合并 | 目前有两套独立实现（见 4.2），计划合并后重新输出 |
| 模型训练 | 代码已完成，**需在 21 只上重新训练** | 此前的训练基于旧资产池 |
| 模型对比、成本、平滑分析 | 代码已完成 | 依赖训练结果，重新训练后重跑 |
| 诊断脚本 | **待上传** | `scripts/` 下的 5 个脚本 |
| 自动更新流程 | **待上传**，且尚未在真实数据上验证 | `live/` 下的 4 个脚本 |
| 测试期回测 | 未开始 | `notebook/05_backtest/` |
| 最终提交物 | 未开始 | `submission/` |

---

## 2. 目录结构

```
fe5209/
├── README.md
├── requirements.txt
├── .gitignore
│
├── docs/                    参考论文、课程项目要求
│
├── data/
│   ├── raw/                 原始数据，不手动修改
│   ├── processed/           特征工程的输出，由代码生成
│   └── cache/               缓存，不上传 git
│
├── src/                     被 import 的模块
│   ├── paths.py             所有文件夹位置的统一定义
│   ├── universe.py          资产池（21 只）
│   ├── neural_config.py     训练和回测的共享配置
│   ├── neural_data.py       数据读取、划分、数据集
│   ├── neural_models.py     MLP / LSTM 定义
│   ├── neural_loss.py       Sharpe 损失
│   ├── train_utils.py       训练循环和早停
│   ├── neural_search.py     随机搜索 + top-K + 多种子集成
│   ├── neural_backtest.py   预测、组合构建、绩效指标
│   └── pipeline_data_processing.py    特征工程
│
├── notebook/                按流程顺序编号
│   ├── 01_data/             （空，特征工程以脚本形式运行）
│   ├── 02_benchmark/        基准策略
│   ├── 03_training/         模型训练
│   ├── 04_evaluation/       模型对比、成本和平滑分析
│   └── 05_backtest/         【未开始】测试期回测
│
├── scripts/                 【待上传】命令行诊断脚本
├── live/                    【待上传】自动更新流程
│
├── results/                 所有运行产出，与 notebook 对应
│   ├── benchmark/replication/
│   ├── benchmark/baz/
│   ├── training/
│   ├── evaluation/
│   └── backtest/
│
└── submission/              【未开始】PPT、资产价格 CSV、绩效指标、分工说明
```

**路径约定。** 所有文件夹位置都定义在 `src/paths.py`，代码里不要手写相对路径。每个 notebook 的第一个代码单元会自动找到项目根目录并把 `src/` 加入 import 路径，所以 notebook 放在 `notebook/` 下任意子文件夹都能运行。修改 `src/` 里的模块后需要重启 notebook 内核。

---

## 3. 环境

- Python 3.10 及以上（开发时使用 3.11）

```
pip install -r requirements.txt
```

---

## 4. 运行流程

按下面的顺序运行。每一步的输出是后面步骤的输入。

| 步骤 | 运行 | 读取 | 输出到 |
|---|---|---|---|
| 1 | `python src/pipeline_data_processing.py`（在项目根目录下） | `data/raw/` | `data/processed/` |
| 2 | `03_training/train_mlp`、`train_lstm` | `data/processed/` | `results/training/` |
| 3 | `02_benchmark/benchmark_baz` | `data/processed/`、`results/training/` | `results/benchmark/baz/` |
| 4 | `04_evaluation/model_compare` | 第 2、3 步的输出 | `results/evaluation/` |
| 5 | `04_evaluation/` 下的其余 notebook | `data/`、`results/training/` | `results/evaluation/` |

`02_benchmark/benchmark_replication` 是独立的，只需要 `data/raw/`，可以随时运行。

第 3 步放在训练之后，是为了让基准和 LSTM 使用完全相同的样本外样本。没有训练结果时它也能运行，会退回到按日期切分。

### 4.1 训练前先做冒烟测试

正式训练很慢（尤其是 LSTM）。先把 `src/neural_config.py` 里的 `QUICK_TEST` 设为 `True` 跑一遍，确认流程不报错；再改回 `False`、重启内核，做正式训练。冒烟测试的结果没有意义。

### 4.2 两套基准实现

| | `benchmark_replication` | `benchmark_baz` |
|---|---|---|
| 数据来源 | 直接读原始价格（`ohlcv_data_adjusted.xlsx`） | 读特征工程后的 `model_ready_dataset.csv` |
| 交易日历 | 每个标的用自己的交易日历，最后对齐 | 沿用特征工程的处理 |
| 样本 | 自行确定 | 与 LSTM 的样本外样本对齐 |
| 交易成本 | 包含（逐资产成本和统一成本） | 不包含（成本在 `cost_eval*` 中另算） |
| 被谁使用 | 暂无 | `model_compare` |

两者计划合并。合并前，模型对比使用 `benchmark_baz` 的输出。

---

## 5. 文件说明

### 5.1 数据

| 文件 | 内容 |
|---|---|
| `data/raw/ohlcv_data_adjusted.xlsx` | 复权后的 OHLCV，每个标的一个 sheet |
| `data/raw/ohlcv_data_by_date_adjusted.xlsx` | 同样的数据按日期排列，特征工程读取其中的 `Close` sheet |
| `data/raw/trading_cost_50_etfs_1.xlsx` | 每个标的的单边交易成本（bp）。本项目使用其中的 `Cost(高流动性20支)` sheet（即 21 只资产池）；另有早期 50 只的列表和流动性分层说明 |
| `data/processed/full_feature_panel.csv`（本地生成） | 全部特征，含没有未来收益的行 |
| `data/processed/model_ready_dataset.csv`（本地生成） | 训练和回测实际使用的数据，约 11.3 万行 |

特征共 8 个：1 日、1 月、3 月、6 月、1 年的波动率标准化收益，以及时间尺度为 (8, 24)、(16, 48)、(32, 96) 的三个 MACD 信号。

### 5.2 配置（`src/neural_config.py`）

| 设置 | 含义 |
|---|---|
| `DATA_PATH` | `model_ready_dataset.csv` 的位置 |
| `FEATURES` | 8 个输入特征 |
| `TARGET_ANNUAL_VOL`、`TRADING_DAYS` | 15% 目标波动率，252 个交易日 |
| `MLP_LOOKBACK`、`LSTM_LOOKBACK` | 每个样本的历史长度：5 天和 63 天 |
| `QUICK_TEST` | `True` 为冒烟测试，`False` 为正式训练 |
| `VAL_FRACTION` | 每个训练块最后 10% 的日期作为验证集 |
| `SEARCH_SPACE` | 随机搜索范围：隐藏层大小、dropout、batch size、学习率、梯度裁剪、weight decay |
| `SEARCH_ITER`、`SEARCH_ITER_LSTM` | 每折的搜索次数（正式训练为 20 和 10） |
| `TOP_K`、`N_SEEDS` | 保留最好的 3 组配置，每组用 3 个随机种子 |
| `MAX_EPOCHS`、`PATIENCE` | 最多 100 轮，25 轮无改善则早停 |

### 5.3 Notebook

| Notebook | 作用 | 主要输出 |
|---|---|---|
| `02_benchmark/benchmark_replication` | 复刻论文的三个基准策略，含交易成本、换手率和单标的层面的检查 | `benchmark_results.xlsx`、毛收益和净收益序列、两张图 |
| `02_benchmark/benchmark_baz` | 在与神经网络相同的样本上计算三个基准 | `{longonly,sgn,macd}_oos_daily_returns.csv` |
| `03_training/train_mlp`、`train_lstm` | walk-forward 训练 | `*_oos_predictions.csv`、`*_oos_daily_returns.csv`、`*_search_log.csv` |
| `04_evaluation/model_compare` | 汇总表和净值曲线 | `neural_model_comparison.csv`、`neural_equity_curves.png` |
| `04_evaluation/cost_eval` | 统一成本 0–10 bp 下 Sharpe 的变化，含“静态仓位”诊断 | `cost_sweep_sharpe.csv` |
| `04_evaluation/cost_eval_assets` | 逐资产成本，成本倍数 0×、0.5×、1×、2× | `cost_assets_universe21.csv` |
| `04_evaluation/smooth_eval` | 事后仓位平滑（EWM 半衰期和不交易带）。参数在第 1–2 折上选，只在第 3–4 折上报告 | `*_smoothing_grid.csv` |

### 5.4 待上传的文件

以下文件在原 README 中有描述，但尚未上传到仓库。上传时请放到对应位置，并把文件读写路径改为使用 `src/paths.py`。

| 位置 | 文件 | 作用 |
|---|---|---|
| `notebook/04_evaluation/` | `class_eval.ipynb` | 按资产类别拆分结果 |
| `notebook/04_evaluation/` | `baz_strategy_transactioncost.ipynb` | 独立的 MACD 策略，带分市场的费率表 |
| `scripts/` | `sig_test.py` | 逐年 Sharpe、Sharpe 差异的 bootstrap 检验 |
| `scripts/` | `universe_check.py` | 资产间相关性、有效独立资产数 |
| `scripts/` | `group_share.py` | 某一组资产对收益、方差、换手的贡献 |
| `scripts/` | `risk_budget_table.py` | 不同目标波动率下的收益和风险 |
| `scripts/` | `competition_metrics.py` | 课程要求的绩效指标 |
| `live/` | `live_config.py`、`live_engine.py`、`final_fit.py`、`daily_update.py` | 自动更新流程 |
| `results/training/`、`results/benchmark/baz/` | 训练和基准的输出 csv | `model_compare` 和成本分析的输入 |

---

## 6. 定义和约定

- **仓位和权重**：模型输出 `X ∈ [-1, 1]`，带杠杆的权重为 `w = X × sigma_tgt / sigma_t`。允许做空和加杠杆。
- **组合收益**：每天对当日可用的资产等权平均；资产有足够历史后才进入组合。
- **Rescaled**：组合收益乘以 `15% / 组合自身的 60 日 EWM 波动率（滞后一天）`。
- **Sharpe 比率**：`均值 / 标准差 × sqrt(252)`，不减无风险利率，与论文一致。
- **下行偏差**：负的日收益的标准差，与常见定义略有不同。
- **交易成本**：`cost_t = Σ_i c_i × |w_i,t − w_i,t−1| / N_t`，`c_i` 为单边成本，即论文式 (35) 换成逐资产成本。每个资产的首日建仓不计成本，假设按收盘价成交。
- **Walk-forward**：每一折用此前全部数据训练，在随后 5 年测试。数据从 2002 年开始，共四折：2007–11、2012–16、2017–21、2022–26。

---

## 7. 已有结果

**以下数字全部基于早期约 50 只的资产池，不适用于当前的 21 只资产池，仅作记录。在 21 只上重新训练和评估后请替换。** 对应的结果文件大部分未上传。Sharpe 均为缩放到 15% 波动率之后的值。样本外约 19 年，Sharpe 的标准误约 0.25，所以 0.1 以内的差异不能和噪声区分。

**不计成本，旧资产池**（`neural_model_comparison.csv`）：

| 策略 | 年化收益 | Sharpe | 最大回撤 |
|---|---|---|---|
| LSTM | 10.0% | 0.64 | 29.0% |
| MLP | 10.1% | 0.63 | 35.8% |
| Long-Only | 7.6% | 0.49 | 37.6% |
| MACD | 5.9% | 0.38 | 37.2% |

- **收益来源**：按资产类别看，模型的增值几乎只来自杠杆/反向债券 ETF 加 `UDN`，且集中在约五个年份（2019、2020、2022、2023、2026）。在股票、商品、外汇上接近静态仓位或 Long-Only。
- **计入成本**（逐资产成本，1 倍，未平滑）：在全部 50 只标的上，神经网络亏损（LSTM −0.97，MLP −1.42），Long-Only 最好（0.46）。只取其中 21 只高流动性标的的仓位（模型未重新训练）时，Long-Only 0.81、LSTM 0.55、MLP 0.19，Long-Only 仍然第一。
- **平滑**：在第 3–4 折上（统一成本），平滑后 LSTM 的年换手从约 54 降到 1.65，不计成本的 Sharpe 从 0.71 变为 0.69；MLP 没有改善。

---

## 8. 局限

- 资产池以美股为主（21 只中的 16 只），彼此高度相关，有效独立资产数远小于论文的 88 个期货。
- 使用 ETF 收益，而不是论文的连续期货合约。
- 成本表只包含买卖价差的一半，没有佣金、融券费用和冲击成本；并且把当前的价差用于整个样本期，低估了早期成本。部分 15 bp 是按流动性分层假设的，不是观测值。
- **论文第 VI-A 节的换手率正则没有实现**，训练目标是不含成本的 Sharpe，用事后平滑代替。
- 平滑参数是在第 1–2 折上按统一 2 bp 成本选的，不是按逐资产成本。
- 没有实现论文中的循环状态 dropout。
- 自动更新流程只在合成数据上做过单元测试，尚未在真实数据上验证；其中的风控阈值是主观设定，没有历史验证。
- 根据测试期结果来选资产池、参数或策略，会使样本外评估失效；这些设置应在测试期开始前固定。

---

## 9. 课程要求

详见 `docs/Group Project.pdf`。

- 组合中所有标的须在 2026 年 8 月 1 日前已交易满一年。
- 需要能自动更新交易策略的模型和方法。
- 提交：项目展示，以及包含资产列表和每日收盘价的 CSV。
- 每位成员须说明自己的贡献。
- 测试期：2026 年 10 月至第 11 周最后一天。
- 建议报告的指标：年化收益、Sharpe、最大回撤、胜率、年化波动率、回撤次数、交易次数、平均持仓期、单笔交易盈利。

---

## 10. 分工

| 成员 | 负责内容 |
|---|---|
| （待填） | |
| （待填） | |
| （待填） | |
| （待填） | |

---

## 参考文献

- Lim, B., Zohren, S., & Roberts, S. (2019). Enhancing Time Series Momentum Strategies Using Deep Neural Networks. *The Journal of Financial Data Science*.
- Moskowitz, T. J., Ooi, Y. H., & Pedersen, L. H. (2012). Time Series Momentum. *Journal of Financial Economics*.
- Baz, J., Granger, N., Harvey, C. R., Le Roux, N., & Rattray, S. (2015). Dissecting Investment Strategies in the Cross Section and Time Series.
