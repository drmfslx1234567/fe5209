# universe.py —— 资产池，全项目统一从这里读取。
#
# 20 只高流动性 ETF（成本表中的 T1 层：近 3 年日成交额中位数 >= 1 亿美元）
# + 新加坡海峡指数 ETF（ES3.SI，课程要求必须包含新加坡标的）。
# 与 data/raw/trading_cost_50_etfs_1.xlsx 第 2 个 sheet「Cost(高流动性20支)」一致。
#
# 筛选标准是流动性，与任何策略的回测表现无关。

UNIVERSE = [
    "SPY", "QQQ", "DIA", "MDY", "IWM",              # 美股大盘 / 风格
    "XLK", "XLV", "XLF", "XLE", "XLY", "XLP",       # 美股行业
    "XLI", "XLB", "XLU", "XLRE", "XLC",
    "GLD", "SLV",                                   # 贵金属
    "TMF",                                          # 3 倍做多美国长债
    "2800.HK",                                      # 香港
    "ES3.SI",                                       # 新加坡
]

COST_SHEET = "Cost(高流动性20支)"                    # 成本表中对应的 sheet 名
