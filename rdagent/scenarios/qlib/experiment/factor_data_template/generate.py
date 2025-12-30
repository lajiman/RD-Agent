import qlib

qlib.init(provider_uri="~/.qlib/qlib_data/soy_continuous", region="us")

from qlib.data import D

from rdagent.scenarios.qlib.experiment.factor_data_template.enrich_daily_py import enrich_daily_pv_h5

'''
无论是修改初级因子，还是修改高阶因子，都需要重新生成 daily_pv_all.h5 和 daily_pv_debug.h5 文件：
cd RD-Agent/rdagent/scenarios/qlib/experiment/factor_data_template
python generate.py
'''

instruments = ["HP_CME_SOY"]
# 当修改了原始数据文件（csv）的初级因子时，需要相应的修改fields列表。所谓初级因子，是指直接从原始数据文件中读取的因子，比如开盘价、收盘价、成交量等。
fields = ["$open", "$close", "$high", "$low", "$volume", "$factor", "$open_interest", "$year", "$month", "$dayofweek", "$dayofyear"]
data = D.features(instruments, fields, freq="day").swaplevel().sort_index().loc["2002-01-01":].sort_index()

data.to_hdf("./daily_pv_all.h5", key="data")
# 顾名思义，enrich_daily_pv_h5 会为每个 instrument 添加额外的“高级因子”。所谓高阶因子，是指基于原始因子计算得到的衍生因子，比如移动平均、动量指标等。
enrich_daily_pv_h5("./daily_pv_all.h5", key="data")

fields = ["$open", "$close", "$high", "$low", "$volume", "$factor", "$open_interest", "$year", "$month", "$dayofweek", "$dayofyear"]
data = (
    (
        D.features(instruments, fields, start_time="2018-01-01", end_time="2019-12-31", freq="day")
        .swaplevel()
        .sort_index()
    )
    .swaplevel()
    .loc[data.reset_index()["instrument"].unique()[:100]]
    .swaplevel()
    .sort_index()
)

data.to_hdf("./daily_pv_debug.h5", key="data")
enrich_daily_pv_h5("./daily_pv_debug.h5", key="data")



# import qlib

# qlib.init(provider_uri="~/.qlib/qlib_data/cn_data")

# from qlib.data import D

# instruments = D.instruments()
# fields = ["$open", "$close", "$high", "$low", "$volume", "$factor"]
# data = D.features(instruments, fields, freq="day").swaplevel().sort_index().loc["2008-12-29":].sort_index()

# data.to_hdf("./daily_pv_all.h5", key="data")


# fields = ["$open", "$close", "$high", "$low", "$volume", "$factor"]
# data = (
#     (
#         D.features(instruments, fields, start_time="2018-01-01", end_time="2019-12-31", freq="day")
#         .swaplevel()
#         .sort_index()
#     )
#     .swaplevel()
#     .loc[data.reset_index()["instrument"].unique()[:100]]
#     .swaplevel()
#     .sort_index()
# )

# data.to_hdf("./daily_pv_debug.h5", key="data")