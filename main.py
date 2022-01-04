from RatioBackTest import *
import os
import numpy as np
import time
import sys
import gzip,pickle
import pandas as pd
sys.path.append("/mnt/feature_test/")
save_path = "/mnt/feature_test/"
from matplotlib import pyplot as plt



decay_coefficients=[45,60,90,120,240,480,1440,2880]
data_tag = "AlphaY2021"
source = "spot"
basic_data_path = f"/share/basic_data_{data_tag}"
ia = {}



if __name__ == "__main__":
    # 读取数据部分，从已经生成的数据读取
    t1 = time.time()
    future_data_dict = pd.read_pickle(os.path.join(basic_data_path,"future_data_dict.pkl"))
    spot_data_dict = pd.read_pickle(os.path.join(basic_data_path,"spot_data_dict.pkl"))
    t2 = time.time()
    print(f"Fetching Data Using {t2 - t1}")

    main_factor_name = "m1_spot_excess_forward_rtn_f45m30"
    main_factor = spot_data_dict[main_factor_name]
    mask_factor_name = 'm1_spot_tradingval_mask_b7200g1440c50'


    RES_ALL = RatioBackTest(spot_data_dict, main_factor_name,  mask_factor_name,LongRatio=0.01, ShortRatio=None, groupNum=None,
                         other_fac_to_zip=None,decay_coefficients=[45,60,90,120,240,480,1440,2880],source="spot")

    #多空画图
    graph = RES_ALL['Long'].mean(axis=0)
    name = ['45','60','90','120','240','480','1440','2880']
    plt.plot(name,graph.values[1:])
    plt.show()




