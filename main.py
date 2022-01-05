from RatioBackTest import *
import RatioBackTest_config as cfg
import os
import time
import sys
import pandas as pd
sys.path.append("/mnt/feature_test/")
save_path = "/mnt/feature_test/"


# 读取数据部分，从已经生成的数据读取
t1 = time.time()
future_data_dict = pd.read_pickle(os.path.join(cfg.basic_data_path,"future_data_dict.pkl"))
spot_data_dict = pd.read_pickle(os.path.join(cfg.basic_data_path,"spot_data_dict.pkl"))
t2 = time.time()
print(f"Fetching Data Using {t2 - t1}")

main_factor = spot_data_dict[cfg.main_factor_name]
RES_ALL = RatioBackTest(spot_data_dict, cfg.main_factor_name,  cfg.mask_factor_name,
                    cfg.LongRatio, cfg.ShortRatio, cfg.groupNum,
                     cfg.other_fac_to_zip,cfg.decay_coefficients,cfg.source,cfg.show_coefficients)




