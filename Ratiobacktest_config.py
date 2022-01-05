
# LSalpha走势图中要看的alpha 必须已经落下因子 否则会报错
decay_coefficients=['45','60','90','120','240','480','1440','2880']
# 整体线性图中要看的alpha 必须已经落下因子 否则会报错
show_coefficients = ['45','90','120']
data_tag = "AlphaY2021"
source = "spot"
basic_data_path = f"/share/basic_data_{data_tag}"
LongRatio = 0.01
ShortRatio = 0.01
groupNum = 100
other_fac_to_zip = None
main_factor_name = "m1_spot_excess_forward_rtn_f45m30"
mask_factor_name = 'm1_spot_tradingval_mask_b7200g1440c50'
