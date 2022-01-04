# import config
# from tools.helper import *
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

def SelectGroupsByRatio(data_dict,  main_factor_name, mask_factor_name=None,LongRatio=None, ShortRatio=None,groupNum = None):
    """
    根据固定百分比筛选股票
    :param data_dict:是一个dict key是各个factor的名字
    :param date:不需要了 此时是唱序列
    :param main_factor_name:
    :param LongRatio:
    :param ShortRatio:
    :param groupNum:总体分组数
    :return: 一个Dict，Dict的key为组名，values是shape为（现货数 ，分钟数）的True False矩阵
    """
    # 导入因子
    main_factor = (data_dict[main_factor_name])

    if not mask_factor_name ==None:
        # 确定要被筛除的样本
        mask = data_dict[mask_factor_name]
        # 删除部分不活跃样本
        rank_factor = pd.DataFrame(np.where(mask, main_factor, np.nan)).rank(pct=True, axis=0, method='first').values
    else:
        rank_factor = pd.DataFrame(main_factor).rank(pct=True, axis=0, method='first').values

    # 计算每组所需部分
    ToCalGroups = []
    SelectedGroups = {}
    if LongRatio is not None:
        ToCalGroups += ['Long']
        SelectedGroups['Long'] = rank_factor > (1 - LongRatio)

    if ShortRatio is not None:
        ToCalGroups += ['Short']
        SelectedGroups['Short'] = rank_factor < ShortRatio

    # groupnum 分组线性再做
    if groupNum is not None:
        groupNum = int(groupNum)
        ToCalGroups += [str(x) for x in range(groupNum)]
        interval = 1/ groupNum
        for i in range(groupNum):
            if i != groupNum - 1:
                SelectedGroups[str(i)] = (rank_factor >=  i * interval) & (rank_factor <  (i + 1) * interval)
            else:
                SelectedGroups[str(i)] = (rank_factor >=  i * interval) & (rank_factor <=  (i + 1) * interval)
    return SelectedGroups



def ZipFactorRtn(data_dict, main_factor_name, SelectedGroups, decay_coefficients=[45,60,90,120,240,480,1440,2880], source = "spot",other_fac_to_zip=None):
    """
    按照选定组，链接因子与alpha收益
    :param data_dict:
    :param date:
    :param main_factor_name:
    :param SelectedGroups:
    :param decay_coefficients:
    :param other_fac_to_zip:
    :return: 返回一个Dict，Dict的key是不同分组名，value是一个df，df中包括每一条被选中的记录，包括当时的因子值、未来n期收益值、以及其他辅助因子值
    """
    # load factor data
    main_factor = (data_dict[main_factor_name])

    # load alpha
    ToZipDict = {}
    for decay in decay_coefficients:
        rtn_name = f"m1_{source}_excess_forward_rtn_f"+str(decay)+"m30"
        temp_alpha_rtn = data_dict[rtn_name]
        ToZipDict['alpha_' + str(decay)] = temp_alpha_rtn

    # load other factors
    if other_fac_to_zip is not None:
        for fac in other_fac_to_zip:
            ToZipDict[fac] = (data_dict[fac])

    # 链接因子与主因子值
    RES = {}
    for Group in SelectedGroups.keys():
        tmp = main_factor.fillna(0)*np.where(SelectedGroups[Group],1,np.nan)
        RES[Group] = {
            'main_factor': pd.DataFrame(tmp.values.reshape((-1, 1))).dropna(),
        }

    # 链接因子与收益
    for ToZip in ToZipDict.keys():
        for Group in RES.keys():
            tmp = ToZipDict[ToZip].fillna(0)*np.where(SelectedGroups[Group],1,np.nan)
            RES[Group][ToZip] = pd.DataFrame(tmp.values.reshape((-1, 1))).dropna()

    # 拼接df
    for Group in RES.keys():
        tmp = pd.DataFrame()
        for name in RES[Group].keys():
            tmp[name] = RES[Group][name]
        #空头收益为负 其他都是正常
        if Group == 'Short':
            RES[Group] = -tmp
        else:
            RES[Group] = tmp
    return RES


# 按照比例回测一天分组情况
def RatioBackTest(data_dict, main_factor_name,  mask_factor_name=None,LongRatio=None, ShortRatio=None, groupNum=None,
                         other_fac_to_zip=None,decay_coefficients=[45,60,90,120,240,480,1440,2880],source="spot"):
    SelectedGroups = SelectGroupsByRatio(data_dict, main_factor_name, mask_factor_name,
                                         LongRatio, ShortRatio,groupNum)

    ZIPRES = ZipFactorRtn(data_dict, main_factor_name, SelectedGroups, decay_coefficients,
                                 source, other_fac_to_zip)
    # 空组测试一般把收益取反向，得到空头收益
    if ShortRatio is not None:
        for c in ZIPRES['Short']:
            if 'alpha_' in c:
                ZIPRES['Short'][c] = -ZIPRES['Short'][c]
    return ZIPRES


# # 对每一天都进行相同回测
# def RatioBackTest(data_dict, main_factor_name, LongRatio, ShortRatio, groupNum=6,decay_coeficients=[10], other_fac_to_zip=None, start_min=5, end_min=237, start_date = None, end_date = None):
#     ZIPRESListDict = {}
#
#     # 选取选定的交易日期
#     if start_date is None:
#         start_date = 0
#     if end_date is None:
#         end_date = 99999999
#
#     selected_tradedates = [x for x in data_dict.factors.keys() if int(start_date) <= int(x) <= int(end_date)]
#
#     # 按天运行回测函数，并且拼接所有分组df
#     for date in selected_tradedates:
#         tempZIPRES = RatioBackTest_Onday(data_dict, date, main_factor_name, decay_coefficients=decay_coeficients,
#                                          LongRatio=LongRatio, ShortRatio=ShortRatio, groupNum=groupNum,
#                                          other_fac_to_zip=other_fac_to_zip)
#         for Group in tempZIPRES.keys():
#             if Group not in ZIPRESListDict.keys():
#                 ZIPRESListDict[Group] = [tempZIPRES[Group]]
#             else:
#                 ZIPRESListDict[Group].append(tempZIPRES[Group])
#
#     for Group in ZIPRESListDict.keys():
#         ZIPRESListDict[Group] = pd.concat(ZIPRESListDict[Group])
#         ZIPRESListDict[Group] = ZIPRESListDict[Group][(ZIPRESListDict[Group]['time'] >=start_min) & (ZIPRESListDict[Group]['time'] <= end_min)]
#
#     return ZIPRESListDict
