import numpy as np
read_path = "E:\storage\cryp\save_features"

def ZipFactorRtn_Oneday(ia, date, main_factor_name, SelectedGroups, decay_coefficients=[1], other_fac_to_zip=None):
    """
    按照选定组，链接因子与alpha收益
    :param ia:
    :param date:
    :param main_factor_name:
    :param SelectedGroups:
    :param decay_coefficients:
    :param other_fac_to_zip:
    :return: 返回一个Dict，Dict的key是不同分组名，value是一个df，df中包括每一条被选中的记录，包括当时的因子值、未来n期收益值、以及其他辅助因子值
    """

    # load factor data
    main_factor = (ia.factors[date][main_factor_name])
    date_factor = np.full(main_factor.shape, date)
    time_factor = np.repeat(np.array(range(main_factor.shape[1])).reshape(1, -1), main_factor.shape[0], axis=0)
    stock_factor = np.repeat(np.array(ia.tickers).reshape(-1, 1), main_factor.shape[1], axis=1)

    # load alpha
    ToZipDict = {}
    for decay in decay_coefficients:
        rtn_name = 'rtn' + str(decay) + 'vwap'
        index_rtn_name = 'index' + str(decay) + 'vwap'
        temp_alpha_rtn = (ia.factors[date][rtn_name] - ia.factors[date][index_rtn_name])
        ToZipDict['alpha_' + str(decay)] = temp_alpha_rtn

    ToZipDict['alpha_2close'] = (ia.factors[date]['2close'] - ia.factors[date]['index2close'])
    ToZipDict['alpha_2nextopen'] = (ia.factors[date]['2nextopen'] - ia.factors[date]['index2nextopen'])
    ToZipDict['alpha_2next'] = (ia.factors[date]['2next'] - ia.factors[date]['index2next'])
    ToZipDict['alpha_2nextnext'] = (ia.factors[date]['2nextnext'] - ia.factors[date]['index2nextnext'])

    # load other factors
    if other_fac_to_zip is not None:
        for fac in other_fac_to_zip:
            ToZipDict[fac] = (ia.factors[date][fac])

    # 链接因子与日期
    RES = {}
    for Group in SelectedGroups.keys():
        RES[Group] = {
            'main_factor': main_factor[SelectedGroups[Group]],
            'date': date_factor[SelectedGroups[Group]],
            'time': time_factor[SelectedGroups[Group]],
            'stock_factor': stock_factor[SelectedGroups[Group]],
        }

    # 链接因子与收益
    for ToZip in ToZipDict.keys():
        for Group in RES.keys():
            RES[Group][ToZip] = ToZipDict[ToZip][SelectedGroups[Group]]

    # 拼接df
    for Group in RES.keys():
        RES[Group] = pd.DataFrame(RES[Group]).sort_values(by=['date', 'time'])

        # 增加两列，为了格式上和隔壁组对齐
        RES[Group]["y_true"] = RES[Group]["alpha_2next"]
        RES[Group]["allocated_cap"] = 50000

    return RES

def SelectGroupsByRatio_Oneday(ia, date, main_factor_name, LongRatio=None, ShortRatio=None, groupNum=None):
    """
    根据固定百分比筛选股票
    :param ia:
    :param date:
    :param main_factor_name:
    :param LongRatio:
    :param ShortRatio:
    :param groupNum:
    :return: 一个Dict，Dict的key为组名，values是shape为（股票数 ，分钟数）的True False矩阵
    """
    # 导入因子
    main_factor = np.load('E:\storage\cryp\save_features\m1_min_vwap_60\\'+'m1_min_vwap_60min_'+str(date)+'.pkl',allow_pickle=True)


    # 要被筛除的样本 delete volume rank > 50
    # todel = np.isnan(ia.factors[date]['2next'])

    # 删除部分na样本
    mask = np.isnan(main_factor)
    rank_factor = pd.DataFrame(np.where(mask, np.nan, main_factor)).rank(pct=True, axis=0, method='first').values

    # 计算每组所需部分
    ToCalGroups = []
    SelectedGroups = {}
    if LongRatio is not None:
        ToCalGroups += ['Long']
        SelectedGroups['Long'] = rank_factor > (1 - LongRatio)
    if ShortRatio is not None:
        ToCalGroups += ['Short']
        SelectedGroups['Short'] = rank_factor < ShortRatio
    if groupNum is not None:
        groupNum = int(groupNum)
        ToCalGroups += [str(x) for x in range(groupNum)]
        interval = 1 / groupNum
        for i in range(groupNum):
            if i != groupNum - 1:
                SelectedGroups[str(i)] = (rank_factor >= i * interval) & (rank_factor < (i + 1) * interval)
            else:
                SelectedGroups[str(i)] = (rank_factor >= i * interval) & (rank_factor <= (i + 1) * interval)

    return SelectedGroups

# 按照比例回测一天分组情况
def RatioBackTest_Onday(ia, date, main_factor_name, LongRatio=None, ShortRatio=None, groupNum=None,decay_coefficients=[1], other_fac_to_zip=None):
    SelectedGroups = SelectGroupsByRatio_Oneday(ia, date, main_factor_name, LongRatio=LongRatio, ShortRatio=ShortRatio,
                                                groupNum=groupNum)
    ZIPRES = ZipFactorRtn_Oneday(ia, date, main_factor_name, SelectedGroups, decay_coefficients=decay_coefficients,
                                 other_fac_to_zip=other_fac_to_zip)
    # 空组测试一般把收益取反向，得到空头收益
    if ShortRatio is not None:
        for c in ZIPRES['Short']:
            if 'alpha_' in c:
                ZIPRES['Short'][c] = -ZIPRES['Short'][c]
    return ZIPRES

# 对每一天都进行相同回测
def RatioBackTest(ia, main_factor_name, LongRatio, ShortRatio, groupNum=6, other_fac_to_zip=None, start_min=5, end_min=283, start_date = None, end_date = None):
    ZIPRESListDict = {}

    # 选取选定的交易日期
    if start_date is None:
        start_date = 0
    if end_date is None:
        end_date = 99999999

    selected_tradedates = [x for x in ia.factors.keys() if int(start_date) <= int(x) <= int(end_date)]

    # 按天运行回测函数，并且拼接所有分组df
    for date in selected_tradedates:
        tempZIPRES = RatioBackTest_Onday(ia, date, main_factor_name, decay_coefficients=config.apd_rtn['rtn'],
                                         LongRatio=LongRatio, ShortRatio=ShortRatio, groupNum=groupNum,
                                         other_fac_to_zip=other_fac_to_zip)
        for Group in tempZIPRES.keys():
            if Group not in ZIPRESListDict.keys():
                ZIPRESListDict[Group] = [tempZIPRES[Group]]
            else:
                ZIPRESListDict[Group].append(tempZIPRES[Group])

    for Group in ZIPRESListDict.keys():
        ZIPRESListDict[Group] = pd.concat(ZIPRESListDict[Group])
        ZIPRESListDict[Group] = ZIPRESListDict[Group][(ZIPRESListDict[Group]['time'] >=start_min) & (ZIPRESListDict[Group]['time'] <= end_min)]

    return ZIPRESListDict
