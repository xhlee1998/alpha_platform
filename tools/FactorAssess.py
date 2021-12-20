from RatioBackTest import *
from ThresBackTest import *
from tqdm import tqdm
from matplotlib import rcParams
from RiskTest import *

rcParams["xtick.labelsize"]='large'
rcParams["ytick.labelsize"]='large'

interval_dict = {
    0: '09:30-09:45',
    1: '09:45-10:00',
    2: '10:00-10:15',
    3: '10:15-10:30',
    4: '10:30-10:45',
    5: '10:45-11:00',
    6: '11:00-11:15',
    7: '11:15-11:30',
    8: '13:00-13:15',
    9: '13:15-13:30',
    10: '13:30-13:45',
    11: '13:45-14:00',
    12: '14:00-14:15',
    13: '14:15-14:30',
    14: '14:30-14:45',
    15: '14:45-15:00',
}

def FactorThresLongPartAssess(ia,main_factor_name,LongThres, MaxTrades=2400, MaxSameStock=30,TopRatio = 0.1, MaxSameTime=10, start_min=1, end_min=240, start_date = None, end_date = None):
    '''
    用于评价单因子
    :param ia:
    :param main_factor_name:
    :param LongThres:
    :param MaxTrades:
    :param MaxSameStock:
    :param MaxSameTime:
    :param start_min:
    :param end_min:
    :return:
    '''

    """
    load data
    """
    # 选取选定的交易日期
    if start_date is None:
        start_date = 0
    if end_date is None:
        end_date = 99999999

    selected_tradedates = [x for x in ia.factors.keys() if int(x) >= int(start_date) and int(x) <= int(end_date)]

    # 开始回测
    RES, POS = ThresBackTest(ia, main_factor_name, LongThres, None, MaxTrades=MaxTrades, MaxSameStock=MaxSameStock, MaxSameTime = MaxSameTime,
                             start_min=start_min, end_min=end_min, start_date=start_date,end_date=end_date)
    alpha_col = [x for x in RES['Long'].columns if 'alpha' in x]

    # 评价标准1：按交易次数统计所有做多机会的收益增长曲线
    record1 = RES['Long'][alpha_col].mean()

    # 评价标准2：按交易次数统计做多,分月收益
    record2 = RES['Long'].groupby(RES['Long']['date'].apply(lambda x: x[:6])).agg(
        {'alpha_2nextopen':'mean','alpha_2next': 'mean', 'alpha_2nextnext': 'mean'})

    # 评价标准3：按天统计做多收益，累计收益
    Tvr = (POS['Long'].sum(axis=1) / MaxTrades)
    Alpha = RES['Long'].groupby(by='date').agg({'alpha_2nextopen':'mean','alpha_2next': 'mean', 'alpha_2nextnext': 'mean'})
    record3 = pd.DataFrame(data=np.concatenate([Tvr.values.reshape(-1, 1), Alpha.values], axis=1),
                           index=Alpha.index.values, columns=['Tvr','alpha_2nextopen', 'alpha_2next', 'alpha_2nextnext'])
    record3['alpha_TvrWeighted'] = record3['Tvr'] * record3['alpha_2next'] + (1 - record3['Tvr']) * \
                                   (record3['alpha_2nextnext'].shift(1) - record3['alpha_2next'].shift(1)) * record3[
                                       'Tvr'].shift(1)
    record3['alpha_cumsum'] = (1 + record3['alpha_TvrWeighted']).cumprod() - 1
    record3['drawdown'] = (
                record3['alpha_TvrWeighted'].cumsum().cummax() - record3['alpha_TvrWeighted'].cumsum()).cummax()

    # 评价标准4：按天统计做多收益，分月表现图
    record4 = record3.groupby(record3.index.map(lambda x: x[:6])).agg(
        {'Tvr': 'mean','alpha_2nextopen':'mean', 'alpha_2next': 'mean', 'alpha_2nextnext': 'mean', 'alpha_TvrWeighted': 'mean'})

    # 评价标准5：因子多头头部日间日内表现图
    RatioRES = RatioBackTest(ia, main_factor_name, TopRatio, None, groupNum=None, start_min=start_min, end_min=end_min, start_date=start_date,end_date=end_date)
    RatioRES['Long']['main_factor_group'] = (RatioRES['Long']['main_factor'].rank(pct=True) * 100).clip(upper=99) // 10
    record5 = RatioRES['Long'].groupby('main_factor_group').agg({'alpha_1': 'mean','alpha_2nextopen': 'mean', 'alpha_2next': 'mean'})

    # 评价标准6：因子多头在不同时间做多的30分钟持仓以及24小时持仓
    RES['Long']['time_interval'] = (RES['Long']['time'] // 15).map(lambda x: interval_dict[x])
    record6 = RES['Long'].groupby(RES['Long']['time_interval']).agg({'alpha_1': 'mean', 'alpha_2next': 'mean','alpha_2nextopen':'mean','alpha_2nextnext': 'mean'})
    record7 = RES['Long'].groupby(RES['Long']['time_interval']).agg({'alpha_2next': 'count'}) / (
                MaxTrades * len(selected_tradedates))
    record7["trades_pct_cumsum"] = record7["alpha_2next"].cumsum()

    # 统计量汇总：
    SUMMARY = {
        'Alpha (Each Trade) Curve': record1,
        'Alpha (Each Trade) By Month': record2,
        'Alpha (Each Day)': pd.concat([record3[['Tvr', 'alpha_2next', 'alpha_2nextnext', 'alpha_TvrWeighted']].mean(),
                                       record3[['alpha_cumsum', 'drawdown']].iloc[-1]]),
        'Alpha (Each Day) By Month': record4,
    }
    """
    plot
    """
    fig = plt.figure(figsize=(25, 50), dpi=cfg.fig1_dpi)
    fig.set_facecolor('#FFFFFF')
    plt.subplots_adjust(wspace=1, hspace=0.25)
    spec = gridspec.GridSpec(ncols=12, nrows=8)
    # --------------Pic.1a
    width = 0.25
    ax = fig.add_subplot(spec[0, 0:6])
    xlabel = [str(np.round(x,2)) for x in cfg.apd_rtn['rtn']] + ["Close","NextOpen","Next24H", "Next48H"]
    plt.plot(record1)
    plt.hlines(record1["alpha_2next"], 0, len(record1) - width * 3, colors="black", linestyles="dashed")
    plt.text(len(record1) - width, record1["alpha_2next"], "%.3f" % (100 * record1["alpha_2next"]) + "%", ha='center',
             va='center', color="red", fontsize=15)
    plt.hlines(record1["alpha_1"], 0, len(record1) - width * 3, colors="grey", linestyles="dashed")
    plt.text(len(record1) - width, record1["alpha_1"], "%.3f" % (100 * record1["alpha_1"]) + "%", ha='center',
             va='center', color="blue", fontsize=15)
    plt.hlines(record1["alpha_2nextopen"], 0, len(record1) - width * 3, colors="grey", linestyles="dashed")
    plt.text(len(record1) - width, record1["alpha_2nextopen"], "%.3f" % (100 * record1["alpha_2nextopen"]) + "%", ha='center',
             va='center', color="turquoise", fontsize=15)

    ax.legend(['平均多组超额 Rtn'],loc='upper left')
    ax.set_xlabel("采样点位置")
    ax.set_xticklabels(xlabel)
    ax.set_title('Decay图，频率：全样本', size=15)
    # --------------Pic.1b
    ax = fig.add_subplot(spec[0, 6:12])
    plt.plot(record5, '--', marker="X", markersize=5)
    ax.legend([f'{cfg.order_time}Min_alpha', 'NO_alpha', '24H_alpha'])
    ax.set_xlabel("分组")
    ax.set_title('头部线性图，频率：全样本', size=15)
    # --------------Pic.2a
    ax = fig.add_subplot(spec[1, 0:6])
    width = 0.25
    plt.text(len(record2) - width, np.nanmean(record2['alpha_2nextopen']),
             "%.3f" % (100 * np.nanmean(record2['alpha_2nextopen'])) + "%", ha='center', va='center', color="turquoise", fontsize=15)
    plt.text(len(record2) - width, np.nanmean(record2['alpha_2next']),
             "%.3f" % (100 * np.nanmean(record2['alpha_2next'])) + "%", ha='center', va='center', color="red", fontsize=15)
    plt.text(len(record2) - width, np.nanmean(record2['alpha_2nextnext']),
             "%.3f" % (100 * np.nanmean(record2['alpha_2nextnext'])) + "%", ha='center', va='center', color="orange", fontsize=15)

    plt.bar(np.arange(len(record2['alpha_2nextopen'])) - width, record2['alpha_2nextopen'], width, color="turquoise", alpha=0.5,
            label="NO_alpha")
    plt.bar(np.arange(len(record2['alpha_2next'])) , record2['alpha_2next'], width, color='red', alpha=0.5,
            label="24H_alpha")
    plt.bar(np.arange(len(record2['alpha_2nextnext'])) + width , record2['alpha_2nextnext'], width, color='orange',
            alpha=0.5, label="48H_alpha")

    plt.hlines(np.nanmean(record2['alpha_2nextopen']), 0 - width, len(record2) - width * 2 , color="turquoise", linestyles="dashed")
    plt.hlines(np.nanmean(record2['alpha_2next']), 0 - width, len(record2) - width * 2, colors="r", linestyles="dashed")
    plt.hlines(np.nanmean(record2['alpha_2nextnext']), 0 - width, len(record2) - width * 2, colors="orange",
               linestyles="dashed")
    plt.legend()
    x_tick = [i + 0.1 for i in np.arange(len(record2['alpha_2next']))]
    xtick_label = list(record2.index)
    plt.xticks(x_tick, xtick_label)
    ax.set_xlabel("月份")
    ax.set_title('按月统计图（按交易次数统计），频率：月度', size=15)
    # --------------Pic.2b
    ax = fig.add_subplot(spec[1, 6:12])
    width = 0.25

    plt.text(len(record4) - width, np.nanmean(record4['alpha_2nextopen']),
             "%.3f" % (100 * np.nanmean(record4['alpha_2nextopen'])) + "%", ha='center', va='center', color="turquoise", fontsize=15)
    plt.text(len(record4) - width, np.nanmean(record4['alpha_2next']),
             "%.3f" % (100 * np.nanmean(record4['alpha_2next'])) + "%", ha='center', va='center', color="red", fontsize=15)
    plt.text(len(record4) - width, np.nanmean(record4['alpha_2nextnext']),
             "%.3f" % (100 * np.nanmean(record4['alpha_2nextnext'])) + "%", ha='center', va='center', color="orange", fontsize=15)


    plt.bar(np.arange(len(record4['alpha_2nextopen'])) - width, record4['alpha_2nextopen'], width, color="turquoise",
            alpha=0.5, label="NO_alpha")
    plt.bar(np.arange(len(record4['alpha_2next'])), record4['alpha_2next'], width, color='red', alpha=0.5,
            label="24H_alpha")
    plt.bar(np.arange(len(record4['alpha_2nextnext'])) + width, record4['alpha_2nextnext'], width, color='orange', alpha=0.5,
            label="48H_alpha")

    plt.hlines(np.nanmean(record4['alpha_2nextopen']), 0 - width, len(record4) - width * 2, colors="turquoise",
               linestyles="dashed")
    plt.hlines(np.nanmean(record4['alpha_2next']), 0 - width, len(record4) - width * 2, colors="r", linestyles="dashed")
    plt.hlines(np.nanmean(record4['alpha_2nextnext']), 0 - width, len(record4) - width * 2, colors="orange",
               linestyles="dashed")

    plt.legend()
    x_tick = [i + 0.1 for i in np.arange(len(record4['alpha_2next']))]
    xtick_label = list(record4.index)
    plt.xticks(x_tick, xtick_label)
    ax.set_xlabel("月份")
    ax.set_title('按月统计图（按天数进行统计），频率：月度', size=15)
    # --------------Pic.3
    ax = fig.add_subplot(spec[2, 0:12])
    width = 0.25
    plt.text(len(record6) - width, np.nanmean(record6['alpha_1']),
             "%.3f" % (100 * np.nanmean(record6['alpha_1'])) + "%", ha='center', va='center', color="blue", fontsize=15)
    plt.text(len(record6) - width, np.nanmean(record6['alpha_2nextopen']),
             "%.3f" % (100 * np.nanmean(record6['alpha_2nextopen'])) + "%", ha='center', va='center', color="turquoise", fontsize=15)
    plt.text(len(record6) - width, np.nanmean(record6['alpha_2next']),
             "%.3f" % (100 * np.nanmean(record6['alpha_2next'])) + "%", ha='center', va='center', color="red", fontsize=15)
    plt.text(len(record6) - width, np.nanmean(record6['alpha_2nextnext']),
             "%.3f" % (100 * np.nanmean(record6['alpha_2nextnext'])) + "%", ha='center', va='center', color="orange", fontsize=15)

    plt.bar(np.arange(len(record6['alpha_1'])) - width , record6['alpha_1'], width/2, color='blue', alpha=0.5,
            label=f'{cfg.order_time}Min_alpha')
    plt.bar(np.arange(len(record6['alpha_2nextopen'])) - width / 2, record6['alpha_2nextopen'], width/2, color="turquoise", alpha=0.5,
            label="NO_alpha")
    plt.bar(np.arange(len(record6['alpha_2next'])), record6['alpha_2next'], width/2, color='red', alpha=0.5,
            label="24H_alpha")
    plt.bar(np.arange(len(record6['alpha_2next'])) + width/2, record6['alpha_2nextnext'], width/2, color="orange", alpha=0.5,
            label="48H_alpha")

    plt.hlines(np.nanmean(record6['alpha_1']), 0 - width, len(record6) - width * 2, colors="blue", linestyles="dashed")
    plt.hlines(np.nanmean(record6['alpha_2nextopen']), 0 - width, len(record6) - width * 2, color="turquoise", linestyles="dashed")
    plt.hlines(np.nanmean(record6['alpha_2next']), 0 - width, len(record6) - width * 2, colors="red",
               linestyles="dashed")
    plt.hlines(np.nanmean(record6['alpha_2nextnext']), 0 - width, len(record6) - width * 2, colors="orange",
               linestyles="dashed")
    plt.legend(loc="upper left")
    ax2 = ax.twinx()
    plt.plot(record7["trades_pct_cumsum"], "--", label="trades_pct_cumsum", color="grey")
    plt.legend(loc="upper right")
    ax.set_xticklabels(list(record6.index))
    ax.set_title('分时收益图，频率：全样本', size=15)
    # --------------Pic.4
    ax = fig.add_subplot(spec[3, 0:12])
    ticker_spacing = 15
    ax.xaxis.set_major_locator(ticker.MultipleLocator(ticker_spacing))
    plt.plot(record3['alpha_cumsum'])
    ax2 = ax.twinx()
    plt.bar(range(len(record3['Tvr'])), record3['Tvr'].values, color='slateblue', alpha=0.3)
    tvrmean = np.nanmean(record3['Tvr'].values)
    plt.hlines(tvrmean,0,len(record3['Tvr'])-2 * width, color='gray', linestyles="dashed")
    plt.text(len(record3['Tvr']) + 1,tvrmean,"Avg Tvr = " + "%.3f" % (100 * tvrmean) + "%", ha='center', va='center', color="gray", fontsize=15)
    ax.legend(['alpha_cumsum', "Tvr"])
    ax.set_xlabel("Date")
    ax.set_ylabel(r"Turnover Weighted Cumsum Return")
    ax2.set_ylabel(r"Turnover Rate")
    ax2.set_ylim(0, 1)
    ax.set_ylim(np.nanmin(record3['alpha_cumsum'].values) - 0.02, np.nanmax(record3['alpha_cumsum'].values) + 0.05)
    ax.set_title('累计收益/换手率图，频率：日度', size=15)
    fig.suptitle(
        f"因子：{main_factor_name}, 调仓周期: {str(cfg.order_time)}分钟, 回测日期: {selected_tradedates[0]} - {selected_tradedates[-1]}\n LongThres: {LongThres}, TopRatio: {TopRatio}, MaxTrades: {MaxTrades}, MaxSameStock: {MaxSameStock}, MaxSameTime: {MaxSameTime}, start_min: {start_min}, end_min: {end_min}",
        fontsize=25, x=0.5, y=1)
    fig.tight_layout()
    plt.show()

    return SUMMARY, RES, POS


def FactorListRatioLongAssess(ia,factor_name_list,Ratio = 0.025,TopRatio = 0.1, start_min=1, end_min=240, start_date = None, end_date = None,direction = 'Long'):
    '''
    不同因子的比较图
    :param ia:
    :param factor_name_list:
    :param Ratio:
    :return:
    '''
    '''
    load data
    '''
    # 选取选定的交易日期
    if start_date is None:
        start_date = 0
    if end_date is None:
        end_date = 99999999

    selected_tradedates = [x for x in ia.factors.keys() if int(x) >= int(start_date) and int(x) <= int(end_date)]

    # 开始回测
    record1List = []
    record2List = []
    record3List = []
    record4List = []
    record5List = []
    record6List = []
    record7List = []
    record8List = []
    record9List = []
    record10List = []
    record11List = []
    record12List = []
    record13List = []
    record14List = []
    record15List = []
    record16List = []
    record17List = []
    record18List = []
    record19List = []
    record20List = []
    record21List = []

    for f in tqdm(factor_name_list):
        if direction == "Long":
            TempRatioRES = RatioBackTest(ia, f, Ratio, None, groupNum=None, start_min=start_min, end_min=end_min, start_date=start_date,end_date=end_date)
        elif direction == "Short":
            TempRatioRES = RatioBackTest(ia, f, None, Ratio, groupNum=None, start_min=start_min, end_min=end_min, start_date=start_date,end_date=end_date)

        alpha_col = [x for x in TempRatioRES[direction].columns if 'alpha' in x]
        record1List.append(TempRatioRES[direction][alpha_col].mean())

        # 分月group
        groupByMonth = TempRatioRES[direction].groupby(TempRatioRES[direction]['date'].apply(lambda x: x[:6])).agg({'alpha_1': 'mean','alpha_2next': 'mean','alpha_2nextopen': 'mean'})
        record2List.append(groupByMonth['alpha_1'])
        record3List.append(groupByMonth['alpha_2next'])
        record8List.append(groupByMonth['alpha_2nextopen'])

        # 分时group
        TempRatioRES[direction]['time_interval'] = (TempRatioRES[direction]['time'] // 15).map(lambda x: interval_dict[x])
        groupByInterval = TempRatioRES[direction].groupby(TempRatioRES[direction]['time_interval']).agg({'alpha_1': 'mean', 'alpha_2next': 'mean','alpha_2nextopen': 'mean'})
        record4List.append(groupByInterval['alpha_1'])
        record5List.append(groupByInterval['alpha_2next'])
        record9List.append(groupByInterval['alpha_2nextopen'])

        # 分日group
        groupByDay = TempRatioRES[direction].groupby(TempRatioRES[direction]['date']).agg({'alpha_2next': 'mean'})
        record11List.append(groupByDay['alpha_2next'].cumsum())

        # 风格分析 holding_median_MV/TR_daily为df,统计每天风格；其余均为数值,默认参照指数为20200331时点中证1000成分股，可通过index_code及date更改指数
        pool_median_MV, pool_median_TR, index_median_MV, index_median_TR, holding_median_MV_daily, holding_median_MV, holding_median_TR_daily, holding_median_TR, pool_MV, pool_TR = cal_risk(
            ia.para_dict["stock_pool"], TempRatioRES[direction], start_date, end_date, index_code='000852',
            date='20200331')

        # 全局头部
        if direction == "Long":
            TempRatioRES = RatioBackTest(ia, f, TopRatio, None, groupNum=None, start_min=start_min, end_min=end_min, start_date=start_date,end_date=end_date)
        elif direction == "Short":
            TempRatioRES = RatioBackTest(ia, f, None, TopRatio, groupNum=None, start_min=start_min, end_min=end_min, start_date=start_date,end_date=end_date)

        TempRatioRES[direction]['main_factor_group'] = (TempRatioRES[direction]['main_factor'].rank(pct=True) * 100).clip(upper=99) // 10
        TopAlpha = TempRatioRES[direction].groupby('main_factor_group').agg({'alpha_1': 'mean','alpha_2next': 'mean','alpha_2nextopen': 'mean'})
        record6List.append(TopAlpha['alpha_2next'])
        record7List.append(TopAlpha['alpha_1'])
        record10List.append(TopAlpha['alpha_2nextopen'])


        # 风格分析继续
        pool_MV_daily = pool_MV.T.median(axis=1)
        pool_TR_daily = pool_TR.T.median(axis=1)

        record12List.append(holding_median_MV_daily)
        record13List.append(holding_median_TR_daily)
        record14List.append(holding_median_MV)
        record15List.append(pool_median_MV)
        record16List.append(index_median_MV)
        record17List.append(holding_median_TR)
        record18List.append(pool_median_TR)
        record19List.append(index_median_TR)
        record20List.append(pool_MV_daily)
        record21List.append(pool_TR_daily)

    '''
    plot
    '''
    fig = plt.figure(figsize=(25, 50), dpi=cfg.fig1_dpi)
    fig.set_facecolor('#FFFFFF')
    plt.subplots_adjust(wspace=1, hspace=0.25)
    spec = gridspec.GridSpec(ncols=12, nrows=8)
    # -------Pic.1a
    ax = fig.add_subplot(spec[0, 0:6])
    xlabel = [str(np.round(x,2)) for x in cfg.apd_rtn['rtn']] + ["Close","NextOpen","Next24H", "Next48H"]

    width = 0.25
    for idx, record in enumerate(record1List):
        plt.plot(record, label=factor_name_list[idx])
        plt.hlines(record['alpha_2next'], 0 - width, len(record) - width * 2,linestyles="dashed")
        plt.text(len(record) - width, record['alpha_2next'],"%.3f" % (100 * record['alpha_2next']) + "%", ha='center', va='center', fontsize=15)
    plt.legend()
    ax.set_xlabel("采样点位置")
    ax.set_xticklabels(xlabel)
    ax.set_title('Decay图比较，频率：全样本', size=15)

    ax = fig.add_subplot(spec[0, 6:12])
    ax.xaxis.set_major_locator(ticker.MultipleLocator(15))
    for idx, record in enumerate(record11List):
        plt.plot(record, label=factor_name_list[idx])
    plt.legend()
    # ax.set_xticklabels(record.index.to_list())
    ax.set_xlabel("日期")
    ax.set_title('累计收益图，频率：全样本', size=15)


    # -------Pic.1b
    ax = fig.add_subplot(spec[1, 0:4])
    for idx, record in enumerate(record7List):
        plt.plot(record, '--', label=factor_name_list[idx], marker="X", markersize=5)
        plt.legend()
    ax.set_xlabel("分组")
    ax.set_title('头部线性图比较30Min，频率：全样本', size=15)
    # -------Pic.1b
    ax = fig.add_subplot(spec[1, 4:8])
    for idx, record in enumerate(record10List):
        plt.plot(record, '--', label=factor_name_list[idx], marker="X", markersize=5)
        plt.legend()
    ax.set_xlabel("分组")
    ax.set_title('头部线性图比较NO，频率：全样本', size=15)
    # -------Pic.1b
    ax = fig.add_subplot(spec[1, 8:12])
    for idx, record in enumerate(record6List):
        plt.plot(record, '--', label=factor_name_list[idx], marker="X", markersize=5)
        plt.legend()
    ax.set_xlabel("分组")
    ax.set_title('头部线性图比较24H，频率：全样本', size=15)

    # -------Pic.2a
    select_idx = pd.concat(record4List,axis=1).dropna().index.to_list()
    ax = fig.add_subplot(spec[2, 0:4])
    for idx, record in enumerate(record4List):
        plt.plot(record.loc[select_idx], '--',label=factor_name_list[idx], marker="X", markersize=5)
        plt.legend()
    ax.set_title('分时30Min收益比较', size=15)
    plt.xticks(rotation=45)
    # -------Pic.2a
    ax = fig.add_subplot(spec[2, 4:8])
    for idx, record in enumerate(record9List):
        plt.plot(record.loc[select_idx], '--',label=factor_name_list[idx], marker="X", markersize=5)
        plt.legend()
    ax.set_title('分时NO收益比较', size=15)
    plt.xticks(rotation=45)
    # -------Pic.2b
    ax = fig.add_subplot(spec[2, 8:12])
    for idx, record in enumerate(record5List):
        plt.plot(record.loc[select_idx], '--',label=factor_name_list[idx], marker="X", markersize=5)
        plt.legend()
    ax.set_title('分时24H收益比较', size=15)
    plt.xticks(rotation=45)

    # -------Pic.3a
    ax = fig.add_subplot(spec[3, 0:4])
    for idx, record in enumerate(record2List):
        plt.plot(record, '--',label=factor_name_list[idx], marker="X", markersize=5)
        plt.legend()
    ax.set_title('按月30Min收益比较', size=15)
    # -------Pic.3a
    ax = fig.add_subplot(spec[3, 4:8])
    for idx, record in enumerate(record8List):
        plt.plot(record, '--', label=factor_name_list[idx], marker="X", markersize=5)
        plt.legend()
    ax.set_title('按月NO收益比较', size=15)
    # -------Pic.3b
    ax = fig.add_subplot(spec[3, 8:12])
    for idx, record in enumerate(record3List):
        plt.plot(record, '--',label=factor_name_list[idx], marker="X", markersize=5)
        plt.legend()
    ax.set_title('按月24H收益比较', size=15)

    # --------------Pic.5a
    ax = fig.add_subplot(spec[4, 0:6])
    ax.xaxis.set_major_locator(ticker.MultipleLocator(15))
    plt.plot(pool_MV_daily, label='pool_MV')
    plt.legend()
    for idx, record in enumerate(record12List):
        plt.plot(record, label=factor_name_list[idx])
        plt.legend()

    # ax.set_xticklabels(record.index.to_list())
    ax.set_xlabel("日期")
    ax.set_title('市值中位数', size=15)

    # --------------Pic.5b
    ax = fig.add_subplot(spec[4, 6:12])
    ax.xaxis.set_major_locator(ticker.MultipleLocator(15))
    plt.plot(pool_TR_daily, label='pool_TR')
    plt.legend()
    for idx, record in enumerate(record13List):
        plt.plot(record, label=factor_name_list[idx])
        plt.legend()
    # ax.set_xticklabels(record.index.to_list())
    ax.set_xlabel("日期")
    ax.set_title('换手率中位数', size=15)

    # --------------Form.6a
    ax = fig.add_subplot(spec[5, 0:6])
    plt.axis('off')
    cell_text = []
    for idx in range(len(record14List)):
        cell_text.append([record14List[idx], record15List[idx], record16List[idx]])
    col_labels = ["Holding Median", "Pool Median", "Index Median"]
    row_labels = [name if len(name) < 20 else name[-20::1] for name in factor_name_list]

    for i in range(len(cell_text)):
        cell_text[i] = [round(x / 100000000, 2) for x in cell_text[i]]
    table = ax.table(cellText=cell_text, rowLabels=row_labels, colLabels=col_labels, loc='center',
                     cellLoc='center', rowLoc='center', colWidths=[0.2, 0.2, 0.2])
    table.auto_set_font_size(False)
    table.set_fontsize(20)
    table.scale(1, 3)
    ax.set_title("流动市值统计（单位：亿）", fontsize=25)

    # --------------Form.6b
    ax = fig.add_subplot(spec[5, 6:12])
    plt.axis('off')
    cell_text = []
    for idx in range(len(record17List)):
        cell_text.append([record17List[idx], record18List[idx], record19List[idx]])
    col_labels = ["Holding Median", "Pool Median", "Index Median"]
    row_labels = [name if len(name) < 20 else name[-20::1] for name in factor_name_list]

    for i in range(len(cell_text)):
        cell_text[i] = [round(x, 2) for x in cell_text[i]]
    table = ax.table(cellText=cell_text, rowLabels=row_labels, colLabels=col_labels, loc='center',
                     cellLoc='center', rowLoc='center', colWidths=[0.2, 0.2, 0.2])
    table.auto_set_font_size(False)
    table.set_fontsize(20)
    table.scale(1, 3)
    ax.set_title("换手率统计", fontsize=25)

    fig.suptitle(
        f"因子：{[x[-10:] for x in factor_name_list]}, \n {direction}Ratio: {Ratio}, TopRatio: {TopRatio},回测日期: {selected_tradedates[0]} - {selected_tradedates[-1]}",
        fontsize=25, x=0.5, y=1)
    fig.tight_layout()
    plt.show()
