# coding=utf-8
import os
import copy
import time
from PqiDataSdk import PqiDataSdk
import shutil
import numpy as np
import pandas as pd
from sys import path
import seaborn as sns
from itertools import groupby
from scipy.stats import rankdata
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
import random
import tools.ThresBackTest as TBT
import tools.RatioBackTest as RBT
from tools.utils import *

logging.basicConfig(level=logging.CRITICAL)

warnings.filterwarnings("ignore")

color_board = ['red', 'orange', 'yellow', 'green', 'blue', 'steelblue', 'purple']
bad_days = ['20210312', '20210108', '20200115', '20190102', '20190103', '20190104', '20190105', '20190106', '20190107', '20190108',
            '20191101', '20191104', '20191105', '20191106', '20191107', '20191108', '20191111', '20191112', '20191113', '20191114',
            '20191115', '20191118', '20191119', '20191120', '20191121', '20191122', '20191125', '20191126', '20191127', '20191128',
            '20191129']


def save_factor(factors, tickers, fac1, output_path):
    """
    因子值数据落地

    :param factors:
    :param tickers:
    :param fac1:
    :param output_path:
    :return:
    """
    fac_val = {date: factors[date][fac1] for date in factors.keys()}
    np.save(save_path(fac1, output_path, 'factor.npy'), fac_val)


def save_IC(factors, tickers, fac1, output_path, using_rank=True, vwap_flag=False):
    """
    计算因子 IC，并且 IC值 数据落地

    :param factors:
    :param tickers:
    :param fac1:
    :param output_path:
    :param using_rank: 是否对因子值求 rank 后再求相关系数，可以减少因子异常值的影响
    :return:
    """

    # 是否采用 vwap return
    if vwap_flag:
        rtn1_ = 'rtn1vwap_'
        rtn1 = 'rtn1vwap'
    else:
        rtn1_ = 'rtn1_'
        rtn1 = 'rtn1'

    tonext = '2next'

    # 计算 IC = corr(fac(t), ret(t+1))
    # 计算 crossIC = corr(fac(t), ret(2next))
    # 存储成 2d array，每一行为一天该因子的 IC 序列

    dates = list(factors.keys())
    ICs = []
    crossICs = []

    # using_rank = True  # 是否对因子值求 rank 后再求 correlation

    for date in dates:
        ret = factors[date][rtn1]
        cross_ret = factors[date][tonext]
        # print(factors[date][fac1])
        if using_rank:
            # 改成求 rank，用 nan 替代 nan
            fac = rankdata(factors[date][fac1], axis=0)
            fac[pd.isna(factors[date][fac1])] = np.nan
        else:
            fac = factors[date][fac1]

        IC = []
        crossIC = []
        for i in range(fac.shape[1] - 1):
            # 取截面因子值和 return
            fac_sec = fac[:, i]
            ret_next_sec = ret[:, i]
            ret_next_day = cross_ret[:, i]
            # drop nan
            fac_sec_rmna = fac_sec[(~pd.isna(fac_sec)) & (~pd.isna(ret_next_sec))]
            ret_next_sec_rmna = ret_next_sec[(~pd.isna(fac_sec)) & (~pd.isna(ret_next_sec))]
            # 计算 IC
            if len(fac_sec_rmna) == 0:
                IC.append(np.nan)
            else:
                if np.all(fac_sec_rmna == fac_sec_rmna[0]) or np.all(ret_next_sec_rmna == ret_next_sec_rmna[0]):
                    # 若 x, y 中有一个为恒定值，则取 correlation 为 0
                    IC.append(np.nan)
                else:
                    IC.append(np.corrcoef(fac_sec_rmna.astype(np.float64), ret_next_sec_rmna.astype(np.float64))[0, 1])

            # 计算跨日IC
            fac_sec_rmna = fac_sec[(~pd.isna(fac_sec)) & (~pd.isna(ret_next_day))]
            ret_next_day_rmna = ret_next_day[(~pd.isna(fac_sec)) & (~pd.isna(ret_next_day))]

            if len(fac_sec_rmna) == 0:
                crossIC.append(np.nan)
            else:
                if (np.all(fac_sec_rmna == fac_sec_rmna[0]) or np.all(ret_next_day_rmna == ret_next_day_rmna[0])):
                    # 若 x, y 中有一个为恒定值，则取 correlation 为 0
                    crossIC.append(np.nan)
                else:
                    crossIC.append(np.corrcoef(fac_sec_rmna.astype(np.float64), ret_next_day_rmna.astype(np.float64))[0, 1])

        ICs.append(IC)
        crossICs.append(crossIC)

    IC_arr = np.array(ICs)
    crossIC_arr = np.array(crossICs)
    np.save(save_path(fac1, output_path, 'IC_arr.npy'), IC_arr)
    np.save(save_path(fac1, output_path, 'crossIC_arr.npy'), crossIC_arr)


def IC_stat(IC_arr):
    IC_mean = np.nanmean(IC_arr)
    IC_mean_positive = np.nanmean(IC_arr[IC_arr > 0])
    IC_mean_negative = np.nanmean(IC_arr[IC_arr < 0])

    IC_std = np.nanstd(IC_arr)
    IC_std_positive = np.nanstd(IC_arr[IC_arr > 0])
    IC_std_negative = np.nanstd(IC_arr[IC_arr < 0])

    IC_positive_ratio = np.sum(IC_arr > 0) / np.sum(~pd.isna(IC_arr))
    IC_negative_ratio = np.sum(IC_arr < 0) / np.sum(~pd.isna(IC_arr))

    IC_significance_ratio = np.sum(abs(IC_arr) > 0.02) / np.sum(~pd.isna(IC_arr))

    IR = IC_mean / IC_std

    num_consecutive_positive_mean = np.mean(
        [sum(1 for _ in group) for key, group in groupby(IC_arr.flatten() > 0) if key])

    return [IC_mean, IC_mean_positive, IC_mean_negative, IC_std, IC_std_positive, IC_std_negative,
            IC_positive_ratio, IC_negative_ratio, IC_significance_ratio, IR, num_consecutive_positive_mean]


def plotFig1(fac1, output_path):
    """
    因子值分布图，IC值分布图 画图函数

    :param factors: 因子值
    :param tickers: ticker 名称
    :param fac1:
    :param output_path: 输出路径

    :param using_rank: 是否求对因子值求rank后再算相关系数
    :return:
    """
    fig = plt.figure(figsize=(25, 50), dpi=cfg.fig1_dpi)
    fig.set_facecolor('#FFFFFF')
    plt.subplots_adjust(wspace=1, hspace=0.25)
    spec = gridspec.GridSpec(ncols=12, nrows=9)

    # --------------------------------------------------------- Load Data --------------------------------------------------
    if len(output_path) >= 1:
        factors = np.load(save_path(fac1, output_path, 'factor.npy'), allow_pickle=True).item()
        IC_arr = np.load(save_path(fac1, output_path, 'IC_arr.npy'))
        crossIC_arr = np.load(save_path(fac1, output_path, 'crossIC_arr.npy'))
    else:
        output_path = get_latest_time(fac1)
        factors = np.load(save_path(fac1, get_latest_time(fac1), 'factor.npy'), allow_pickle=True).item()
        IC_arr = np.load(save_path(fac1, get_latest_time(fac1), 'IC_arr.npy'), allow_pickle=True)
        crossIC_arr = np.load(save_path(fac1, get_latest_time(fac1), 'crossIC_arr.npy'), allow_pickle=True)

    # -----------------------------------------------------
    #
    # # 图1. 因子值时序 violinplot
    #
    # ax = fig.add_subplot(spec[0, 0:12])
    #
    # Get labels and factor values
    dates = list(factors.keys())
    all_data = np.array([factors[date].reshape(-1) for date in dates])

    # # Make df for violinplot
    # # factor = np.array(all_data).flatten()
    # factor = all_data.flatten()
    # n = len(all_data[0])
    # date = [label[4:] for label in dates for i in range(n)]
    # df = pd.DataFrame({'value': factor, 'date': date})
    #
    # # Violinplot
    # # print(df)
    # sns.violinplot(x='date', y='value', data=df)
    # ax.set_title('Daily Factor Values')
    # for tick in ax.get_xticklabels():
    #     tick.set_rotation(90)

    # -----------------------------------------------------

    # 图3b. 因子值时段 violinplot
    # 时段划分取 9:30-10:30, 10:30-11:30, 13:00-14:00, 14:00-15:00

    ax = fig.add_subplot(spec[1, 0:8])

    labels = ['9:30:00-10:30:00', '10:30:00-11:30:00', '13:00:00-14:00:00', '14:00:00-15:00:00']
    sample_freq = cfg.sample_freq  # x min采样
    num_fac_group = int(60 / sample_freq)  # 每个group的因子数
    num_group = 4

    fac_group_list = []
    for i in range(num_group):
        arr = np.array([val[int(random.random() * 20)::20, i * num_fac_group:(i + 1) * num_fac_group].flatten()
                        for val in factors.values()]).flatten()

        arr = arr[~pd.isna(arr)]
        fac_group_list.append(arr)

    # Make df for violinplot
    factor = np.concatenate(fac_group_list)
    group = [labels[i] for i in range(len(fac_group_list)) for j in range(fac_group_list[i].shape[0])]
    df = pd.DataFrame({'value': factor, 'time': group})

    # Violinplot
    sns.violinplot(x='time', y='value', data=df)
    ax.set_title('Intraday Factor Values')

    # -----------------------------------------------------

    # 图3c. 因子值时段 violinplot，早盘15min与次日相同时间节点15min

    ax = fig.add_subplot(spec[1, 8:12])

    sample_freq = cfg.sample_freq  # x min采样
    num_fac_group = 15 / sample_freq  # 每个group的因子数
    labels = ['9:30:00-9:45:00', '14:45:00-15:00:00']

    fac_group_list = []
    # 提取 9:30:00-9:45:00 因子值
    fac_group_list.append(np.array([val[:, :int(num_fac_group)].flatten()
                                    for val in factors.values()]).flatten())
    # 提取 14:45:00-15:00:00 因子值
    fac_group_list.append(np.array([val[:, int(15 * num_fac_group):].flatten()
                                    for val in factors.values()]).flatten())
    # Drop nan
    fac_group_list_rmna = [arr[~pd.isna(arr)] for arr in fac_group_list]

    # Make df for violinplot
    factor = np.concatenate(fac_group_list_rmna)
    group = [labels[i] for i in range(len(fac_group_list_rmna)) for j in range(fac_group_list_rmna[i].shape[0])]
    df = pd.DataFrame({'value': factor, 'time': group})

    # Violinplot
    sns.violinplot(x='time', y='value', data=df)
    ax.set_title('Intraday Factor Values')

    # -----------------------------------------------------

    # 图2. 因子值分布图

    ax = fig.add_subplot(spec[2, 0:6])

    factor_1d = np.array(all_data).reshape(-1)
    factor_1d_dropna = factor_1d[~pd.isna(factor_1d)]

    ax.hist(factor_1d_dropna, bins=20, edgecolor='white')
    ax.set_xlabel('factor value')
    ax.set_ylabel('count')
    ax.set_title('Total Factor Values')

    # ----------------------------------------------------

    # 图5. 因子值 mean-std 散点图，每个ticker对应一个点

    ax = fig.add_subplot(spec[2, 6:12])

    dates = list(factors.keys())
    fac_arr = np.concatenate([factors[date] for date in dates], axis=1)

    # Calculate mean, std
    ticker_fac_mean = np.nanmean(fac_arr, axis=1)
    ticker_fac_std = np.nanstd(fac_arr, axis=1)
    ticker_fac_CV = ticker_fac_std / ticker_fac_mean

    df = pd.DataFrame({'mean': ticker_fac_mean, 'std': ticker_fac_std, 'CV': ticker_fac_CV})

    # scatter plot
    sns.scatterplot(data=df, x='mean', y='std', size='CV', hue='CV')
    ax.set_title('Factor Coefficient of Variation (CV) for tickers')

    # --------------------------------------------------------

    # IC_arr = np.load(save_path(fac1, output_path, 'IC_arr.npy'))

    # 图6. IC 值时序日频 violinplot

    ax = fig.add_subplot(spec[3, :])

    dates = list(factors.keys())

    # Calculate daily mean IC
    daily_mean = np.nanmean(IC_arr, axis=1)

    # Make df for violinplot
    IC = IC_arr.flatten()
    n = IC_arr.shape[1]
    date = [date[4:] for date in dates for i in range(n)]
    df = pd.DataFrame({'IC': IC, 'date': date})

    # Violinplot
    sns.violinplot(x='date', y='IC', data=df)
    ax.set_title('Daily IC values')
    for tick in ax.get_xticklabels():
        tick.set_rotation(90)

    plt.twinx()
    plt.plot(np.cumsum(daily_mean), '--', label='Daily Mean IC Cumsum')
    plt.legend()
    plt.ylabel('Daily Mean IC Cumsum')

    # --------------------------------------------------------

    # 图7. IC 值分布图

    ax = fig.add_subplot(spec[4, 0:3])

    IC_1d = IC_arr.flatten()

    ax.hist(IC_1d, bins=20, edgecolor='white')
    ax.axvline(x=0, color='red')
    ax.set_xlabel('IC')
    ax.set_ylabel('count')
    ax.set_title('Total IC values')

    # --------------------------------------------------------

    # 图8a. IC值分时段 violinplot
    ax = fig.add_subplot(spec[4, 3:9])

    sample_freq = cfg.sample_freq  # x min采样
    num_IC_group = int(60 / sample_freq)  # 每个group的因子数
    # num_group = math.ceil(factors['20200401'][fac1].shape[1] / num_IC_group)  # group数量，现在为 4
    num_group = 4

    IC_group_list = []
    for i in range(num_group):
        arr = IC_arr[:, i * num_IC_group:(i + 1) * num_IC_group].flatten()
        IC_group_list.append(arr)

    # Make df for violinplot
    labels = ['9:30:00-10:30:00', '10:30:00-11:30:00', '13:00:00-14:00:00', '14:00:00-15:00:00']
    IC = np.concatenate(IC_group_list)
    group = [labels[i] for i in range(len(IC_group_list)) for j in range(IC_group_list[i].shape[0])]
    df = pd.DataFrame({'IC': IC, 'time': group})

    # Violinplot
    sns.violinplot(x='time', y='IC', data=df)
    ax.set_title('Intraday IC values')

    # --------------------------------------------------------

    ax = fig.add_subplot(spec[4, 9:12])

    # 图8b. IC值时段 violinplot，早盘15min

    sample_freq = cfg.sample_freq  # x min采样
    num_per_group = 15 / sample_freq  # 每个group的因子数
    labels = ['9:30:00-9:45:00', '14:45:00-15:00:00']

    IC_group_list = []
    # 提取 9:30:00-9:45:00 因子值
    IC_group_list.append(IC_arr[:, :int(num_per_group)].flatten())
    # # 提取 14:45:00-15:00:00 因子值
    # IC_group_list.append(IC_arr[:, 15*num_per_group:].flatten())
    # Drop nan
    # fac_group_list_rmna = [arr[~pd.isna(arr)] for arr in fac_group_list]

    # Make df for violinplot
    IC = np.concatenate(IC_group_list)
    group = [labels[i] for i in range(len(IC_group_list)) for j in range(IC_group_list[i].shape[0])]
    df = pd.DataFrame({'IC': IC, 'time': group})

    # Violinplot
    sns.violinplot(x='time', y='IC', data=df)
    ax.set_title('Intraday IC values')

    # ------------------------------------------------------------

    # IC 相关统计量计算
    # 全天 IC
    ax = fig.add_subplot(spec[5, :])

    stat_list = IC_stat(IC_arr)
    stat_list_round = [str('%.4f' % x) for x in stat_list]
    stat_str = (' ' * 18).join(stat_list_round)
    # 分时 IC
    sample_freq = cfg.sample_freq  # x min采样
    num_IC_group = int(60 / sample_freq)  # 每个group的因子数
    num_group = 4

    IC_stats_str_list = []
    for i in range(num_group):
        arr = IC_arr[:, i * num_IC_group:(i + 1) * num_IC_group]
        temp_stat_list = IC_stat(arr)
        IC_stats_str_list.append((' ' * 18).join([str('%.4f' % x) for x in temp_stat_list]))

    name_list = ['t', 'IC_mean', 'IC_mean_pos', 'IC_mean_neg', 'IC_std', 'IC_std_pos', 'IC_std_neg',
                 'IC_pos_ratio', 'IC_neg_ratio', 'IC_sig_ratio', 'IR', 'IC_num_con_pos_mean']
    name_list_filled = [f"{s:^20}" for s in name_list]
    name_str = (' ' * 2).join(name_list_filled)

    table_str = '-' * 320

    s = ""
    s += '\n' + table_str
    s += '\n' + name_str
    s += '\n' + table_str
    s += "\n" + 'All Day          ' + stat_str
    s += '\n' + table_str
    s += "\n" + '9:30-10:30       ' + IC_stats_str_list[0]
    s += '\n' + table_str
    s += "\n" + '10:30-11:30      ' + IC_stats_str_list[1]
    s += '\n' + table_str
    s += "\n" + '13:00-14:00      ' + IC_stats_str_list[2]
    s += '\n' + table_str
    s += "\n" + '14:00-15:00      ' + IC_stats_str_list[3]
    s += '\n' + table_str

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.text(0.5, 1, s,
            horizontalalignment='center',
            verticalalignment='top',
            transform=ax.transAxes,
            fontsize=14)
    ax.set_title('IC Related Statistics', fontsize=20)

    ICTable = pd.DataFrame(columns=name_list[1:], index=['All Day', '9:30-10:30', '10:30-11:30', '13:00-14:00', '14:00-15:00'])
    ICTable.iloc[0, :] = stat_str.split(' ' * 18)
    ICTable.iloc[1, :] = IC_stats_str_list[0].split(' ' * 18)
    ICTable.iloc[2, :] = IC_stats_str_list[1].split(' ' * 18)
    ICTable.iloc[3, :] = IC_stats_str_list[2].split(' ' * 18)
    ICTable.iloc[4, :] = IC_stats_str_list[3].split(' ' * 18)

    # ------------------------------- 跨日IC图
    # 图9. crossIC 值时序日频 violinplot

    ax = fig.add_subplot(spec[6, :])

    dates = list(factors.keys())

    # Calculate daily mean IC
    daily_mean = np.nanmean(crossIC_arr, axis=1)

    # Make df for violinplot
    crossIC = crossIC_arr.flatten()
    n = crossIC_arr.shape[1]
    date = [date[4:] for date in dates for i in range(n)]
    df = pd.DataFrame({'crossIC': crossIC, 'date': date})

    # Violinplot
    sns.violinplot(x='date', y='crossIC', data=df)
    ax.set_title('Daily crossIC values')
    for tick in ax.get_xticklabels():
        tick.set_rotation(90)

    plt.twinx()
    plt.plot(np.cumsum(daily_mean), '--', label='Daily Mean crossIC Cumsum')
    plt.legend()
    plt.ylabel('Daily Mean crossIC Cumsum')

    # --------------------------------------------------------

    # 图10. crossIC 值分布图

    ax = fig.add_subplot(spec[7, 0:3])

    crossIC_1d = crossIC_arr.flatten()

    ax.hist(crossIC_1d, bins=20, edgecolor='white')
    ax.axvline(x=0, color='red')
    ax.set_xlabel('crossIC')
    ax.set_ylabel('count')
    ax.set_title('Total crossIC values')

    # --------------------------------------------------------

    # 图11a. crossIC值分时段 violinplot

    ax = fig.add_subplot(spec[7, 3:9])

    sample_freq = cfg.sample_freq  # x min采样
    num_crossIC_group = int(60 / sample_freq)  # 每个group的因子数
    # num_group = math.ceil(factors['20200401'][fac1].shape[1] / num_IC_group)  # group数量，现在为 4
    num_group = 4

    crossIC_group_list = []
    for i in range(num_group):
        arr = crossIC_arr[:, i * num_crossIC_group:(i + 1) * num_crossIC_group].flatten()
        crossIC_group_list.append(arr)

    # Make df for violinplot
    labels = ['9:30:00-10:30:00', '10:30:00-11:30:00', '13:00:00-14:00:00', '14:00:00-15:00:00']
    crossIC = np.concatenate(crossIC_group_list)
    group = [labels[i] for i in range(len(crossIC_group_list)) for j in range(crossIC_group_list[i].shape[0])]
    df = pd.DataFrame({'crossIC': crossIC, 'time': group})

    # Violinplot
    sns.violinplot(x='time', y='crossIC', data=df)
    ax.set_title('Intraday crossIC values')

    # --------------------------------------------------------

    ax = fig.add_subplot(spec[7, 9:12])

    # 图8b. crossIC值时段 violinplot，早盘15min

    sample_freq = cfg.sample_freq  # x min采样
    num_per_group = 15 / sample_freq  # 每个group的因子数
    labels = ['9:30:00-9:45:00', '14:45:00-15:00:00']

    crossIC_group_list = []
    # 提取 9:30:00-9:45:00 因子值
    crossIC_group_list.append(crossIC_arr[:, :int(num_per_group)].flatten())
    # # 提取 14:45:00-15:00:00 因子值
    # IC_group_list.append(IC_arr[:, 15*num_per_group:].flatten())
    # Drop nan
    # fac_group_list_rmna = [arr[~pd.isna(arr)] for arr in fac_group_list]

    # Make df for violinplot
    crossIC = np.concatenate(crossIC_group_list)
    group = [labels[i] for i in range(len(crossIC_group_list)) for j in range(crossIC_group_list[i].shape[0])]
    df = pd.DataFrame({'crossIC': crossIC, 'time': group})

    # Violinplot
    sns.violinplot(x='time', y='crossIC', data=df)
    ax.set_title('Intraday crossIC values')

    # ------------------------------------------------------------

    # crossIC 相关统计量计算
    # 统计量表格
    ax = fig.add_subplot(spec[8, :])

    stat_list = IC_stat(crossIC_arr)
    stat_list_round = [str('%.4f' % x) for x in stat_list]
    stat_str = (' ' * 18).join(stat_list_round)
    # 分时 crossIC
    sample_freq = cfg.sample_freq  # x min采样
    num_crossIC_group = int(60 / sample_freq)  # 每个group的因子数
    num_group = 4

    crossIC_stats_str_list = []
    for i in range(num_group):
        arr = crossIC_arr[:, i * num_crossIC_group:(i + 1) * num_crossIC_group]
        temp_stat_list = IC_stat(arr)
        crossIC_stats_str_list.append((' ' * 18).join([str('%.4f' % x) for x in temp_stat_list]))

    name_list = ['t', 'crossIC_mean', 'crossIC_mean_pos', 'crossIC_mean_neg', 'crossIC_std', 'crossIC_std_pos', 'crossIC_std_neg',
                 'crossIC_pos_ratio', 'crossIC_neg_ratio', 'crossIC_sig_ratio', 'crossIR', 'crossIC_num_con_pos_mean']
    name_list_filled = [f"{s:^20}" for s in name_list]
    name_str = (' ' * 2).join(name_list_filled)

    table_str = '-' * 300

    s = ""
    s += '\n' + table_str
    s += '\n' + name_str
    s += '\n' + table_str
    s += "\n" + 'All Day            ' + stat_str
    s += '\n' + table_str
    s += "\n" + '9:30-10:30         ' + crossIC_stats_str_list[0]
    s += '\n' + table_str
    s += "\n" + '10:30-11:30        ' + crossIC_stats_str_list[1]
    s += '\n' + table_str
    s += "\n" + '13:00-14:00        ' + crossIC_stats_str_list[2]
    s += '\n' + table_str
    s += "\n" + '14:00-15:00        ' + crossIC_stats_str_list[3]
    s += '\n' + table_str

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.text(0.5, 1, s,
            horizontalalignment='center',
            verticalalignment='top',
            transform=ax.transAxes,
            fontsize=14)
    ax.set_title('crossIC Related Statistics', fontsize=20)

    crossICTable = pd.DataFrame(columns=name_list[1:], index=['All Day', '9:30-10:30', '10:30-11:30', '13:00-14:00', '14:00-15:00'])
    crossICTable.iloc[0, :] = stat_str.split(' ' * 18)
    crossICTable.iloc[1, :] = crossIC_stats_str_list[0].split(' ' * 18)
    crossICTable.iloc[2, :] = crossIC_stats_str_list[1].split(' ' * 18)
    crossICTable.iloc[3, :] = crossIC_stats_str_list[2].split(' ' * 18)
    crossICTable.iloc[4, :] = crossIC_stats_str_list[3].split(' ' * 18)

    ICTable = pd.concat([ICTable, crossICTable], axis=1)
    ICTable.index.name = "Interval"
    ICTable.to_csv(save_path(fac1, output_path, "ICTable.csv"))
    fig.suptitle("因子：" + fac1 + " 调仓周期: " + str(cfg.order_time) + "分钟 回测日期: " + cfg.start_date + " - " + cfg.end_date, fontsize=35)
    fig.tight_layout()
    plt.savefig(save_path(fac1, output_path, 'ic.png'))
    if os.path.exists('./pics/' + output_path):
        shutil.copy(save_path(fac1, output_path, 'ic.png'), './pics/' + output_path + '/{}_ic.png'.format(fac1))
    else:
        os.makedirs('./pics/' + output_path)
        shutil.copy(save_path(fac1, output_path, 'ic.png'), './pics/' + output_path + '/{}_ic.png'.format(fac1))
    # fig.savefig(output_path + '/{}ic.png'.format(fac1))


def group_longshort_backtest(ia, fac_name, num_group_backtest, l_ratio, s_ratio, output_path=None, other_fac_to_zip=None):
    ZIPRESListDict = RBT.RatioBackTest(ia, fac_name, LongRatio=l_ratio, ShortRatio=s_ratio, groupNum=num_group_backtest,
                                       other_fac_to_zip=other_fac_to_zip)
    group_sequence = get_rtn_sequence(ZIPRESListDict, cfg.beginTime, cfg.endTime, cfg.order_time)
    np.save(save_path(fac_name, output_path, 'group_sequence'), group_sequence)


def save_parameters(atp, tickers, fac_name, output_path='', using_rank=True, num_group_backtest=cfg.num_group_backtest,
                    l_ratio=cfg.long_ratio, s_ratio=cfg.short_ratio, refresh=True, other_fac_to_zip=None):
    vwap_flag = cfg.whether_vwap
    if refresh and (output_path != ''):
        time0 = time.time()
        save_factor(atp.factors, tickers, fac_name, output_path)
        print('save fac')
        print(time.time() - time0)
        save_IC(atp.factors, tickers, fac_name, output_path, using_rank=using_rank, vwap_flag=vwap_flag)
        print('IC')
        print(time.time() - time0)
        group_longshort_backtest(atp, fac_name, num_group_backtest, l_ratio=l_ratio, s_ratio=s_ratio, output_path=output_path,
                                 other_fac_to_zip=other_fac_to_zip)
        print(time.time() - time0)


def get_rtn_sequence(ZIPRES, beginTime, endTime, order_time):
    """
    得到某个返回值的时序涨跌序列，shape 为 日 x 日内期数
    """
    time0 = time.time()
    SEQ_indexs = list(ZIPRES.keys())
    SEQRES = {}

    beginTime = beginTime * order_time
    endTime = endTime * order_time

    # 对每个 rtn 项
    for SEQ_index in SEQ_indexs:
        SEQRES[SEQ_index] = None
        ZIPRES_ = ZIPRES[SEQ_index]

        # 时间限制，注意开闭
        ZIPRES_ = ZIPRES_[ZIPRES_['time'] >= beginTime]
        ZIPRES_ = ZIPRES_[ZIPRES_['time'] < endTime]

        ZIPRES_['unique_dt'] = ZIPRES_['date'] + ZIPRES_['time'].astype('str')
        tmp_res = ZIPRES_.groupby('unique_dt').mean()
        tmp_res.sort_values('unique_dt', inplace=True)
        SEQRES[SEQ_index] = tmp_res
        SEQRES[SEQ_index].reset_index(drop=True, inplace=True)
    return SEQRES


def Group_Plot(fig, spec, group_sequence):

    mean_np = []
    mean_nd = []
    group_num = len(group_sequence.keys()) - 2
    for group in range(group_num):
        group_mean = group_sequence[str(group)]
        mean_np.append(group_mean['alpha_1'].mean())
        mean_nd.append(group_mean['alpha_2next'].mean())

    # 图 1 ： Next 分层单调性图
    ax = fig.add_subplot(spec[1:3, 0:4])
    ax.set_title("分层测试（分层组数）每层平均 Rtn (超额，下一期 Rtn)")
    ax.plot(mean_np, c='b', marker='o', label='每层平均 Rtn（超额）')
    ax.set_xlabel("分层测试的层")
    ax.set_ylabel("平均 Rtn")
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.legend()
    plt.yticks(size=15)

    # 图 2 ： Next 分层时序收益图
    ax = fig.add_subplot(spec[3:5, 0:6])
    ax.set_title("分层测试（分层组数）每层 Rtn (超额，下一期 Rtn) 时序累加")
    for group in range(group_num):
        if group < 7:
            ax.plot(group_sequence[str(group)]['alpha_1'].cumsum(),
                    label='第 ' + str(group) + ' 组累计超额 Rtn', alpha=1, c=color_board[group], linewidth=0.9)
        else:
            ax.plot(group_sequence[str(group)]['alpha_1'].cumsum(),
                    label='第 ' + str(group) + ' 组累计超额 Rtn', alpha=1, linewidth=0.9)
    ax.set_xlabel("期数")
    ax.set_ylabel("超额 Rtn")
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    if group_num < 7:
        ax.legend(fontsize=18)
    else:
        ax.legend()
    plt.yticks(size=15)

    # 图 3 ： Next 分层每时间段收益图
    by_min_np = []
    by_min_nd = []
    for group in range(group_num):
        group_seq = group_sequence[str(group)]
        by_min_np.append(group_seq.groupby('time').mean()['alpha_1'])
        by_min_nd.append(group_seq.groupby('time').mean()['alpha_2next'])

        # 总的有数据的分钟数
    num_minute = by_min_np[0].shape[0]

    # 半小时一根柱子
    num_bar = num_minute // 30
    if (num_minute % 30) > 5:
        num_bar += 1

    # 横轴的总宽和柱宽
    X = np.array(list(range(1, num_bar + 1))) * 4
    width = 4 / (group_num + 3)

    ax = fig.add_subplot(spec[5:7, 0:6])
    ax.set_title("分层测试每层（分层组数） Rtn (超额，下一期 Rtn) 日内每隔 " + str(30) + " 分钟求均值")
    ax.set_xlabel("时间轴 | 每个柱状区间为 " + str(30) + " 分钟")
    ax.set_ylabel("Rtn (超额)")

    for group in range(group_num):
        # 每次画一层的，分成 num_bar 组
        group_plot = []
        data = by_min_np[group]
        for bar_index in range(num_bar):
            group_plot.append(data.iloc[30 * bar_index: 30 * (bar_index + group)].mean())
        ax.bar(X + group * width, group_plot, label='第 ' + str(group) + ' 组的平均 Rtn （超额）', width=width)

    # 横轴时间
    time_gap = 30

    begin_minute = by_min_np[0].index[0]
    x_ticks_pre = generate_ts(begin_minute, time_gap, num_bar + 1)
    x_ticks = []
    for i in range(len(x_ticks_pre) - 1):
        x_ticks.append(x_ticks_pre[i] + " - " + x_ticks_pre[i + 1])
    ax.set_xticks(X + 1 * width)
    ax.set_xticklabels(x_ticks, rotation=10)
    ax.legend()
    plt.yticks(size=15)

    # 图 4 ： nextSame 分层测试单调性图
    ax = fig.add_subplot(spec[1:3, 4:8])
    ax.set_title("分层测试（分层组数）每层平均 Rtn (超额，到次日相同时间节点 Rtn)")
    ax.plot(mean_nd, c='b', marker='o', label='平均 Rtn (超额）')
    ax.set_xlabel("分层测试的每层")
    ax.set_ylabel("平均 Rtn (超额）")
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.legend()
    plt.yticks(size=15)

    # 图 5 ： nextSame 分层时序收益图
    ax = fig.add_subplot(spec[3:5, 6:12])
    ax.set_title("分层测试（分层组数）每层 Rtn (超额，到次日相同时间节点 Rtn) 时序累加")
    for group in range(group_num):
        if group < 7:
            ax.plot(group_sequence[str(group)]['alpha_2next'].cumsum(),
                    label='第 ' + str(group) + ' 组累计超额 Rtn', alpha=1, c=color_board[group], linewidth=0.9)
        else:
            ax.plot(group_sequence[str(group)]['alpha_2next'].cumsum(),
                    label='第 ' + str(group) + ' 组累计超额 Rtn', alpha=1, linewidth=0.9)
    ax.set_xlabel("期数")
    ax.set_ylabel("超额")
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    if group_num < 7:
        ax.legend(fontsize=18)
    else:
        ax.legend()
    plt.yticks(size=15)

    # 图 6 ： nextSame 分层每时间段收益图

    ax = fig.add_subplot(spec[5:7, 6:12])
    ax.set_title("分层测试（分层组数）每层 Rtn (超额，到次日相同时间节点 Rtn) 日内每隔 " + str(30) + " 分钟求均值")
    ax.set_xlabel("时间轴 | 每个柱状区间为 " + str(30) + " 分钟")
    ax.set_ylabel("Rtn (超额)")

    for group in range(group_num):
        # 每次画一层的，分成 num_bar 组
        group_plot = []
        data = by_min_nd[group]
        for bar_index in range(num_bar):
            group_plot.append(data.iloc[30 * bar_index: 30 * (bar_index + 1)].mean())
        ax.bar(X + group * width, group_plot, label='超额 for Group ' + str(group), width=width)

    time_gap = 30
    begin_minute = cfg.order_time * cfg.beginTime
    x_ticks_pre = generate_ts(begin_minute, time_gap, num_bar + 1)
    x_ticks = []
    for i in range(len(x_ticks_pre) - 1):
        x_ticks.append(x_ticks_pre[i] + " - " + x_ticks_pre[i + 1])
    ax.set_xticks(X + 1 * width)
    ax.set_xticklabels(x_ticks, rotation=10)
    ax.legend()
    plt.yticks(size=15)
    fig.tight_layout()


def LS_Plot(fig, spec, group_sequence, fac1, output_path=None):
    if output_path is None:
        output_path = get_latest_time(fac1)
    # 图 7 ： Next 多空测试，按照用户给定的比例
    next_long_alpha = np.array(group_sequence['Long']['alpha_1'])
    next_short_alpha = np.array(group_sequence['Short']['alpha_1'])

    ax = fig.add_subplot(spec[7:9, 0:6])
    ax.set_title("多空测试（多空组 ratio）时序 Rtn (超额，下一期 Rtn) 累加", fontsize=12)
    ax.set_xlabel("期数", fontsize=12)
    ax.set_ylabel("Return（超额）", fontsize=12)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    line1 = ax.plot((next_long_alpha + next_short_alpha).cumsum() / 2, c='blue', linewidth=0.9)
    line2 = ax.plot(next_long_alpha.cumsum(), c='red', linewidth=0.9)
    line3 = ax.plot(next_short_alpha.cumsum(), c='green', linewidth=0.9)
    ax.legend(line1 + line2 + line3, ['多空组合 Rtn', '多组超额 Rtn', '空组超额 Rtn'])
    plt.yticks(size=15)

    # 图 8 ： nextSame 多空测试
    nextSame_long_alpha = np.array(group_sequence['Long']['alpha_2next'])
    nextSame_short_alpha = np.array(group_sequence['Short']['alpha_2next'])

    ax = fig.add_subplot(spec[7:9, 6:12])
    ax.set_title("多空测试（多空组 ratio）时序 Rtn（超额，到次日相同时间节点）累加", fontsize=12)
    ax.set_xlabel("期数", fontsize=12)
    ax.set_ylabel("Rtn（超额）", fontsize=12)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    line1 = ax.plot((nextSame_long_alpha + nextSame_short_alpha).cumsum() / 2, c='blue', linewidth=0.9)
    line2 = ax.plot(nextSame_long_alpha.cumsum(), c='red', linewidth=0.9)
    line3 = ax.plot(nextSame_short_alpha.cumsum(), c='green', linewidth=0.9)
    ax.legend(line1 + line2 + line3, ['多空组合 Rtn', '多组超额 Rtn', '空组超额 Rtn'])
    plt.yticks(size=15)

    # 图 9 ： 因子衰减
    ax = fig.add_subplot(spec[1:3, 8:12])

    long_mean = group_sequence['Long'].mean()
    short_mean = group_sequence['Short'].mean()

    decay_coefficients = cfg.apd_rtn['rtn']
    decay_coefficients.sort()
    decay_array_long = []
    decay_array_short = []
    for coef in decay_coefficients:
        decay_array_long.append(long_mean['alpha_' + str(coef)])
        decay_array_short.append(short_mean['alpha_' + str(coef)])
    decay_array_long.append(long_mean['alpha_2close'])
    decay_array_short.append(short_mean['alpha_2close'])
    decay_array_long.append(long_mean['alpha_2nextopen'])
    decay_array_short.append(short_mean['alpha_2nextopen'])
    decay_array_long.append(long_mean['alpha_2next'])
    decay_array_short.append(short_mean['alpha_2next'])
    decay_array_long = np.array(decay_array_long)
    decay_array_short = np.array(decay_array_short)

    baseline = np.zeros(decay_array_long.shape)
    ax.set_title("多空测试（多空组 ratio）对下一期进行时间采样的平均 Rtn（超额）")
    x_ticks = decay_coefficients.copy()
    for i in range(3):
        x_ticks.extend([x_ticks[-1] + 0.2])
    x_labels = []
    for decay_coefficient in decay_coefficients:
        x_labels.append(round(decay_coefficient, 2))
    x_labels.extend(['close', 'nextopen', 'next'])
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels)
    ax.plot(x_ticks, decay_array_short, c='green', marker='o', label='空组平均 Rtn')
    ax.plot(x_ticks, decay_array_long, c='red', marker='o', label='多组平均 Rtn')
    ax.plot(x_ticks, (decay_array_short + decay_array_long) / 2, c='blue', marker='o', label='多空组平均 Rtn')
    ax.plot(x_ticks, baseline, linestyle="--")

    ax.set_xlabel("采样点位置")
    ax.set_ylabel("平均 Rtn （超额）")
    ax.legend()
    plt.yticks(size=15)

    # 表 1 ：统计量表格

    # 分层测试统计量保存
    num_group = len(group_sequence.keys()) - 2
    std_group = []
    mean_group = []
    max_group = []
    min_group = []
    for i in range(num_group):
        std_group.append(group_sequence[str(i)]['alpha_1'].std())
        mean_group.append(group_sequence[str(i)]['alpha_1'].mean())
        max_group.append(group_sequence[str(i)]['alpha_1'].max())
        min_group.append(group_sequence[str(i)]['alpha_1'].min())

    std_group_nextSame = []
    mean_group_nextSame = []
    max_group_nextSame = []
    min_group_nextSame = []
    for i in range(num_group):
        std_group_nextSame.append(group_sequence[str(i)]['alpha_2next'].std())
        mean_group_nextSame.append(group_sequence[str(i)]['alpha_2next'].mean())
        max_group_nextSame.append(group_sequence[str(i)]['alpha_2next'].max())
        min_group_nextSame.append(group_sequence[str(i)]['alpha_2next'].min())

    # 表格
    ax = fig.add_subplot(spec[9:11, 0:12])
    ax.axis('off')
    # 表格行名
    rowLabels = ['', "Long Short", "Long Alpha", "Short Alpha"]
    # 表格每一行数据
    cellText = [['rtn_std', 'rtn_mean', 'rtn_max', 'rtn_min', 'rtn_nextSame_std', 'rtn_nextSame_mean', 'rtn_nextSame_max',
                 'rtn_nextSame_min']]
    celltext_longshort(cellText, next_long_alpha + next_short_alpha, nextSame_long_alpha + nextSame_short_alpha)
    celltext_longshort(cellText, next_long_alpha, nextSame_long_alpha)
    celltext_longshort(cellText, next_short_alpha, nextSame_short_alpha)

    for i in range(len(std_group)):
        rowLabels.append(str(i))
        cellText.append([])
        cellText[-1].append(str(float('%.3g' % std_group[i])))
        cellText[-1].append(str(float('%.3g' % mean_group[i])))
        cellText[-1].append(str(float('%.3g' % max_group[i])))
        cellText[-1].append(str(float('%.3g' % min_group[i])))
        cellText[-1].append(str(float('%.3g' % std_group_nextSame[i])))
        cellText[-1].append(str(float('%.3g' % mean_group_nextSame[i])))
        cellText[-1].append(str(float('%.3g' % max_group_nextSame[i])))
        cellText[-1].append(str(float('%.3g' % min_group_nextSame[i])))

    table = ax.table(cellText=cellText, rowLabels=rowLabels, loc='center', cellLoc='center', rowLoc='center')
    table.scale(0.8, 2.5)
    table.auto_set_font_size(False)
    table.set_fontsize(15)
    fig.suptitle("因子：" + fac1 + " 调仓周期: " + str(cfg.order_time) + "分钟 回测日期: " + cfg.start_date + " - " + cfg.end_date, fontsize=25)
    fig.tight_layout()
    print("如果中文显示为方框，命令行运行 rm -rf ~/.cache/matplotlib/* ")
    plt.savefig(save_path(fac1, output_path, 'group_result.png'))
    if os.path.exists('./pics/' + output_path):
        shutil.copy(save_path(fac1, output_path, 'group_result.png'),
                    './pics/' + output_path + '/{}_group_result.png'.format(fac1))
    else:
        os.makedirs('./pics/' + output_path)
        shutil.copy(save_path(fac1, output_path, 'group_result.png'),
                    './pics/' + output_path + '/{}_group_result.png'.format(fac1))


def plotFig2(fac1, output_path=''):
    if len(output_path) >= 1:
        group_sequence = np.load(save_path(fac1, output_path, 'group_sequence.npy'), allow_pickle=True).item()
    else:
        group_sequence = np.load(save_path(fac1, get_latest_time(fac1), 'group_sequence.npy'), allow_pickle=True).item()

    fig = plt.figure(figsize=(25, 50), dpi=cfg.fig2_dpi)
    fig.set_facecolor('#FFFFFF')
    plt.subplots_adjust(wspace=1, hspace=0.2)
    spec = gridspec.GridSpec(ncols=12, nrows=16)

    Group_Plot(fig, spec, group_sequence)
    LS_Plot(fig, spec, group_sequence, fac1)


def plotFigs(fac_name, output_path=''):
    plotFig1(fac_name, output_path)
    plotFig2(fac_name, output_path)


def industry_IC_stat(factors, tickers, fac_name, industry_dict, using_rank=True, vwap_flag=False):
    """
    分行业进行 IC 统计
    :param factors:
    :param tickers:
    :param fac_name:
    :param industry_dict:
    :param using_rank:
    :param vwap_flag:

    :return:
    """

    # 是否采用 vwap return
    if vwap_flag:
        rtn1 = 'rtn1vwap'
    else:
        rtn1 = 'rtn1'

    # 回测日期
    dates = list(factors.keys())

    # 计算 IC = corr(fac(t), ret(t+1))
    # 存储成 2d array，每一行为一天该因子的 IC 序列

    ticker_dict = {}
    for i, ticker in enumerate(tickers):
        ticker_dict[ticker] = i

    industry_IC_list = []
    for industry in industry_dict.keys():
        ticker_list = industry_dict[industry]
        idx_list = [ticker_dict[ticker] for ticker in ticker_list if ticker in ticker_dict.keys()]

        ICs = []
        for date in dates:
            ret = factors[date][rtn1][idx_list, :]
            # print(factors[date][fac1])
            if using_rank:
                # 改成求 rank，用 nan 替代 nan
                fac = rankdata(factors[date][fac_name][idx_list, :], axis=0)
                fac[pd.isna(factors[date][fac_name][idx_list, :])] = np.nan
            else:
                fac = factors[date][fac_name][idx_list, :]

            IC = []
            for i in range(fac.shape[1] - 1):
                # 取截面因子值和 return
                fac_sec = fac[:, i]
                ret_next_sec = ret[:, i + 1]
                # drop nan
                fac_sec_rmna = fac_sec[(~pd.isna(fac_sec)) & (~pd.isna(ret_next_sec))]
                ret_next_sec_rmna = ret_next_sec[(~pd.isna(fac_sec)) & (~pd.isna(ret_next_sec))]
                # 计算 IC
                if len(fac_sec_rmna) == 0:
                    IC.append(np.nan)
                else:
                    if np.all(fac_sec_rmna == fac_sec_rmna[0]) or np.all(ret_next_sec_rmna == ret_next_sec_rmna[0]):
                        # 若 x, y 中有一个为恒定值，则取 correlation 为 0
                        IC.append(np.nan)
                    else:
                        IC.append(np.corrcoef(fac_sec_rmna, ret_next_sec_rmna)[0, 1])
            ICs.append(IC)

        IC_arr = np.array(ICs)
        industry_IC_list.append(IC_arr)

    # IC 相关统计量计算
    name_list = ['        ', 'IC_mean', 'IC_mean_pos', 'IC_mean_neg', 'IC_std', 'IC_std_pos', 'IC_std_neg',
                 'IC_pos_ratio', 'IC_neg_ratio', 'IC_sig_ratio', 'IR', 'IC_num_con_pos_mean']
    total_stat_list = []
    for IC_arr, industry in zip(industry_IC_list, industry_dict.keys()):
        IC_mean = np.nanmean(IC_arr)
        IC_mean_positive = np.nanmean(IC_arr[IC_arr > 0])
        IC_mean_negative = np.nanmean(IC_arr[IC_arr < 0])

        IC_std = np.nanstd(IC_arr)
        IC_std_positive = np.nanstd(IC_arr[IC_arr > 0])
        IC_std_negative = np.nanstd(IC_arr[IC_arr < 0])

        IC_positive_ratio = np.sum(IC_arr > 0) / np.sum(~pd.isna(IC_arr))
        IC_negative_ratio = np.sum(IC_arr < 0) / np.sum(~pd.isna(IC_arr))

        IC_significance_ratio = np.sum(abs(IC_arr) > 0.02) / np.sum(~pd.isna(IC_arr))

        IR = IC_mean / IC_std

        num_consecutive_positive_mean = np.mean(
            [sum(1 for _ in group) for key, group in groupby(IC_arr.flatten() > 0) if key])

        # 统计量表格
        stat_list = [IC_mean, IC_mean_positive, IC_mean_negative, IC_std, IC_std_positive, IC_std_negative,
                     IC_positive_ratio, IC_negative_ratio, IC_significance_ratio, IR, num_consecutive_positive_mean]
        stat_list_round = [industry] + [str(round(x, 4)) for x in stat_list]
        total_stat_list.append(stat_list_round)

    # 按 IC_mean 从大到小排序
    total_stat_list.sort(key=lambda x: x[1], reverse=True)

    # 去除nan（有的行业中所有票都不在票池中）
    total_stat_list_dropna = [x for x in total_stat_list if 'nan' not in x]

    # 画表格
    fig, ax = plt.subplots(figsize=(15, 5))
    ax.axis('off')
    table = ax.table(cellText=total_stat_list_dropna, colLabels=name_list)
    table.scale(1, 2)
    table.auto_set_font_size(False)
    table.set_fontsize(12)

    fig.tight_layout()
    plt.show()


def concat_ICTables(factor_names):
    '''
    批量读取存储ICTables add at 20200331 yjsun
    :param factor_names:
    :return:
    '''
    # 必须在plotfigs 运行之后使用
    res = []
    for f in factor_names:
        try:
            df = pd.read_csv(save_path(f, get_latest_time(f), 'ICTable.csv'))
            df['factorName'] = f
            res.append(df)
        except:
            print(save_path(f, get_latest_time(f), 'ICTable.csv'))
    df = pd.concat(res)
    return df


def concat_Decay(factor_names):
    res = []

    for factor in factor_names:
        try:
            decay = np.load(save_path(factor, get_latest_time(factor), 'decay_dict.npy'), allow_pickle=True).item()

            inday_short = (-decay['decay_rtn'][0] + decay['decay_index_rtn'][0])[10]
            inday_long = (decay['decay_rtn'][1] - decay['decay_index_rtn'][0])[10]
            crossday_short = (-decay['decay_rtn'][0] + decay['decay_index_rtn'][0])[-1]
            crossday_long = (decay['decay_rtn'][1] - decay['decay_index_rtn'][0])[-1]

            data = {'factorName': factor,
                    'IndayLong': inday_long,
                    'IndayShort': inday_short,
                    'CrossdayLong': crossday_long,
                    'CrossdayShort': crossday_short
                    }
            res.append(data)

        except:
            print(save_path(factor, get_latest_time(factor), 'decay_dict.npy'))

    return pd.DataFrame(res)


# 主因子与辅助因子收益关系
def zip_main_support(ia, date, main_factor_name, support_factor_name, longshort_ratio=None, time='n'):
    if longshort_ratio is None:
        longshort_ratio = cfg.longshort_ratio
    # load data
    if time == 'n':
        main_factor = (ia.factors[date][main_factor_name]).astype(np.float64)[:, 30:-30]
        support_factor = (ia.factors[date][support_factor_name]).astype(np.float64)[:, 30:-30]
        alpha_rtn1 = (ia.factors[date]['rtn1vwap'] - ia.factors[date]['indexrtn']).astype(np.float64)[:, 30:-30]
        alpha_rtnnext = (ia.factors[date]['2next'] - ia.factors[date]['index2next']).astype(np.float64)[:, 30:-30]
    elif time == 'e':
        main_factor = (ia.factors[date][main_factor_name]).astype(np.float64)[:, :30]
        support_factor = (ia.factors[date][support_factor_name]).astype(np.float64)[:, :30]
        alpha_rtn1 = (ia.factors[date]['rtn1vwap'] - ia.factors[date]['indexrtn']).astype(np.float64)[:, :30]
        alpha_rtnnext = (ia.factors[date]['2next'] - ia.factors[date]['index2next']).astype(np.float64)[:, :30]
    elif time == 'l':
        main_factor = (ia.factors[date][main_factor_name]).astype(np.float64)[:, -30:]
        support_factor = (ia.factors[date][support_factor_name]).astype(np.float64)[:, -30:]
        alpha_rtn1 = (ia.factors[date]['rtn1vwap'] - ia.factors[date]['indexrtn']).astype(np.float64)[:, -30:]
        alpha_rtnnext = (ia.factors[date]['2next'] - ia.factors[date]['index2next']).astype(np.float64)[:, -30:]
    else:
        print("因子分组图 time 参数有误")
        raise
    # main_factor = (ia.factors[date][main_factor_name]).astype(np.float64)[:, 30:]
    # support_factor = (ia.factors[date][support_factor_name]).astype(np.float64)[:, 30:]
    # alpha_rtn1 = (ia.factors[date]['rtn1vwap'] - ia.factors[date]['indexrtn']).astype(np.float64)[:, 30:]
    # alpha_rtnnext = (ia.factors[date]['2next'] - ia.factors[date]['index2next']).astype(np.float64)[:, 30:]

    # 日内分辨度
    mask = np.isnan(main_factor) | np.isnan(alpha_rtn1)
    mask_main_factor = pd.DataFrame(np.where(mask, np.nan, main_factor)).rank(pct=True, axis=0).values

    longdf_inday = pd.DataFrame(data={'alpha': alpha_rtn1[mask_main_factor > (1 - longshort_ratio)],
                                      'support_factor': support_factor[
                                          mask_main_factor > (1 - longshort_ratio)]})
    shortdf_inday = pd.DataFrame(data={'alpha': alpha_rtn1[mask_main_factor < (longshort_ratio)],
                                       'support_factor': support_factor[mask_main_factor < (longshort_ratio)]})

    # 日间分辨度
    mask = np.isnan(main_factor) | np.isnan(alpha_rtnnext)
    mask_main_factor = pd.DataFrame(np.where(mask, np.nan, main_factor)).rank(pct=True, axis=0).values

    longdf_crossday = pd.DataFrame(data={'alpha': alpha_rtnnext[mask_main_factor > (1 - longshort_ratio)],
                                         'support_factor': support_factor[
                                             mask_main_factor > (1 - longshort_ratio)]})
    shortdf_crossday = pd.DataFrame(data={'alpha': alpha_rtnnext[mask_main_factor < (longshort_ratio)],
                                          'support_factor': support_factor[
                                              mask_main_factor < (longshort_ratio)]})

    return longdf_inday, shortdf_inday, longdf_crossday, shortdf_crossday


def zip_main_support_thrd(ia, date, main_factor_name, support_factor_name, longshort_ratio=None, time='n'):
    if longshort_ratio is None:
        longshort_ratio = cfg.longshort_ratio
    # load data
    if time == 'n':
        main_factor = (ia.factors[date][main_factor_name]).astype(np.float64)[:, 30:210]
        support_factor = (ia.factors[date][support_factor_name]).astype(np.float64)[:, 30:210]
        alpha_rtn1 = (ia.factors[date]['rtn1vwap'] - ia.factors[date]['indexrtn']).astype(np.float64)[:, 30:210]
        alpha_rtnnext = (ia.factors[date]['2next'] - ia.factors[date]['index2next']).astype(np.float64)[:, 30:210]
    elif time == 'e':
        main_factor = (ia.factors[date][main_factor_name]).astype(np.float64)[:, :30]
        support_factor = (ia.factors[date][support_factor_name]).astype(np.float64)[:, :30]
        alpha_rtn1 = (ia.factors[date]['rtn1vwap'] - ia.factors[date]['indexrtn']).astype(np.float64)[:, :30]
        alpha_rtnnext = (ia.factors[date]['2next'] - ia.factors[date]['index2next']).astype(np.float64)[:, :30]
    elif time == 'l':
        main_factor = (ia.factors[date][main_factor_name]).astype(np.float64)[:, 210:]
        support_factor = (ia.factors[date][support_factor_name]).astype(np.float64)[:, 210:]
        alpha_rtn1 = (ia.factors[date]['rtn1vwap'] - ia.factors[date]['indexrtn']).astype(np.float64)[:, 210:]
        alpha_rtnnext = (ia.factors[date]['2next'] - ia.factors[date]['index2next']).astype(np.float64)[:, 210:]
    else:
        print("因子分组图 time 参数有误")
        raise

    # 日内分辨度
    mask = np.isnan(main_factor) | np.isnan(alpha_rtn1)
    mask_main_factor = pd.DataFrame(np.where(mask, np.nan, main_factor)).rank(pct=True, axis=0).values

    longdf_inday = pd.DataFrame(data={'alpha': alpha_rtn1[mask_main_factor > (1 - longshort_ratio)],
                                      'support_factor': support_factor[
                                          mask_main_factor > (1 - longshort_ratio)]})
    shortdf_inday = pd.DataFrame(data={'alpha': alpha_rtn1[mask_main_factor < (longshort_ratio)],
                                       'support_factor': support_factor[mask_main_factor < (longshort_ratio)]})

    # 日间分辨度
    mask = np.isnan(main_factor) | np.isnan(alpha_rtnnext)
    mask_main_factor = pd.DataFrame(np.where(mask, np.nan, main_factor)).rank(pct=True, axis=0).values

    longdf_crossday = pd.DataFrame(data={'alpha': alpha_rtnnext[mask_main_factor > (1 - longshort_ratio)],
                                         'support_factor': support_factor[
                                             mask_main_factor > (1 - longshort_ratio)]})
    shortdf_crossday = pd.DataFrame(data={'alpha': alpha_rtnnext[mask_main_factor < (longshort_ratio)],
                                          'support_factor': support_factor[
                                              mask_main_factor < (longshort_ratio)]})

    return longdf_inday, shortdf_inday, longdf_crossday, shortdf_crossday


def plot_main_support(ia, main_factor_name, support_factor_name, longshort_ratio=None, time='n'):
    longdf_inday_list = []
    shortdf_inday_list = []
    longdf_crossday_list = []
    shortdf_crossday_list = []
    for date in ia.factors.keys():
        longdf_inday, shortdf_inday, longdf_crossday, shortdf_crossday = zip_main_support(ia, date, main_factor_name,
                                                                                          support_factor_name,
                                                                                          longshort_ratio=longshort_ratio, time=time)
        longdf_inday_list.append(longdf_inday)
        shortdf_inday_list.append(shortdf_inday)
        longdf_crossday_list.append(longdf_crossday)
        shortdf_crossday_list.append(shortdf_crossday)

    marker = None
    if time == 'n':
        marker = "去掉早尾盘"
    elif time == 'l':
        marker = "尾盘"
    elif time == 'e':
        marker = "早盘"
    else:
        print("你能看到这句话的话，项目代码肯定错了")

    gapDict = {}
    # 作图
    fig = plt.figure(figsize=(15, 10), dpi=128)
    fig.set_facecolor('#FFFFFF')
    plt.subplots_adjust(wspace=0.25, hspace=0.25)
    spec = gridspec.GridSpec(ncols=2, nrows=2)
    fig.suptitle(marker + f"\n Long/Short Part of {main_factor_name}\n vs\n Rank of {support_factor_name}", fontsize=15)
    # plt.grid(False)

    # 日内超额与辅助因子关系
    ax = fig.add_subplot(spec[0, 0])
    df = pd.concat(longdf_inday_list)
    df['support_factor_group'] = np.round(df['support_factor'].rank(pct=True) * 10, 0)
    groupdf = df.groupby('support_factor_group').mean().sort_values("support_factor")
    gap = groupdf['alpha'].max() - groupdf['alpha'].min()
    gapDict['Inday Long Gap'] = gap
    gapDict['Inday Long Corr'] = df[['support_factor', 'alpha']].corr().iloc[0, 1]
    ax.plot([0, 10], [0, 0], '--r', c='gray')
    ax.plot(groupdf.index, groupdf['alpha'], '--r', c='red', marker="X", markersize=5)
    ax.set_title(f"Inday Long Part Alpha,\n gap = {np.round(gap, 3)}, corr = {np.round(gapDict['Inday Long Corr'], 3)}")

    # 日间超额与辅助因子关系
    ax = fig.add_subplot(spec[0, 1])
    df = pd.concat(longdf_crossday_list)
    df['support_factor_group'] = np.round(df['support_factor'].rank(pct=True) * 10, 0)
    groupdf = df.groupby('support_factor_group').mean().sort_values("support_factor")
    gap = groupdf['alpha'].max() - groupdf['alpha'].min()
    gapDict['Crossday Long Gap'] = gap
    gapDict['Crossday Long Corr'] = df[['support_factor', 'alpha']].corr().iloc[0, 1]
    ax.plot([0, 10], [0, 0], '--r', c='gray')
    ax.plot(groupdf.index, groupdf['alpha'], '--r', c='red', marker="X", markersize=5)
    ax.set_title(f"Crossday Long Part Alpha,\n gap = {np.round(gap, 3)}, corr = {np.round(gapDict['Crossday Long Corr'], 3)}")

    mean = groupdf['alpha'].mean()

    # 日内超额与辅助因子关系
    ax = fig.add_subplot(spec[1, 0])
    df = pd.concat(shortdf_inday_list)
    df['support_factor_group'] = np.round(df['support_factor'].rank(pct=True) * 10, 0)
    groupdf = df.groupby('support_factor_group').mean().sort_values("support_factor")
    gap = groupdf['alpha'].max() - groupdf['alpha'].min()
    gapDict['Inday Short Gap'] = gap
    gapDict['Inday Short Corr'] = df[['support_factor', 'alpha']].corr().iloc[0, 1]
    ax.plot([0, 10], [0, 0], '--r', c='gray')
    ax.plot(groupdf.index, groupdf['alpha'], '--r', c='green', marker="X", markersize=5)
    ax.set_title(f"Inday Short Part Alpha,\n gap = {np.round(gap, 3)}, corr = {np.round(gapDict['Inday Short Corr'], 3)}")

    # 日内超额与辅助因子关系
    ax = fig.add_subplot(spec[1, 1])
    df = pd.concat(shortdf_crossday_list)
    df['support_factor_group'] = np.round(df['support_factor'].rank(pct=True) * 10, 0)
    groupdf = df.groupby('support_factor_group').mean().sort_values("support_factor")
    gap = groupdf['alpha'].max() - groupdf['alpha'].min()
    gapDict['Crossday Short Gap'] = gap
    gapDict['Crossday Short Corr'] = df[['support_factor', 'alpha']].corr().iloc[0, 1]
    ax.plot([0, 10], [0, 0], '--r', c='gray')
    ax.plot(groupdf.index, groupdf['alpha'], '--r', c='green', marker="X", markersize=5)
    ax.set_title(f"Crossday Short Part Alpha,\n gap = {np.round(gap, 3)}, corr = {np.round(gapDict['Crossday Short Corr'], 3)}")

    plt.show()

    return gapDict, mean


def get_random_tickers(sample_date='20200101', n=1200, seed=0, save=False):
    """
    全市场中选取随机股池
    :param sample_date: 用来判定是否次新、是否ST的日期
    :param n: 随机选取的票数
    :param seed: 随机种子
    :return:
    """
    connector = PqiDataSdk(user=cfg.user, pool_type="mt", offline=True)
    sample_df = connector.get_eod_history(fields=['OpenPrice', 'STStatus', 'TradeStatus'], source="stock", start_date='20180101',
                                          end_date=sample_date)
    fmtickers = list(sample_df['OpenPrice'][sample_date].index)

    # 去除ST,退市
    ST = sample_df['STStatus'][~(sample_df['STStatus'][sample_date] == 0)].index.tolist()

    # 去除新上市的，粗略定义为过去60日均没有开盘价的股票
    New = (sample_df['OpenPrice'].isnull() | (sample_df['OpenPrice'] == 0)).rolling(60, axis=1).sum()
    New = New[sample_date][New[sample_date] == 60].index.tolist()

    # 在剩下的股票中挑选n支票
    remain = [x for x in fmtickers if (x not in New and x not in ST)]
    np.random.seed(seed)
    random_tickers = np.random.choice(remain, n, replace=False)

    # 保存为npy
    if save:
        np.save(f"stock_pool_{sample_date}_{n}_{seed}", random_tickers, allow_pickle=True)
    return random_tickers


def zip_main_support_GroupIC(ia, date, main_factor_name, support_factor_name):
    # load data
    main_factor = (ia.factors[date][main_factor_name]).astype(np.float64)[:, 30:-30]
    support_factor = (ia.factors[date][support_factor_name]).astype(np.float64)[:, 30:-30]
    alpha_rtn1 = (ia.factors[date]['rtn1vwap'] - ia.factors[date]['indexrtn']).astype(np.float64)[:, 30:-30]
    alpha_rtnnext = (ia.factors[date]['2next'] - ia.factors[date]['index2next']).astype(np.float64)[:, 30:-30]

    # 日内分辨度
    mask = np.isnan(main_factor) | np.isnan(alpha_rtn1)
    mask_main_factor = pd.DataFrame(np.where(mask, np.nan, main_factor)).rank(pct=True, axis=0).values

    df_inday = pd.DataFrame(data={'alpha': alpha_rtn1[mask_main_factor > (-1)],
                                  'support_factor': support_factor[mask_main_factor > (-1)],
                                  'main_factor': mask_main_factor[mask_main_factor > (-1)]})
    # 日间分辨度
    mask = np.isnan(main_factor) | np.isnan(alpha_rtnnext)
    mask_main_factor = pd.DataFrame(np.where(mask, np.nan, main_factor)).rank(pct=True, axis=0).values

    df_crossday = pd.DataFrame(data={'alpha': alpha_rtnnext[mask_main_factor > (-1)],
                                     'support_factor': support_factor[mask_main_factor > (-1)],
                                     'main_factor': mask_main_factor[mask_main_factor > (-1)]})
    return df_inday, df_crossday


def plot_main_support_GroupIC(ia, main_factor_name, support_factor_name):
    df_inday_list = []
    df_crossday_list = []
    for date in ia.factors.keys():
        df_inday, df_crossday = zip_main_support_GroupIC(ia, date, main_factor_name, support_factor_name)
        df_inday_list.append(df_inday)
        df_crossday_list.append(df_crossday)

    groupIC_Dict = {}

    # 作图
    fig = plt.figure(figsize=(15, 10), dpi=128)
    fig.set_facecolor('#FFFFFF')
    plt.subplots_adjust(wspace=0.25, hspace=0.25)
    spec = gridspec.GridSpec(ncols=2, nrows=2)
    fig.suptitle(f"Group Rank IC of {main_factor_name}\n vs\n Rank of {support_factor_name}", fontsize=15)

    # 日内超额与辅助因子关系
    ax = fig.add_subplot(spec[0, 0])
    df = pd.concat(df_inday_list)
    df['support_factor_group'] = np.round(df['support_factor'].rank(pct=True) * 10, 0)
    res = pd.DataFrame(index=list(set(df["support_factor_group"])), columns=["group_IC"])
    # 计算分组IC
    for group_num in range(11):
        df_num = df[df["support_factor_group"] == group_num]
        res.loc[group_num, "group_IC"] = np.corrcoef(df_num["alpha"].astype(np.float64), df_num["main_factor"].astype(np.float64))[0, 1]
    ax.plot([0, 10], [0, 0], '--r', c='gray')
    ax.plot(res.index, res['group_IC'], '--r', c='blue', marker="X", markersize=5)
    ax.set_title(f"Inday Group IC")
    groupIC_Dict["Inday"] = res

    # 日间超额与辅助因子关系
    ax = fig.add_subplot(spec[0, 1])
    df = pd.concat(df_crossday_list)
    df['support_factor_group'] = np.round(df['support_factor'].rank(pct=True) * 10, 0)
    res = pd.DataFrame(index=list(set(df["support_factor_group"])), columns=["group_IC"])
    # 计算分组IC
    for group_num in range(11):
        df_num = df[df["support_factor_group"] == group_num]
        res.loc[group_num, "group_IC"] = np.corrcoef(df_num["alpha"].astype(np.float64), df_num["main_factor"].astype(np.float64))[0, 1]
    ax.plot([0, 10], [0, 0], '--r', c='gray')
    ax.plot(res.index, res['group_IC'], '--r', c='orange', marker="X", markersize=5)
    ax.set_title(f"Crossday Group IC")
    groupIC_Dict["Crossday"] = res

    plt.show()
    return groupIC_Dict


def plot_ls_group(ia, factor_name, longshort_ratio=None):
    period_return = {}
    _, tmp = plot_main_support(ia, factor_name, factor_name, longshort_ratio=longshort_ratio, time='e')
    period_return['早盘买入24小时'] = tmp
    _, tmp = plot_main_support(ia, factor_name, factor_name, longshort_ratio=longshort_ratio, time='n')
    period_return['去除早晚盘买入24小时'] = tmp
    _, tmp = plot_main_support(ia, factor_name, factor_name, longshort_ratio=longshort_ratio, time='l')
    period_return['晚盘买入24小时'] = tmp
    return pd.DataFrame(period_return, index=[factor_name])


def nosample_long_plot(factor_names, thr_flag=None):
    """
    :param thr_flag: 是否有阈值
    :param factor_names: 因子名
    :return: 作图
    """
    for factor_name in factor_names:
        output_path = get_latest_path(factor_name)
        end_minute = cfg.end_minute
        order_time = cfg.order_time

        if thr_flag is None:
            longshort_dict = np.load(output_path + '/longshort_dict.npy', allow_pickle=True).item()
            inday_long_alpha = np.mean(longshort_dict['next_nodown_ls'], axis=0)[:, 1] - np.mean(longshort_dict['next_index_nodown_ls'], axis=0)
            inday_long_alpha = np.append(inday_long_alpha, [np.nan for i in range(order_time)])
            fac_cp_time = end_minute - len(inday_long_alpha)
            # modified 20210608, nan is in front of alpha rtn, thats wrong
            inday_long_alpha = np.append(inday_long_alpha, [np.nan for i in range(fac_cp_time)])

            cross_long_alpha = np.mean(longshort_dict['tommorrow_nodown_ls'], axis=0)[:, 1] - np.mean(longshort_dict['tommorrow_index_nodown_ls'], axis=0)
            close_long_alpha = np.mean(longshort_dict['close_nodown_ls'], axis=0)[:, 1] - np.mean(longshort_dict['close_index_nodown_ls'], axis=0)
        else:
            longshort_dict = np.load(output_path + '/longshort_dict_thr.npy', allow_pickle=True).item()
            inday_long_alpha = np.nanmean(longshort_dict['next_nodown_thrd'], axis=0)[:, 1] - np.nanmean(longshort_dict['next_index_nodown_thrd'], axis=0)
            inday_long_alpha = np.append(inday_long_alpha, [np.nan for i in range(order_time)])
            fac_cp_time = end_minute - len(inday_long_alpha)
            inday_long_alpha = np.append(inday_long_alpha, [np.nan for i in range(fac_cp_time)])

            cross_long_alpha = np.nanmean(longshort_dict['nextSame_nodown_thrd'], axis=0)[:, 1] - np.nanmean(longshort_dict['nextSame_index_nodown_thrd'],
                                                                                                      axis=0)
            close_long_alpha = np.nanmean(longshort_dict['close_nodown_thrd'], axis=0)[:, 1] - np.nanmean(longshort_dict['close_index_nodown_thrd'], axis=0)

        inday_long_alpha = np.where(np.isnan(inday_long_alpha), close_long_alpha, inday_long_alpha)

        assert len(inday_long_alpha) == end_minute
        assert len(cross_long_alpha) == end_minute

        fig = plt.figure(figsize=(15, 6), dpi=cfg.fig2_dpi)
        fig.set_facecolor('#FFFFFF')
        plt.subplots_adjust(wspace=1, hspace=0.2)
        spec = gridspec.GridSpec(ncols=2, nrows=1)

        ax = fig.add_subplot(spec[0, 0:1])
        ax.grid()
        ax.set_title("因子每分钟多组 {} 分钟 rtn，最后 {} 分钟 rtn 用 close rtn 代替".format(order_time, order_time), fontsize=12)
        ax.set_xlabel("时间", fontsize=12)
        ax.set_ylabel("Return（超额）", fontsize=12)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        line1 = ax.plot(inday_long_alpha, c='red', linewidth=1.1)
        X = [30 * i for i in range(9)]
        labels = ['9:30', '10:00', '10:30', '11:00', '11:30', '13:30', '14:00', '14:30', '15:00']
        ax.set_xticks(X)
        ax.set_xticklabels(labels)
        ax.legend(line1, ['每分钟多组 30 分钟 rtn'])
        plt.yticks(size=15)

        ax = fig.add_subplot(spec[0, 1:2])
        ax.grid()
        ax.set_title("因子每分钟多组跨日 rtn", fontsize=12)
        ax.set_xlabel("时间", fontsize=12)
        ax.set_ylabel("Return（超额）", fontsize=12)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        line1 = ax.plot(cross_long_alpha, c='red', linewidth=1.1)
        X = [30 * i for i in range(9)]
        labels = ['9:30', '10:00', '10:30', '11:00', '11:30', '13:30', '14:00', '14:30', '15:00']
        ax.set_xticks(X)
        ax.set_xticklabels(labels)
        ax.legend(line1, ['每分钟多组跨日 rtn'])
        plt.yticks(size=15)

        fig.tight_layout()


def sample_month_table(factor_names, sample_flag=True, time=None, thr_flag=None):
    """
    sample_flag 为 true 则处理采样后，为 False 则看 time 参数
    :param thr_dict:
    :param thrd_dict:
    :param sample_flag:
    :param time: n 是去掉早午盘，e 是早盘，n 是晚盘
    :param factor_names: 因子名
    :return: 作图
    """
    paths = []
    for fac in factor_names:
        paths.append(get_latest_path(fac))

    long_30_dict = {}
    short_30_dict = {}
    long_next_same_dict = {}
    short_next_same_dict = {}
    for output_path_index in range(len(paths)):
        output_path = paths[output_path_index]
        if thr_flag is None:
            longshort_dict = np.load(output_path + '/longshort_dict.npy', allow_pickle=True).item()
            if sample_flag:
                if time is not None:
                    print("time 参数仅在非降采样中有效")
                next_ls, next_index_ls, _, _, _, nextSame_ls, nextSame_index_ls, _, _, _, _, _, _ \
                    = list(longshort_dict.values())
            else:
                try:
                    assert time is not None
                except AssertionError:
                    print("非降采样必须设置 time 参数")
                _, _, _, _, _, _, _, next_ls, next_index_ls, \
                nextSame_ls, nextSame_index_ls, _, _ = list(longshort_dict.values())
        else:
            if sample_flag:
                if time is not None:
                    print("time 参数仅在非降采样中有效")
                longshort_dict = np.load(output_path + 'longshort_dict_thr.npy', allow_pickle=True).item()
                next_ls, next_index_ls, _, _, nextSame_ls, nextSame_index_ls, _, _, _, _, _, _, _, _ \
                    = list(longshort_dict.values())
            else:
                try:
                    assert time is not None
                except AssertionError:
                    print("非降采样必须设置 time 参数")
                longshort_dict = np.load(output_path + 'longshort_dict_thr.npy', allow_pickle=True).item()
                _, _, _, _, _, _, next_ls, next_index_ls, nextSame_ls, nextSame_index_ls, _, _, _, _ \
                    = list(longshort_dict.values())

        connector = PqiDataSdk(user='cchen')
        dates = connector.get_trade_dates(start_date=cfg.start_date, end_date=cfg.end_date, market='stock')
        dates = list(set(dates).difference(set(bad_days)))
        dates.sort()
        month = None
        month_indexs = [0]
        month_name = [cfg.start_date[:6]]
        for date_index in range(len(dates)):
            if month is None:
                month = dates[date_index][:6]
            else:
                if month == dates[date_index][:6]:
                    continue
                else:
                    month = dates[date_index][:6]
                    month_indexs.append(date_index)
                    month_name.append(month)

        next_long_group_ls = next_ls[:, :, -1]
        next_short_group_ls = next_ls[:, :, 0]
        next_index_down_ls = next_index_ls
        next_long_alpha = next_long_group_ls - next_index_down_ls
        next_short_alpha = next_index_down_ls - next_short_group_ls

        if not sample_flag:
            next_long_alpha = np.concatenate([next_long_alpha, np.ones((next_long_alpha.shape[0], cfg.order_time)) * np.nan], axis=1)
            next_short_alpha = np.concatenate([next_short_alpha, np.ones((next_long_alpha.shape[0], cfg.order_time)) * np.nan], axis=1)

        nextSame_long_group_ls = nextSame_ls[:, :, -1]
        nextSame_short_group_ls = nextSame_ls[:, :, 0]
        nextSame_index_down_ls = nextSame_index_ls
        nextSame_long_alpha = (nextSame_long_group_ls - nextSame_index_down_ls)
        nextSame_short_alpha = - nextSame_short_group_ls + nextSame_index_down_ls

        if not sample_flag:
            if time == 'n':
                next_long_alpha = next_long_alpha[:, 30:210]
                next_short_alpha = next_short_alpha[:, 30:210]
                nextSame_long_alpha = nextSame_long_alpha[:, 30:210]
                nextSame_short_alpha = nextSame_short_alpha[:, 30:210]
            elif time == 'e':
                next_long_alpha = next_long_alpha[:, :30]
                next_short_alpha = next_short_alpha[:, :30]
                nextSame_long_alpha = nextSame_long_alpha[:, :30]
                nextSame_short_alpha = nextSame_short_alpha[:, :30]
            elif time == 'l':
                next_long_alpha = next_long_alpha[:, 210:]
                next_short_alpha = next_short_alpha[:, 210:]
                nextSame_long_alpha = nextSame_long_alpha[:, 210:]
                nextSame_short_alpha = nextSame_short_alpha[:, 210:]
            else:
                print('time 参数设置错误')
                raise ValueError

        assert next_long_alpha.shape[0] == nextSame_long_alpha.shape[0]

        next_month_long_alpha = []
        next_month_short_alpha = []
        nextSame_month_long_alpha = []
        nextSame_month_short_alpha = []
        month_index = None
        for month_index in range(len(month_indexs) - 1):
            next_month_long_alpha.append(np.nanmean(next_long_alpha[month_indexs[month_index]:month_indexs[month_index + 1], :]))
            next_month_short_alpha.append(np.nanmean(next_short_alpha[month_indexs[month_index]:month_indexs[month_index + 1], :]))
            nextSame_month_long_alpha.append(np.nanmean(nextSame_long_alpha[month_indexs[month_index]:month_indexs[month_index + 1], :]))
            nextSame_month_short_alpha.append(np.nanmean(nextSame_short_alpha[month_indexs[month_index]:month_indexs[month_index + 1], :]))

        next_month_long_alpha.append(np.nanmean(next_long_alpha[month_indexs[month_index + 1]:, :]))
        next_month_short_alpha.append(np.nanmean(next_short_alpha[month_indexs[month_index + 1]:, :]))
        nextSame_month_long_alpha.append(np.nanmean(nextSame_long_alpha[month_indexs[month_index + 1]:, :]))
        nextSame_month_short_alpha.append(np.nanmean(nextSame_short_alpha[month_indexs[month_index + 1]:, :]))
        long_30_dict[factor_names[output_path_index]] = next_month_long_alpha
        short_30_dict[factor_names[output_path_index]] = next_month_short_alpha
        long_next_same_dict[factor_names[output_path_index]] = nextSame_month_long_alpha
        short_next_same_dict[factor_names[output_path_index]] = nextSame_month_short_alpha

    return pd.DataFrame(long_30_dict, index=month_name), pd.DataFrame(long_next_same_dict, index=month_name), \
           pd.DataFrame(short_30_dict, index=month_name), pd.DataFrame(short_next_same_dict, index=month_name)
