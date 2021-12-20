import sys

sys.path.append('./tools')
from tools.alphaAssist import alphaAssist, multi, save_fac, load_fac
from PqiDataSdk import *
import pandas as pd
import numpy as np
import time
import config
import os
import shutil
import matplotlib.pyplot as plt
import tools.helper as hpl
from tools.FactorAssess import *

import seaborn as sns
import itertools
from matplotlib.cbook import _reshape_2D

'''
factors_describe为绘制因子箱线图的函数，my_boxplot_stats仅为修改源码让主函数能够绘制任意分位数的箱线图
'''


def my_boxplot_stats(X, whis=1.5, bootstrap=None, labels=None,
                     autorange=False, percents=[25, 75]):
    def _bootstrap_median(data, N=5000):
        # determine 95% confidence intervals of the median
        M = len(data)
        percentiles = [2.5, 97.5]

        bs_index = np.random.randint(M, size=(N, M))
        bsData = data[bs_index]
        estimate = np.median(bsData, axis=1, overwrite_input=True)

        CI = np.percentile(estimate, percentiles)
        return CI

    def _compute_conf_interval(data, med, iqr, bootstrap):
        if bootstrap is not None:
            # Do a bootstrap estimate of notch locations.
            # get conf. intervals around median
            CI = _bootstrap_median(data, N=bootstrap)
            notch_min = CI[0]
            notch_max = CI[1]
        else:

            N = len(data)
            notch_min = med - 1.57 * iqr / np.sqrt(N)
            notch_max = med + 1.57 * iqr / np.sqrt(N)

        return notch_min, notch_max

    # output is a list of dicts
    bxpstats = []

    # convert X to a list of lists
    X = _reshape_2D(X, "X")

    ncols = len(X)
    if labels is None:
        labels = itertools.repeat(None)
    elif len(labels) != ncols:
        raise ValueError("Dimensions of labels and X must be compatible")

    input_whis = whis
    for ii, (x, label) in enumerate(zip(X, labels)):

        # empty dict
        stats = {}
        if label is not None:
            stats['label'] = label

        # restore whis to the input values in case it got changed in the loop
        whis = input_whis

        # note tricksyness, append up here and then mutate below
        bxpstats.append(stats)

        # if empty, bail
        if len(x) == 0:
            stats['fliers'] = np.array([])
            stats['mean'] = np.nan
            stats['med'] = np.nan
            stats['q1'] = np.nan
            stats['q3'] = np.nan
            stats['cilo'] = np.nan
            stats['cihi'] = np.nan
            stats['whislo'] = np.nan
            stats['whishi'] = np.nan
            stats['med'] = np.nan
            continue

        # up-convert to an array, just to be safe
        x = np.asarray(x)

        # arithmetic mean
        stats['mean'] = np.mean(x)

        # median
        med = np.percentile(x, 50)
        ## Altered line
        q1, q3 = np.percentile(x, (percents[0], percents[1]))

        # interquartile range
        stats['iqr'] = q3 - q1
        if stats['iqr'] == 0 and autorange:
            whis = 'range'

        # conf. interval around median
        stats['cilo'], stats['cihi'] = _compute_conf_interval(
            x, med, stats['iqr'], bootstrap
        )

        # lowest/highest non-outliers
        if np.isscalar(whis):
            if np.isreal(whis):
                loval = q1 - whis * stats['iqr']
                hival = q3 + whis * stats['iqr']
            elif whis in ['range', 'limit', 'limits', 'min/max']:
                loval = np.min(x)
                hival = np.max(x)
            else:
                raise ValueError('whis must be a float, valid string, or list '
                                 'of percentiles')
        else:
            loval = np.percentile(x, whis[0])
            hival = np.percentile(x, whis[1])

        # get high extreme
        wiskhi = np.compress(x <= hival, x)
        if len(wiskhi) == 0 or np.max(wiskhi) < q3:
            stats['whishi'] = q3
        else:
            stats['whishi'] = np.max(wiskhi)

        # get low extreme
        wisklo = np.compress(x >= loval, x)
        if len(wisklo) == 0 or np.min(wisklo) > q1:
            stats['whislo'] = q1
        else:
            stats['whislo'] = np.min(wisklo)

        # compute a single array of outliers
        stats['fliers'] = np.hstack([
            np.compress(x < stats['whislo'], x),
            np.compress(x > stats['whishi'], x)
        ])

        # add in the remaining stats
        stats['q1'], stats['med'], stats['q3'] = q1, med, q3

    return bxpstats


def factors_describe(ia, factor_list, start_date=None, end_date=None, start_min=5, end_min=237, picture_save_path='',
                     whis=10000000, \
                     percents=[1, 99], save_csv=False):
    '''
    percents用以表示上下分位数的具体percentile
    save_csv用以控制是否输出按天、按分钟范围统计的每日（每分钟范围）因子stats（min、q1、median、q3、max）
    '''

    # 选取选定的日期范围
    if start_date is None:
        start_date = 0
    if end_date is None:
        end_date = 99999999

    # 选取选定的分钟范围
    if start_min < 1:
        start_min = 1
    if end_min > 237:
        end_min = 237

    selected_tradedates = sorted([x for x in ia.factors.keys() if int(start_date) <= int(x) <= int(end_date)])

    for factor in factor_list:

        min_upper = start_min + 5 if start_min + 5 <= end_min else end_min  # 记录切割min的上界

        plt.figure(figsize=(20, 20))
        '''
        画按天describe的图
        '''
        ax1 = plt.subplot(2, 1, 1)
        factor_value_by_day = pd.DataFrame(columns=selected_tradedates)  # 列名为每天的日期
        # 将每日的因子值展成一维向量，放到factor_value_by_day里
        for day in selected_tradedates:
            factor_value_by_day[day] = ia.factors[day][factor][:, start_min - 1:end_min].flatten()  # 分钟维度是纵向，做一下切分
        #         sns.boxplot(data=factor_value_by_day, whis=100000000)
        stats = my_boxplot_stats(factor_value_by_day, percents=percents, whis=whis, labels=factor_value_by_day.columns)
        ax1.bxp(stats)
        q1_list, q3_list, med_list = [], [], []
        for dic in stats:
            q1_list.append(dic['q1'])
            q3_list.append(dic['q3'])
            med_list.append(dic['med'])
        csv_df1 = pd.DataFrame(columns=['min', 'q1', 'median', 'q3', 'max'], index=factor_value_by_day.columns)
        csv_df1[['q1', 'median', 'q3']] = np.concatenate(
            [np.array(q1_list).reshape(-1, 1), np.array(med_list).reshape(-1, 1), \
             np.array(q3_list).reshape(-1, 1)], axis=1)
        csv_df1['min'] = factor_value_by_day.min().values
        csv_df1['max'] = factor_value_by_day.max().values
        q1_mean, q3_mean, med_mean = np.mean(q1_list), np.mean(q3_list), np.mean(med_list)
        ax1.plot([q1_mean] * len(factor_value_by_day.columns), color='r', linestyle='--')
        ax1.plot([q3_mean] * len(factor_value_by_day.columns), color='blue', linestyle='--')
        ax1.plot([med_mean] * len(factor_value_by_day.columns), color='green', linestyle='--')
        shift_length = -(len(factor_value_by_day.columns) / 21 / 2.2)
        ax1.annotate('%.3f' % q1_mean, xy=(shift_length, q1_mean), color='r', fontsize=14)
        ax1.annotate(' %.3f' % q3_mean, xy=(shift_length, q3_mean), color='blue', fontsize=14)
        ax1.annotate(' %.3f' % med_mean, xy=(shift_length * 1.35, med_mean), color='green', fontsize=14)
        plt.title(factor + ' boxplot by day', fontsize=18)
        plt.xlabel('date', fontsize=18)
        ticker_spacing = 15
        ax1.set_xticklabels(factor_value_by_day.columns.tolist())
        ax1.xaxis.set_major_locator(ticker.MultipleLocator(ticker_spacing))
        plt.ylabel('values', fontsize=18)

        '''
        画按分钟period describe的图
        '''
        ax2 = plt.subplot(2, 1, 2)
        factor_value_by_min = pd.DataFrame()  # 列名为min范围
        while (min_upper <= end_min):
            lower = min_upper - 5
            # 将每日该min范围的因子值展成一维向量并拼到一起，放到factor_value_by_min里
            array_list = []
            for day in selected_tradedates:
                array_list.append(ia.factors[day][factor][:, lower:min_upper].flatten())  # 分钟维度是纵向，做一下切分
            this_min_period = np.concatenate(array_list)
            factor_value_by_min[str(lower) + '_' + str(min_upper)] = this_min_period
            min_upper += 5

        #         sns.boxplot(data=factor_value_by_min, whis=100000000)
        stats = my_boxplot_stats(factor_value_by_min, percents=percents, whis=whis, labels=factor_value_by_min.columns)
        ax2.bxp(stats)
        q1_list, q3_list, med_list = [], [], []
        for dic in stats:
            q1_list.append(dic['q1'])
            q3_list.append(dic['q3'])
            med_list.append(dic['med'])
        csv_df2 = pd.DataFrame(columns=['min', 'q1', 'median', 'q3', 'max'], index=factor_value_by_min.columns)
        csv_df2[['q1', 'median', 'q3']] = np.concatenate(
            [np.array(q1_list).reshape(-1, 1), np.array(med_list).reshape(-1, 1), \
             np.array(q3_list).reshape(-1, 1)], axis=1)
        csv_df2['min'] = factor_value_by_min.min().values
        csv_df2['max'] = factor_value_by_min.max().values
        q1_mean, q3_mean, med_mean = np.mean(q1_list), np.mean(q3_list), np.mean(med_list)
        ax2.plot([q1_mean] * len(factor_value_by_min.columns), color='r', linestyle='--')
        ax2.plot([q3_mean] * len(factor_value_by_min.columns), color='blue', linestyle='--')
        ax2.plot([med_mean] * len(factor_value_by_min.columns), color='green', linestyle='--')
        shift_length = -(len(factor_value_by_min.columns) / 21 / 2.2)
        ax2.annotate('%.3f' % q1_mean, xy=(shift_length, q1_mean), color='r', fontsize=14)
        ax2.annotate(' %.3f' % q3_mean, xy=(shift_length, q3_mean), color='blue', fontsize=14)
        ax2.annotate(' %.3f' % med_mean, xy=(shift_length, med_mean), color='green', fontsize=14)
        plt.title(factor + ' boxplot by min period', fontsize=18)
        plt.xlabel('minute period', fontsize=18)
        plt.xticks(rotation=90)
        plt.ylabel('values', fontsize=18)

        plt.savefig(picture_save_path + factor + '_describe.png')

        if save_csv:
            csv_df1.to_csv(picture_save_path + factor + '_stats_table_by_day.csv')
            csv_df2.to_csv(picture_save_path + factor + '_stats_table_by_minute_period.csv')

    return