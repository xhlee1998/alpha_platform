# coding=utf8
import numpy as np
import pandas as pd
from PqiDataSdk import PqiDataSdk
from collections import defaultdict
import copy
import config as cfg
import multiprocessing
import time
import tools.helper as hpl
import os
import shutil
import time
import warnings
import logging

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.CRITICAL)

conn = PqiDataSdk(user=cfg.public_user, size=1, pool_type="mt", offline=True, str_map=False)
eod_fac_names_dir = os.listdir('/home/shared/Data/data/shared/inday_alpha_local/high_freq_eod/eod_feature')
eod_fac_names = [ele.strip('.h5') for ele in eod_fac_names_dir]

# 有异常数据的日期删掉
bad_days = ['20210315', '20210312', '20200115', '20210108', '20190102', '20190103', '20190104', '20190105', '20190106', '20190107',
            '20190108', '20190109', '20191101', '20191104', '20191105', '20191106', '20191107', '20191108', '20191111', '20191112',
            '20191113', '20191114', '20191115', '20191118', '20191119', '20191120', '20191121', '20191122', '20191125', '20191126',
            '20191127', '20191128', '20191129']

PathDict = {
    '/data/local_data/stats/92': ['201901', '201902', '201903', '201904', '201905', '201906'],
    '/data/local_data/stats/102': ['201907', '201908', '201909', '201910', '201911', '201912'],
    '/data/local_data/stats/96': ['202001', '202002', '202003', '202004', '202005', '202006'],
    '/data/local_data/stats/98': ['202007', '202008', '202009', '202010', '202011', '202012'],
    '/data/local_data/stats/99': ['202101', '202102', '202103', '202104', '202105', '202106'],
}

reversePathDict = {}
for key in PathDict:
    for YrMth in PathDict[key]:
        reversePathDict[YrMth] = key


class EodData(object):
    def __init__(self, eod_data):
        # eod_data.swapaxes(0, 1)
        self.data = eod_data

    def __getitem__(self, eod_name):
        encrypt_name = list(conn.encrypt_feature(real_names=eod_name).values())[0]
        return self.data[encrypt_name]


class alphaAssist():
    def __init__(self, mp=True, para_dict={}):
        if mp:
            print("========================== initialize =======================")
        self.para_dict = {}
        if para_dict != {}:
            self.para_dict = copy.deepcopy(para_dict)
        else:
            self.para_dict['stock_pool'] = cfg.stock_pool

            self.para_dict['start_date'] = cfg.start_date
            self.para_dict['end_date'] = cfg.end_date
            self.para_dict['sample_freq'] = cfg.sample_freq
        self.mp = mp
        self.tickers = self.para_dict['stock_pool']
        self.para_dict['stock_pool'].sort()
        self.factors_ = defaultdict(dict)  # multiprocessing.Manager().dict()
        self.factors = defaultdict(dict)
        self.factors_multi = {}
        self.group_rtn = None
        self.cut = False
        if mp:
            print("=================== initialize  PqiDataSdk ==================")
            self.myconnector = PqiDataSdk(user=cfg.user, size=20, pool_type="mp", offline=True, str_map=False)
            print("=================== initialize finished =====================")
        else:
            self.myconnector = PqiDataSdk(user=cfg.user, size=1, pool_type="mt", offline=True, str_map=False)

    def __getstate__(self):
        self_dict = self.__dict__.copy()
        del self_dict['pool']
        return self_dict

    def __setstate__(self, state):
        self.__dict__.update(state)

    def generate_map_dict(self, collist, collist_idx):
        """
        生成 df的columns的映射map
        """
        self.map_stock = dict(zip(collist, range(len(collist))))
        self.map_index = dict(zip(collist_idx, range(len(collist_idx))))

    def fetch_eod(self):
        """
        获取eod数据
        """
        global eod_fac_names
        connector = PqiDataSdk(user=cfg.public_user, size=1, pool_type="mt", offline=True, str_map=False)
        dates = connector.get_trade_dates(start_date='20190102', end_date=cfg.end_date)
        eod = connector.get_eod_feature(tickers=copy.deepcopy(cfg.stock_pool), dates=dates,
                                        fields=eod_fac_names, where='/home/shared/Data/data/shared/inday_alpha_local/high_freq_eod/')
        eod.swapaxes(1, 2)
        skipList = ['eod_OvNtRtn']
        encrypt_skipList = list(conn.encrypt_feature(real_names=skipList).values())[0]
        eodday = {}
        for day in self.dates:
            df = pd.DataFrame(index=self.tickers, columns=eod.header[0])
            for facname in eod.header[0]:
                try:
                    if facname in encrypt_skipList:
                        df[facname] = eod[facname].to_dataframe().loc[day, self.tickers]
                    else:
                        date = connector.get_prev_trade_date(trade_date=day)
                        # df[facname] = eod[facname][day].loc[self.tickers]
                        df[facname] = eod[facname].to_dataframe().loc[date, self.tickers]
                except:
                    continue
            eodday[day] = df

        self.eod_facs = copy.deepcopy(eodday)
        self.eod_facs_names = df.columns

    def fetch_date(self, data_type='tick', son=False):
        """
        获取depth数据
        """
        self.data_type = data_type
        collist = cfg.minute_col_list
        self.collist = copy.deepcopy(collist)
        if not son:
            connector = PqiDataSdk(user=cfg.user, size=20, pool_type="mp", offline=True, str_map=False)
        else:
            connector = PqiDataSdk(user=cfg.user, size=1, pool_type="mt", offline=True, str_map=False)
        df = connector.get_mins_history(tickers=self.para_dict['stock_pool'],
                                        start_date=str(self.para_dict['start_date']),
                                        end_date=str(self.para_dict['end_date']), fields=cfg.minute_col_list)
        self.depthDataDict = {}
        dates = list(list(df.values())[0].keys())
        dates = list(set(dates).difference(set(bad_days)))
        dates.sort()
        std_df = None
        # 有一些日期的df是空的，因此需要一个标准的，内部全为nan的数据去填充他
        for ticker in self.tickers:
            if len(df[ticker][dates[0]]) > 2:
                std_df = pd.DataFrame(index=df[ticker][dates[0]].index, columns=df[ticker][dates[0]].columns)
                break

        # 从 minute 数据到一个 np array， 截面数据
        self.map_stock = dict(zip(self.collist, range(len(self.collist))))
        for date in dates:
            minuteData = [df[ticker][date].values.T if (len(df[ticker][date]) > 0) else std_df.values.T for ticker in cfg.stock_pool]
            data_process = np.array(minuteData).swapaxes(1, 0)[:, :, :cfg.end_minute]
            self.depthDataDict[date] = data_process.astype(float)
        self.tickers = cfg.stock_pool
        self.dates = copy.deepcopy(dates)
        self.fetch_eod()

        # 过滤价格异常点
        for day in self.depthDataDict.keys():
            self.depthDataDict[day][self.map_stock['MpClose']] = np.where(self.depthDataDict[day][self.map_stock['MpClose']] == 0, np.nan,self.depthDataDict[day][self.map_stock['MpClose']])
        return


def neutralize(ret, eodday, neutral_industry=False, neutral_mktcap=False):
    """
    市值或行业中性化函数
    :param ret: 原始的因子矩阵
    :param eodday: eod数据
    :param neutral_industry: 是否行业中性化，1代表申万一级行业，2代表申万二级行业
    :param neutral_mktcap: 是否市值中性化，其中用log对市值进行处理
    :return: 中性化后的因子矩阵
    """
    matrix_path1 = '/data/shared/industry_index_data/primary_industry_matrix.csv'
    industry_matrix1 = pd.read_csv(matrix_path1)
    industry_matrix1.set_index(keys="Unnamed: 0", inplace=True)

    matrix_path2 = '/data/shared/industry_index_data/secondary_industry_matrix.csv'
    industry_matrix2 = pd.read_csv(matrix_path2)
    industry_matrix2.set_index(keys="Unnamed: 0", inplace=True)
    if neutral_industry | neutral_mktcap:
        class_var = pd.DataFrame(index=ret.index)
        if neutral_mktcap:
            class_var['logmktcap'] = np.log(pd.DataFrame(eodday['eod_TotalMarketValue'], index=ret.index))
            class_var = class_var.fillna(0)
        ## 行业中性化
        if neutral_industry == 1:
            class_var = pd.concat([class_var, industry_matrix1.iloc[:, :-1].reset_index(drop=True)], axis=1)
        elif neutral_industry == 2:
            class_var = pd.concat([class_var, industry_matrix2.iloc[:, :-1].reset_index(drop=True)], axis=1)
        x = np.hstack((np.ones((len(ret), 1)), class_var.values))
        new_ret = pd.DataFrame(index=ret.index, columns=ret.columns)
        for col in ret.columns:
            y = ret.iloc[:, col].values
            beta = np.linalg.pinv(x.T.dot(x)).dot(x.T).dot(y)
            fit = x.dot(beta)
            new_ret[col] = y - fit
        return new_ret
    else:
        return ret


def single_test_alpha_minute(alpha_func, factor_name, date, eodday, neutral_industry=False, neutral_mktcap=False, **kwargs):
    # 分钟级别的回测其中的单进程
    atp = alphaAssist(mp=False)
    atp.para_dict['start_date'] = date
    atp.para_dict['end_date'] = date
    atp.fetch_date('minute', True)
    rm = alpha_func(atp.depthDataDict[date], atp.map_stock, atp.tickers, date, eodday, **kwargs)
    atp.factors_[factor_name][date] = np.array(rm)
    return atp.factors_[factor_name][date]


def read_features_from_ds(tickers,features,strat_date,end_date,ds_path):
    ds = PqiDataSdk(user=cfg.public_user, size=50, offline=True, str_map=False)
    dates = ds.get_trade_dates(start_date=strat_date,end_date=end_date)

    output_dict = {} # 第一层date 第二层factor
    if ds_path is not None:
        print(ds_path)
        res = ds.get_mins_feature(tickers=tickers,dates=dates,fields=features,where=ds_path)
        for d in dates:
            if d not in output_dict.keys():
                output_dict[d] = {}

            for f in features:
                output_dict[d][f] = res[f][d].values[:,:cfg.end_minute]

    else:
        # YrMthList = list(set([x[:6] for x in ds.get_trade_dates(start_date=strat_date,end_date=end_date)]))
        # for YrMth in YrMthList:
        #     start = YrMth + '01'
        #     end = YrMth + '31'
        subdates = ds.get_trade_dates(start_date = strat_date, end_date = end_date)
        res = ds.get_mins_feature(tickers=tickers, dates=subdates, fields=features, where='/home/shared/Data/data/shared/inday_alpha_local/inday_fatcors/')
        for d in subdates:
            if d not in output_dict.keys():
                output_dict[d] = {}

            for f in features:
                output_dict[d][f] = res[f][d].values[:,:cfg.end_minute]

    return output_dict


def multi(atp, alpha_list, factor_names=None, using_rank=True, neutral_industry=False, neutral_mktcap=False, only_fac=False, thr_dict=None,
          given_factor_path=None, l_ratio=cfg.long_ratio, s_ratio=cfg.short_ratio, alpha_params=None,from_ds=False):
    if thr_dict is None:
        thr_dict = {}
    if alpha_params is None:
        alpha_params = {}
    time0 = time.time()
    num_group_backtest = cfg.num_group_backtest
    atp.output_path = time.strftime('%Y%m%d_%H:%M:%S')
    if factor_names is None:
        factor_names = []
        for func in alpha_list:
            factor_names.append(func.__name__)
    '''
    -------------------------------------------保存因子文件------------------------------------------
    计算中，用户可能实时修改代码文件，因此每次都会把代码文件保存一下
    '''
    for factor_name in factor_names:
        dirctry = os.path.join(cfg.output_path, factor_name)
        fac_path = os.path.join(dirctry, atp.output_path)
        if atp.data_type == 'tick':
            if not os.path.exists(fac_path):
                os.makedirs(fac_path)
                shutil.copyfile('./factors_def.py', os.path.join(fac_path, 'factors_def.py'))
                shutil.copyfile('./cfg.py', os.path.join(fac_path, 'faccfg.py'))
                if os.path.exists('./demo.ipynb'):
                    shutil.copyfile('./demo.ipynb', os.path.join(fac_path, 'demo.ipynb'))
                    shutil.copyfile('./cfg.py', os.path.join(fac_path, 'faccfg.py'))
            else:
                os.makedirs(fac_path)
                shutil.copyfile('./factors_def.py', os.path.join(fac_path, 'factors_def.py'))
                shutil.copyfile('./cfg.py', os.path.join(fac_path, 'faccfg.py'))
                if os.path.exists('./minute.ipynb'):
                    shutil.copyfile('./minute.ipynb', os.path.join(fac_path, 'minute.ipynb'))
                    shutil.copyfile('./cfg.py', os.path.join(fac_path, 'faccfg.py'))
    pool = multiprocessing.Pool(cfg.pool_num)
    '''
    -------------------------------------------保存因子文件------------------------------------------
    '''

    '''
    -------------------------------------------多进程过程------------------------------------------
    '''
    r = defaultdict(dict)
    connector = PqiDataSdk(user=cfg.public_user, size=1, pool_type="mt", offline=True, str_map=False)
    if from_ds:
        # ds_output = read_features_from_ds(atp.tickers, factor_names, atp.para_dict['start_date'],
        #                                   atp.para_dict['end_date'], ds_path= "/home/shared/Data/data/shared/inday_alpha_local/inday_fatcors")

        ds_output = read_features_from_ds(atp.tickers, factor_names, atp.para_dict['start_date'],
                                          atp.para_dict['end_date'], ds_path= None)

    ML_factors = []
    if not (given_factor_path is None):
        if (not (type(given_factor_path) == list)) or (len(given_factor_path) != len(alpha_list)):
            if len(alpha_list) == 0:
                print("因子名将按照传入文件进行命名")
                for p in given_factor_path:
                    exec(f"def {p.split('/')[-1][:-4]}(): pass")
                    alpha_list.append(eval(p.split('/')[-1][:-4]))
            else:
                print('传入的路径应为list，且长度与因子数量一致，且应一一对应，请核验准确')
                raise
        else:
            for i in range(len(alpha_list)):
                ML_factors.append(pd.read_pickle(given_factor_path[i]))

    date = None
    for date in atp.dates:
        for i in range(len(alpha_list)):
            if factor_names[i] in alpha_params.keys():
                kwargs = alpha_params[factor_names[i]]
            else:
                kwargs = {}
            # 如果已经用因子平台生成好了，从因子平台取因子
            if from_ds:
                continue
            if not (given_factor_path is None):
                if date in ML_factors[i].keys():
                    r[date][factor_names[i]] = ML_factors[i][date]
                else:
                    r[date][factor_names[i]] = np.full((len(atp.tickers),cfg.end_minute),np.nan)
            else:
                r[date][factor_names[i]] = pool.apply_async(single_test_alpha_minute,
                                                            args=(copy.deepcopy(alpha_list[i]), factor_names[i], date,
                                                                  EodData(atp.eod_facs[date]), neutral_industry, neutral_mktcap,), kwds=kwargs)
    pool.close()
    pool.join()
    print('因子值相关计算、降采样完成', time.time() - time0)

    # 获取一些到明日的rtn所需的eod数据
    train_dates = connector.get_trade_dates(start_date='20010101', end_date='21000101')
    train_dates = list(set(train_dates).difference(set(bad_days)))
    # 升序
    train_dates.sort()
    end_date = train_dates[train_dates.index(np.array(train_dates)[np.array(train_dates) <= date][-1]) + 3]
    dates = connector.get_trade_dates(start_date='20190102', end_date=end_date)
    eod = connector.get_eod_feature(tickers=copy.deepcopy(cfg.stock_pool), dates=dates, fields=eod_fac_names,
                                    where='/home/shared/Data/data/shared/inday_alpha_local/high_freq_eod/')

    eod_AdjFactor_name = list(connector.encrypt_feature(real_names='eod_AdjFactor').values())[0]
    AdjFactor = eod[eod_AdjFactor_name].to_dataframe()
    date_list = atp.dates
    if not atp.cut:
        date_list = atp.dates[:-2]
    for date in date_list:
        for i in range(len(alpha_list)):
            if from_ds: # 如果来自ds，直接读取
                atp.factors[date][factor_names[i]] = np.array(ds_output[date][factor_names[i]]).astype(np.float64)
                continue

            if not (given_factor_path is None):
                atp.factors[date][factor_names[i]] = np.array(r[date][factor_names[i]]).astype(np.float64)
            else:
                if neutral_industry | neutral_mktcap:
                    atp.factors[date][factor_names[i]] = np.array(
                        neutralize(pd.DataFrame(np.array(r[date][factor_names[i]].get())).fillna(0), atp.eod_facs[date], neutral_industry,
                                   neutral_mktcap).values).astype(np.float64)
                else:
                    atp.factors[date][factor_names[i]] = np.array(r[date][factor_names[i]].get()).astype(np.float64)

        '''
        -------------------------------------------多进程过程------------------------------------------
        '''
        if not 'number' in atp.factors[date].keys():
            '''
            -------------------------------------------指数计算------------------------------------------
            计算中，指数是由个股等权合成的，最后几个分钟没办法达到30min的 用当前时刻到收盘时候的rtn来代替
            '''

            tick = int(cfg.order_time * 1 * cfg.sample_freq)
            mp = pd.DataFrame(atp.depthDataDict[date][atp.map_stock['MpClose']])
            rtn_df = (mp.diff(tick, axis=1).shift(-tick, axis=1) / mp)
            index2close = pd.DataFrame((atp.depthDataDict[date][atp.map_stock['MpClose']][:, -1:] -
                                        atp.depthDataDict[date][atp.map_stock['MpClose']]) /
                                       atp.depthDataDict[date][atp.map_stock['MpClose']])
            rtn_df = rtn_df.fillna(index2close)
            # 去掉异常值
            rtn_df[abs(rtn_df) > 1] = np.nan
            index2close[abs(index2close) > 1] = np.nan

            indexrtn = np.nanmean(rtn_df.values, axis=0, keepdims=True)
            index2close = np.nanmean(index2close.values, axis=0, keepdims=True)
            atp.factors[date]['indexrtn'] = indexrtn
            atp.factors[date]['index2close'] = index2close

            '''
            -------------------------------------------指数计算------------------------------------------
            '''

            '''
            -------------------------------------------计算到明日今时的价格------------------------------------------
            '''
            # 到明天的rtn计算，包括个股的和index的 rtn，有用 VWAP计算的 和 用 实时计算的，对于实时计算的，如果有停牌的，就用后天同一时刻
            tomorrow = train_dates[train_dates.index(np.array(train_dates)[np.array(train_dates) <= date][-1]) + 1]
            day_after_tomorrow = train_dates[train_dates.index(np.array(train_dates)[np.array(train_dates) <= date][-1]) + 2]

            # 避免停牌是为了分红送股
            tomorrow_mp = atp.depthDataDict[tomorrow][atp.map_stock['MpClose']] * AdjFactor.loc[atp.tickers, tomorrow].values.reshape(-1, 1).astype(
                np.float64)
            tomorrow_mp = tomorrow_mp.astype(np.float64)
            nxt_num = atp.dates.index(tomorrow) + 1

            while (np.sum(pd.isna(tomorrow_mp)) > 0) & (nxt_num <= (len(atp.dates) - 2)):
                tomorrow_mp = pd.DataFrame(tomorrow_mp).fillna(pd.DataFrame(
                    atp.depthDataDict[atp.dates[nxt_num]][atp.map_stock['MpClose']] * AdjFactor.loc[atp.tickers,
                                                                                                    atp.dates[nxt_num]].values.reshape(-1, 1))).values
                nxt_num += 1

            next_tomorrow_mp = atp.depthDataDict[day_after_tomorrow][atp.map_stock['MpClose']] * AdjFactor.loc[
                atp.tickers, day_after_tomorrow].values.reshape(-1, 1).astype(np.float64)
            next_tomorrow_mp = next_tomorrow_mp.astype(np.float64)
            nxt_num = atp.dates.index(day_after_tomorrow) + 1
            while (np.sum(pd.isna(next_tomorrow_mp)) > 0) and (nxt_num <= (len(atp.dates) - 2)):
                next_tomorrow_mp = pd.DataFrame(next_tomorrow_mp).fillna(pd.DataFrame(
                    atp.depthDataDict[atp.dates[nxt_num]][atp.map_stock['MpClose']] * AdjFactor.loc[atp.tickers,
                                                                                                    atp.dates[nxt_num]].values.reshape(-1, 1))).values
                nxt_num += 1

            # OpenPrice * adj，eod已经shift过了所以用后天的
            en_adjfactor = atp.myconnector.encrypt_feature(real_names=['eod_AdjFactor'])['eod_AdjFactor']
            en_openprice = atp.myconnector.encrypt_feature(real_names=['eod_OpenPrice'])['eod_OpenPrice']
            next_open_mp = (atp.eod_facs[day_after_tomorrow][en_adjfactor] * atp.eod_facs[day_after_tomorrow][en_openprice]).values.reshape(-1, 1)
            atp.factors[date]['2nextopen_mp'] = next_open_mp

            mp = atp.depthDataDict[date][atp.map_stock['MpClose']] * AdjFactor.loc[atp.tickers, date].values.reshape(-1, 1)
            if atp.data_type == 'tick':
                today_ap1_tick = atp.depthDataDict[date][atp.map_stock['ap1']]
            else:
                today_ap1_tick = copy.deepcopy(atp.depthDataDict[date][atp.map_stock['UpLimit']]) - 1
                today_ap1_tick = np.abs(today_ap1_tick)

            # vwap = pd.DataFrame(atp.depthDataDict[date][atp.map_stock['Vwap']]) * AdjFactor.loc[atp.tickers, date].values.reshape(-1, 1)
            # vwap = vwap.shift(-1, axis=1)
            # vwap = vwap.replace(0, np.nan)
            # vwap = vwap.fillna(pd.DataFrame(mp)).values

            vwaptick = int(cfg.vwap_period)

            # 原来的vwap算法
            # val = pd.DataFrame(atp.depthDataDict[date][atp.map_stock['ActBuyVal']] +
            #                    atp.depthDataDict[date][atp.map_stock['ActSellVal']]).rolling(vwaptick, axis=1).sum().round(2)
            # vol = pd.DataFrame(atp.depthDataDict[date][atp.map_stock['ActBuyVol']] +
            #                    atp.depthDataDict[date][atp.map_stock['ActSellVol']]).rolling(vwaptick, axis=1).sum().round(0)
            # val = pd.DataFrame(np.where(vol < 100, 0, val))
            # vol = pd.DataFrame(np.where(vol < 100, 1, vol))
            # vwap = (val / vol).shift(-vwaptick, axis=1)

            # replace nan with close rtn
            TradeVolume = pd.DataFrame(atp.depthDataDict[date][atp.map_stock['TotalTradeVolume']].astype(float))
            TradeValue = pd.DataFrame(atp.depthDataDict[date][atp.map_stock['TotalTradeValue']].astype(float))
            TradeVolumeShift = TradeVolume.shift(-vwaptick, axis=1).fillna(method='ffill', axis=1)
            TradeValueShift = TradeValue.shift(-vwaptick, axis=1).fillna(method='ffill', axis=1)
            zero_select = (TradeVolumeShift - TradeVolume) < 100
            # 计算vwap
            vwap = (TradeValueShift - TradeValue) / (TradeVolumeShift - TradeVolume)
            vwap = vwap.where(~zero_select, np.nan)
            abnormal_select = np.abs(vwap.values / atp.depthDataDict[date][atp.map_stock['MpClose']] - 1) > 0.2
            vwap = vwap.where(~abnormal_select, np.nan)
            # vwap = vwap.fillna(pd.DataFrame(mp)).fillna(method='ffill', axis=1)

            # vwap = vwap.replace(0, np.nan)
            vwap = vwap.values * AdjFactor.loc[atp.tickers, date].values.reshape(-1, 1)
            vwap = pd.DataFrame(vwap).fillna(pd.DataFrame(mp)).fillna(method='ffill', axis=1)

            # vwap = vwap.fillna(method='ffill', axis=1)
            vwap_liu = vwap.values
            vwap = np.where(today_ap1_tick == 0, np.nan, vwap)
            vwap[:,-10:] = np.nan  # 最后10分钟不进行买入卖出

            vwap_liu = pd.DataFrame(vwap_liu)
            vwap_liu.iloc[:, -10:] = np.nan  # 最后10分钟不进行买入卖出

            atp.factors[date]['vwap'] = vwap
            atp.factors[date]['vwap_liu'] = vwap_liu.values


            # atp.factors[date]['indexvwap'] = np.nanmean(atp.factors[date]['vwap'], axis=0)

            calcrtn_ = (- vwap + tomorrow_mp) / vwap

            atp.factors[date]['2next_liu'] = calcrtn_
            # 涨停的不买
            calcrtn_ = np.where(today_ap1_tick == 0, np.nan, calcrtn_)
            atp.factors[date]['2next'] = calcrtn_
            calcrtn_ = (- vwap + next_tomorrow_mp) / vwap
            atp.factors[date]['2nextnext_liu'] = calcrtn_
            calcrtn_ = np.where(today_ap1_tick == 0, np.nan, calcrtn_)
            atp.factors[date]['2nextnext'] = calcrtn_
            calcrtn_ = (- vwap + next_open_mp) / vwap
            atp.factors[date]['2nextopen_liu'] = calcrtn_
            calcrtn_ = np.where(today_ap1_tick == 0, np.nan, calcrtn_)
            atp.factors[date]['2nextopen'] = calcrtn_

            # 去除异常值
            if (abs(atp.factors[date]['2next_liu']) >= 1).sum() > 0:
                # print(np.where(abs(atp.factors[date]['2next_liu']) >= 1))
                print('rtn 有异常值，已被替换为 nan，发生日期为' + date)

            atp.factors[date]['2next'][abs(atp.factors[date]['2next']) > 1] = np.nan
            atp.factors[date]['2nextopen'][abs(atp.factors[date]['2nextopen']) > 1] = np.nan
            atp.factors[date]['2nextnext'][abs(atp.factors[date]['2nextnext']) > 1] = np.nan
            atp.factors[date]['2next_liu'][abs(atp.factors[date]['2next_liu']) > 1] = np.nan
            atp.factors[date]['2nextopen_liu'][abs(atp.factors[date]['2nextopen_liu']) > 1] = np.nan
            atp.factors[date]['2nextnext_liu'][abs(atp.factors[date]['2nextnext_liu']) > 1] = np.nan

            indexrtn = np.nanmean(atp.factors[date]['2next_liu'], axis=0, keepdims=True)
            atp.factors[date]['index2next'] = indexrtn
            indexrtn = np.nanmean(atp.factors[date]['2nextnext_liu'], axis=0, keepdims=True)
            atp.factors[date]['index2nextnext'] = indexrtn
            indexrtn = np.nanmean(atp.factors[date]['2nextopen'], axis=0, keepdims=True)
            atp.factors[date]['index2nextopen'] = indexrtn

            '''
            -------------------------------------------计算到明日今时的价格------------------------------------------
            '''

            '''
            -------------------------------------------------计算vwap----------------------------------------------
            '''
            mp_val = atp.depthDataDict[date][atp.map_stock['MpClose']]
            mp = pd.DataFrame(mp_val)
            # mp[mp .] = np.nan

            atp.factors[date]['MpClose'] = mp_val
            atp.factors[date]['ap1'] = mp_val
            atp.factors[date]['bp1'] = mp_val
            # atp.factors[date]['2close'] = ((-vwap + vwap[:, -1:]) / vwap)
            en_closeprice = atp.myconnector.encrypt_feature(real_names=['eod_ClosePrice'])['eod_ClosePrice']
            # 今天的eod在明天
            today_close_mp = (atp.eod_facs[tomorrow][en_adjfactor] * atp.eod_facs[tomorrow][en_closeprice]).values.reshape(-1, 1)
            atp.factors[date]['2close'] = ((-vwap + today_close_mp) / vwap)
            atp.factors[date]['2close'][abs(atp.factors[date]['2close']) > 1] = np.nan



            '''
            -------------------------------------------------计算vwap----------------------------------------------
            '''

            '''
            -------------------------------------------------计算其他rtn----------------------------------------------
            '''
            for key in cfg.apd_rtn.keys():
                for period in cfg.apd_rtn[key]:
                    if 'avg' in key:
                        # mp[mp == 0] = np.nan
                        atp.factors[date][key + str(period)] = mp_val.shift(-int(cfg.order_time * period), axis=1)
                    elif key == 'rtn':
                        calcrtn = (mp.diff(int(cfg.order_time * period), axis=1).shift(-int(cfg.order_time * period), axis=1) / mp)
                        atp.factors[date][key + str(period) + '_liu'] = calcrtn.values
                        calcrtn = np.where(today_ap1_tick == 0, np.nan, calcrtn)
                        atp.factors[date][key + str(period)] = calcrtn

                        calcrtn = (vwap_liu.diff(int(cfg.order_time * period), axis=1).shift(-int(cfg.order_time * period), axis=1) / vwap_liu)
                        atp.factors[date][key + str(period) + 'vwap_liu'] = calcrtn.values
                        calcrtn = np.where(today_ap1_tick == 0, np.nan, calcrtn)
                        atp.factors[date][key + str(period) + 'vwap'] = calcrtn

                        # 去除异常值
                        atp.factors[date][key + str(period) + 'vwap'][abs(atp.factors[date][key + str(period) + 'vwap']) > 1] = np.nan
                        atp.factors[date][key + str(period) + 'vwap_liu'][abs(atp.factors[date][key + str(period) + 'vwap_liu']) > 1] = np.nan

                        atp.factors[date]['index' + str(period) + 'vwap'] = np.nanmean(
                            atp.factors[date][key + str(period) + 'vwap'], axis=0)

                        atp.factors[date]['index' + str(period) + 'vwap_liu'] = np.nanmean(
                            atp.factors[date][key + str(period) + 'vwap_liu'], axis=0)

                if key == 'rtn':
                    calcrtn = (vwap_liu.diff(int(cfg.order_time * cfg.sample_freq), axis=1).shift(
                        -int(cfg.order_time * cfg.sample_freq), axis=1) / vwap_liu)
                    atp.factors[date][key + 'vwaporder_liu'] = calcrtn.values
                    calcrtn = np.where(today_ap1_tick == 0, np.nan, calcrtn)
                    #                         calcrtn = np.where(atp.depthDataDict[date][atp.map_stock['bp1']] == 0, np.nan,calcrtn)

            atp.factors[date]['number'] = np.arange(atp.factors[date]['rtn1'].shape[1]).reshape(1, -1) + \
                                          int(cfg.beginTime * cfg.order_time * cfg.sample_freq)

    print('rtn 计算完成 ' + str(time.time() - time0))
    if not only_fac:
        for factor_name in factor_names:
            if (type(thr_dict) == dict) and len(thr_dict.keys()) > 0:
                raise NotImplemented
            else:
                hpl.save_parameters(atp, atp.tickers, factor_name, atp.output_path, using_rank=using_rank, l_ratio=l_ratio, s_ratio=s_ratio,
                                    num_group_backtest=num_group_backtest)
    print("总时长")
    print(time.time() - time0)
    return atp.factors, atp.tickers, atp.output_path


def save_fac(ia, fac_names):
    data = {}
    for ticker in ia.tickers:
        data[ticker] = {}
        for date in ia.dates:
            df = pd.DataFrame()
            fac_npy = ia.factors[date]
            df['TimeStamp'] = range(ia.factors[date]['MpClose'].shape[1])

            for fac in fac_names:
                if fac_npy[fac].shape[0] == 1:
                    print('can not save ' + fac + ' shape rejected!')
                    raise
                df[fac] = fac_npy[fac][ia.tickers.index(ticker)]
            data[ticker][date] = df

    for fac in fac_names:
        ia.myconnector.save_fcs(data, field=fac, ns='example')  # 保存因子时会自动根据key来判断日期和ticker


def load_fac(ia, fac_names):
    res = {}
    for fac in fac_names:
        res[fac] = ia.myconnector.get_fcs_history(tickers=ia.tickers, start_date=cfg.start_date, end_date=cfg.end_date,
                                                  field=fac, ns='example')  # 一次只能读一个因子
    for fac in fac_names:
        fac_now = res[fac]
        for date in ia.dates:
            factor = []
            for ticker in ia.tickers:
                factor.append(fac_now[ticker][date].tolist())
            ia.factors[date][fac] = np.array(factor)
    return ia.factors
