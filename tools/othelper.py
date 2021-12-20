import pandas as pd
import numpy as np
import copy
import os
from PqiDataSdk import *
import config as cfg
import multiprocessing

def processfunc(ticker, date):
    myconnector = PqiDataSdk(cfg.user, pool_type="mt", log=False)
    try:
        df_norm = myconnector.get_3to1_history(ticker, date, date)[ticker][date]
    except:
        return
    df_norm["T_Val"] = df_norm["TradePrice"] * df_norm["TradeVolume"] / 10000.  # 金额归到：万元
    df_norm["O_Val"] = df_norm["OrderPrice"] * df_norm["OrderVolume"] / 10000.  # 金额归到：万元
    act_buy = 0.;
    act_sell = 0.;
    act_b_num = 0;
    act_s_num = 0
    pd_b = 0.;
    pd_s = 0.;
    pd_b_num = 0;
    pd_s_num = 0;
    T_id = 0

    od_cancel_b = 0
    od_cancel_s = 0

    mp = (df_norm[df_norm.marker == 'D'].ap1.tolist()[0] + df_norm[df_norm.marker == 'D'].bp1.tolist()[0]) / 2

    BP5 = 0.995 * mp
    AP5 = 1.005 * mp

    whole_len = len(df_norm)
    # 得到每两个tick之间的金额，交易量
    for i, row in df_norm.iterrows():
        if row['marker'] == 'O':
            if (row['Side'] == '1') & (row['OrderPrice'] >= BP5):
                if i < (whole_len - 10):
                    if not ((df_norm.loc[i + 1, 'marker'] == 'T') & (df_norm.loc[i + 1, 'ExecType'] == 'F')):
                        pd_b += row['O_Val']
                        pd_b_num += 1

            elif (row['Side'] == '2') & (row['OrderPrice'] <= AP5):
                if i < (whole_len - 10):
                    if not ((df_norm.loc[i + 1, 'marker'] == 'T') & (df_norm.loc[i + 1, 'ExecType'] == 'F')):
                        pd_s += row['O_Val']
                        pd_s_num += 1

        elif (row['marker'] == 'T') & (row['ExecType'] == 'F'):
            if row['SellIndex'] > row['BuyIndex']:
                act_sell += row['T_Val']
                if T_id == row['TradeIndex'] - 1:
                    T_id = row['TradeIndex']
                else:
                    act_s_num += 1
                    T_id = row['TradeIndex']

            else:
                act_buy += row['T_Val']
                if T_id == row['TradeIndex'] - 1:
                    T_id = row['TradeIndex']
                else:
                    act_b_num += 1
                    T_id = row['TradeIndex']
        elif (row['marker'] == 'T') & (row['ExecType'] != 'F'):
            if row['SellIndex'] > row['BuyIndex']:
                od_cancel_s += row['TradeVolume']
            else:
                od_cancel_b += row['TradeVolume']

        elif row['marker'] == 'D':
            mp = (row['ap1'] + row['bp1']) / 2

            BP5 = 0.995 * mp
            AP5 = 1.005 * mp
            df_norm.loc[i, "act_b"] = act_buy
            df_norm.loc[i, "act_s"] = act_sell
            df_norm.loc[i, "act_b_num"] = act_b_num
            df_norm.loc[i, "act_s_num"] = act_s_num
            df_norm.loc[i, "pd_b"] = pd_b
            df_norm.loc[i, "pd_s"] = pd_s
            df_norm.loc[i, "pd_b_num"] = pd_b_num
            df_norm.loc[i, "pd_s_num"] = pd_s_num

            df_norm.loc[i, "od_cancel_b"] = od_cancel_b
            df_norm.loc[i, "od_cancel_s"] = od_cancel_s

            act_buy, act_sell, act_b_num, act_s_num = 0, 0, 0, 0  # 这边配tradenum的原因是tick数据不准
            pd_b, pd_s, pd_b_num, pd_s_num = 0, 0, 0, 0
            od_cancel_b = 0
            od_cancel_s = 0

    df_tk = df_norm[df_norm.marker == 'D']
    df_tk.reset_index(drop=True, inplace=True)
    df_tk[['TimeStamp',"act_b", "act_s", "act_b_num", "act_s_num", "pd_b", "pd_s", "pd_b_num", "pd_s_num", "od_cancel_b",
                  "od_cancel_s"]].to_csv(cfg.direction_3to1 + '{}_{}.csv'.format(ticker,date))


def generateOTfac(ia, func, pool_num = 32):
    if not os.path.exists(cfg.direction_3to1):
        os.makedirs(cfg.direction_3to1)
    fac_list = ["act_b", "act_s", "act_b_num", "act_s_num", "pd_b", "pd_s", "pd_b_num", "pd_s_num", "od_cancel_b",
                  "od_cancel_s"]

    pool = multiprocessing.Pool()
    for ticker in cfg.stock_pool:
        if ticker < '600000':
            for date in ia.dates:
                if not os.path.exists(cfg.direction_3to1 + '{}_{}.csv'.format(ticker,date)):
                    pool.apply_async(func,(ticker,date,))
    pool.close()
    pool.join()



def getOFfac(ia):
    if not os.path.exists(cfg.direction_OTFac):
        os.mkdir(cfg.direction_OTFac)
    extend_ids = ["act_b", "act_s", "act_b_num", "act_s_num", "pd_b", "pd_s", "pd_b_num", "pd_s_num", "od_cancel_b",
                  "od_cancel_s"]
    for date in ia.dates:
        print(date)
        act_b = []
        act_s = []
        act_b_num = []
        act_s_num = []
        pd_b = []
        pd_s = []
        pd_b_num = []
        pd_s_num = []
        od_cancel_b = []
        od_cancel_s = []
        for ticker in cfg.stock_pool:
            if not os.path.exists(cfg.direction_3to1 + '{}_{}.csv'.format(ticker,date)):
                act_b.append([np.nan for i in range(4801)])
                act_s.append([np.nan for i in range(4801)])
                act_b_num.append([np.nan for i in range(4801)])
                act_s_num.append([np.nan for i in range(4801)])
                pd_b.append([np.nan for i in range(4801)])
                pd_s.append([np.nan for i in range(4801)])
                pd_b_num.append([np.nan for i in range(4801)])
                pd_s_num.append([np.nan for i in range(4801)])
                od_cancel_b.append([np.nan for i in range(4801)])
                od_cancel_s.append([np.nan for i in range(4801)])
            else:
                df = pd.read_csv(cfg.direction_3to1 + '{}_{}.csv'.format(ticker,date))
                act_b.append(df['act_b'].to_list())
                act_s.append(df['act_s'].to_list())
                act_b_num.append(df['act_b_num'].to_list())
                act_s_num.append(df['act_s_num'].to_list())
                pd_b.append(df['pd_b'].to_list())
                pd_s.append(df['pd_s'].to_list())
                pd_b_num.append(df['pd_b_num'].to_list())
                pd_s_num.append(df['pd_s_num'].to_list())
                od_cancel_b.append(df['od_cancel_b'].to_list())
                od_cancel_s.append(df['od_cancel_s'].to_list())

        act_b = np.array(act_b)
        act_s = np.array(act_s)
        act_b_num = np.array(act_b_num)
        act_s_num = np.array(act_s_num)
        pd_b = np.array(pd_b)
        pd_s = np.array(pd_s)
        pd_b_num = np.array(pd_b_num)
        pd_s_num = np.array(pd_s_num)
        od_cancel_b = np.array(od_cancel_b)
        od_cancel_s = np.array(od_cancel_s)

        np.save(cfg.direction_OTFac + 'act_b{}.npy'.format(date), act_b)
        np.save(cfg.direction_OTFac + 'act_s{}.npy'.format(date), act_s)
        np.save(cfg.direction_OTFac + 'act_b_num{}.npy'.format(date), act_b_num)
        np.save(cfg.direction_OTFac + 'act_s_num{}.npy'.format(date), act_s_num)
        np.save(cfg.direction_OTFac + 'pd_b{}.npy'.format(date), pd_b)
        np.save(cfg.direction_OTFac + 'pd_s{}.npy'.format(date), pd_s)
        np.save(cfg.direction_OTFac + 'pd_b_num{}.npy'.format(date), pd_b_num)
        np.save(cfg.direction_OTFac + 'pd_s_num{}.npy'.format(date), pd_s_num)
        np.save(cfg.direction_OTFac + 'od_cancel_b{}.npy'.format(date), od_cancel_b)
        np.save(cfg.direction_OTFac + 'od_cancel_s{}.npy'.format(date), od_cancel_s)


