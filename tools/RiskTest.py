from PqiDataSdk import PqiDataSdk
import pandas as pd

ds = PqiDataSdk(user="mtyang", pool_type="mp",size =16)

# 生成参照指数成分股代码，默认返回20200331那天中证1000的成分股名称
def gen_index_namelist(index_code,date):
    index_weight = ds.get_index_weight(ticker=index_code,start_date=date,day_count=1)
    index_namelist = sorted(index_weight['StockTicker'].to_list())
    return index_namelist



def cal_index_feature(index_name,start_date,end_date):
    eod = ds.get_eod_history(tickers=index_name, fields=['FloatMarketValue','TurnoverRate'],start_date = start_date, end_date = end_date)
    MV = pd.DataFrame(eod['FloatMarketValue'])
    median_MV = MV.median(axis=1).median()
    TurnoverRate = pd.DataFrame(eod['TurnoverRate'])
    median_TurnoverRate = TurnoverRate.median(axis=1).median()
    return MV,median_MV,TurnoverRate,median_TurnoverRate


# 接收RatioBackTest的结果，按次数将其转为每天的权重
def cal_position(ratio_result):
    res_df = pd.DataFrame(ratio_result)
    count_df = pd.DataFrame(res_df.groupby(["date", "stock_factor"])['main_factor'].count()).reset_index()
    count_df = pd.pivot_table(count_df, index=['stock_factor'], values=['main_factor'], columns=['date'])
    count_df.columns = res_df['date'].unique()
    count_df.fillna(0, inplace=True) #pivot_table会有nan值产生
    position= count_df / count_df.sum()
    return position


def cal_feature(feature, position):
    median_dict=dict()
    for date in position.columns:
        merge_data = pd.concat([position[date],feature[date]],axis=1)
        merge_data.columns=['position','feature']
        sort_data = merge_data.sort_values(by='feature')
        sort_data['cum_pos'] = sort_data['position'].cumsum()
        median_value = sort_data.loc[sort_data['cum_pos']>=0.5,'feature'][0]
        median_dict[date]=median_value
    median_daily=pd.DataFrame(median_dict,index=['median']).T
    median_res=median_daily.median()[0]
    return median_daily,median_res


def cal_risk(stock_pool, ratio_result, start_date, end_date, index_code='000852', date='20200331'):
    pool_MV, pool_median_MV, pool_TR, pool_median_TR = cal_index_feature(stock_pool,start_date,end_date)
    index_namelist = gen_index_namelist(index_code, date)
    _, index_median_MV, _, index_median_TR = cal_index_feature(index_namelist,start_date,end_date)
    position = cal_position(ratio_result)
    holding_median_MV_daily,holding_median_MV = cal_feature(pool_MV, position)
    holding_median_TR_daily,holding_median_TR = cal_feature(pool_TR, position)
    return pool_median_MV,pool_median_TR,index_median_MV,index_median_TR,holding_median_MV_daily,holding_median_MV,holding_median_TR_daily,holding_median_TR,pool_MV,pool_TR










