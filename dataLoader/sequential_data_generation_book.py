import pandas as pd
import numpy as np

def calc_wap1(df):
    # Function to calculate first WAP
    wap = (df['bid_price1'] * df['ask_size1'] + df['ask_price1'] * df['bid_size1']) / (df['bid_size1'] + df['ask_size1'])
    return wap

def calc_wap2(df):
    # Function to calculate second WAP
    wap = (df['bid_price2'] * df['ask_size2'] + df['ask_price2'] * df['bid_size2']) / (df['bid_size2'] + df['ask_size2'])
    return wap

def log_return(series):
    # Function to calculate the log of the return
    return np.log(series).diff()

def realized_volatility(series):
    # Calculate the realized volatility
    return np.sqrt(np.sum(series**2))

def realized_volatility_HL(series_price, interval_size = 30):
    # Calculate the realized volatility based on high/low price
    temp_series = series_price
    temp_series = temp_series.reset_index(drop = True)
    high = temp_series.groupby(temp_series.index // interval_size).max()
    low = temp_series.groupby(temp_series.index // interval_size).min()
    hl_ret = np.log(high/low)
    ## annualize the volatility to make it bigger for better convergence
    return np.sqrt(np.mean(hl_ret**2)/4/np.log(2)) * np.sqrt(252*60/interval_size*390)

def realized_volatility_GK(series_price, interval_size = 30):
    # Calculate the realized volatility based on Garman Klass Paper
    temp_series = series_price
    temp_series = temp_series.reset_index(drop = True)
    high = temp_series.groupby(temp_series.index // interval_size).max()
    low = temp_series.groupby(temp_series.index // interval_size).min()
    open = temp_series.groupby(temp_series.index // interval_size).first()
    close = temp_series.groupby(temp_series.index // interval_size).last()
    hl_ret = np.log(high/low)
    co_ret = np.log(close/open)
    ## annualize the volatility to make it bigger for better convergence
    return np.sqrt(0.5*np.mean(hl_ret**2) - (2*np.log(2)-1)*np.mean(co_ret**2)) * np.sqrt(252*60/interval_size*390)

def count_unique(series):
    # Function to count unique elements of a series
    return len(np.unique(series))

def expand_and_ffill(df):
    df = df.set_index('seconds_in_bucket')
    df = df.reindex(range(600))#
    df = df.reset_index()
    df.ffill(inplace=True, axis=0)
    return df
def book_preprocessor(file_path):
    # Function to preprocess book data (for each stock id)

    df = pd.read_parquet(file_path)

    df = df.groupby(['time_id']).apply(expand_and_ffill)
    df = df.reset_index(drop=True)
    # Calculate Wap
    df['wap1'] = calc_wap1(df)
    df['wap2'] = calc_wap2(df)

    # Calculate log returns
    df['log_return1'] = df.groupby(['time_id'])['wap1'].apply(log_return).reset_index()["wap1"]
    df['log_return2'] = df.groupby(['time_id'])['wap2'].apply(log_return).reset_index()["wap2"]

    # Calculate wap balance
    df['wap_balance'] = abs(df['wap1'] - df['wap2'])

    # Calculate spread
    df['price_spread'] = (df['ask_price1'] - df['bid_price1']) / ((df['ask_price1'] + df['bid_price1']) / 2)

    df['total_volume'] = (df['ask_size1'] + df['ask_size2']) + (df['bid_size1'] + df['bid_size2'])
    df['volume_imbalance'] = abs((df['ask_size1'] + df['ask_size2']) - (df['bid_size1'] + df['bid_size2']))

    feature_list = ['wap1', 'wap2', 'log_return1', 'log_return2','wap_balance','price_spread','total_volume','volume_imbalance']
    def get_ewm_stats(look_back_window, add_suffix=True):
        df_rolling_feature_mean = df.groupby(['time_id'])[feature_list].ewm(halflife=look_back_window).mean()
        df_rolling_feature_std = df.groupby(['time_id'])[feature_list].ewm(halflife=look_back_window).std()
        # Rename columns joining suffix
        df_rolling_feature_mean.columns = [col+'_mean' for col in df_rolling_feature_mean.columns]
        df_rolling_feature_std.columns = [col+'_std' for col in df_rolling_feature_std.columns]
        df_rolling_feature = pd.concat([df_rolling_feature_mean, df_rolling_feature_std],axis=1)
        df_rolling_feature = df_rolling_feature.reset_index(level=0, names=['time_id', None])
        df_rolling_feature['seconds_in_bucket'] = df['seconds_in_bucket']
        df_rolling_feature = df_rolling_feature.set_index(['time_id', 'seconds_in_bucket'])
        df_rolling_feature.fillna(0,inplace=True)
        # Add a suffix to differentiate windows
        if add_suffix:
            df_rolling_feature = df_rolling_feature.add_suffix('_' + str(look_back_window))
        return df_rolling_feature

    # Get the stats for different windows

    df_feature_300 = get_ewm_stats(look_back_window=300, add_suffix=True)
    df_feature_60 = get_ewm_stats(look_back_window=60, add_suffix=True)
    df = df.set_index(['time_id', 'seconds_in_bucket'])
    df_feature = pd.concat([df, df_feature_300, df_feature_60],axis=1)
    df_feature = df_feature.reset_index()
    # Create row_id so we can merge
    stock_id = file_path.split('=')[1]
    df_feature['row_id'] = df_feature['time_id'].apply(lambda x: f'{stock_id}-{x}')

    return df_feature

#book_feature = book_preprocessor('optiver-realized-volatility-prediction/book_train.parquet/stock_id=0')
