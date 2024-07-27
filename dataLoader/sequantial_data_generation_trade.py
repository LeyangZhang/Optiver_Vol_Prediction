import pandas as pd
import numpy as np

def expand(df):
    df = df.set_index('seconds_in_bucket')
    df = df.reindex(range(600))#
    df = df.reset_index()
    df['price'] = df['price'].ffill()
    df['time_id'] = df['time_id'].bfill()
    df['time_id'] = df['time_id'].ffill()
    df.fillna(0,inplace=True)
    return df

def log_return(series):
    # Function to calculate the log of the return
    return np.log(series).diff()
def trade_preprocessor(file_path):
    # Function to preprocess trade data (for each stock id)

    df = pd.read_parquet(file_path)
    df['log_return'] = df.groupby('time_id')['price'].apply(log_return).reset_index()["price"]
    df = df.groupby(['time_id']).apply(expand)
    df = df.reset_index(drop=True)


    feature_list = ['log_return', 'size', 'order_count']

    def get_ewm_stats(look_back_window, add_suffix=True):
        df_rolling_feature_mean = df.groupby(['time_id'])[feature_list].ewm(halflife=look_back_window).mean()
        df_rolling_feature_std = df.groupby(['time_id'])[feature_list].ewm(halflife=look_back_window).std()
        df_rolling_feature_sum = df.groupby(['time_id'])[feature_list].ewm(halflife=look_back_window).sum()

        # Rename columns joining suffix
        df_rolling_feature_mean.columns = [col + '_mean' for col in df_rolling_feature_mean.columns]
        df_rolling_feature_std.columns = [col + '_std' for col in df_rolling_feature_std.columns]
        df_rolling_feature_sum.columns = [col + '_sum' for col in df_rolling_feature_sum.columns]

        df_rolling_feature = pd.concat([df_rolling_feature_mean, df_rolling_feature_std,df_rolling_feature_sum], axis=1)
        df_rolling_feature = df_rolling_feature.reset_index(level=0, names=['time_id', None])
        df_rolling_feature['seconds_in_bucket'] = df['seconds_in_bucket']
        df_rolling_feature = df_rolling_feature.set_index(['time_id', 'seconds_in_bucket'])
        df_rolling_feature.fillna(0, inplace=True)
        # Add a suffix to differentiate windows
        if add_suffix:
            df_rolling_feature = df_rolling_feature.add_suffix('_' + str(look_back_window))
        return df_rolling_feature

    df_feature_600 = get_ewm_stats(look_back_window=600, add_suffix=True)
    df_feature_300 = get_ewm_stats(look_back_window=300, add_suffix=True)
    df_feature_60 = get_ewm_stats(look_back_window=60, add_suffix=True)
    df = df.set_index(['time_id', 'seconds_in_bucket'])
    df_feature = pd.concat([df, df_feature_600, df_feature_300, df_feature_60], axis=1)
    df_feature = df_feature.reset_index()
    # Create row_id so we can merge
    stock_id = file_path.split('=')[1]
    df_feature['row_id'] = df_feature['time_id'].apply(lambda x: f'{stock_id}-{x}')

    return df_feature

#trade_feature = trade_preprocessor('optiver-realized-volatility-prediction/trade_train.parquet/stock_id=0')
