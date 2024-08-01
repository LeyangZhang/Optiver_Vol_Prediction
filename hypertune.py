import gc
import torch
import einops
import pandas as pd
import numpy as np
import xarray as xr
import pytorch_lightning as pl
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib as mpl
import ipywidgets as widgets
import optiver
import itertools
import json
import os
import time

from torch import nn
from tqdm.auto import tqdm
from joblib import Parallel, delayed
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import Dataset, DataLoader

def prepare_data(stock_id, stock_ind, set, time_ids, coarsen, norm, out):
    df_book = pd.read_parquet(f'{data_dir}/book_{set}.parquet/stock_id={stock_id}')
    df_min_second = df_book.groupby('time_id').agg(min_second=('seconds_in_bucket', 'min'))
    df_book = df_book.merge(df_min_second, left_on='time_id', right_index=True) \
        .eval('seconds_in_bucket = seconds_in_bucket - min_second') \
        .drop('min_second', axis=1)
    df_trade = pd.read_parquet(f'{data_dir}/trade_{set}.parquet/stock_id={stock_id}') \
        .merge(df_min_second, left_on='time_id', right_index=True) \
        .eval('seconds_in_bucket = seconds_in_bucket - min_second') \
        .drop('min_second', axis=1)
    df = pd.merge(df_book, df_trade, on=['time_id', 'seconds_in_bucket'], how='outer')
    df['stock_id'] = stock_id
    df = df.set_index(['stock_id', 'time_id', 'seconds_in_bucket'])
    df = df.to_xarray().astype('float32')
    df = df.reindex({'time_id': time_ids, 'seconds_in_bucket': np.arange(n_seconds)})
    for name in ['bid_price1', 'bid_price2', 'ask_price1', 'ask_price2',
         'bid_size1', 'bid_size2', 'ask_size1', 'ask_size2']:
        df[name] = df[name].ffill('seconds_in_bucket')
    df['wap1'] = (df.bid_price1 * df.ask_size1 + df.ask_price1 * df.bid_size1) / (df.bid_size1 + df.ask_size1)
    df['wap2'] = (df.bid_price2 * df.ask_size2 + df.ask_price2 * df.bid_size2) / (df.bid_size2 + df.ask_size2)
    df['log_return1'] = np.log(df.wap1).diff('seconds_in_bucket')
    df['log_return2'] = np.log(df.wap2).diff('seconds_in_bucket')
    df['current_vol'] = (df.log_return1 ** 2).sum('seconds_in_bucket') ** 0.5
    df['current_vol_2nd_half'] = (df.log_return1[..., 300:] ** 2).sum('seconds_in_bucket') ** 0.5
    if coarsen > 1:
        mean_features = ['ask_price1', 'ask_price2', 'bid_price1', 'bid_price2',  'ask_size1', 'ask_size2',
               'bid_size1', 'bid_size2', 'price']
        sum_features = ['size', 'order_count']

        df = xr.merge((df[mean_features].coarsen({'seconds_in_bucket': coarsen}, coord_func='min').mean(),
                       df[sum_features].coarsen({'seconds_in_bucket': coarsen}, coord_func='min').sum(),
                       df[['current_vol', 'current_vol_2nd_half']]))
        df['wap1'] = (df.bid_price1 * df.ask_size1 + df.ask_price1 * df.bid_size1) / (df.bid_size1 + df.ask_size1)
        df['wap2'] = (df.bid_price2 * df.ask_size2 + df.ask_price2 * df.bid_size2) / (df.bid_size2 + df.ask_size2)
        df['log_return1'] = np.log(df.wap1).diff('seconds_in_bucket')
        df['log_return2'] = np.log(df.wap2).diff('seconds_in_bucket')

    df['spread1'] = df.ask_price1 - df.bid_price1
    df['spread2'] = df.ask_price2 - df.ask_price1
    df['spread3'] = df.bid_price1 - df.bid_price2
    df['total_volume'] = df.ask_size1 + df.ask_size2 + df.bid_size1 + df.bid_size2
    df['volume_imbalance1'] = df.ask_size1 + df.ask_size2 - df.bid_size1 - df.bid_size2
    df['volume_imbalance2'] = (df.ask_size1 + df.ask_size2 - df.bid_size1 - df.bid_size2) / df.total_volume
    for name in ['bid_size1', 'bid_size2', 'ask_size1', 'ask_size2', 'size', 'order_count', 'total_volume']:
        df[name] = np.log1p(df[name])
    df['volume_imbalance1'] = np.sign(df['volume_imbalance1']) * np.log1p(abs(df['volume_imbalance1']))

    for field in ['ask_price1','ask_price2','bid_price1', 'bid_price2', 'ask_size1','ask_size2', 'bid_size1','bid_size2','wap1', 'wap2',\
               'log_return1', 'log_return2', 'spread1', 'spread2', 'spread3', 'total_volume',
               'volume_imbalance1', 'volume_imbalance2']:
        df[field] = df[field].ffill(dim='seconds_in_bucket')

    df = df.fillna({'price': 1, 'size': 0, 'order_count': 0,'current_vol': 0, 'current_vol_2nd_half': 0})

    # fill the missing NAs
    df = df.fillna({'ask_price1': 1, 'ask_price2': 1, 'bid_price1': 1, 'bid_price2': 1,  'ask_size1': 0, 'ask_size2': 0,
               'bid_size1': 0, 'bid_size2': 0, 'price': 1, 'size': 0, 'order_count': 0, 'wap1': 1, 'wap2': 1,
               'log_return1': 0, 'log_return2': 0, 'spread1': 0, 'spread2': 0, 'spread3': 0, 'total_volume': 0,
               'volume_imbalance1': 0, 'volume_imbalance2': 0, 'current_vol': 0, 'current_vol_2nd_half': 0})

    features = ['ask_price1', 'ask_price2', 'bid_price1', 'bid_price2',  'ask_size1', 'ask_size2',
               'bid_size1', 'bid_size2', 'price', 'size', 'order_count', 'wap1', 'wap2',
               'log_return1', 'log_return2', 'spread1', 'spread2', 'spread3', 'total_volume',
               'volume_imbalance1', 'volume_imbalance2']
    extra = ['current_vol', 'current_vol_2nd_half']

    if norm is not None:
        mean = norm['mean'].sel(stock_id=stock_id)
        std = norm['std'].sel(stock_id=stock_id)
    else:
        mean = df.mean(('time_id', 'seconds_in_bucket')).drop(['current_vol', 'current_vol_2nd_half'])
        std = df.std(('time_id', 'seconds_in_bucket')).drop(['current_vol', 'current_vol_2nd_half'])

    df.update((df - mean) / std)
    df = df.astype('float32')

    out[:, stock_ind] = einops.rearrange(df[features].to_array().values, 'f () t sec -> t sec f')
    return df[extra], {'mean': mean, 'std': std}

class OptiverDataset(Dataset):
    def __init__(self, features_data, extra_data, mode, time_ids):
        self.features_data = features_data
        self.extra_data = extra_data
        self.time_ids = time_ids
        self.mode = mode

    def __len__(self):
        if self.mode == 'single-stock':
            return len(self.time_ids) * n_stocks
        elif self.mode == 'multi-stock':
            return len(self.time_ids)

    def __getitem__(self, i):
        if self.mode == 'single-stock':
            time_id = self.time_ids[i // n_stocks]
            time_ind = self.extra_data.indexes['time_id'].get_loc(time_id)
            stock_ind = i % n_stocks
            stock_id = self.extra_data.indexes['stock_id'][stock_ind]
            return {
                'data': self.features_data[time_ind], # (112, 600, 21)
                'target': self.extra_data['target'].values[time_ind, stock_ind],  # (1,)
                'current_vol': self.extra_data['current_vol'].values[time_ind, stock_ind],  # (1,)
                'current_vol_2nd_half': self.extra_data['current_vol_2nd_half'].values[time_ind, stock_ind],  # (1,)
                'time_id': time_id,
                'stock_id': stock_id,
                'stock_ind': stock_ind
            }
        elif self.mode == 'multi-stock':
            time_id = self.time_ids[i]
            time_ind = self.extra_data.indexes['time_id'].get_loc(time_id)
            return {
                'data': self.features_data[time_ind], # (112, 600, 21)
                'target': self.extra_data['target'].values[time_ind],  # (112,)
                'current_vol': self.extra_data['current_vol'].values[time_ind],  # (112,)
                'current_vol_2nd_half': self.extra_data['current_vol_2nd_half'].values[time_ind],  # (112,)
                'time_id': time_id,
            }
        
data_dir = 'optiver-realized-volatility-prediction'
n_features = 21
n_stocks = 112
n_seconds = 600
coarsen = 1

df_train = pd.read_csv(f'{data_dir}/train.csv')

train_data = np.memmap(f'{data_dir}/train.npy', 'float16', 'w+',
                       shape=(df_train.time_id.nunique(), n_stocks, n_seconds // coarsen, n_features))

res = Parallel(n_jobs=4, verbose=51)(
    delayed(prepare_data)(stock_id, stock_ind, 'train', df_train.time_id.unique(), coarsen, None, train_data)
    for stock_ind, stock_id in enumerate(df_train.stock_id.unique())
)

train_extra = xr.concat([x[0] for x in res], 'stock_id')
train_extra['target'] = df_train.set_index(['time_id', 'stock_id']).to_xarray()['target'].astype('float32')
train_extra = train_extra.transpose('time_id', 'stock_id')

train_data = np.array(train_data)       

train_time_ids = df_train[df_train["time_id"] < 22385]["time_id"].unique().tolist()
val_time_ids = df_train[(df_train["time_id"] >= 22385) & (df_train["time_id"] < 29239)]["time_id"].unique().tolist()
test_time_ids = df_train[df_train["time_id"] >= 29239]["time_id"].unique().tolist()

train_ds = OptiverDataset(train_data, train_extra, 'multi-stock', train_time_ids)
train_dl = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=1, pin_memory=True, persistent_workers=True)

val_ds = OptiverDataset(train_data, train_extra, 'multi-stock', val_time_ids)
val_dl = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=1, pin_memory=True, persistent_workers=True)

test_ds = OptiverDataset(train_data, train_extra, 'multi-stock', test_time_ids)
test_dl = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=1, pin_memory=True, persistent_workers=True)

dim_list = [16, 32]
conv1_kernel_list = [2,3]
rnn_layers_list = [2,3]
dropout_list = [0, 0.25, 0.5]

all_combinations = list(itertools.product(dim_list, conv1_kernel_list, rnn_layers_list, dropout_list))

all_stats_list = []

for i, combination in tqdm(enumerate(all_combinations)):
    start_time = time.time()
    print(f"running {i}-th combination")
    dim, conv1_kernel, rnn_layers, dropout = combination
    filename = f"res/stats_{dim}_{conv1_kernel}_{rnn_layers}_{dropout}.json"
    if os.path.exists(filename):
        print(f"skipped {i}-th combination")
        continue     

    model = optiver.OptiverModel(mode='multi-stock', dim=dim, conv1_kernel=conv1_kernel, rnn_layers=rnn_layers, rnn_dropout = dropout, aux_loss_weight=0)
    trainer = pl.Trainer(accelerator='auto', 
                     precision=16, 
                     max_epochs=40, 
                     enable_progress_bar=True) # multi-stock    
    trainer.fit(model, train_dl, val_dl)
    min_loss = np.min([x["val_loss_epoch"] for x in model.stats])
    print(f"finished dim={dim}, conv1_kernel={conv1_kernel}, rnn_layers={rnn_layers}, dropout={dropout}")
    print(f"min_loss={min_loss}")
    stats_to_save = model.stats
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"The combination took {elapsed_time:.1f} seconds to execute.")


    with open(filename, 'w') as json_file:
        json.dump(stats_to_save, json_file)