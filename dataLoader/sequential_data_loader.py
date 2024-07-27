import torch
from torch.utils.data import Dataset
import os
import pandas as pd
from sequential_data_generation_book import book_preprocessor
from sequantial_data_generation_trade import trade_preprocessor
from joblib import Parallel, delayed

data_dir = '../optiver-realized-volatility-prediction/'
def preprocessor():
    def for_joblib(stock_id):
        if os.path.isfile(f'../processed_seq_data/feature_parquet_stock_id={stock_id}.parquet'):
            return
        print(f'generating sequential features for {stock_id}')
        file_path_book = data_dir + "book_train.parquet/stock_id=" + str(stock_id)
        file_path_trade = data_dir + "trade_train.parquet/stock_id=" + str(stock_id)
        book_features = book_preprocessor(file_path_book)
        trade_features = trade_preprocessor(file_path_trade)
        book_features = book_features.set_index(['row_id','time_id','seconds_in_bucket'])
        trade_features = trade_features.set_index(['row_id','time_id','seconds_in_bucket'])
        df_tmp = pd.concat([book_features,trade_features], axis=1).reset_index()
        df_tmp['stock_id'] = stock_id
        df_tmp.to_parquet(f'../processed_seq_data/feature_parquet_stock_id={stock_id}.parquet')
        print("Processed Data:", stock_id)


    train = pd.read_csv('../optiver-realized-volatility-prediction/train.csv')
    Parallel(n_jobs=-1, verbose=1)(delayed(for_joblib)(stock_id) for stock_id in train['stock_id'])
    return

# preprocessor()


# class BookAndTradeLoader(Dataset):
#     def __init__(self, mode='training'):
#
#
#     def __len__(self):
#         return len(self.img_labels)
#
#     def __getitem__(self, idx):
#         img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
#         image = read_image(img_path)
#         label = self.img_labels.iloc[idx, 1]
#         if self.transform:
#             image = self.transform(image)
#         if self.target_transform:
#             label = self.target_transform(label)
#         return image, label