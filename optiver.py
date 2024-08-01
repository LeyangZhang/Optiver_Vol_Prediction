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

from torch import nn
from tqdm.auto import tqdm
from joblib import Parallel, delayed
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import Dataset, DataLoader

n_features = 21
n_stocks = 112
n_seconds = 600
coarsen = 1

class TimeAttention(nn.Module):
    def __init__(self, steps):
        super().__init__()
        self.steps = steps
        self.weights = nn.Parameter(torch.zeros(steps))

    def forward(self, x):
        # x: (b, st, t, f)
        attn = F.softmax(self.weights, 0)
        x = torch.einsum('b s t f, t -> b s f', x, attn)
        return x

# You can experiment with other ideas for stock attention: maybe it could be
# something like MultiHeadAttention module with keys and queries that depends on current input,
# maybe it could be a linear combination of all stocks (full connected layer),
# maybe you can try sparse softmax
class StockAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros((n_stocks, n_stocks)))
        self.bias = nn.Parameter(torch.zeros(n_stocks))
        self.fc_combine = nn.Linear(dim * 2, dim)

    def forward(self, x):
        # x: (b, st, t, f)
        attn = F.softmax(self.weight + self.bias[None, :], dim=-1) # (st, st)
        y = torch.einsum('b i ..., j i -> b j ...', x, attn)
        x = torch.cat((x, y), -1)
        x = self.fc_combine(x)
        return x

class OptiverModel(pl.LightningModule):
    def __init__(self, mode='multi-stock', dim=32, conv1_kernel=3, rnn_layers=2, rnn_dropout=0.3,
                 n_features=21, aux_loss_weight=1.0):
        super().__init__()
        self.save_hyperparameters()
        self.stock_emb = nn.Embedding(n_stocks, dim)
        self.stock_emb.weight.data.normal_(0, 0.2)
        self.conv1 = nn.Conv1d(n_features, dim, conv1_kernel, conv1_kernel)
        self.conv2 = nn.Conv1d(dim, dim, 1, 1)
        self.norm1 = nn.LayerNorm([n_stocks, dim])
        self.norm2 = nn.LayerNorm([n_stocks, dim])
        # self.rnn = nn.GRU(dim, dim, rnn_layers, batch_first=True, dropout=rnn_dropout)
        self.rnn = nn.LSTM (dim, dim, rnn_layers, batch_first=True, dropout=rnn_dropout)

        self.timesteps_attn = TimeAttention(600 // conv1_kernel // coarsen)
        self.timesteps_attn2 = TimeAttention(300 // conv1_kernel // coarsen)
        self.stock_attn = StockAttention(dim)
        self.fc_out1 = nn.Linear(dim, 1)
        self.fc_out2 = nn.Linear(dim, 1)
        self.history = pd.DataFrame()

        self.vol_predictions = []
        self.vol_target = []

        self.stats = []

    def forward(self, x, stock_ind=None):
        # x: (b, st, t, f)
        x = einops.rearrange(x, 'b st t f -> (b st) f t')
        x = self.conv1(x)
        x = einops.rearrange(x, '(b st) f t -> b t st f', st=n_stocks)
        x = F.gelu(x)
        x = self.norm1(x)
        x = einops.rearrange(x, 'b t st f -> (b st) f t')
        x = self.conv2(x)
        x = einops.rearrange(x, '(b st) f t -> b t st f', st=n_stocks)
        x = F.gelu(x)
        x = self.norm2(x)
        x = einops.rearrange(x, 'b t st f -> b st t f')
        x = self.stock_attn(x)
        x = x + self.stock_emb.weight[None, :, None, :]
        if self.hparams.mode == 'single-stock':
            x = x[torch.arange(len(x)), stock_ind][:, None]
        x = einops.rearrange(x, 'b st t f -> (b st) t f')
        x = self.rnn(x)[0]
        x = einops.rearrange(x, '(b st) t f -> b st t f', st=n_stocks if self.hparams.mode == 'multi-stock' else 1)
        x1 = self.timesteps_attn(x)
        x2 = self.timesteps_attn2(x[:, :, :self.timesteps_attn2.steps, :])
        x1 = self.fc_out1(x1)
        x2 = self.fc_out2(x2)
        x1 = x1 * 0.63393 - 5.762331
        x2 = x2 * 0.67473418 - 6.098946
        x1 = torch.exp(x1)
        x2 = torch.exp(x2)
        if self.hparams.mode == 'single-stock':
            return {
                'vol': x1[:, 0, 0], # (b,)
                'vol2': x2[:, 0, 0] # (b,)
            }
        else:
            return {
                'vol': x1[..., 0], # (b, st)
                'vol2': x2[..., 0] # (b, st)
            }

    def training_step(self, batch, batch_ind):
        res = self.common_step(batch, 'train')
        loss = res['loss']
        target = res['target']
        pred = res["vol"]

        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, batch_ind):
        res = self.common_step(batch, 'valid')
        loss = res['loss']
        target = res['target']
        pred = res["vol"]

        self.vol_target.append(target)    
        self.vol_predictions.append(pred)
        
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def common_step(self, batch, stage):
        out = self(batch['data'], batch['stock_ind'] if self.hparams.mode == 'single-stock' else None)
        mask1 = ~torch.isnan(batch['target'])
        target1 = torch.where(mask1, batch['target'], torch.tensor(1.0, device=self.device))
        vol_loss = (((out['vol'] - target1) / target1) ** 2)[mask1].mean() ** 0.5
        loss = vol_loss 

        return {
            'loss': loss,
            'target': batch['target'],
            'vol': out['vol'].detach(),
            'time_id': batch['time_id']
        }

    def common_epoch_end(self, outs, stage):
        print("outs:", outs)  # Add this line to print the contents of outs

        target = torch.cat([x['target'] for x in outs])
        vol = torch.cat([x['vol'] for x in outs])
        time_ids = torch.cat([x['time_id'] for x in outs])
        mask = ~torch.isnan(target)
        target = torch.where(mask, target, torch.tensor(1.0, device=self.device))
        rmspe = (((vol - target) / target) ** 2)[mask].mean() ** 0.5
        self.log(f'{stage}/rmspe', rmspe, prog_bar=True, on_step=False, on_epoch=True)
        self.history.loc[self.trainer.current_epoch, f'{stage}/rmspe'] = rmspe.item()


    def on_validation_epoch_end(self):
        all_preds = torch.cat(self.vol_predictions, dim=0)
        all_targets = torch.cat(self.vol_target, dim=0)
        mask = ~torch.isnan(all_targets)
        rmspe = (((all_preds - all_targets) / all_targets) ** 2)[mask].mean() ** 0.5
        self.log('rmspe_val', rmspe, prog_bar=True, logger=True)

        self.vol_predictions.clear()
        self.vol_target.clear()

        self.stats.append({k: v for k, v in self.trainer.progress_bar_metrics.items()})

    def on_validation_epoch_end(self):
        # all_preds = torch.cat(self.vol_predictions, dim=0)
        # all_targets = torch.cat(self.vol_target, dim=0)
        # mask = ~torch.isnan(all_targets)
        # rmspe = (((all_preds - all_targets) / all_targets) ** 2)[mask].mean() ** 0.5
        # self.log('rmspe_train', rmspe, prog_bar=True, logger=True)

        # self.vol_predictions.clear()
        # self.vol_target.clear()

        self.stats.append({k: v for k, v in self.trainer.progress_bar_metrics.items()})


    def configure_optimizers(self):
        opt = AdamW(self.parameters(), lr=1e-3)
        sched = {
            'scheduler': ExponentialLR(opt, 0.93),
            'interval': 'epoch'
        }
        return [opt], [sched]


