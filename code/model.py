import sys
sys.path.insert(1, '/content/ED-MTT/code')
import dataloader
import os
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR,ReduceLROnPlateau

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
import torchmetrics
from torchmetrics.functional import accuracy
from pytorch_lightning.tuner.lr_finder import lr_find
from pytorch_lightning.callbacks import EarlyStopping
import hydra
from omegaconf import DictConfig, OmegaConf
from torchsampler import ImbalancedDatasetSampler
import logging
logger = logging.getLogger(__name__)

import wandb
from pytorch_lightning.loggers import WandbLogger

os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:2"
wandb.login()


class SequenceModel(nn.Module):
  def __init__(self, n_features, hparam: DictConfig, freeze_lstm = False):
    super().__init__()
    self.n_hidden = hparam.lstm.n_hidden
    self.n_layers = hparam.lstm.n_layers
    self.dropout = hparam.train.dropout
    self.freeze_lstm = freeze_lstm
    pl.seed_everything(hparam.seed,workers=True)
    def weight_init(m):
      if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    
    self.mlp = nn.Sequential(
      nn.Flatten(),
      nn.Linear(self.n_hidden*2 , hparam.mlp.h1),
      nn.ReLU(),
      nn.Linear(hparam.mlp.h1, hparam.mlp.h2),
      nn.ReLU(),
      nn.Linear(hparam.mlp.h2,hparam.mlp.out),
      #nn.ReLU()
    )
    #self.mlp.apply(weight_init)
    self.lstm = nn.LSTM(
         input_size= n_features,
         hidden_size=self.n_hidden,
         num_layers=self.n_layers,
         batch_first=True,
         dropout = self.dropout,
         bidirectional = True
         )
    if freeze_lstm:
      for param in self.lstm.parameters():
        param.requires_grad = False
    


  def forward(self, x):
    #xr = torch.reshape(self.cnn1d(x),(-1,12,8))
    xr=x
    #xr= self.batch_norm(x)
    h0 = torch.zeros(self.n_layers*2, xr.size(0), self.n_hidden)
    h0= h0.type_as(x)
    c0 = torch.zeros(self.n_layers*2, xr.size(0), self.n_hidden)
    c0= c0.type_as(x)
    out,_ = self.lstm(xr,(h0,c0))
    out_lstm= out.type_as(x)
    out_mlp = self.mlp(out_lstm[:,-1, :])
    return out_lstm,out_mlp
  
  


class EngagementPredictor(pl.LightningModule):

  def __init__(self, n_features: int,hparam: DictConfig,):
    super().__init__()
    self.hparam = hparam
    self.model=SequenceModel(n_features,self.hparam)
    self.criterion_reg = nn.MSELoss()
    self.criterion_triplet = nn.TripletMarginWithDistanceLoss(margin=self.hparam.train.triplet_margin)
    self.save_hyperparameters()
    self.batch_norm = nn.BatchNorm1d(n_features)
  def forward(self, x, labels=None):
    x=torch.permute(x, (0, 2, 1))
    x=self.batch_norm(x)
    x=torch.permute(x, (0, 2, 1))
    out_lstm, out_mlp = self.model(x)
    return out_lstm, out_mlp
  
  def training_step(self, batch, batch_idx):
    anchor_sequence = batch["anchor_sequence"]
    anchor_label = batch["anchor_label"]
    positive_sequence = batch["positive_sequence"]
    positive_label = batch["positive_label"]
    negative_sequence = batch["negative_sequence"]
    negative_label = batch["negative_label"]
    out_anchor_lstm, out_anchor_mlp = self.forward(anchor_sequence, anchor_label)
    out_positive_lstm, out_positive_mlp = self.forward(positive_sequence, positive_label)
    out_negative_lstm, out_negative_mlp = self.forward(negative_sequence, negative_label)
    loss_reg = self.criterion_reg(out_anchor_mlp,anchor_label.unsqueeze(dim=1))
    loss_trip = self.criterion_triplet(out_anchor_lstm,out_positive_lstm,out_negative_lstm)
    loss = loss_reg + loss_trip
    #sch = self.lr_schedulers()
    #if self.current_epoch%30==0:
      #step = sch.step()

    self.log("train_loss",loss, prog_bar=True, logger=True)
    self.log("train_loss_reg",loss_reg, prog_bar=True, logger=True)
    #self.log("lr", sch.get_last_lr()[0], prog_bar=True, logger=True)
    return loss

  def validation_step(self, batch, batch_idx):
    anchor_sequence = batch["anchor_sequence"]
    anchor_label = batch["anchor_label"]
    out_anchor_lstm, out_anchor_mlp = self.forward(anchor_sequence, anchor_label)
    loss_reg = self.criterion_reg(out_anchor_mlp,anchor_label.unsqueeze(dim=1))
    loss = loss_reg 
    #sch = self.lr_schedulers()
    #step = sch.step(loss)
    self.log("validation_loss",loss, prog_bar=True, logger=True)
    #self.log("lr", sch.get_last_lr()[0], prog_bar=True, logger=True)
    return loss

  def test_step(self, batch, batch_idx):
    anchor_sequence = batch["anchor_sequence"]
    anchor_label = batch["anchor_label"]
    out_anchor_lstm, out_anchor_mlp = self.forward(anchor_sequence, anchor_label)
    loss_reg = self.criterion_reg(out_anchor_mlp,anchor_label.unsqueeze(dim=1))
    loss = loss_reg 

    self.log("test_loss",loss, prog_bar=True, logger=True)
    return loss

  def configure_optimizers(self):
    adam_optimizer = optim.AdamW(self.parameters(), lr=self.hparam.train.lr)
    #lr_scheduler = StepLR(adam_optimizer, step_size=10, gamma=0.95)
    #lr_scheduler = ReduceLROnPlateau(adam_optimizer, factor = 0.8,
     #patience = 10, min_lr = 0.00001, verbose=True)
    return [adam_optimizer]#, [lr_scheduler]
    #return {
    #    "optimizer": adam_optimizer,
    #    "lr_scheduler": {
    #        "scheduler": lr_scheduler,
    #        "monitor": "validation_loss"
    #    },
    #}

  """def on_train_batch_start(self, batch, batch_idx):
    val_loss = self.validation_step(batch, batch_idx)
    if val_loss > self.hparam.train.threshold:
      return -1"""
    