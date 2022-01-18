import sys
sys.path.insert(1, '/content/ED-MTT/code')
import dataloader
import model
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



@hydra.main(config_path="/content/ED-MTT/configs/",config_name="batchnorm_default")
def train(cfg: DictConfig):

  config = {
        # 'frame_size': cfg.data.frame_size,
        # 'step_size': cfg.data.step_size,
        # "lstm_hidden_size": cfg.model.lstm.n_hidden ,
        # "lstm_hidden_layers":  cfg.model.lstm.n_layers,
        # "mlp_layer_1_size": cfg.model.mlp.h1,
        # "mlp_layer_2_size":  cfg.model.mlp.h2,
        # "lr": cfg.model.train.lr,
        # "triplet_margin": cfg.model.train.triplet_margin,
        # "batch_size": cfg.data.batch_size
        "attributes":cfg.data.attributes
    }
    
  wandb.init(project="Engagement Detection", config=config)

  # cfg.data.frame_size = wandb.config.frame_size
  # cfg.data.step_size = wandb.config.step_size
  # cfg.model.lstm.n_hidden = wandb.config.lstm_hidden_size
  # cfg.model.lstm.n_layers = wandb.config.lstm_hidden_layers
  # cfg.model.mlp.h1 = wandb.config.mlp_layer_1_size
  # cfg.model.mlp.h2 = wandb.config.mlp_layer_2_size
  # cfg.model.train.lr = wandb.config.lr
  # cfg.model.train.triplet_margin = wandb.config.triplet_margin
  # cfg.data.batch_size = wandb.config.batch_size
  cfg.data.attributes = wandb.config.attributes

  logger.info(OmegaConf.to_yaml(cfg))
  
  data_module = dataloader.create_data_module(cfg)
  wandb_logger = WandbLogger(project="Engagement Detection")
  early_stopping = EarlyStopping('validation_loss',
  divergence_threshold =cfg.model.train.threshold)
  checkpoint_callback = ModelCheckpoint(
      dirpath="/content/ED-MTT/checkpoints",
      filename="best-checkpoint",
      save_top_k=1,
      verbose = True,
      monitor="validation_loss",
      mode="min"
  )
  pred_model = model.EngagementPredictor(
      data_module.train_dataset[0]["anchor_sequence"].shape[1],
      cfg.model)
  #lr_monitor = LearningRateMonitor(logging_interval='epoch')
  trainer= pl.Trainer(
      logger=wandb_logger,  
      callbacks = checkpoint_callback,
      max_epochs=cfg.model.train.n_epochs,
      gpus=1,
      progress_bar_refresh_rate=30,
      deterministic=True,
      sync_batchnorm=True

  )
  trainer.fit(pred_model,data_module)
  wandb.finish()
if __name__ == "__main__":

  train()
