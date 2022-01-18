import sys
sys.path.insert(1, '/content/ED-MTT/code')
import dataloader
import model
import utils
import numpy as np
import os
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm.auto import tqdm
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torchmetrics
from torchmetrics.functional import accuracy
import hydra
from omegaconf import DictConfig, OmegaConf
import logging
logger = logging.getLogger(__name__)

os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:2"
os.environ["HYDRA_FULL_ERROR"]="1"
@hydra.main(config_path="/content/ED-MTT/configs/",config_name="batchnorm_default")
def test(cfg: DictConfig):

  logger.info(OmegaConf.to_yaml(cfg))
  
  data_module = dataloader.create_data_module(cfg)

  best_model_path = "/content/drive/MyDrive/Multi-task_Engagement_Detection/tripletloss_checkpoints/best-checkpoint-v574.ckpt"
  trained_model = model.EngagementPredictor.load_from_checkpoint(
      best_model_path,
      n_features = data_module.train_dataset[0]["anchor_sequence"].shape[1],
      hparam = cfg.model)

  trained_model.freeze()
  predictions =[]
  labels = []

  for item in tqdm(data_module.val_dataloader()):
    sequence = item["anchor_sequence"]
    label = item["anchor_label"]
    #print(sequence,label)
    out_anchor_lstm, out_anchor_mlp = trained_model(sequence,label)
    predictions.append(out_anchor_mlp[0].numpy().mean())
    labels.append(label.item())
    print(predictions[-1])
    print(labels[-1])

  print("mse ",mean_squared_error(labels,predictions))
  np.save("/content/drive/MyDrive/Multi-task_Engagement_Detection/predictions.npy", predictions)
  np.save("/content/drive/MyDrive/Multi-task_Engagement_Detection/labels.npy", labels)
if __name__ == "__main__":

  test()
  #utils.plot_graph(predictions,labels)
  
