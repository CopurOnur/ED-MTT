import os
import math
import random
import pandas as pd
import numpy as np
from tqdm.auto import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

from sklearn.preprocessing import StandardScaler
from tsfresh.feature_extraction import extract_features, MinimalFCParameters,feature_calculators
from torchsampler import ImbalancedDatasetSampler
import hydra
from omegaconf import DictConfig

os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:2"

class OpenFaceDataset(Dataset):
    def __init__(self, cfg:DictConfig,case):
      pl.seed_everything(cfg.model.seed,workers=True)
      root = cfg.data.root
      l_dir = cfg.data.l_dir

      self.case=case
      self.file_list = []
      self.label_list = []
      self.columns=[]
      self.level = cfg.data.level
      labels = pd.read_excel(l_dir,header=None)
      self.frame_size = cfg.data.frame_size
      self.step_size = cfg.data.step_size
      self.gaze_range= cfg.data.gaze_range
      self.head_range = cfg.data.head_range
      self.rot_range = cfg.data.rot_range
      self.aus_range = cfg.data.aus_range
      self.attributes = cfg.data.attributes
      self.functions = cfg.data.functions
      self.cols = cfg.data.cols
      


      for face_feature in os.listdir(root + self.case):
        if "txt" not in face_feature:
          face_feature = face_feature + ".txt"

        label = labels[labels[0]==face_feature.split(".")[0]]
        if len(label)==0:
          continue
        self.label_list.append(label[1].values[0]) 
        self.file_list.append(os.path.join(root + self.case,face_feature))

      self.col_names = pd.read_csv(self.file_list[0], delimiter=',').columns
      gaze_cols = self.col_names[self.gaze_range[0]:self.gaze_range[1]]
      head_cols = self.col_names[self.head_range[0]:self.head_range[1]]
      rot_cols = self.col_names[self.rot_range[0]:self.rot_range[1]]
      aus_cols = self.col_names[self.aus_range[0]:self.aus_range[1]]

      for i in self.cols:
        for f in self.functions:
          for j in locals()[i]:
            self.columns.append(j + "_" + f)

      self.all_features = self.get_feature()
      self.label_dict = self.group_labels()

    def group_labels(self):

      sequences_dict ={}
      zero=[]
      one = []
      two = []
      three= []

      for i in range(len(self.label_list)):
        if abs(self.label_list[i]-0)<0.01:
          zero.append(i)
        elif abs(self.label_list[i]-0.33)<0.01:
          one.append(i)

        elif abs(self.label_list[i]-0.66)<0.01:
          two.append(i)

        elif abs(self.label_list[i]-1)<0.01:
          three.append(i)

      sequences_dict["zero_one"] = zero + one
      sequences_dict["two_three"] = two + three 

      return sequences_dict

    def get_feature(self):
        features = []
        def norm_rad(x):
          return math.sin(math.radians(x))
        for idx in range(len(self.label_list)):
            # segment video to 10 segments, return features
            file_dir, label = self.file_list[idx], self.label_list[idx]
            v_data = pd.read_csv(file_dir, delimiter=',')
            v_data = np.array(v_data.iloc[90:-90,:])
            v_data = np.delete(v_data, 0, 0)    # delete table caption
            v_data = v_data.astype(np.float)   # gaze / pose
            

            # remove nan
            v_data = v_data[~np.isnan(v_data).any(axis=1)]

            #scaler = RobustScaler()
            #scaler = StandardScaler()
            #v_data = scaler.fit_transform(v_data)
       
            interval = int(v_data.shape[0]/self.frame_size)
            feature = []
            for i in range(self.frame_size):
                seg = v_data[i*interval:int((i+self.step_size)*interval),:]
                gaze_seg = seg[:,self.gaze_range[0]:self.gaze_range[1]]
                head_seg = seg[:,self.head_range[0]:self.head_range[1]]
                rot_seg = seg[:,self.rot_range[0]:self.rot_range[1]]
                aus_seg = seg[:,self.aus_range[0]:self.aus_range[1]]

                selected_feature=[]
                for att in self.attributes:
                  for func in self.functions:
                    method_to_call = getattr(feature_calculators, func)
                    selected_feature.append(np.apply_along_axis(method_to_call,0,locals()[att]))

                feature.append(torch.FloatTensor(np.concatenate(selected_feature)))
            features.append(feature)

        return features


    def __getitem__(self, idx):
        anchor_x, anchor_y = self.all_features[idx], torch.tensor(self.label_list[idx]).float()

        if idx in self.label_dict["zero_one"]:
          positive_idx = random.choice(self.label_dict["zero_one"])
          negative_idx = random.choice(self.label_dict["two_three"])
        else:
          positive_idx = random.choice(self.label_dict["two_three"])
          negative_idx = random.choice(self.label_dict["zero_one"])

        positive_x, positive_y = self.all_features[positive_idx], torch.tensor(self.label_list[positive_idx]).float()
        negative_x, negative_y = self.all_features[negative_idx], torch.tensor(self.label_list[negative_idx]).float()

        
        def shape_data(x):
          data = torch.zeros((self.frame_size,len(x[0])))
          for i in range(self.frame_size):
              data[i,:] = x[i]
          return data
          
        
        return dict(
          anchor_sequence = shape_data(anchor_x), 
          anchor_label = anchor_y,
          positive_sequence = shape_data(positive_x),
          positive_label = positive_y,
          negative_sequence = shape_data(negative_x),
          negative_label = negative_y
          )


    def __len__(self):
        return len(self.label_list)

    def get_labels(self):
  
      return self.label_list




class OpenFaceDataModule(pl.LightningDataModule):
  def __init__(self, batch_size,cfg:DictConfig):
    super().__init__()
    self.batch_size = batch_size
    self.cfg=cfg
    self.setup()
  def setup(self):
    self.train_dataset = OpenFaceDataset(self.cfg,case= "Train")
    self.test_dataset = OpenFaceDataset(self.cfg, case = "validation")

  def train_dataloader(self):
    return DataLoader(
        self.train_dataset,
        batch_size=self.batch_size,
        sampler=ImbalancedDatasetSampler(self.train_dataset),
        shuffle=False,
        num_workers=2 #cpu_count(),
        #worker_init_fn=seed_worker,
        #generator=g

    )
  
  def val_dataloader(self):
    return DataLoader(
        self.test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2
    )
  
  def test_dataloader(self):
    return DataLoader(
        self.test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2
    )
  
  

#@hydra.main(config_path="/content/drive/MyDrive/Multi-task_Engagement_Detection/configs/",config_name="default")
def create_data_module(cfg:DictConfig):
  BATCH_SIZE = cfg.data.batch_size
  #print(type(cfg.data))
  #print(cfg.data)
  data_module = OpenFaceDataModule(BATCH_SIZE,cfg)
  #data_module.setup(cfg)
  return data_module
"""if __name__ == "__main__":
  create_data_module()"""
