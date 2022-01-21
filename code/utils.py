import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
#from captum.attr import LayerConductance, LayerActivation, LayerIntegratedGradients
#from captum.attr import IntegratedGradients, DeepLift, GradientShap, NoiseTunnel, FeatureAblation

pd.set_option('display.max_columns', None)
def plot_graph(predictions,labels,names):
  zer0_pred = []
  zer0_label = []
  one_pred = []
  one_label = []
  two_pred = []
  two_label = []
  three_pred = []
  three_label = []
  for i in range(len(predictions)):
    if labels[i]==0:
      zer0_label.append(labels[i])
      zer0_pred.append(predictions[i])
    elif abs(labels[i]-0.33)<=0.01:
      one_label.append(labels[i])
      one_pred.append(predictions[i])

    elif abs(labels[i]-0.66)<=0.01:
      two_label.append(labels[i])
      two_pred.append(predictions[i])
    
    elif abs(labels[i]-1)<=0.01:
      three_label.append(labels[i])
      three_pred.append(predictions[i])

  preds = zer0_pred + one_pred + two_pred + three_pred
  labels = zer0_label + one_label + two_label + three_label


  plt.scatter(names,np.array(preds),color="red")
  plt.scatter(np.arange(len(preds)),labels,color = "green")
  plt.axvline(x=len(zer0_label))
  plt.axhline(y = sum(zer0_pred)/len(zer0_pred),xmin=0,xmax=(len(zer0_label)+2)/len(labels),color="yellow")
  plt.axhline(y = sum(one_pred)/len(one_pred),xmin=(len(zer0_label)+2)/len(labels),xmax=(len(one_label)+5)/len(labels),color="yellow")
  plt.axhline(y = sum(two_pred)/len(two_pred),xmin=(len(one_label)+5)/len(labels),xmax=(len(two_label)+14)/len(labels),color="yellow")
  plt.axhline(y = sum(three_pred)/len(three_pred),xmin=(len(two_label)+14)/len(labels),xmax=(len(three_label)+30)/len(labels),color="yellow")
  plt.axvline(x=len(zer0_label)+len(one_label))
  plt.axvline(x=len(zer0_label)+len(one_label)+len(two_label))
  plt.tick_params(axis='x', which='major',rotation=80)
  



def feature_importance(eng_level,data_module,trained_model,
column_names,X_train,y_test):
  STEP_AMOUNT = 100
  SAMPLE_DIM = len(column_names)
  ig = IntegratedGradients(trained_model.forward)
  alphas=0
  df=0
  count=0
  for i in range(len(y_test)):
    if abs(y_test[i]-eng_level)<0.01:
      count+=1
      ts = data_module.test_dataset
      val = ts.__getitem__(i)["anchor_sequence"]
      imp_m = ig.attribute(inputs=val.unsqueeze(0),
                          baselines=X_train.mean(dim=0)[0].unsqueeze(0),
                          #baselines=torch.zeros(val.unsqueeze(0).shape),
                          n_steps=STEP_AMOUNT)
          
      
      
      #print(y_test[i])
      #print(np.flip(column_names[np.argsort(abs(imp_m.squeeze(0).numpy()).sum(axis=0))[-20:]]))
      #print(np.flip(imp_m.squeeze(0).numpy().sum(axis=0)[np.argsort(abs(imp_m.squeeze(0).numpy()).sum(axis=0))[-20:]]))
      #print(np.argsort(abs(imp_m.squeeze(0).numpy()).sum(axis=0))[-9:])
      alphas += np.array(imp_m.squeeze(0)).T
      df+= pd.DataFrame(np.array(val),columns=column_names)
  alphas=alphas/count
  #df=df/count
  alpha_df = pd.DataFrame(alphas.T, columns=column_names)
  #print(alpa_df)
  return alpha_df #sns.heatmap(alpha_df, cmap ='RdYlGn', linewidths = 0.30, annot = True)

  #cor = df.corr()
  #sns.heatmap(alpa_df, annot=False, cmap=plt.cm.Reds)
  #plt.show()
  #fig, ax = plt.subplots(figsize=(80, 80))
  #im = ax.imshow(alphas[:,-10:])
  #tight_layout()
  #ax.set_xticks(np.arange(10))
  #ax.set_yticks(np.arange(len(column_names)))
  #ax.set_xticklabels(["t-"+str(i) for i in np.arange(X_train.shape[1], -1, -1)])
  #ax.set_yticklabels(column_names)

"""  length_flag=0
  for i in range(len(column_names)):
      if "length" in column_names[i]:
        length_flag = 1
      if length_flag==1 and "length" in column_names[i]:
        continue
      for j in range(10):
          text = ax.text(j, i, round(alphas[i, j], 7),fontsize=14,
                        ha="center", va="center", color="w")
  ax.set_title("Importance of features and timesteps")
  #fig.tight_layout()
  plt.show()
"""

