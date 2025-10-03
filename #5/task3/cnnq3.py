#要求1
import torch
import torch.nn as nn
import cnnmodel2
import cnndataloader
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

dir_path="cnn.pth"
val_path='../../picture/custom_image_dataset/val'
core_num={"first":16,"end":80}

model=cnnmodel2.cnnmodel()
model.load_state_dict(torch.load(dir_path))
#为什么不用val_path,打的时候忘了
val_data=cnndataloader.get_loader('../../picture/custom_image_dataset/val',shuffle=False)
firstL2record=[]
endL2record=[]
labels_L2=[]
output_every={}
model.eval()
with torch.no_grad():
    for val_,labels in tqdm(val_data):
        middle_output={}
        x=model.CNN[0](val_)
        middle_output["first"]=x
        x=model.CNN[1](x)
        x=model.CNN[2](x)
        for i in range(3,len(model.CNN)):
            x=model.CNN[i](x)
            if i==len(model.CNN)-2:
                middle_output["end"]=x
        firstL2=torch.norm(middle_output["first"],p=2,dim=[2,3])    
        endL2=torch.norm(middle_output["end"],p=2,dim=[2,3])
        firstL2record.append(firstL2.numpy())
        endL2record.append(endL2.numpy())
        labels_L2.append(labels.numpy())
#计算平均激活强度
#合并数据
first_L2_all=np.concatenate(firstL2record,axis=0)
end_L2_all=np.concatenate(endL2record,axis=0)
all_labels = np.concatenate(labels_L2, axis=0)
class_num=10
#初始化存储数组
first_save=np.zeros((class_num,core_num["first"]))
end_save=np.zeros((class_num,core_num["end"]))
#
for cls in range(class_num):
    """"""
    clsidx=np.where(all_labels==cls)[0]
    if(len(clsidx)>0):
          first_save[cls] = np.mean(first_L2_all[clsidx], axis=0)
          end_save[cls] = np.mean(end_L2_all[clsidx], axis=0)

def plot_barmap(save):
    """"""
    plt.figure(figsize=(15,10))
    for i in range(5):
        plt.subplot(2,3,i+1)
        plt.bar(range(class_num),save[:,i])
        plt.xlabel="class"
        plt.ylabel="meanL2"
        plt.title=f"卷积核{i}"
    plt.show()
plot_barmap(first_save)
plot_barmap(end_save)   
    
