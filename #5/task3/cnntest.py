import torch
import torch.nn as nn
import cnnmodel2
import cnndataloader
from tqdm import tqdm
import pandas as pd
device=torch
dir_path="cnn.pth"
test_root_dir='../../picture/custom_image_dataset/test_unlabeled'

"""数据处理"""
test_loader=cnndataloader.get_loader(test_root_dir,shuffle=False,is_test=True)

model=cnnmodel2.cnnmodel()
model.load_state_dict(torch.load(dir_path))
model.eval()
"""正向计算"""
torch.no_grad()
for image,label in test_loader:
    """正向计算"""
    y=model(image)
    _,y=torch.max(y,1)
    for i in range(len(label)):
        record=f"{label[i]} : {y[i]}"
        record=pd.DataFrame([record])
        record.to_csv('result.csv',mode='a',header=False,index=False)
   
        
    

    
