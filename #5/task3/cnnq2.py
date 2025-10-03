#要求1
import torch
import torch.nn as nn
import cnnmodel2
import cnndataloader
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
device=torch
dir_path="cnn.pth"

model=cnnmodel2.cnnmodel()
model.load_state_dict(torch.load(dir_path))
firstcnn=model.CNN[0].weight.data
plt.figure(figsize=(12, 12))

# 遍历16个卷积核并可视化
for i in range(16):   
    kernel =firstcnn[i]  
    
    # 归一化到0-1范围
    kernel = kernel - kernel.min()
    kernel = kernel / kernel.max()
    
    kernel = kernel.permute(1, 2, 0).numpy()
    
    plt.subplot(4, 4, i+1)
    plt.imshow(kernel)
    plt.title(f'Kernel {i+1}')

plt.show()


