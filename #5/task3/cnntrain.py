import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import cnndataloader
import cnnmodel2
import time
from torch.optim.lr_scheduler import StepLR
#参数设置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备：{device}")
epoches=32
lr=0.001
def train_model(model,train_loder,val_loder,epoches=epoches,lr=lr):
    """"""
    #实例化优化器与损失函数
    lossfunction=nn.CrossEntropyLoss()
    optimizer=optim.Adam(model.parameters(),lr=lr)
    schedule=StepLR(optimizer, step_size=3, gamma=0.5)
   
    history={
        'train_acc':[],
        'train_loss':[],
        'val_acc':[],
        'val_loss':[]
    }

    #训练
    time_start=time.time()
    for epoch in range(epoches):
         #记录
        trainacc=0
        traintotal=0
        trainloss=0
        valacc=0
        valtotal=0
        valloss=0
        """"""
        model.train()
        for image,label in tqdm(train_loder):
            image,label=image.to(device),label.to(device)
            """正向计算"""
            y=model(image)
            loss=lossfunction(y,label)
            _,y=torch.max(y,1)
            

            """反向传播"""
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            """记录train"""
            trainacc+=(y==label).sum().item()
            traintotal+=label.size(0)
            trainloss+=loss.item()

        #计算并记录
        history['train_acc'].append(trainacc/traintotal)
        history['train_loss'].append(trainloss/traintotal)

        #验证集
        model.eval()
        torch.no_grad()
        for image,label in tqdm(val_loder):
            image,label=image.to(device),label.to(device)
            """"""
            """正向计算"""
            y=model(image)
            valloss+=lossfunction(y,label).item()
            _,y=torch.max(y,1)
            valacc+=(y==label).sum().item()
            valtotal+=label.size(0)

        history['val_acc'].append(valacc/valtotal)
        history['val_loss'].append(valloss/valtotal)   
        schedule.step()
    print(f"训练时间:{time.time()-time_start}") 
    return history

def plotdata(train_acc,train_loss,val_acc,val_loss):
    plt.figure(figsize=(8,6))
    plt.subplot(221)
    plt.plot(train_acc)
    plt.title('train_acc')
    plt.xlabel('epoch')
    plt.ylabel('train_acc')
    plt.subplot(222)
    plt.plot(train_loss)
    plt.title('train_loss')
    plt.xlabel('epoch')
    plt.ylabel('train_loss')
    plt.subplot(223)
    plt.plot(val_acc)
    plt.title('val_acc')
    plt.xlabel('epoch')
    plt.ylabel('val_acc')
    plt.subplot(224)
    plt.plot(val_loss)
    plt.title('val_loss')
    plt.xlabel('epoch')
    plt.ylabel('val_loss')
    plt.show()

def main():
    train_loader=cnndataloader.get_loader(root_dir='../../picture/custom_image_dataset/train')
    val_loader=cnndataloader.get_loader(root_dir='../../picture/custom_image_dataset/val')
    model=cnnmodel2.cnnmodel().to(device)
    history=train_model(model,train_loader,val_loader)
    plotdata(history['train_acc'],history['train_loss'],history['val_acc'],history['val_loss'])
    torch.save(model.state_dict(), 'cnn.pth')

if __name__=='__main__':
    main()    

            




