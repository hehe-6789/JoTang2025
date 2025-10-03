#处理csv数据
import pandas as pd
#
import numpy as np
#可视化
import matplotlib.pyplot as plt
#混淆矩阵
from sklearn.metrics import confusion_matrix
#数据预处理
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
#pytorch框架
import torch
import torch.nn as nn
#优化器
import torch.optim as optim
#处理时间
import time

#生成种子，确保结果可复现
torch.manual_seed(42)
np.random.seed(42)
#model设置
#Titanic数据
data_name='../datasets/train_and_test2.csv'
label='2urvived'
#隐藏层数及隐藏层神经元数
hidenum=1
hnunum=26
#学习率
lr=0.01
#轮数
epoches=500

def create_data(data_name,label):
    """利用pandas获取Titanic数据"""
    data=pd.read_csv(data_name)
    #处理缺失值
    if data.isnull().any().any():
        print("检测到缺失值，正在处理...")
        # 对数值型列用中位数填充，类别型列用众数填充
        for col in data.columns:
            if data[col].dtype == 'float64' or data[col].dtype == 'int64':
                data[col].fillna(data[col].median(), inplace=True)
            else:
                data[col].fillna(data[col].mode()[0], inplace=True)
    x=data.drop(['2urvived','Passengerid'],axis=1)
    y=data.loc[:,label]
    """转化为np数组，避免pytorch无法识别"""
    x=np.array(x)
    y=np.array(y)

    """预处理数据（归一化与划分数据集）"""
    standardscaler=StandardScaler()
    x_scale=standardscaler.fit_transform(x) 
    x_train,x_test,y_train,y_test=train_test_split(x_scale,y,test_size=0.2,random_state=42)

    """转化为张量便于使用"""
    x_train=torch.FloatTensor(x_train)
    x_test=torch.FloatTensor(x_test)
    y_train=torch.FloatTensor(y_train)
    y_test=torch.FloatTensor(y_test)

    """返回数据集"""
    return x_train,x_test,y_train,y_test

class Titanicmodel(nn.Module):
    """初始化"""
    def __init__(self,hnunum=hnunum):
        super().__init__()
        #构建模型
        self.linear1=nn.Linear(26,hnunum)
        self.relu=nn.ReLU()
        self.linear2=nn.Linear(hnunum,1)
        self.sigmoid=nn.Sigmoid()
        self.init_weight()


    def init_weight(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m,nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self,x):
        """前向计算"""
        x=self.relu(self.linear1(x))
        x=self.linear2(x)
        return x

    def predict(self,x):
        """计算预测值"""
        return self.sigmoid(self.forward(x))

def train_model(model,x_train,x_test,y_train,y_test):
    """定义损失函数和优化器"""
    lossfunction=nn.BCEWithLogitsLoss()
    optimer=optim.Adam(model.parameters(),lr=lr)
    """记录"""
    train_acc=[]
    test_acc=[]
    """时间开始"""
    time_start=time.time()
    """训练"""
    model.train()
    for epoch in range(epoches):
        """正向计算"""
        y_pred=model(x_train).squeeze()
        loss=lossfunction(y_pred,y_train)
        """反向传播"""
        optimer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimer.step()
        """计算准确率(训练集)"""
        y_pred=model.predict(x_train).squeeze()
        y_train_pred=(y_pred>=0.5).float()
        y_train_acc=(y_train_pred==y_train).sum().item()/y_train.size(0)
        """测试集"""
        model.eval()
        torch.no_grad()
        y_pred=model.predict(x_test).squeeze()
        y_test_pred=(y_pred>=0.5).float()
        y_test_acc=(y_test_pred==y_test).sum().item()/y_test.size(0)
        """记录"""
        train_acc.append(y_train_acc)
        test_acc.append(y_test_acc)
        """打印"""
        if((epoch+1)%100==0):
            print(loss.item())
    """计算花费时间并打印"""
    print(f"本次训练共耗时：{time.time()-time_start}")
    """返回画图所需值"""        
    return train_acc,test_acc,y_train_pred,y_test_pred

def plot_data(acc_train,acc_test):
    plt.figure()
    plt.subplot(121)
    plt.title('train')
    plt.xlabel('epoch')
    plt.ylabel('acc_rate')
    plt.plot(acc_train)
    plt.subplot(122)
    plt.title('test')
    plt.xlabel('epoch')
    plt.ylabel('acc_test')
    plt.plot(acc_test)
    plt.show()

def plot_cpr(y_train,y_train_pred,y_test,y_test_pred):
    print(confusion_matrix(y_train,y_train_pred))
    print(confusion_matrix(y_test,y_test_pred))

def printacc(acc_train,acc_test):
    print(f"acc:{acc_train[-1]}")
    print(f"acc:{acc_test[-1]}")

def main():   
    x_train,x_test,y_train,y_test=create_data(data_name,label)
    model=Titanicmodel()
    acc_train,acc_test,y_train_pred,y_test_pred=train_model(model, x_train,x_test,y_train,y_test)
    plot_data(acc_train,acc_test)
    plot_cpr(y_train,y_train_pred,y_test,y_test_pred)
    printacc(acc_train,acc_test)
    #记录实验结果
    data_record=[hidenum,hnunum,lr,epoches,acc_train[-1],acc_test[-1]]
    date_new=pd.DataFrame([data_record])
    date_new.to_csv('../datasets/titanicdata.csv',mode='a',header=False,index=False)
if __name__=='__main__':
    main()
