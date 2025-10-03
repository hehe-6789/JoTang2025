import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import time

#随机种子
torch.manual_seed(42)
np.random.seed(42)

#设置
noise=0.1
n_samples=500
lr=0.001

#数据集
def create_database(n_samples=n_samples,noise=noise):
    """创建数据集"""
    X,Y=make_moons(n_samples=n_samples,noise=noise,random_state=42)
    """标准化数据"""
    standardscaler=StandardScaler()
    x_scale=standardscaler.fit_transform(X)
    """划分数据集"""
    x_train,x_test,y_train,y_test=train_test_split(x_scale,Y,test_size=0.1,random_state=42)
    """转化为张量"""
    x_train=torch.FloatTensor(x_train)
    x_test=torch.FloatTensor(x_test)
    y_train=torch.FloatTensor(y_train)
    y_test=torch.FloatTensor(y_test)
    """返回数据集"""
    return x_train,x_test,y_train,y_test


#定义神经网络model
class NeuralNetwork(nn.Module):
    """继承nn.Module"""
    
    def __init__(self):
        super().__init__()
        self.hNeur=8
        """构建所需层"""
        self.linear1=nn.Linear(2,self.hNeur)
        self.linear2=nn.Linear(self.hNeur,1)
        self.relu=nn.ReLU()
        self.sigmoid=nn.Sigmoid()
        """初始化参数"""
        self.initialize_weights()

    def initialize_weights(self):
        """初始化"""
        for m in self.modules():
            if isinstance(m,nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self,x):
        x=self.relu(self.linear1(x))
        x=self.linear2(x)
        return x
    
    def predict_proba(self,x):
        """"""
        return self.sigmoid(self.forward(x))

#训练
def train_model(model,x_train,x_test,y_train,y_test,lr=0.01,epochs=500):
    """定义损失和优化器"""
    losses=nn.BCEWithLogitsLoss()
    optimer=optim.Adam(model.parameters(),lr=lr)
    """记录数组"""
    xtrain_loss=[]
    xtest_loss=[]
    xtrain_acc=[]
    xtest_acc=[]
    """计时"""
    timestart=time.time()
    """训练部分"""
    for epoch in range(epochs):
        model.train()
        """前向传播"""
        y_p=model(x_train).squeeze()
        loss=losses(y_p,y_train)
        """反向传播"""
        optimer.zero_grad()
        loss.backward()
        optimer.step()
        """计算准确率"""
        y_pr=model.predict_proba(x_train).squeeze()
        y_pred=(y_pr>=0.5).float()
        trainacc=(y_pred==y_train).sum().item()/y_train.size(0)

    
        """测试部分"""
        model.eval()
        torch.no_grad()
        y_p2=model(x_test).squeeze()
        loss2=losses(y_p2,y_test)
        y_pr2=model.predict_proba(x_test).squeeze()
        y_pred2=(y_pr2>=0.5).float()
        testacc=(y_pred2==y_test).sum().item()/y_test.size(0)
        """记录数据"""
        xtrain_loss.append(loss.item())
        xtest_loss.append(loss2.item())
        xtrain_acc.append(trainacc)
        xtest_acc.append(testacc)
    
    "计算训练时间"
    times=time.time()-timestart
    print(f"训练完成，耗时：{times}")
    return y_pred,y_pred2,xtrain_loss,xtest_loss,xtrain_acc,xtest_acc    


"""可视化"""
def plot_data(x,y,title):
    """画出点图"""
    plt.figure()
    plt.scatter(x[y==0,0],x[y==0,1],c='red')
    plt.scatter(x[y==1,0],x[y==1,1],c='blue')
    plt.title(title)
    plt.show()

def plot_train_test(xtrain_loss,xtest_loss,xtrain_acc,xtest_acc  ):  
    """"""
    plt.figure()
    plt.subplot(221)
    plt.plot(xtrain_loss,label='train_loss')
    plt.title("xtrain_loss")
    plt.subplot(222)
    plt.plot(xtest_loss,label='test_loss') 
    plt.title("xtest_loss")
    plt.subplot(223)
    plt.plot(xtrain_acc,label='train_acc')
    plt.title("xtrain_acc")
    plt.subplot(224)
    plt.plot(xtest_acc,label='test_acc')
    plt.title("xtest_acc")
    plt.show()


def main():
    x_train,x_test,y_train,y_test=create_database()
    model=NeuralNetwork()
    y_pred,y_pred2,xtrain_loss,xtest_loss,xtrain_acc,xtest_acc=train_model(model,x_train,x_test,y_train,y_test,lr=0.01,epochs=500)
    plot_data(x=x_train,y=y_pred,title='train')
    plot_data(x=x_test,y=y_pred2,title='test')
    plot_train_test(xtrain_loss,xtest_loss,xtrain_acc,xtest_acc)

if __name__=='__main__':
    main()

