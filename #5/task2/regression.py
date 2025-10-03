#加载模块
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.model_selection import train_test_split
#加州数据集
from sklearn.datasets import fetch_california_housing

califoria=fetch_california_housing()
x=califoria.data
y=califoria.target
print(x.shape)
print(y.shape)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
#sklearn实现
model=LinearRegression()
model.fit(x_train,y_train)
print(f"参数：{model.coef_}")
print(f"截距（偏置项）：{model.intercept_}")
#train
MSE=mean_squared_error(y_train,model.predict(x_train))
R2=r2_score(y_train,model.predict(x_train))
print(MSE,R2)
#test
MSE=mean_squared_error(y_test,model.predict(x_test))
R2=r2_score(y_test,model.predict(x_test))
print(MSE,R2)

#绘图
plt.figure()
#train
plt.subplot(1,2,1)
plt.scatter(y_train,model.predict(x_train))
plt.xlabel('real_price')
plt.ylabel('pred_price')
#test
plt.subplot(1,2,2)
plt.scatter(y_test,model.predict(x_test))
plt.xlabel('real_price')
plt.ylabel('pred_price')
plt.show()
