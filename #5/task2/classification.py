#加载模块
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
#加载莺尾花数据集
from sklearn.datasets import load_iris

iris=load_iris()
x=iris.data
y=iris.target
print(x.shape,y.shape)
print(len(iris.target_names))

#划分数据集
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

#sklearn实现
model=LogisticRegression()
model.fit(x_train,y_train)

#预测
y_train_pred=model.predict(x_train)
y_test_pred=model.predict(x_test)
print(accuracy_score(y_train,y_train_pred))
print(accuracy_score(y_test,y_test_pred))
print(confusion_matrix(y_train,y_train_pred))
print(confusion_matrix(y_test,y_test_pred))
