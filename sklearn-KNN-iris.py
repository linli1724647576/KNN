#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# 导入算法包以及数据集
from sklearn import neighbors
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import random


# In[8]:


# 载入数据
iris = datasets.load_iris()
print(iris)


# In[29]:


# 打乱数据切分数据集
# x_train,x_test,y_train,y_test = train_test_split(iris.data, iris.target, test_size=0.2) #分割数据0.2为测试数据，0.8为训练数据

#打乱数据
data_size = iris.data.shape[0]
index = [i for i in range(data_size)] 
random.shuffle(index)  
iris.data = iris.data[index]
iris.target = iris.target[index]

#切分数据集
test_size = 40
x_train = iris.data[test_size:]
x_test =  iris.data[:test_size]
y_train = iris.target[test_size:]
y_test = iris.target[:test_size]

# 构建模型
model = neighbors.KNeighborsClassifier(n_neighbors=3)
model.fit(x_train, y_train)
prediction = model.predict(x_test)

print(classification_report(y_test, prediction))


# In[ ]:





# In[ ]:




