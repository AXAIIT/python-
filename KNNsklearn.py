from sklearn.model_selection import train_test_split
from sklearn import neighbors
from sklearn.preprocessing import StandardScaler, scale
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

# 表示小数不需要以科学计数法的形式输出
np.set_printoptions(suppress=True)

# 设置显示汉字
plt.rcParams['font.sans-serif'] = ['SimHei']

# 读入文件
csvdata = pd.read_csv('glass.csv')

# 可视化数据，看数据内部情况
print(csvdata.shape)
print(csvdata.describe())
print(csvdata.isnull().sum())
count_class = pd.value_counts(csvdata['Type'],sort=True).sort_index()
count_class.plot(kind = 'bar')
plt.title("各类数据含量")
plt.xlabel("种类")
plt.ylabel("含量")
# plt.show()

# 选取除Type列以外的作为一组数据
traindata = csvdata.iloc[:, csvdata.columns != 'Type']
# traindata = np.array(traindata)

# 选取Type列作为标签
targetdata = csvdata.iloc[:, csvdata.columns == 'Type']
# targetdata = np.array(targetdata)
# print(traindata)

# sklearn划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(traindata, targetdata, test_size=0.3)
# print(y_train)
X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)
# print(y_train)

# print(csvdata)

# 利用sklearn标准化数据
x_scale = scale(X=X_train, with_mean=True, with_std=True, copy=True)
# print(x_scale)
y_scale = scale(X=X_test, with_mean=True, with_std=True, copy=True)
# print(y_train)

# 利用sklearn调用knn分类器
knn = KNeighborsClassifier(n_neighbors=8)
knn.fit(X_train, y_train.ravel())
'''print(knn.predict(X_test))#预测测试集特征
print(y_test)#真实特征值'''
y_predict = knn.score(X_test, y_test)
print(y_predict)
