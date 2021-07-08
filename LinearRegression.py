import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.svm import SVR
from sklearn.preprocessing import scale
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LinearRegression, Ridge, Lasso

# 设置输出结果不带省略号
pd.set_option('display.max_colwidth', 1000)
pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)
np.set_printoptions(threshold=10000)
np.set_printoptions(threshold=10000)

# 读入数据
csvdata = pd.read_csv('ford_second_hand_car.csv')
# print(csvdata)

# 可视化数据，观察数据总体情况
print(csvdata.head())  # 打印前几行
print('\n')
print(csvdata.shape)  # 打印数据行列总数
print('\n')
print(csvdata.isnull().sum())  # 打印数据每列缺失值的总数
print('\n')
print(csvdata.dtypes)  # 打印数据每列的数据类型
print('\n')
print(csvdata['transmission'].value_counts())
print('\n')
print(csvdata['fuelType'].value_counts())
print('\n')
print(csvdata['model'].value_counts())
print('\n')
'''plt.figure(figsize=(12, 10))
ax = sns.heatmap(csvdata.corr())
fig = plt.figure(figsize=(20, 15))
ax = fig.gca()
csvdata.hist(ax=ax)'''
# plt.show()

# 选取特征
feature_data = csvdata.loc[:, ['year', 'price', 'mileage', 'tax', 'mpg', 'engineSize']]
textdata = csvdata.loc[:, ['transmission', 'model', 'fuelType']]
# print(textdata)
# testdata = csvdata.iloc[:, csvdata.columns == 'price']

# 增加新特征:汽车已使用时间
feature_data["age_of_car"] = 2020 - feature_data["year"]
feature_data = feature_data.drop(columns=["year"])

# 将文本标签数据转换为数值的方法
pd.get_dummies(textdata, prefix=['transmission', 'model', 'fuelType'])
feature_data = feature_data.join(pd.get_dummies(textdata, prefix=['transmission', 'model', 'fuelType']))
# print(featuredata.head())

# 标准化数据
feature_data_scale = scale(X=feature_data, with_mean=True, with_std=True, copy=True)
feature_data_scale = pd.DataFrame(feature_data_scale, columns=feature_data.columns)
print(feature_data_scale.head())

# 标准化数据集另一种方式
'''std = StandardScaler()
data_vw_expanded_std = std.fit_transform(traindata)
data_vw_expanded_std = pd.DataFrame(data_vw_expanded_std, columns=traindata.columns)
print(data_vw_expanded_std.shape)
print(data_vw_expanded_std.head())'''

# 划分训练集和测试集
train_data = feature_data_scale.iloc[:, feature_data_scale.columns != 'price']
test_data = feature_data_scale.iloc[:, feature_data_scale.columns == 'price']
X_train, X_test, y_train, y_test = train_test_split(train_data, test_data, test_size=0.2)
X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)
'''print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)'''

# 选取拟合程度最佳的特征
no_of_features = []
r_squared_train = []
r_squared_test = []

for k in range(3, 36, 2):
    selector = SelectKBest(f_regression, k=k)  # f_regression是python提取特征的一种方法
    X_train_transformed = selector.fit_transform(X_train, y_train.ravel())  # fit_transform返回提取特征转换后的数组
    X_test_transformed = selector.transform(X_test)  # 提取特征
    regressor = LinearRegression()  # 线性模型
    regressor.fit(X_train_transformed, y_train.ravel())  # 传入数据
    no_of_features.append(k)  # 添加k值到no_of_features里面数据最后
    r_squared_train.append(regressor.score(X_train_transformed, y_train.ravel()))
    r_squared_test.append(regressor.score(X_test_transformed, y_test.ravel()))
print(sns.lineplot(x=no_of_features, y=r_squared_train, legend='full'))
print(sns.lineplot(x=no_of_features, y=r_squared_test, legend='full'))
plt.show()

# 建立模型
selector = SelectKBest(f_regression, k=23)
X_train_transformed = selector.fit_transform(X_train, y_train.ravel())
X_test_transformed = selector.transform(X_test)


def regression_model(model):
    regressor = model
    regressor.fit(X_train_transformed, y_train.ravel())
    score = regressor.score(X_test_transformed, y_test)
    return regressor, score


model_performance = pd.DataFrame(columns=["Features", "Model", "Score"])
models_to_evaluate = [LinearRegression(), Ridge(), Lasso(), SVR(), RandomForestRegressor(), MLPRegressor()]
# MLP是多层感知器,使用的是前馈神经网络,MLPRegressor主要用来做回归;LinearRegression()回归最简单的模型,Ridge()岭回归,
# Lasso(Least absolute shrinkage and selection operator)方法是以缩小变量集（降阶）为思想的压缩估计方,
for model in models_to_evaluate:
    regressor, score = regression_model(model)
    model_performance = model_performance.append({"Features": "Linear", "Model": model, "Score": score}, ignore_index=True)
print(model_performance)

