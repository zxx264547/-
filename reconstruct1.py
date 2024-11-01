import lightgbm as lgbm
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error
import scipy.io as sio
import matplotlib.pyplot as plt
import pandas as pd

"""
利用lightgbm回归模型对异常点的光伏功率值进行重构
"""

# 字体设置
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 读取数据
data = sio.loadmat('data_GSGF4_new2.mat')
data = data['data']  # 0辐照度 1温度 2湿度 3功率（含异常值） 4真实标签 5功率（不含异常值）
Label = np.load('Label.npy').item()
label = Label['LSCP']

# 根据异常标签生成训练集和测试集
list_bad = np.where(label == 1)
list_good = np.where(label == 0)
list_bad = list_bad[0]
list_good = list_good[0]
data = np.delete(data, 4, axis=1)
data_train = data[list_good, :]
data_train = data_train.reshape(-1, 5)
data_predict = data[list_bad, :]
data_predict = data_predict.reshape(-1, 5)
data_true = data[:, 4]

# 归一化处理
scaler = MinMaxScaler()
data_train = scaler.fit_transform(data_train)
data_predict = scaler.transform(data_predict)

# 输入输出特征
input = [0, 1, 2]

# 输入输出样本
X_train = data_train[:, input]
y_train = data_train[:, 4]
X_predict = data_predict[:, input]
y_true = data_predict[:, 4]

# 基于默认参数进行预测
# model = lgbm.LGBMRegressor(
#     objective='regression',
#     max_depth=5,
#     num_leaves=30,
#     learning_rate=0.1,
#     n_estimators=1000,
#     min_child_samples=80,
#     subsample=0.8,
#     colsample_bytree=1,
#     reg_alpha=0,
#     reg_lambda=0,
#     random_state=np.random.randint(10e6))
model = lgbm.LGBMRegressor(objective='regression')
model.fit(X_train, y_train)
y_predict1 = model.predict(X_predict)

# 采用优化后的参数进行预测
# gridParams = {
#     'max_depth': [5, 6, 8, 10],
#     'num_leaves': [25, 30, 40, 60, 100],
#     'learning_rate': [0.01, 0.02, 0.05, 0.1, 0.15],
#     'feature_fraction': [0.6, 0.7, 0.8, 0.9, 0.95],
#     'bagging_fraction': [0.6, 0.7, 0.8, 0.9, 0.95],
#     'bagging_freq': [2, 4, 5, 6, 8],
#     'lambda_l1': [0, 0.1, 0.4, 0.5, 0.6],
#     'lambda_l2': [0, 10, 15, 35, 40],
#     'cat_smooth': [1, 10, 15, 20, 35]
# }
# gridParams = {
#     'max_depth': [-1, 5, 6, 8, 10],
#     'num_leaves': [25, 30, 40, 60, 100],
#     'learning_rate': [0.01, 0.02, 0.05, 0.1, 0.15, 0.2],
#     'n_estimators': [50, 100, 200, 500],
#     'min_child_samples': [10, 20, 30, 50],
#     'reg_alpha': [0, 0.1, 0.2, 0.4, 1, 10],
#     'reg_lambda': [0, 0.1, 0.2, 0.4, 1, 10]
# }
gridParams = {
    'max_depth': [10, 20, 30, 40, 50, 100, 200],
    'n_estimators': [100, 200, 300, 400, 500],
    'num_leaves': [25, 31, 40, 60, 70, 100, 150],
    'learning_rate': [0.01, 0.02, 0.05, 0.1, 0.15, 0.2],
    'min_child_samples': [10, 20, 30],
    'min_child_weight': [1e-4, 1e-3, 1e-2],
    'colsample_bytree': [0.1, 0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 0.95, 1],
    'subsample': [0.1, 0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 0.95, 1],
    'subsample_freq': [0, 2, 4, 5, 6, 8],
    'reg_alpha': [0, 0.1, 1, 10],
    'reg_lambda': [0, 1, 10, 20, 30]
}
model = lgbm.LGBMRegressor(objective='regression')
random_search = RandomizedSearchCV(model, param_distributions=gridParams, n_iter=1, scoring='neg_mean_absolute_error', cv=5)
random_search.fit(X_train, y_train)
y_predict2 = random_search.predict(X_predict)
# model = lgbm.LGBMRegressor(
#     objective='regression',
#     max_depth=5,
#     num_leaves=30,
#     learning_rate=0.1)
# model.fit(X_train, y_train)
# y_predict2 = model.predict(X_predict)

# 预测误差
mae1 = mean_absolute_error(y_true, y_predict1)
mse1 = mean_squared_error(y_true, y_predict1)
mae2 = mean_absolute_error(y_true, y_predict2)
mse2 = mean_squared_error(y_true, y_predict2)
print('默认参数的mse误差为：', mse1)
print('优化参数的mse误差为：', mse2)
print('默认参数的mae误差为：', mae1)
print('优化参数的mae误差为：', mae2)

# 将预测输出值以及真实值反归一化
data_max, data_min = scaler.data_max_, scaler.data_min_
y_max, y_min = data_max[len(input)-1], data_min[len(input)-1]
y_predict1 = y_predict1 * (y_max - y_min) + y_min
y_predict2 = y_predict2 * (y_max - y_min) + y_min
y_true = y_true * (y_max - y_min) + y_min

# 得到修复后的光伏功率值
y_rec1 = np.copy(data[:, 3])
y_rec1[list_bad] = y_predict1
y_rec2 = np.copy(data[:, 3])
y_rec2[list_bad] = y_predict2
e = mean_absolute_error(y_rec1, y_rec2)
f = mean_absolute_error(y_predict1, y_predict2)

# 绘图
# fig3 = plt.figure()
# ax1 = fig3.add_subplot(211)
# ax1.plot(p_0, c='red', label='异常处理前')
# ax1.plot(p_1, c='black', label='准确值')
# ax2 = fig3.add_subplot(212)
# ax2.plot(p_rec, c='blue', label='异常处理后')
# ax2.plot(p_1, c='black', label='准确值')
# ax2 = fig3.add_subplot(212)
# ax2.plot(p_rec, c='blue', label='异常处理后')
# ax2.plot(p_1, c='black', label='准确值')
plt.plot(data[:, 3], c='green', label='处理前')
# plt.plot(y_rec2, c='black', label='优化参数')
plt.plot(y_rec1, c='red', label='处理后')
plt.plot(data_true, c='blue', label='真实值')
# y_before = data[list_bad, 3]
# y_before = y_before.flatten()
# plt.plot(y_before, c='green', label='处理前')
# plt.plot(y_predict1, c='red', label='处理后')

plt.xlabel('时间采样点/30min')
plt.ylabel('光伏功率/MW')
plt.legend()
plt.show()

pass
