import lightgbm as lgbm
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error
import scipy.io as sio
import matplotlib.pyplot as plt
import pandas as pd
import xlsxwriter

"""
利用lightgbm回归模型对异常点的光伏功率值进行重构
"""


# 生成输入输出样本
def split_sequence_test(seq, label):
    X, y = list(), list()
    loc = list()
    for i in range(len(seq)):
        ii = i+1
        if ii+1 > len(seq)-1:
            break
        seq_x, seq_y = seq[[ii-1, ii+1], :], seq[ii, -1]
        seq_x = seq_x.flatten()
        xx = seq[ii, 0:-1]
        xx = xx.flatten()
        seq_x = np.hstack((seq_x, xx))
        if (label[ii-1] == 0) & (label[ii] == 1) & (label[ii+1] == 0):
            X.append(seq_x)
            y.append(seq_y)
            loc.append(ii)
    return np.array(X), np.ravel(np.array(y)), np.array(loc)


# 生成输入输出样本
def split_sequence_train(seq, label):
    X, y = list(), list()
    loc = list()
    for i in range(len(seq)):
        ii = i+1
        if ii+1 > len(seq)-1:
            break
        seq_x, seq_y = seq[[ii-1, ii+1], :], seq[ii, -1]
        seq_x = seq_x.flatten()
        xx = seq[ii, 0:-1]
        xx = xx.flatten()
        seq_x = np.hstack((seq_x, xx))
        if (label[ii-1] == 0) & (label[ii] == 0) & (label[ii+1] == 0):
            X.append(seq_x)
            y.append(seq_y)
            loc.append(ii)
    return np.array(X), np.ravel(np.array(y)), np.array(loc)


# 字体设置
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 读取数据
data = sio.loadmat('data_GSGF4_new4.mat')
data = data['data']  # 0辐照度 1温度 2湿度 3功率（含异常值） 4真实标签 5功率（不含异常值）
# Label = np.load('Label.npy').item()
# label = Label['LSCP']
label = data[:, 4]
data0 = np.copy(data)

# 根据异常标签生成训练集和测试集
list_bad = np.where(label == 1)
list_good = np.where(label == 0)
list_bad = list_bad[0]
list_good = list_good[0]

# 模型1——不考虑前一时刻
# 生成训练和测试集
input = [0, 1, 2]
X_train = data[list_good, :]
X_train = X_train[:, input]
y_train = data[list_good, 3]
X_test = data[list_bad, :]
X_test = X_test[:, input]
y_test = data[list_bad, 5]
# 归一化处理
scaler_Xtrain = MinMaxScaler()
X_train = scaler_Xtrain.fit_transform(X_train)
X_test = scaler_Xtrain.transform(X_test)
scaler_ytrain = MinMaxScaler()
y_train = scaler_ytrain.fit_transform(y_train.reshape(-1, 1))
# 建立模型（默认参数）
model = lgbm.LGBMRegressor(objective='regression')
# 训练
model.fit(X_train, y_train.ravel())
# 预测
y_predict = model.predict(X_test)
# 反归一化
y_predict = scaler_ytrain.inverse_transform(y_predict.reshape(-1, 1))
# 修复
y_fixed1 = np.copy(data[:, 3])
y_fixed1[list_bad] = y_predict.ravel()
# 误差
bad_fixed1 = y_fixed1[list_bad]
mae1 = mean_absolute_error(y_test, bad_fixed1)
rmse1 = np.sqrt(mean_squared_error(y_test, bad_fixed1))
print('模型1的rmse误差为：', rmse1)
print('模型1的mae误差为：', mae1)

# 模型2——考虑前一时刻
# 生成预测模型2的测试集与训练集（输入特征涉及前一时刻）
X_test2, y_test2, loc_test = split_sequence_test(data[:, [0, 1, 2, 3]], label)
X_train2, y_train2, loc_train = split_sequence_train(data[:, [0, 1, 2, 3]], label)
# 归一化处理
scaler_Xtrain2 = MinMaxScaler()
X_train2 = scaler_Xtrain2.fit_transform(X_train2)
X_test2 = scaler_Xtrain2.transform(X_test2)
scaler_ytrain2 = MinMaxScaler()
y_train2 = scaler_ytrain2.fit_transform(y_train2.reshape(-1, 1))
# 建立模型（默认参数）
model2 = lgbm.LGBMRegressor(objective='regression')
# 训练
model.fit(X_train2, y_train2.ravel())
# 预测
y_predict2 = model.predict(X_test2)
# 反归一化
y_predict2 = scaler_ytrain2.inverse_transform(y_predict2.reshape(-1, 1))
# 修复
y_fixed2 = np.copy(y_fixed1)
y_fixed2[loc_test] = y_predict2.ravel()
# 误差
bad_fixed2 = y_fixed2[list_bad]
mae2 = mean_absolute_error(y_test, bad_fixed2)
rmse2 = np.sqrt(mean_squared_error(y_test, bad_fixed2))
print('模型2的rmse误差为：', rmse2)
print('模型2的mae误差为：', mae2)

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
plt.plot(data[:, 3], c='red', label='修复前')
plt.plot(y_fixed1, c='green', label='模型1')
# plt.plot(y_rec2, c='black', label='优化参数')
plt.plot(y_fixed2, c='orange', label='模型2')
plt.plot(data[:, 5], c='blue', label='真实值')
# y_before = data[list_bad, 3]
# y_before = y_before.flatten()
# plt.plot(y_before, c='green', label='处理前')
# plt.plot(y_predict1, c='red', label='处理后')

plt.xlabel('时间采样点/30min')
plt.ylabel('光伏功率/MW')
plt.legend()
plt.show()

Result = pd.DataFrame([data[:, 0], data[:, 1], data[:, 2], data[:, 3], data[:, 5], y_fixed1, y_fixed2, label],
                      index=['辐照度', '温度', '湿度', '修复前', '真值', '修复1', '修复2', '标签'])
df = pd.DataFrame(Result).T
df.to_excel('Result_fix.xlsx', engine='xlsxwriter')

pass
