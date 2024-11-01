# 功能：电流值预测
# 预测方法：lightgbm
# 预测类型：点预测
# 输入：预测点时刻预报辐照度、温度、湿度、预测点前一时刻功率
import lightgbm as lgbm
import numpy as np
from keras.layers import LSTM
from keras.layers import Dense, Dropout, Bidirectional
from keras import losses
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error
import scipy.io as sio
import xlsxwriter
import matplotlib.pyplot as plt
import pandas as pd

# 字体设置
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 载入数据
data = sio.loadmat('GSGF4-181-61-4.mat')
data = data['data']  # 三维数据：181*61*4
# 输入特征
input = [0, 1, 2, 3]  # 辐照度 温度 湿度 功率
# 测试数据
p_start = 0
p_end = 181
data_predict = data[p_start:p_end, :, input]
data_predict = data_predict.reshape((-1, len(input)))
# 训练数据
t_start = 0
t_end = 100
data_train = data[t_start:t_end, :, input]
data_train = data_train.reshape((-1, len(input)))


# 将数据拆分为样本形式
# 输入变量形式：n_steps_in*【I,T,H,P】
# 输出变量形式：n_steps_out*P
def split_sequence(seq, n_steps_in, n_steps_out):
    X, y = list(), list()
    for i in range(len(seq)):
        ii = i + 1
        end_ix = ii + n_steps_in
        out_end_ix = end_ix + n_steps_out
        if out_end_ix > len(seq)+1:
            break
        seq_x, seq_y = seq[ii:end_ix, 0:-1], seq[end_ix-1:out_end_ix-1, -1]
        seq_x = np.array(seq_x)
        seq_x = np.c_[seq_x, seq[ii-1:end_ix-1, -1]]
        seq_x = np.array(seq_x)
        seq_x = seq_x.flatten()
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.ravel(np.array(y))


# 对数据进行归一化处理
scaler = MinMaxScaler()
data_train = scaler.fit_transform(data_train)
data_predict = scaler.transform(data_predict)

# 形成样本
n_steps_in, n_steps_out = 5, 1
X_train, y_train = split_sequence(data_train, n_steps_in, n_steps_out)
X_predict, y_true = split_sequence(data_predict, n_steps_in, n_steps_out)

# 样本的特征个数
# n_features = X_train.shape[2]

# 搭建lightgbm模型
# model = lgbm.LGBMRegressor(
#     objective='regression',
#     max_depth=5,
#     num_leaves=25,
#     learning_rate=0.007,
#     n_estimators=1000,
#     min_child_samples=80,
#     subsample=0.8,
#     colsample_bytree=1,
#     reg_alpha=0,
#     reg_lambda=0,
#     random_state=np.random.randint(10e6))
model = lgbm.LGBMRegressor(objective='regression')

# 训练模型
model.fit(X_train, y_train)

# 预测
y_predict1 = model.predict(X_predict)


# 参数优化范围
# gridParams = {
#     'max_depth': [8, 9, 10, 15, 20],
#     'num_leaves': [25, 31, 40, 60, 70, 100],
#     'learning_rate': [0.01, 0.02, 0.05, 0.1, 0.15, 0.2],
#     'min_child_samples': [10, 20, 30],
#     'min_child_weight': [1e-4, 1e-3, 1e-2],
#     'colsample_bytree': [0.1, 0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 0.95, 1],
#     'subsample': [0.1, 0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 0.95, 1],
#     'subsample_freq': [0, 2, 4, 5, 6, 8],
#     'reg_alpha': [0, 0.1, 1, 10],
#     'reg_lambda': [0, 1, 10, 20, 30]
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

# 超参数优化
model = lgbm.LGBMRegressor(objective='regression')
random_search = RandomizedSearchCV(model, param_distributions=gridParams, n_iter=50, scoring='neg_mean_absolute_error', cv=5)
random_search.fit(X_train, y_train)
y_predict2 = random_search.predict(X_predict)
print(random_search.best_params_)

# 计算误差
# mae = np.mean(np.abs(y_true - y_predict))
# rmse = np.sqrt(np.mean(np.square(y_true - y_predict)))
# mse = (np.mean(np.square(y_true - y_predict)))
# print('预测误差MAE为：', mae)
# print('预测误差RMSE为：', rmse)
# print('预测误差MSE为：', mse)

# 将预测输出值以及真实值反归一化
data_max, data_min = scaler.data_max_, scaler.data_min_
y_max, y_min = data_max[len(input)-1], data_min[len(input)-1]
y_predict1 = y_predict1 * (y_max - y_min) + y_min
y_predict2 = y_predict2 * (y_max - y_min) + y_min
y_true = y_true * (y_max - y_min) + y_min

mae1 = mean_absolute_error(y_true, y_predict1)
rmse1 = np.sqrt(mean_squared_error(y_true, y_predict1))
mae2 = mean_absolute_error(y_true, y_predict2)
rmse2 = np.sqrt(mean_squared_error(y_true, y_predict2))
print('默认参数的rmse误差为：', rmse1)
print('优化参数的rmse误差为：', rmse2)
print('默认参数的mae误差为：', mae1)
print('优化参数的mae误差为：', mae2)

# 绘图
# fig3 = plt.figure()
# ax1 = fig3.add_subplot(211)
# ax1.plot(p_0, c='red', label='异常处理前')
# ax1.plot(p_1, c='black', label='准确值')
# ax2 = fig3.add_subplot(212)
# ax2.plot(p_rec, c='blue', label='异常处理后')
# ax2.plot(p_1, c='black', label='准确值')
plt.plot(y_predict1, c='red', label='默认参数')
plt.plot(y_predict2, c='black', label='优化参数')
plt.plot(y_true, c='blue', label='真实值')
plt.legend()
plt.show()

# 预测误差
# result = {'mae': mae, 'mse': mse, 'y_predict': y_predict, 'y_true': y_true}

# 保存结果：辐照度、温度、湿度、光伏功率、功率预测值
results = {'真实值': y_true, '默认参数': y_predict1, '优化参数': y_predict2}
df = pd.DataFrame(results)
df.to_excel('lightgbm调参对比.xlsx', engine='xlsxwriter')
# input = [0, 1, 2, 3]  # 辐照度 温度 湿度 功率
# data_save = data[p_start:p_end, :, input]
# data_save = data_save.reshape((-1, len(input)))
# data_save = data_save[n_steps_in:, :]
# data_save = np.c_[data_save, y_predict]
# data_save = pd.DataFrame(data_save)
# data_save.to_csv('Current_predict3_30min.csv')  # 电流，风速，温度，湿度，总辐照，散射，功率，电流预测值
# data_save.to_csv('Power_predict_30min.csv')
# data_save.to_csv('Data_train_GSGF4.csv')  # 电流，风速，温度，湿度，总辐照，散射，功率，功率预测值
pass

