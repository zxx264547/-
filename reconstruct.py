import lightgbm as lgbm
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense, Dropout, Bidirectional
from keras import losses
from sklearn.preprocessing import MinMaxScaler
import scipy.io as sio
import matplotlib.pyplot as plt
import pandas as pd


def Reconstruct(data, label):
    """
    利用lightgbm回归模型对异常点的光伏功率值进行重构
    data： 0辐照度 1温度 2湿度 3功率（含异常值） 4真实标签 5功率（不含异常值）
    """
    # 根据异常标签生成训练集和测试集
    list_bad = np.where(label == 1)
    list_good = np.where(label == 0)
    data = np.delete(data, 4, axis=1)
    data_train = data[list_good, :]
    data_train = data_train.reshape(-1, 5)
    data_predict = data[list_bad, :]
    data_predict = data_predict.reshape(-1, 5)
    # 归一化处理
    scaler = MinMaxScaler()
    data_train = scaler.fit_transform(data_train)
    data_predict = scaler.transform(data_predict)
    # 输入输出特征
    input = [0, 1, 2]
    # 输入输出样本
    X_train = data_train[:, input]
    y_train = data_train[:, 3]
    X_predict = data_predict[:, input]
    y_true = data_predict[:, 4]

    # 搭建lightgbm模型
    model = lgbm.LGBMRegressor(
    objective='regression',
    max_depth=5,
    num_leaves=30,
    learning_rate=0.1,
    n_estimators=1000,
    min_child_samples=80,
    subsample=0.8,
    colsample_bytree=1,
    reg_alpha=0,
    reg_lambda=0,
    random_state=np.random.randint(10e6))
    # 训练模型
    model.fit(X_train, y_train)
    # 预测
    y_predict = model.predict(X_predict)

    # 反归一化
    data_max, data_min = scaler.data_max_, scaler.data_min_
    y_max, y_min = data_max[3], data_min[3]
    y_true = y_true * (y_max - y_min) + y_min
    y_predict = y_predict * (y_max - y_min) + y_min
    # 得到修复后的光伏功率值
    y_rec = data[:, 3]
    y_rec[list_bad] = y_predict
    # 计算误差
    mae = np.mean(np.abs(y_true - y_predict))
    rmse = np.sqrt(np.mean(np.square(y_true - y_predict)))
    mse = (np.mean(np.square(y_true - y_predict)))
    print('重构误差MAE为：', mae)
    print('重构误差RMSE为：', rmse)
    print('重构误差MSE为：', mse)

    return y_rec

# if __name__ =='__main__':
#     a = timeseries(start='06:15:00',end='20:00:00')
#     print(a)
# pass
