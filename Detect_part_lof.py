from pyod.models.lof import LOF
import numpy as np
from pyod.utils.data import generate_data
from pyod.utils.data import evaluate_print
from pyod.utils.example import visualize
import scipy.io as sio
import matplotlib.pyplot as plt

if __name__ == "__main__":

    # 读取数据
    data = sio.loadmat('data_GSGF4_new.mat')
    data = data['data']  # 辐照度 温度 湿度 功率
    input = [0, 3]
    X_train = data[:, input]
    # 选取原散点图中右上角区域
    index = np.where(X_train[:, 0] > 800)
    X_train = X_train[index, :]
    X_train = X_train.reshape(-1, 2)
    index = np.where(X_train[:, 1] > 50)
    X_train = X_train[index, :]
    X_train = X_train.reshape(-1, 2)

    # train kNN detector
    clf_name = 'LOF'
    clf = LOF()
    clf.fit(X_train)

    # get the prediction labels and outlier scores of the training data
    y_train_pred = clf.labels_  # binary labels (0: inliers, 1: outliers)
    # y_train_scores = clf.decision_scores_  # raw outlier scores

    # 识别结果可视化
    plt.scatter(X_train[:, 0], X_train[:, 1], marker='o', c=y_train_pred, cmap='rainbow', s=10)
    plt.savefig('LOF-part')
    plt.show()
