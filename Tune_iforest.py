from pyod.models.lof import LOF
from pyod.models.knn import KNN
from KNN_new import KnnNew
from pyod.models.iforest import IForest
from pyod.models.lscp import LSCP
from sklearn.metrics import roc_curve, auc, f1_score
import numpy as np
import scipy.io as sio
import xlsxwriter
import collections
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score, make_scorer
from reconstruct import Reconstruct as Rec
from pyod.utils.utility import score_to_label, precision_n_scores

if __name__ == "__main__":

    # 字体设置
    # plt.rcParams['font.sans-serif'] = ['SimHei']
    # plt.rcParams['axes.unicode_minus'] = False

    # 读取数据
    data = sio.loadmat('data_GSGF4_new2.mat')
    data = data['data']  # 0辐照度 1温度 2湿度 3功率（含异常值） 4真实标签 5功率（不含异常值）
    input = [0, 3]
    X_train = data[:, input]
    y_train = data[:, 4]

    # making the instance
    # clf = KnnNew()
    clf = IForest()
    AUC = make_scorer(roc_auc_score)
    # Hyper Parameters Set
    # leaf_size不影响精度
    n_estimators = list(range(10, 210, 10))
    # max_samples = list([5, 50, 100, 150, 200, 256, 300, 350, 400])
    # max_samples = list([600, 650, 700, 750, 800])
    max_samples = list([450, 500, 550])
    # n_estimators = list(range(10, 20, 10))
    # max_samples = list([5, 50])
    params = {'n_estimators': n_estimators,
              'max_samples': max_samples}
    # Making models with hyper parameters sets
    model = GridSearchCV(clf, param_grid=params, scoring=AUC, n_jobs=1)
    # Learning
    model.fit(X_train, y_train)

    # 输出结果并展示
    print("Best Hyper Parameters:\n", model.best_params_)
    # results = model.cv_results_
    # score = model.cv_results_['mean_test_score']
    # score = score.reshape((3, -1))
    # fig, ax = plt.subplots()
    # ax.plot(k_list, score[0, :], c='red', label='largest')
    # ax.plot(k_list, score[1, :], c='blue', label='mean')
    # ax.plot(k_list, score[2, :], c='green', label='median')
    # ax.plot(k_list[3], score[2, 3], marker='o', color='r')
    # ax.text(k_list[3], score[2, 3], 'Best k: %d Best method: median' % 20)
    # ax.text(k_list[3], score[0, 3], 'Default k: %d Default method: largest' % 20)
    # ax.set_ylabel('AUC')
    # ax.set_xlabel('k')
    # plt.legend()
    # plt.show()
    # fig.savefig('KNN调参过程')

    # 保存结果
    score = model.cv_results_['mean_test_score']
    score = score.reshape((-1, len(max_samples)), order='F')
    df = pd.DataFrame(score, index=n_estimators, columns=max_samples)
    df.to_excel('tune_iforest3.xlsx', engine='xlsxwriter')

    pass
    # Best Hyper Parameters: n_neighbors=130, leaf_size=5
    # fig, ax = plt.subplots()
    #     # ax.plot(p_0, c='red', label='异常处理前')
    #     # ax.plot(p_rec, c='blue', label='异常处理后')
    #     # plt.show()
