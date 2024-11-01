from pyod.models.lof import LOF
from pyod.models.knn import KNN
from KNN_new import KnnNew
from pyod.models.iforest import IForest
from pyod.models.lscp import LSCP
from sklearn.metrics import roc_curve, auc, f1_score
import numpy as np
import scipy.io as sio
import collections
import matplotlib.pyplot as plt
import xlsxwriter
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
    clf = LOF()
    AUC = make_scorer(roc_auc_score)
    # Hyper Parameters Set
    # leaf_size不影响精度
    k_list = list(range(10, 200, 10))
    params = {'n_neighbors': k_list}
    # Making models with hyper parameters sets
    model = GridSearchCV(clf, param_grid=params, scoring=AUC, n_jobs=1)
    # Learning
    model.fit(X_train, y_train)
    # The best hyper parameters set
    print("Best Hyper Parameters:\n", model.best_params_)

    # Best Hyper Parameters: n_neighbors=20, method='median', leaf_size=5
    results = {'n_neighbors': list(range(10, 200, 10)), 'score': model.cv_results_['mean_test_score']}
    df = pd.DataFrame(results)
    df.to_excel('tune_lof.xlsx', engine='xlsxwriter')

    fig, ax = plt.subplots()
    score = model.cv_results_['mean_test_score']
    ax.plot(k_list, score, c='blue')
    ax.set_ylabel('AUC')
    ax.set_xlabel('k')
    ax.plot(k_list[12], score[12], marker='o', color='r')
    ax.text(k_list[12], score[12], 'Best k: %d' % 130)
    plt.show()
    fig.savefig('LOF调参过程')
    pass
