from pyod.models.lof import LOF
from pyod.models.knn import KNN
from pyod.models.iforest import IForest
from pyod.models.lscp import LSCP
from sklearn.metrics import roc_curve, auc, f1_score
import numpy as np
import scipy.io as sio
import collections
import matplotlib.pyplot as plt
import pandas as pd
from reconstruct import Reconstruct as Rec
from pyod.utils.utility import score_to_label, precision_n_scores

if __name__ == "__main__":

    # 读取数据
    data = sio.loadmat('data_GSGF4_new2.mat')
    data = data['data']  # 0辐照度 1温度 2湿度 3功率（含异常值） 4真实标签 5功率（不含异常值）
    input = [0, 3]
    X_train = data[:, input]

    # 训练各个检测器
    clfs = collections.OrderedDict()
    # 检测器 1
    clf = KNN(n_neighbors=20, method='median', leaf_size=5)
    clf.fit(X_train)
    clfs['KNN1'] = clf
    # 检测器 2
    clf = KNN(n_neighbors=100)
    clf.fit(X_train)
    clfs['KNN2'] = clf
    # 检测器 3
    clf = LOF(n_neighbors=200)
    clf.fit(X_train)
    clfs['LOF1'] = clf
    # 检测器 4
    clf = LOF(n_neighbors=130, leaf_size=5)
    clf.fit(X_train)
    clfs['LOF2'] = clf

    # 其他
    fig1 = plt.figure()
    fig2 = plt.figure()
    count = 1
    F1 = list()
    AUC = list()
    Rate = list()
    Rank_n = list()
    # Name = ['KNN', 'LOF', 'IForest', 'LSCP']
    Label = collections.OrderedDict()
    for name, clf in clfs.items():

        # 根据真实标签计算fpr，tpr，thr
        y_true = data[:, 4]
        y_score = clf.decision_scores_
        fpr, tpr, thr = roc_curve(y_true, y_score)

        # 根据ROC曲线计算最佳阈值
        y_diff = tpr - fpr
        loc = np.argmax(y_diff)
        thr_best = thr[loc]

        # 计算最佳异常比例
        y_score_sorted = np.sort(y_score)
        loc_2 = np.argmin(np.abs(y_score_sorted - thr_best))
        rate = (8385 - loc_2) / 8385

        # 绘制ROC曲线
        ax1 = fig1.add_subplot(2, 2, count)
        point = [fpr[loc], tpr[loc]]
        roc_auc = auc(fpr, tpr)
        ax1.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
        ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax1.plot(point[0], point[1], marker='o', color='r')
        ax1.text(point[0], point[1], 'Best Threshold: %.2f Best Rate: %.2f' % (thr_best, rate))
        ax1.set_xlim((0.0, 1.0))
        ax1.set_ylim((0.0, 1.05))
        ax1.set_xlabel('False Positive Rate')
        ax1.set_ylabel('True Positive Rate')
        ax1.set(title=name)
        ax1.legend(loc="lower right")

        # 绘制异常辨识结果（散点图）
        ax2 = fig2.add_subplot(2, 2, count)
        y_train_pred = score_to_label(y_score, outliers_fraction=rate)  # binary labels (0: inliers, 1: outliers)
        ax2.scatter(data[:, 0], data[:, 3], marker='o', c=y_train_pred, cmap='rainbow', s=10)
        ax2.set(title=name)

        # 记录评价指标
        f1 = f1_score(y_true, y_train_pred)
        rank_n = precision_n_scores(y_true, y_score)
        F1.append(f1)
        AUC.append(roc_auc)
        Rate.append(rate)
        Rank_n.append(rank_n)
        Label[name] = y_train_pred
        print('方法%s的F1指标为：%.2f' % (name, f1))
        print('方法%s的AUC指标为：%.2f' % (name, roc_auc))
        print('方法%s的Rank_n指标为：%.2f' % (name, rank_n))

        count = count + 1

    # 评价指标对比
    # Result = pd.DataFrame([Rank_n, F1, AUC, Rate], index=['Rank_n', 'F1', 'AUC', 'Rate'], columns=Name)

    # 展示绘图
    fig1.tight_layout()
    fig2.tight_layout()
    plt.show()
    # fig1.savefig('ROC曲线对比图')
    # fig2.savefig('异常辨识结果对比图')

    # # 对异常值进行修复
    # p_rec = Rec(data, Label['LSCP'])
    # # 异常修复前
    # p_0 = data[:, 3]
    # # 准确值
    # p_1 = data[:, 5]
    # # 绘图
    # fig3 = plt.figure()
    # ax1 = fig3.add_subplot(211)
    # ax1.plot(p_0, c='red', label='异常处理前')
    # ax1.plot(p_1, c='black', label='准确值')
    # ax2 = fig3.add_subplot(212)
    # ax2.plot(p_rec, c='blue', label='异常处理后')
    # ax2.plot(p_1, c='black', label='准确值')
    # fig4, ax = plt.subplots()
    # ax.plot(p_0, c='red', label='异常处理前')
    # ax.plot(p_rec, c='blue', label='异常处理后')
    # plt.show()

    pass

