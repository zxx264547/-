from pyod.models.lof import LOF
from pyod.models.knn import KNN
from pyod.models.iforest import IForest
from pyod.models.lscp import LSCP
from pyod.models.copod import COPOD
from sklearn.metrics import roc_curve, auc, f1_score, precision_score, recall_score, accuracy_score
import numpy as np
import scipy.io as sio
import xlsxwriter
import collections
import matplotlib.pyplot as plt
import pandas as pd
from reconstruct import Reconstruct as Rec
from pyod.utils.utility import score_to_label, precision_n_scores

if __name__ == "__main__":

    # 字体设置
    # plt.rcParams['font.sans-serif'] = ['SimHei']
    # plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    # 读取数据
    data = sio.loadmat('data_GSGF4_earlydetected.mat')
    data = data['data']  # 0辐照度 1温度 2湿度 3功率（含异常值） 4真实标签 5功率（不含异常值）
    input = [0, 3]
    X_train = data[:, input]

    # 训练各个检测器
    clfs = collections.OrderedDict()
    # 检测器 1-knn
    clf = KNN()
    clf.fit(X_train)
    clfs['(a) KNN'] = clf
    # 检测器 2-lof
    clf = LOF()
    clf.fit(X_train)
    clfs['(b) LOF'] = clf
    # 检测器 3-iforest
    clf = IForest()
    clf.fit(X_train)
    clfs['(c) IForest'] = clf
    # 检测器 4-LSCP
    # clf = COPOD()
    # clf.fit(X_train)
    # clfs['COPOD'] = clf
    detector_list = list()
    # k_list = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140,
    #           150, 160, 170, 180, 190, 200]
    # k_list = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    # k_list = [20, 40, 60, 80, 100]
    # for k in k_list:
    #     detector_list.append(LOF(n_neighbors=k))
    # for k in k_list:
    #     detector_list.append(KNN(n_neighbors=k))
    # detector_list.append(IForest())
    # detector_list.append(KNN())
    # detector_list = [KNN(), LOF(), IForest()]
    detector_list = [KNN(n_neighbors=20, method='median'), LOF(n_neighbors=130), LOF(n_neighbors=130), IForest(), KNN()]
    clf = LSCP(detector_list)
    clf.fit(X_train)
    clfs['(d) LSCP'] = clf

    # 其他
    fig1 = plt.figure()
    fig2 = plt.figure()
    count = 1
    F1 = list()
    AUC = list()
    Rate = list()
    Rank_n = list()
    Precision = list()
    Recall = list()
    Accuracy = list()
    Name = ['(a) KNN', '(b) LOF', '(c) IForest', '(d) LSCP']
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
        rate = (len(y_score) - loc_2) / len(y_score)

        # 绘制ROC曲线
        ax1 = fig1.add_subplot(2, 2, count)
        point = [fpr[loc], tpr[loc]]
        roc_auc = auc(fpr, tpr)
        ax1.plot(fpr, tpr, color='darkorange', lw=2, label='ROC曲线 (AUC = %0.2f)' % roc_auc)
        ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax1.plot(point[0], point[1], marker='o', color='r')
        ax1.text(point[0], point[1], '最优阈值: %.2f' % thr_best)
        ax1.set_xlim((0.0, 1.0))
        ax1.set_ylim((0.0, 1.05))
        ax1.set_xlabel('假阳性率\n%s' % name)
        ax1.set_ylabel('真阳性率')
        # ax1.set(title=name)
        ax1.legend(loc="lower right")

        # 绘制异常辨识结果（散点图）
        ax2 = fig2.add_subplot(2, 2, count)
        y_train_pred = score_to_label(y_score, outliers_fraction=rate)  # binary labels (0: inliers, 1: outliers)
        ax2.scatter(data[:, 0], data[:, 3], marker='o', c=y_train_pred, cmap='rainbow', s=10)
        ax2.set(title=name)

        # 记录评价指标
        f1 = f1_score(y_true, y_train_pred)
        rank_n = precision_n_scores(y_true, y_score)
        precision = precision_score(y_true, y_train_pred)
        recall = recall_score(y_true, y_train_pred)
        accuracy = accuracy_score(y_true, y_train_pred)
        F1.append(f1)
        AUC.append(roc_auc)
        Rate.append(rate)
        Rank_n.append(rank_n)
        Precision.append(precision)
        Recall.append(recall)
        Accuracy.append(accuracy)
        Label[name] = y_train_pred
        print('方法%s的F1指标为：%.2f' % (name, f1))
        print('方法%s的AUC指标为：%.2f' % (name, roc_auc))
        print('方法%s的Rank_n指标为：%.2f' % (name, rank_n))

        count = count + 1

    # 评价指标对比
    Result = pd.DataFrame([Rank_n, F1, AUC, Precision, Recall, Accuracy], index=['Rank_n', 'F1', 'AUC', 'Precision', 'Recall', 'Accuracy'], columns=Name)
    df = pd.DataFrame(Result)
    df.to_excel('Result.xlsx', engine='xlsxwriter')

    # 保存异常标签结果
    # df = pd.DataFrame(Label)
    # df.to_excel('Label.xlsx', engine='xlsxwriter')
    # np.save('Label.npy', Label)

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
