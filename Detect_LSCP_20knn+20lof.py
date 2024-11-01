from pyod.models.lof import LOF
from pyod.models.knn import KNN
from sklearn.metrics import roc_curve, auc
from pyod.models.iforest import IForest
import numpy as np
from pyod.models.lscp import LSCP
import scipy.io as sio
import matplotlib.pyplot as plt
from pyod.utils.utility import score_to_label

if __name__ == "__main__":

    # 读取数据
    data = sio.loadmat('data_GSGF4_new.mat')
    data = data['data']  # 辐照度 温度 湿度 功率
    input = [0, 3]
    X_train = data[:, input]
    # X_train = data

    # train detector
    # detector_list = [LOF(), LOF()]
    # detector_list = list()
    # k_list = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140,
    #           150, 160, 170, 180, 190, 200]
    # for k in k_list:
    #     detector_list.append(KNN(n_neighbors=k))
    # for k in k_list:
    #     detector_list.append(LOF(n_neighbors=k))
    # detector_list.append(IForest())
    detector_list = [KNN(), LOF()]
    clf = LSCP(detector_list, n_bins=2)
    clf.fit(X_train)

    # 绘制ROC曲线
    y_true = data[:, -1]
    y_score = clf.decision_scores_
    fpr, tpr, thr = roc_curve(y_true, y_score)
    # 根据ROC曲线计算最佳阈值
    y_diff = tpr - fpr
    loc = np.argmax(y_diff)
    thr_best = thr[loc]
    # 计算最佳异常比例
    y_score_sorted = np.sort(y_score)
    loc_2 = np.argmin(np.abs(y_score_sorted - thr_best))
    rate = (8385-loc_2)/8385
    # 绘图
    point = [fpr[loc], tpr[loc]]
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.plot(point[0], point[1], marker='o', color='r')
    plt.text(point[0], point[1], 'Best Threshold: %.2f Best Rate: %.2f' % (thr_best, rate))
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.savefig('roc曲线-LSCP_knn+lof')
    plt.show()

    #  在最佳阈值的异常标签
    y_train_pred = score_to_label(y_score, outliers_fraction=rate)  # binary labels (0: inliers, 1: outliers)
    # 识别结果可视化
    plt.scatter(data[:, 0], data[:, 3], marker='o', c=y_train_pred, cmap='rainbow', s=10)
    plt.savefig('LSCP_knn+lof')
    plt.show()

    pass
