import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import scipy.io as sio

from pyod.models.knn import KNN
from pyod.models.combination import aom, moa, average, maximization, median
from pyod.utils.utility import standardizer
from pyod.utils.data import generate_data
from pyod.utils.data import evaluate_print
from pyod.utils.utility import score_to_label

if __name__ == "__main__":

    # 读取数据
    data = sio.loadmat('data_GSGF4_new.mat')
    data = data['data']  # 辐照度 温度 湿度 功率
    input = [0, 3]
    X_train = data[:, input]

    # standardizing data for processing
    X_train_norm = standardizer(X_train)

    n_clf = 20  # number of base detectors

    # Initialize 20 base detectors for combination
    k_list = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140,
              150, 160, 170, 180, 190, 200]

    train_scores = np.zeros([X_train.shape[0], n_clf])

    print('Combining {n_clf} kNN detectors'.format(n_clf=n_clf))

    for i in range(n_clf):
        k = k_list[i]

        clf = KNN(n_neighbors=k, method='largest')
        clf.fit(X_train_norm)

        train_scores[:, i] = clf.decision_scores_

    # Decision scores have to be normalized before combination
    train_scores_norm = standardizer(train_scores)

    # Combination by average
    y_by_average = average(train_scores_norm)

    # Combination by max
    y_by_maximization = maximization(train_scores_norm)

    # Combination by median
    y_by_median = median(train_scores_norm)

    # Combination by aom
    y_by_aom = aom(train_scores_norm, n_buckets=5)

    # Combination by moa
    y_by_moa = moa(train_scores_norm, n_buckets=5)

    label_moa = score_to_label(y_by_moa, outliers_fraction=0.152)

    # 识别结果可视化
    plt.scatter(X_train[:, 0], X_train[:, 1], marker='o', c=label_moa, cmap='rainbow', s=10)
    plt.savefig('Combination_15,2%_moa_newdata-knn集成方法')
    plt.show()

    # 异常分数可视化
    y_by_moa_sorted = np.sort(y_by_moa)
    min_max_scalar = MinMaxScaler(feature_range=(1, 2))
    y_by_moa_sorted_minmax = min_max_scalar.fit_transform(y_by_moa_sorted.reshape(-1, 1))
    y_by_moa_sorted_log = np.log(y_by_moa_sorted_minmax)
    plt.plot(y_by_moa_sorted, label='without log')
    plt.plot(y_by_moa_sorted_log, label='with log')
    plt.legend()
    plt.savefig('异常分数')
    plt.show()

    # 绘制ROC曲线
    y_true = data[:, -1]
    y_score = y_by_moa
    fpr, tpr, thr = roc_curve(y_true, y_score)
    # 根据ROC曲线计算最佳阈值
    y_diff = tpr - fpr
    loc = np.argmax(y_diff)
    thr_best = thr[loc]
    point = [fpr[loc], tpr[loc]]
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.plot(point[0], point[1], marker='o', color='r')
    plt.text(point[0], point[1], 'Best Threshold: %.2f' % thr_best)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.savefig('roc曲线-knn集成方法')
    plt.show()

    pass
