from pyod.models.knn import KNN
from pyod.models.loci import LOCI
from pyod.models.lof import LOF
from pyod.models.sos import SOS
from pyod.models.hbos import HBOS
from pyod.models.abod import ABOD
from pyod.models.vae import VAE
from pyod.models.feature_bagging import FeatureBagging
from pyod.utils.data import generate_data
from pyod.utils.data import evaluate_print
from pyod.utils.example import visualize
import scipy.io as sio
import matplotlib.pyplot as plt

if __name__ == "__main__":

    # 读取数据
    data = sio.loadmat('data_GSGF4.mat')
    data = data['data']  # 辐照度 温度 湿度 功率
    input = [0, 3]
    X_train = data[:, input]

    # train kNN detector
    clf = FeatureBagging(base_estimator=KNN())
    clf.fit(X_train)

    # get the prediction labels and outlier scores of the training data
    y_train_pred = clf.labels_  # binary labels (0: inliers, 1: outliers)
    # y_train_scores = clf.decision_scores_  # raw outlier scores

    # 识别结果可视化
    plt.scatter(X_train[:, 0], X_train[:, 1], marker='o', c=y_train_pred, cmap='rainbow', s=10)
    plt.savefig('FeatureBagging_KNN')
    plt.show()
