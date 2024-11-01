from pyod.models.knn import KNN

# class Foo:
#
#   def __init__(self, name, age):
#     self.name = name
#     self.age = age
#
#   def detail(self):
#     print(self.name)
#     print(self.age)


class KnnNew:

    def __init__(self, contamination=0.1, n_neighbors=5, method='largest', algorithm='auto', leaf_size=30,
                 metric='minkowski', p=2, metric_params=None, n_jobs=1,
                 **kwargs):
        self.contamination = contamination
        self.n_neighbors = n_neighbors
        self.method = method
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.metric = metric
        self.p = p
        self.metric_params = metric_params
        self.n_jobs = n_jobs

    def fit(self, X, y=None):
        clf = KNN(contamination=self.contamination, n_neighbors=self.n_neighbors, method=self.method, leaf_size=self.leaf_size)
        clf.fit(X)
        return self

    def score(self, X, y):
        clf = KNN(contamination=self.contamination, n_neighbors=self.n_neighbors, method=self.method, leaf_size=self.leaf_size)
        score_auc = clf.fit_predict_score(X, y, scoring='roc_auc_score')
        return score_auc

    def get_params(self, deep=True):
        clf = KNN(contamination=self.contamination, n_neighbors=self.n_neighbors, method=self.method, leaf_size=self.leaf_size)
        params = clf.get_params(deep)
        # clf.
        return params


