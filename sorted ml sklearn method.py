#coding=utf-8
import sklearn

#1.增加数据


#2.处理缺失值和异常值
#2.1 imputation of missing values
sklearn.preprocessing.Imputer(missing_values='NaN',    #缺失值的占位符
                              strategy='mean',         #替换缺失值的方法还有median, most_frequent
                              axis=0,                  #填补缺失值的方向 0表示列，1表示行
                              verbose=0,
                              copy=True)
#使用情况
from sklearn.preprocessing import Imputer
import numpy as np
imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
imp.fit([[1, 2],[np.nan, 3],[7, 6]])
X = [[np.nan, 2],[6, np.nan],[7,6]]
print imp.transform(X)




#3.变量范围从原始范围变为从0到1(数据标准化)
#3.1 标准化方法: scale
sklearn.preprocessing.scale(X          #要进行标准化的数据
                           ,axis=0    
                           ,with_mean=True    #是否计算均值
                           ,with_std=True     #是否计算标准差
                           ,copy=True)
#使用情况
from sklearn import preprocessing
import numpy as np
X = np.array([[1., -1., 2.],[2., 0., 0.],[0., 1., -1.]])
X_scaled = preprocessing.scale(X)
print X_scaled
print X_scaled.mean(axis=0)
print X_scaled.std(axis=0)
#3.2 MinMaxScaler
sklearn.preprocessing.MinMaxScaler(feature_range=(0,1), copy=True)
#使用情况
from sklearn import preprocessing
import numpy as np
X_train = np.array([[1., -1., 2.],[2., 0., 0.],[0., 1., -1.]])
min_max_scaler = preprocessing.MinMaxScaler()
X_train_minmax = min_max_scaler.fit_transform(X_train)
print X_train_minmax
print min_max_scaler.fit_transform([-3., -1., 4.])
print min_max_scaler.scale_
print min_max_scaler.min_

#3.3 Normalize
sklearn.preprocessing.normalize(X,norm='l2',axis=1,copy=True)
#使用情况
from sklearn import preprocessing
X = [[1., -1., 2.],[2., 0., 0.],[0., 1., -1]]
X_normalized = preprocessing.normalize(X, norm='l2')
print X_normalized
#Normalizer用于实现Transformer API
from sklearn import preprocessing
X = [[1., -1., 2.],[2., 0., 0.],[0., 1., -1]]
normalizer = preprocessing.Normalizer().fit(X)
print normalizer.transform(X)
print normalizer.transform([-1., 1., 0.])




#4.有些算法对于正态分布数据表现好，需要去掉变量的偏向，用对数，平方根，倒数等方法来修正偏斜




#5.特征选择
#5.1 Removing features with low variance
sklearn.feature_selection.VarianceThreshold(threshold=0.0)
#使用情况
from sklearn.feature_selection import VarianceThreshold
X = [[0,0,1],[0,1,0],[1,0,0],[0,1,1],[0,1,0],[0,1,1]]
sel = VarianceThreshold(threshold=(.8*(1-.8)))
sel.fit_transform(X)
#5.2 Univariate feature selection
#5.2.1 selectKBest
sklearn.feature_selection.SelectKBest(score_func=0,k=10)
#5.2.2 selectPercentile
sklearn.feature_selection.SelectPercentile(score_func=1,percentile=10)
#5.2.3 GenericUnivariateSelect
sklearn.feature_selection.GenericUnivariateSelect(score_func=2, mode='percentile', param=1e-05)
#5.3 Tree-based feature selection
#使用情况
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.datasets import load_iris
iris = load_iris()
X, y = iris.data, iris.target
X.shape
clf = ExtraTreesClassifier()
X_new = clf.fit(X, y).transform(X)
clf.feature_importances_
X_new.shape




#6.算法调参
#6.1 Grid Search: Searching for estimator parameters
#6.1.1
sklearn.grid_search.GridSearchCV(estimator
                                , param_grid
                                ,scoring=None
                                ,loss_func=None
                                ,score_func=None
                                ,fit_params=None
                                ,n_jobs=1
                                ,iid=True
                                ,refit=True
                                ,cv=None
                                ,verbose=0
                                ,pre_dispatch='2*n_jobs')
#使用情况
from sklearn import svm, grid_search, datasets
iris = datasets.load_iris()
parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
svr = svm.SVC()
clf = grid_search.GridSearchCV(svr, parameters)
clf.fit(iris.data, iris.target)
#6.2 Randomized Parameter Optimization
sklearn.grid_search.RandomizedSearchCV(estimator
                                      ,param_distributions
                                      ,n_iter=10
                                      ,scoring=None
                                      ,fit_params=None
                                      ,n_jobs=1
                                      ,iid=True
                                      ,refit=True
                                      ,cv=None
                                      ,verbose=0
                                      ,pre_dispatch='2*n_jobs'
                                      ,random_state=None)
#使用情况
from sklearn import svm, grid_search, datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import RandomizedSearchCV
iris = datasets.load_iris()
X, y = iris.data, iris.target
clf = RandomForestClassifier(n_estimators=20)
param_dist = {"max_depth": [3, None],
              "max_features": sp_randint(1, 11),
              "min_sample_split": sp_randint(1, 11),
              "min_sample_leaf": sp_randint(1, 11),
              "bootstrap": [True, True],
              "criterion": ["gini", "entropy"]}
n_iter_search = 20
random_search = RandomizedSearchCV(clf, param_distributions=param_dist, n_iter=n_iter_search)
random_search.fit(X, y)




#7.集成模型
#7.1 BaggingClassifier
clf = sklearn.ensemble.BaggingClassifier(base_estimator=None
                                  ,n_estimators=10
                                  ,max_samples=1.0
                                  ,max_features=1.0
                                  ,bootstrap=True
                                  ,bootstrap_features=False
                                  ,oob_score=False
                                  ,n_jobs=1
                                  ,random_state=None
                                  ,verbose=0)
#7.2 RandomForestsClassifier
clf = sklearn.ensemble.RandomForestClassifier(n_estimators=10
                                       ,criterion='gini'
                                       ,max_depth=None
                                       ,min_samples_split=2
                                       ,min_samples_leaf=1 
                                       ,max_features='auto'
                                       ,max_leaf_nodes=None
                                       ,bootstrap=True
                                       ,oob_score=False
                                       ,n_jobs=1
                                       ,random_state=None
                                       ,verbose=0
                                       ,min_density=None
                                       ,compute_importances=None)

#7.3 Adaboost
sklearn.ensemble.AdaBoostClassifier(base_estimator=None
                                   ,n_estimators=50
                                   ,learning_rate=1.0
                                   ,algorithm='SAMME.R'
                                   ,random_state=None)
#使用情况
from sklearn.cross_validation import cross_val_score
from sklearn.datasets import load_iris
from sklearn.ensemble import AdaBoostClassifier
iris = load_iris()
clf = AdaBoostClassifier(n_estimators=100)
scores = cross_val_score(clf, iris.data, iris.target)
scores.mean()

#7.4 Gradient Tree Boosting
sklearn.ensemble.GradientBoostingClassifier(loss='deviance'
                                           ,learning_rate=0.1
                                           ,n_estimators=100
                                           ,subsample=1.0
                                           ,min_samples_split=2
                                           ,min_samples_leaf=1
                                           ,max_depth=3
                                           ,init=None
                                           ,random_state=None
                                           ,max_features=None
                                           ,verbose=0
                                           ,max_leaf_nodes=None
                                           ,warm_start=False)
#使用情况1
from sklearn.datasets import make_hastie_10_2
from sklearn.ensemble import GradientBoostingClassifier
X,y = make_hastie_10_2(random_state=0)
X_train, X_test = X[:2000],X[2000:]
y_train, y_test = y[:2000],y[2000:]
clf = GradientBoostingClassifier(n_estimators=100,learning_rate=1.0,max_depth=1,random_state=0).fit(X_train,y_train)
clf.score(X_test, y_test)
#使用情况2
from sklearn.ensemble import GradientBoostingClassifier
gbdt = GradientBoostingClassifier()
gbdt.fit(X_train, y_train)
print gbdt.feature_importances_
print gbdt.feature_importances_.shape
y_pred = gbdt.predict(X_test)
#根据重要性保留部分特征

X_train_new = X_train[:, feature_importances>0]
print X_train_new.shape
X_test_new = X_test[:, feature_importances>0]
print X_test_new.shape
#7.5使用多项式贝叶斯
from sklearn.naive_bayes import MultinomialNB
bayes = MultinomialNB()
bayes.fit(X_train, y_train)
y_pred = bayes.predict(X_test)
#7.6使用伯努利贝叶斯
from sklearn.naive_bayes import BernoulliNB
bayes_nb = BernoulliNB()
bayes_nb.fit(X_train, y_train)
y_pred = bayes_nb.predict(X_test)






#8.交叉验证
#8.1 KFold
sklearn.cross_validation.KFold(n
                              ,n_folds=3
                              ,indices=None
                              ,shuffle=False
                              ,random_state=None)
#使用情况
import numpy as np
from sklearn.cross_validation import KFold
kf = KFold(4, n_folds=2)
for train_test in kf:
    print("%s %s" % (train, test))
#8.2 Leave-One-Out -LOO
sklearn.cross_validation.LeaveOneOut(n
                                    ,indices=None)
#使用情况
from sklearn.cross_validation import LeaveOneOut
loo = LeaveOneOut(4)
for train, test in loo:
    print("%s %s" % (train, test))






#9 Model persistence
#实现持久化的类是pickle
#使用情况
from sklearn import svm
from sklearn import datasets
clf = svm.SVC()
iris = datasets.load_iris()
X, y = iris.data, iris.target
clf.fit(X, y)
import pickle
s = pickle.dumps(clf)
clf2 = pickle.load(s)
print clf2.predict(X[0])




#10 validation curves: ploting scores to evalue models
#使用情况
import numpy as np
from sklearn.learning_curve import validation_curve
from sklearn.datasets import load_iris
from sklearn.linear_model import Ridge
np.random.seed(0)
iris = load_iris()
X, y = iris.data, iris.target
indices = np.arange(y.shape[0])
np.random.shuffle(indices)
X, y = X[indices], y[indices]
train_scores, valid_scores = validation_curve(Ridge(), X, y,'alpha', 
                                             np.logspace(-7, 3, 3))
print train_scores
print valid_scores

#Learning curve
#使用情况
import numpy as np 
from sklearn.learning_curve import learning_curve
from sklearn.datasets import load_iris
from sklearn.svm import SVC
np.random.seed(0)
iris = load_iris()
X, y = iris.data, iris.target
indices = np.arange(y.shape[0])
np.random.shuffle(indices)
X, y = X[indices], y[indices]
train_sizes, train_scores, valid_scores = learning_curve(SVC(kernel='linear'), X, y, train_sizes=[50, 80, 110], cv = 5)
print train_sizes
print train_scores
print valid_scores

#11 特征特别多的情况下需要减少特征
#11.1 PCA
sklearn.decomposition.PCA(n_components=None)
#使用情况
import numpy as np
from sklearn.decomposition import PCA
X = np.array([[-1,1],[-2,-1],[-3,-2],[1,1],[2,1],[3,2]])
pca = PCA(n_components=2)
pca.fit(X)
print pca.explained_variance_ratio_

