from sklearn import datasets
import numpy as np
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from scipy.stats import sem
from sklearn.cross_validation import cross_val_score, KFold, LeaveOneOut
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn import feature_selection
import pylab as pl
from sklearn.externals import joblib
from sklearn.cross_validation import ShuffleSplit
from sklearn.grid_search import IterGrid
from IPython.parallel import Client

def data_valid_check(data):  #使用sklearn里面的分类器，必须保证训练集和标签规格符合分类器的要求，为了统一，此处统一要求为numpy.ndarray!!!
    if type(data) == np.ndarray:
        print 'the type of data is ok!'
    else:
        print "There may be some mistakes! To ensure the validation of data, you'd better transfer the type."


def data_skew_check(data): 
    print 'ok'


def data_preprocessing(X_datasets, Y_datasets, test_size, random_state=33):
    X_train, X_test, Y_train, Y_test = train_test_split(X_datasets, Y_datasets, test_size, random_state)   #这部分是不是应该放在公共部分
    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)    
    return X_train, X_test, Y_train, Y_test

def optimal_features_select_from_data(X_train, X_test, Y_train, Y_test):
    optimal_percentil, results, pecentils = optimal_percentile_find(X_train, X_test, Y_train, Y_test)
    fs = feature_selection.SelectPercentile(feature_selection.chi2, percentile=pecentils[optimal_percentil])
    X_train_fs = fs.fit_transform(X_train, Y_train)   #fit_transform区别
    X_test_fs = fs.transform(X_test)                  #transform区别
    return X_train_fs, X_test_fs
    #dt.fit(X_train_fs, y_train)   
    #y_pred_fs = dt.predict(X_test_fs)
    #print "Accuracy:{0:.3f}".format(metrics.accuracy_score(y_test,y_pred_fs)),"\n"

def optimal_percentile_find(X_train, Y_train, X_test, Y_test):        
    percentiles = range(1, X_train.shape[1], 5)
    results = []
    for i in range(1,X_train.shape[1],5):
        fs = feature_selection.SelectPercentile(feature_selection.chi2, percentile=i)     #SelectPercentile选择排名排在前n%的变量
        X_train_fs = fs.fit_transform(X_train, Y_train)  
        scores = cross_val_score(clf, X_train_fs, Y_train, cv=5) 
        results = np.append(results, scores.mean())
    optimal_percentil = np.where(results == results.max())[0] 
    print "Optimal number of features:{0}".format(percentiles[optimal_percentil]), "\n"
    return optimal_percentil, results, percentiles
def clf_train_and_predict(clf, X_datasets, Y_datasets, test_size, random_state=33):
    X_train, X_test, Y_train, Y_test = data_preprocessing(X_datasets, Y_datasets, test_size, random_state)
    clf.fit(X_train, Y_train)
    Y_train_pred = clf.predict(X_train)
    Y_test_pred = clf.predict(X_test)
    train_predict_accuracy = metrics.accuracy_score(Y_train, Y_train_pred)
    train_confusion_matrix = metrics.confusion_matrix(Y_train, Y_train_pred)
    train_classification_report = metrics.classification_report(Y_train, Y_train_pred)           #需要添加target_names吗？
    test_predict_accuracy = metrics.accuracy_score(Y_test, Y_test_pred)
    test_confusion_matrix = metrics.confusion_matrix(Y_test, Y_test_pred)
    test_classification_report = metrics.classification_report(Y_test, Y_test_pred)              #需要添加target_names吗？    
    print 'linear classification trian accuracy is: ', train_predict_accuracy*100, '%'
    print 'linear classification test accuracy is: ', test_predict_accuracy*100, '%'
    print 'linear classification train confusion matrix: \n', train_confusion_matrix
    print 'linear classification test confusion matrix: \n', test_confusion_matrix
    print 'linear classification train confusion matrix: \n', train_classification_report
    print 'linear classification test confusion matrix: \n', test_classification_report    
    return X_train, X_test, Y_train, Y_test



def loo_cv(X_train, Y_train, clf):
    loo = LeaveOneOut(X_train[:].shape[0])
    scores = np.zeros(X_train[:].shape[0])
    for train_index, test_index in loo:
        X_train_cv, X_test_cv = X_train[train_index], X_train[test_index]
        Y_train_cv, Y_test_cv = Y_train[train_index], Y_train[test_index]
        clf = clf.fit(X_train_cv, Y_train_cv)
        Y_pred = clf.predict(X_test_cv)
        scores[test_index] = metrics.accuracy_score(Y_test_cv.astype(int), Y_pred.astype(int))  #这里astype(int)有问题吗？
        print ("Loo_cv mean score: {0:.3f} (+/-{1:.3f})").format(np.mean(scores), sem(scores))


def plot_cv_accuracy_with_feature_number(X_train, Y_train, X_test, Y_test):  
    # Plot number of features VS. cross-validation scores
    _, results, pecentils = optimal_percentile_find(X_train, X_test, Y_train, Y_test)
    pl.figure()
    pl.xlabel("Number of features selected")
    pl.ylabel("Cross-validation accuracy)")
    pl.plot(percentiles, results)    

def mean_scores(scores):
    return ("Mean score: {0:.3f} (+/-{1:.3f})").format(np.mean(scores), sem(scores))


def persist_cv_splits(X, y, K=3, name='data',suffix="_cv_%03d.pkl"):
    """Dump K folds to filesystem."""
    cv_split_filenames = []
    # create KFold cross validation
    cv = KFold(n_samples, K, shuffle=True, random_state=0)              #??? 
    # iterate over the K folds
    for i, (train, test) in enumerate(cv):
        cv_fold = ([X[k] for k in train], y[train], [X[k] for k in test], y[test])
        cv_split_filename = name + suffix % i
        cv_split_filename = os.path.abspath(cv_split_filename)
        joblib.dump(cv_fold, cv_split_filename)
        cv_split_filenames.append(cv_split_filename)
    return cv_split_filenames
    #cv_filenames = persist_cv_splits(X, y, name='news')    


def compute_evaluation(cv_split_filename, clf, params):
    # All module imports should be executed in the worker namespace
    # load the fold training and testing partitions from the filesystem
    X_train, y_train, X_test, y_test = joblib.load(cv_split_filename, mmap_mode='c') 
    clf.set_params(**params)
    clf.fit(X_train, y_train)
    test_score = clf.score(X_test, y_test)
    return test_score


def parallel_grid_search(lb_view, clf, cv_split_filenames, param_grid):
    all_tasks = []
    all_parameters = list(IterGrid(param_grid))
    # iterate over parameter combinations
    for i, params in enumerate(all_parameters):
        task_for_params = []
        # iterate over the K folds
        for j, cv_split_filename in enumerate(cv_split_filenames):
            t = lb_view.apply(compute_evaluation, cv_split_filename, clf,  params)
            task_for_params.append(t)
        all_tasks.append(task_for_params)
    return all_parameters, all_tasks



def print_progress(tasks):
    progress = np.mean([task.ready() for task_group in tasks for task in task_group])
    print "Tasks completed: {0}%".format(100 * progress)


def find_best_parameters(all_parameters, all_tasks, n_top=5):
    """Compute the mean score of the completed tasks"""
    mean_scores = []
    for param, task_group in zip(all_parameters, all_tasks):
        scores = [t.get() for t in task_group if t.ready()]
        if len(scores) == 0:
            continue
        mean_scores.append((np.mean(scores), param))
    return sorted(mean_scores, reverse=True)[:n_top]








def get_lb_view():
    client = Client()
    lb_view = Client.load_balanced_view()
    return lb_view



iris = datasets.load_iris()
x_iris, y_iris = iris.data, iris.target
data_valid_check(x_iris)
#print x_iris.shape, y_iris.shape
#print type(x_iris), type(y_iris) 
#print x_iris
#print y_iris

######### machine learning method - linear classification ##########
def linear_classification(X_datasets, Y_datasets, test_size, random_state=33):
    print 'start to train linear classification!'
    from sklearn.linear_model import SGDClassifier
    clf = SGDClassifier()
    X_train, X_test, Y_train, Y_test = clf_train_and_predict(clf, X_datasets, Y_datasets, test_size)
    clf = Pipeline([('scaler', preprocessing.StandardScaler()),('linear_model', SGDClassifier())])
    cv = KFold(n)                                                                               #怎样设置？？？
    scores = cross_val_score(clf, X_train, Y_train, cv=cv)
    print scores
    print mean_scores(scores)
    print 'linear classification training and classifying end!'

######### machine learning method - SVM  classification ##########

def svm_classification(X_datasets, Y_datasets, test_size, random_state=33):                        #参数寻优！！！要加入
    print 'start to train svm classification!'
    from sklearn.svm import SVC
    svc_1 = SVC(kernel='linear')
    X_train, X_test, Y_train, Y_test = clf_train_and_predict(svc_1, X_datasets, Y_datasets, test_size)
    cv = KFold(n)                                                                               #怎样设置？？？
    scores = cross_val_score(svc_1, X_train, Y_train, cv=cv)
    print scores
    print mean_scores(scores)    
    print 'svm classification training and classifying end!'


######### machine learning method - RF  classification ##########

def RF_classification(X_datasets, Y_datasets, test_size, n_trees, random_state=33):                        
    print 'start to train RF classification!'
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(n_estimators=n_trees)
    X_train, X_test, Y_train, Y_test = clf_train_and_predict(clf, X_datasets, Y_datasets, test_size)
    cv = KFold(n)                                                                               #怎样设置？？？
    scores = cross_val_score(svc_1, X_train, Y_train, cv=cv)
    print scores
    print mean_scores(scores)    
    print 'svm classification training and classifying end!'
    clf = RandomForestClassifier(n_estimators=n_trees)
    clf.fit(X_train, Y_train)
    loo_cv(X_train, Y_train, clf)