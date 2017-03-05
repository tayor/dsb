# usage: python classify_nodes.py nodes.npy

import numpy as np
import scipy as sp

from sklearn.cross_validation import StratifiedKFold as KFold
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier as RF
import xgboost as xgb

data_path = '../../../data/'
results_path = data_path+'results/'


def logloss(act, pred):
    epsilon = 1e-15
    pred = sp.maximum(epsilon, pred)
    pred = sp.minimum(1-epsilon, pred)
    ll = sum(act*sp.log(pred) + sp.subtract(1,act)*sp.log(sp.subtract(1,pred)))
    ll = ll * -1.0/len(act)
    return ll

def logregobj(y_true, y_pred):
    y_pred = 1.0 / (1.0 + np.exp(-y_pred))
    grad = y_pred - y_true
    hess = y_pred * (1.0-y_pred)
    return grad, hess


def train_classifier():
    X = np.load(results_path+'dataX.npy')
    Y = np.load(results_path+'dataY.npy')

    kf = KFold(Y, n_folds=3)
    y_pred = Y * 0
    for train, test in kf:
        X_train, X_test, y_train, y_test = X[train,:], X[test,:], Y[train], Y[test]
        clf = RF(n_estimators=100, n_jobs=3)
        clf.fit(X_train, y_train)
        y_pred[test] = clf.predict(X_test)
    print classification_report(Y, y_pred, target_names=["No Cancer", "Cancer"])

    y_pred[y_pred > 0.95] = 0.75
    y_pred[y_pred < 0.05] = 0.25
    print("logloss",logloss(Y, y_pred))

    # All Cancer
    print "Predicting all positive"
    y_pred = np.ones(Y.shape)
    print classification_report(Y, y_pred, target_names=["No Cancer", "Cancer"])
    print("logloss",logloss(Y, y_pred))

    # No Cancer
    print "Predicting all negative"
    y_pred = Y*0
    print classification_report(Y, y_pred, target_names=["No Cancer", "Cancer"])
    print("logloss",logloss(Y, y_pred))

    # try XGBoost
    print ("XGBoost")
    kf = KFold(Y, n_folds=3)
    y_pred = Y * 0
    for train, test in kf:
        X_train, X_test, y_train, y_test = X[train,:], X[test,:], Y[train], Y[test]
        clf = xgb.XGBClassifier(learning_rate =0.1, n_estimators=250, max_depth=9, min_child_weight=1, gamma=0.1,
                                 subsample=0.85, colsample_bytree=0.75, objective= 'binary:logistic', nthread=4,
                                  scale_pos_weight=1, seed=27, reg_alpha=0.01)
        # param = {'num_class': 2}
        # clf.set_params(**param)
        clf.fit(X_train, y_train)
        y_pred[test] = clf.predict(X_test)
    # print classification_report(Y, y_pred, target_names=["No Cancer", "Cancer"])
    params = {'objective': 'multi:softmax',
              'num_class': 2}

    dtrain = xgb.DMatrix(X, Y)
    clf = xgb.train(params, dtrain)
    y_pred = clf.predict(xgb.DMatrix(X))

    y_pred[y_pred > 0.95] = 0.95
    y_pred[y_pred < 0.05] = 0.05

    print y_pred[0:100]
    print("logloss",logloss(Y, y_pred))

    print "Predicting all 0.25"
    y_pred = np.ones(Y.shape)*0.25
    print("logloss",logloss(Y, y_pred))


def predict_test():

    X = np.load(results_path+'dataX.npy')
    Y = np.load(results_path+'dataY.npy')
    X_test = np.load(results_path+'testX.npy')
    X_ids = np.load(results_path+'testId.npy')

    params = {'objective': 'multi:softmax',
              'num_class': 2}

    dtrain = xgb.DMatrix(X, Y)
    clf = xgb.train(params, dtrain)

    y_pred = clf.predict(xgb.DMatrix(X_test))

    y_pred[y_pred > 0.90] = 0.90
    y_pred[y_pred < 0.05] = 0.05

    subm = np.stack([X_ids, y_pred], axis=1)

    subm_file_name = results_path+'subm3.csv'
    np.savetxt(subm_file_name, subm, fmt='%s,%.5f', header='id,cancer', comments='')
    print('Saved predictions in {}'.format(subm_file_name))


if __name__ == "__main__":
    # train_classifier()
    predict_test()
