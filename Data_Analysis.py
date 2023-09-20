import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier
from sklearn import tree,svm
from sklearn.linear_model import Lasso, Ridge, LogisticRegression
from sklearn.ensemble import StackingClassifier

import time
from sklearn.model_selection import train_test_split, cross_val_score, ShuffleSplit, KFold, StratifiedKFold
# from sklearn.model_selection import cross_val_score
from sklearn.metrics import *
# from sklearn.model_selection import ShuffleSplit
from sklearn import preprocessing
from scipy.stats import norm
from collections import defaultdict
from scipy import stats

# read data
def read_excel(FilePath):
    data = pd.read_excel(FilePath, index_col=None, header=0)

    # drop_list = ['Sample', 'Class1', 'Subject', 'Infection_status', 'Infection_status_code', 'Sex_code', 'Sex', 'Age', 'Heart_disease', 'Hypertension', 'Diabetes', 'Dyslipidemia', 'Obesity', 'CRD', 'COPD', 'SpO2']
    # x_data = data.drop(labels=drop_list, axis=1)

    sig = np.array(pd.read_csv('./data/match_mz.txt', index_col=None, header=None))
    # sig = np.array(pd.read_csv('./data/shap/01/fea.txt', index_col=None, header=None))

    sig = sig.squeeze()
    x_data = data[sig]

    drop_list = [
        179.0912706, 103.0388071, 130.0497797, 153.0769488,
        105.0333624, 151.0613655,
        166.0497778, 131.0895226,
        135.0439309,
        139.0865057,
        138.0912264,
        399.3271824, 219.0476056, 163.0607952, 137.1324043,
        148.0967053, 265.2161462,
        194.1174761, 177.1021238,
        211.1884765,
                 ]
    x_data = x_data.drop(labels=drop_list, axis=1)

    print(x_data.shape)

    y_data = data['Infection_status_code']

    return x_data, y_data


def standardize(x_data):
    for i in list(x_data.columns):
        # x_data[i] = (x_data[i] - x_data[i].mean()) / x_data[i].std()
        x_data[i] = (x_data[i] - x_data[i].min()) / (x_data[i].max() - x_data[i].min())
    return x_data

if __name__ == '__main__':

    filePath = "./data/death data - s&f.xlsx"

    data_x, data_y = read_excel(filePath)
    features = np.array(data_x.columns)

    data_x = np.array(data_x)
    data_y = np.array(data_y)

    scaler = preprocessing.MinMaxScaler()
    # scaler = preprocessing.StandardScaler()

    models = {
        'rfc': RandomForestClassifier(n_estimators=200, random_state=None),
        'lgb': LGBMClassifier(n_estimators=200)
    }

    for model_key in models.keys():
        print('\nthe classifier is:', model_key)
        auc = []
        acc = []
        for i in range(0, 20):
            AUC_score = []
            ACC_score = []

            # K-fold cross-validation
            n_splits = 10
            # kf = KFold(n_splits, shuffle = True, random_state=None)
            kf = StratifiedKFold(n_splits, shuffle=True, random_state=None)
            for train_index, test_index in kf.split(data_x, data_y):

                X_train, X_test = data_x[train_index], data_x[test_index]
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)
                y_train, y_test = data_y[train_index], data_y[test_index]

                model = models[model_key]

                begin = time.perf_counter()
                model.fit(X_train, y_train)
                elapsed = time.perf_counter() - begin

                if model_key == 'svm':
                    pred = model.decision_function(X_test)
                else:
                    pred = model.predict_proba(X_test)[:, 1]
                roc_score = roc_auc_score(y_test, pred)
                acc_score = model.score(X_test, y_test)

                AUC_score.append(roc_score)
                ACC_score.append(acc_score)

            auc.append(np.mean(AUC_score))
            acc.append(np.mean(ACC_score))

        # print(auc)
        # print(acc)
        print('The AUC score is:', np.mean(auc))
        print('The ACC score is:', np.mean(acc))
