# Bagged Decision Trees for Classification
import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
import time as t
from sklearn.pipeline import make_pipeline
from sklearn import metrics


names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = pd.read_csv('pima-indians-diabetes.csv', names=names)
array = dataframe.values

def scale_data_standardscaler(df_):
    scaler_train =StandardScaler()
    df_scaled = scaler_train.fit_transform(np.array(df_).astype('float64'))
    df_scaled = pd.DataFrame(df_scaled)
    return df_scaled
# array = scale_data_standardscaler(array)

X = array[:,0:8]
Y = array[:,8]
seed = 7

kfold = model_selection.KFold(n_splits=10, random_state=seed)

my_models = {'cart': DecisionTreeClassifier(),
             'svm': make_pipeline(StandardScaler(), LinearSVC(max_iter=1500, dual=False)),
             'knn': KNeighborsClassifier(),
             'LR': LogisticRegression(solver='lbfgs', max_iter=1200),
             'NB': GaussianNB(),
             'XGB': xgb.XGBClassifier(),
             'RF': RandomForestClassifier(),
             }

num_trees = 100

for clf, value in my_models.items():
    print(f'Train model {clf}')
    t1 = t.time()
    model = BaggingClassifier(base_estimator=value, n_estimators=num_trees, random_state=seed)
    results = model_selection.cross_val_score(model, X, Y, cv=kfold)
    # model.fit(X,Y)
    # y_predict = model.predict(X)
    # print(metrics.accuracy_score(y_predict, Y))
    t2 = round(t.time() - t1, 3)
    print(f'Time taken by training {t2} seconds')
    print(results.mean())