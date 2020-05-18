# use feature importance for feature selection
from numpy import loadtxt
from numpy import sort
from xgboost import XGBClassifier
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectFromModel
import pandas as pd
import operator
from xgboost import plot_importance
from matplotlib import pyplot
# load data
dataset = loadtxt('pima-indians-diabetes.csv', delimiter=",")
# split data into X and y
X = dataset[:,0:8]
Y = dataset[:,8]
# split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=7)
# fit model on all training data
model = XGBClassifier()
model.fit(X_train, y_train)

# make predictions for test data and evaluate
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

# plot feature importance
# importance_type = ['weight', 'gain', 'cover', 'total_gain', 'total_cover']
# plot_importance(model, importance_type='weight')
# pyplot.show()

f_gain = model.get_booster().get_score(importance_type='gain')
importance = sorted(f_gain.items(), key=operator.itemgetter(1))
print(importance)
df = pd.DataFrame(importance, columns=['feature', 'fscore'])
fscore_list = list(df['fscore'])
# Fit model using each importance as a threshold
# thresholds = sorted(model.feature_importances_, reverse=True)
# print(thresholds)
# for thresh in fscore_list:
# 	# select features using threshold
# 	selection = SelectFromModel(model, threshold=thresh, prefit=True)
# 	select_X_train = selection.transform(X_train)
# 	# train model
# 	selection_model = XGBClassifier()
# 	selection_model.fit(select_X_train, y_train)
# 	# eval model
# 	select_X_test = selection.transform(X_test)
# 	y_pred = selection_model.predict(select_X_test)
# 	predictions = [round(value) for value in y_pred]
# 	accuracy = accuracy_score(y_test, predictions)
# 	print("Thresh=%.3f, n=%d, Accuracy: %.2f%%" % (thresh, select_X_train.shape[1], accuracy*100.0))

for fscore in fscore_list:
    model = XGBClassifier()
    model.fit(X_train, y_train)

    # make predictions for test data and evaluate
    y_pred = model.predict(X_test)
    predictions = [round(value) for value in y_pred]
    accuracy = accuracy_score(y_test, predictions)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))