# importing necessary libraries
from sklearn import datasets
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import xgboost as xgb

# loading the iris dataset
iris = datasets.load_iris()

# X -> features, y -> label
X = iris.data
y = iris.target

# dividing X, y into train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)

# training a Naive Bayes classifier
xgb_clf = xgb.XGBClassifier().fit(X_train, y_train)
xgb_predictions = xgb_clf.predict(X_test)
print(xgb_predictions)
# accuracy on X_test
accuracy = xgb_clf.score(X_test, y_test)
print(accuracy)

# creating a confusion matrix
cm = confusion_matrix(y_test, xgb_predictions)
print(cm)
