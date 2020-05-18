# importing necessary libraries
from sklearn import datasets
from sklearn.metrics import confusion_matrix, precision_score, recall_score, roc_auc_score, classification_report
from sklearn.model_selection import train_test_split

# loading the iris dataset
iris = datasets.load_iris()

# X -> features, y -> label
X = iris.data
y = iris.target

# dividing X, y into train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)

# training a linear SVM classifier
from sklearn.svm import SVC
svm_model_linear = SVC(kernel = 'linear', C = 1).fit(X_train, y_train)
svm_predictions = svm_model_linear.predict(X_test)
print(svm_predictions)
# model accuracy for X_test
accuracy = svm_model_linear.score(X_test, y_test)
# precision = precision_score(y_test, svm_predictions)
# recall = recall_score(y_test, svm_predictions)
# print(classification_report(y_test, svm_predictions, digits=3))

report = classification_report(y_test, svm_predictions, digits=3)
print(report)

# report = list(report.split("\n"))
# print(report)
# print(report[3])
# print(report[4])

def extract_each_class_metric_from_report(report):
    report = list(report.split("\n"))

    mydict2 = {}
    mydict = {}
    index = 0
    for line in range(len(report)):
        if report[line] != '':
            values_list = report[line].split(' ')
            mydict[index] = values_list
            index+=1
    count=0
    for value in mydict:
        mylist = []
        if value != 0:
            for item in range(len(mydict[value])):
                if mydict[value][item] != '':
                    mylist.append(mydict[value][item])
            mydict2[count] = mylist
            count+=1
    return mydict2[0], mydict2[1], mydict2[2], mydict2[3]

def extract_metric_from_report(report):
    report = list(report.split("\n"))
    report = report[-2].split(' ')
    # print(report)
    mylist = []
    for i in range(len(report)):
        if report[i] != '':
            mylist.append(report[i])
    print(mylist)
    return mylist[2], mylist[3], mylist[4]

# precision, recall, fscore = extract_metric_from_report(report)
label1, label2, label3 = extract_metric_from_report(report)


print(label1)
print(label2)
print(label3)

# creating a confusion matrix
cm = confusion_matrix(y_test, svm_predictions)
print(cm)