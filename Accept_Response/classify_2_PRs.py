import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_selection import SelectFromModel
import matplotlib.pyplot as plt
# plt.rc("font", size=14)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC
from sklearn.dummy import DummyClassifier
from sklearn import metrics
from sklearn.metrics import precision_recall_fscore_support as score
from statistics import mean
import xgboost as xgb
import operator
from xgboost import plot_importance
import pickle
from sklearn.utils import shuffle
from imblearn.under_sampling import RandomUnderSampler, TomekLinks, ClusterCentroids
from imblearn.over_sampling import RandomOverSampler, SMOTE

import seaborn as sns
# sns.set(style="white")
# sns.set(style='whitegrid', color_codes=True)

df = pd.read_csv("E:\\Documents\\Research Work\\datasets\\3_multilabel.csv", sep=',', encoding='utf-8')

# 'cmssw'
project_list = ['react', 'django', 'nixpkgs', 'scikit-learn', 'yii2', 'cdnjs', 'terraform', 'cmssw', 'salt', 'tensorflow', 'pandas',
                'symfony', 'moby', 'rails', 'rust', 'kubernetes', 'angular.js', 'laravel', 'opencv',
                ]


df = df[(df.Project_Name != 'githubschool') & (df.Project_Name != 'curriculum')]

print(df.shape)
# print(list(df.columns))
# print(df.Project_Name.unique())


scoring = ['precision', 'recall', 'f1', 'roc_auc', 'accuracy']

def get_classifiers():
    return {
        # 'RandomForest': RandomForestClassifier(n_jobs=4, n_estimators=200, bootstrap=False, class_weight='balanced',
        #                                        max_depth=11, max_features='sqrt', min_samples_leaf=2),
        # 'LinearSVC': LinearSVC(max_iter=2000), #max_depth=11, max_features='sqrt', min_samples_leaf=2
        # 'LogisticRegression': LogisticRegression(solver='lbfgs', n_jobs=4, multi_class='auto', max_iter=1200),
        'XGBoost': xgb.XGBClassifier(max_depth=11, min_child_weight=9), #max_depth=11, min_child_weight=9
    }


params = {
    'objective': 'binary:logistic',
    'eta': 0.08,
    'colsample_bytree': 0.886,
    'min_child_weight': 1.1,
    'max_depth': 7,
    'subsample': 0.886,
    'gamma': 0.1,
    'lambda':10,
    'verbose_eval': True,
    'eval_metric': 'auc',
    'scale_pos_weight':6,
    'seed': 201703,
    'missing':-1
}


def encode_labels(df1, column_name):
    encoder = LabelEncoder()
    df1[column_name] = [str(label) for label in df1[column_name]]
    encoder.fit(df1[column_name])
    one_hot_vector = encoder.transform(df1[column_name])
    return one_hot_vector


df['Language'] = encode_labels(df, 'Language')
df['Project_Domain'] = encode_labels(df, 'Project_Domain')
df['src_churn'] = df['Additions'] + df['Deletions']
df['num_comments'] = df['Review_Comments_Count'] + df['Comments_Count']

# {0: 'directly_accepted', 1: 'response_required', 2: 'rejected',}

project_features = ['Project_Age', 'Team_Size', 'Stars', 'File_Touched_Average', 'Forks_Count', 'Watchers', 'Language',
                    'Project_Domain', 'Contributor_Num', 'Comments_Per_Closed_PR', 'Additions_Per_Week', 'Deletions_Per_Week',
                    'Merge_Latency', 'Comments_Per_Merged_PR', 'Churn_Average', 'Close_Latency', 'Project_Accept_Rate',
                    'Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Workload', 'Commits_Average',
                    'Open_Issues', 'PR_Time_Created_At', 'PR_Date_Closed_At', 'PR_Time_Closed_At', 'PR_Date_Created_At',
                    'Project_Name', 'PR_accept']
PR_features = ['Intra_Branch', 'Assignees_Count', 'Label_Count', 'Files_Changed', 'Contain_Fix_Bug', 'Wait_Time', 'Day',
               'src_churn', 'Deletions', 'Commits_PR', 'first_response_time', 'first_response',
               'latency_after_first_response', 'conflict', 'X1_0', 'X1_1', 'X1_2', 'X1_3', 'X1_4', 'X1_5', 'X1_6', 'X1_7',
               'X1_8', 'X1_9', 'PR_Latency', 'title_words_count', 'body_words_count','Point_To_IssueOrPR', 'PR_Time_Created_At',
               'PR_Date_Closed_At', 'PR_Time_Closed_At', 'PR_Date_Created_At', 'Project_Name', 'PR_accept']
integrator_features = ['Participants_Count', 'num_comments', 'Last_Comment_Mention',
                       'line_comments_count', 'comments_reviews_words_count', 'PR_Time_Created_At', 'PR_Date_Closed_At',
                       'PR_Time_Closed_At', 'PR_Date_Created_At', 'Project_Name', 'PR_accept']
contributor_features = ['Followers', 'Closed_Num', 'Contributor', 'Public_Repos', 'Organization_Core_Member',
                        'Contributions', 'User_Accept_Rate', 'Accept_Num', 'Closed_Num_Rate', 'Prev_PRs', 'Following',
                        'PR_Time_Created_At', 'PR_Date_Closed_At', 'PR_Time_Closed_At', 'PR_Date_Created_At',
                        'Project_Name', 'PR_accept']

# user_features = ['Participants_Count', 'Comments_Count', 'Last_Comment_Mention']
# issue_features = ['Point_To_IssueOrPR', 'Open_Issues']
# other_features = ['Workload', 'Commits_Average']
# integrator_features = ['Review_Comments_Count', 'line_comments_count', 'comments_reviews_words_count']

all_features = ['Closed_Num_Rate', 'Label_Count', 'num_comments', 'Following', 'Stars', 'Contributions', 'Merge_Latency',
                'Followers',  'Workload', 'Wednesday', 'Closed_Num', 'Public_Repos',
                'Deletions_Per_Week', 'Contributor', 'File_Touched_Average', 'Forks_Count', 'Organization_Core_Member',
                'Monday', 'Contain_Fix_Bug', 'src_churn', 'Team_Size', 'Last_Comment_Mention', 'Sunday',
                'Thursday', 'Project_Age', 'Open_Issues', 'Intra_Branch', 'Saturday', 'Participants_Count',
                'Comments_Per_Closed_PR', 'Watchers', 'Project_Accept_Rate', 'Point_To_IssueOrPR', 'Accept_Num',
                'Close_Latency', 'Contributor_Num', 'Commits_Average', 'Assignees_Count', 'Friday', 'Commits_PR',
                'Wait_Time', 'line_comments_count', 'Prev_PRs', 'Comments_Per_Merged_PR', 'Files_Changed', 'Day',
                'Churn_Average', 'Language', 'Tuesday', 'Additions_Per_Week', 'User_Accept_Rate', 'X1_0', 'X1_1',
                'X1_2', 'X1_3', 'X1_4', 'X1_5', 'X1_6',  'X1_7', 'X1_8', 'X1_9', 'PR_Latency', 'Project_Name',
                'PR_Date_Created_At', 'PR_Time_Created_At', 'PR_Date_Closed_At',
                'PR_Time_Closed_At', 'first_response_time', 'first_response', 'latency_after_first_response',
                'title_words_count', 'body_words_count', 'comments_reviews_words_count',
                'Project_Domain', 'PR_accept']

selected_features = ['Contributor', 'Project_Accept_Rate', 'User_Accept_Rate', 'Deletions_Per_Week', 'Wait_Time', 'Stars',
                     'Additions_Per_Week', 'Saturday', 'latency_after_first_response', 'Assignees_Count', 'Wednesday',
                     'comments_reviews_words_count', 'PR_Latency', 'Project_Domain', 'Team_Size', 'Language', 'Commits_PR',
                     'Label_Count', 'Contributions', 'num_comments', 'Last_Comment_Mention', 'Comments_Per_Merged_PR',
                     'Contributor_Num', 'Watchers', 'first_response', 'Files_Changed', 'Comments_Per_Closed_PR', 'Monday',
                     'line_comments_count', 'Open_Issues', 'Sunday',
                     'PR_Date_Created_At', 'PR_Time_Created_At', 'PR_Date_Closed_At', 'PR_Time_Closed_At', 'PR_accept'
                     ]

# df = df[integrator_features]

# Previous work features
accept_baseline = ['src_churn', 'Commits_PR', 'Files_Changed', 'num_comments', 'Participants_Count', 'conflict',
                   'Team_Size', 'Project_Size', 'File_Touched_Average', 'Commits_Average', 'Prev_PRs',
                   'User_Accept_Rate', 'PR_Time_Created_At', 'PR_Date_Closed_At', 'PR_Time_Closed_At',
                   'PR_Date_Created_At', 'Project_Name', 'PR_accept']


df = df[integrator_features]

df = df.sort_values(by=['PR_Date_Closed_At', 'PR_Time_Closed_At'], ascending=True)

target = 'PR_accept'
start_date = '2017-09-01'
end_date = '2018-02-28'

X_test = df.loc[(df['PR_Date_Created_At'] >= start_date) & (df['PR_Date_Created_At'] <= end_date)]
y_test = X_test[target]
X_train = df.loc[(df['PR_Date_Created_At'] < start_date)]
X_train = X_train
y_train = X_train[target]

print('Test dataset')
print(X_test[target].value_counts())
accepted, rejected = X_test.PR_accept.value_counts()
print('Percentage of accepted PRs {}'.format((accepted*100)/X_test.shape[0]))
print('Percentage of rejected PRs {}'.format((rejected*100)/X_test.shape[0]))

predictors = [x for x in df.columns if x not in [target, 'PR_Date_Created_At', 'PR_Time_Created_At', 'PR_Date_Closed_At',
                                                 'PR_Time_Closed_At', 'Project_Name']]

predictors_with_label = [x for x in df.columns if x not in ['PR_accept', 'PR_Date_Created_At', 'PR_Time_Created_At',
                                                            'Project_Name']]

# X = df
# y = df[target]
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, stratify=y, random_state=143)


# def get_under_sampled_dataset():
#     # Class count
#     count_class_1, count_class_0 = X_train[target].value_counts()
#
#     print(f'0 count: {count_class_0}, 1 count: {count_class_1}')
#     # Divide by class
#     df_class_0 = X_train[X_train[target] == 0]
#     df_class_1 = X_train[X_train[target] == 1]
#     print(f'Negative samples shape: {df_class_0.shape}')
#     print(f'Positive samples shape: {df_class_1.shape}')
#
#     df_class_1_under = df_class_1.sample(count_class_0)
#     df_test_under = pd.concat([df_class_1_under, df_class_0], axis=0)
#
#     print('Random under-sampling:')
#     print(df_test_under[target].value_counts())
#     shuffle(df_test_under)
#     y_df = df_test_under[target]
#
#     return df_test_under, y_df


def get_over_sampled_dataset(df_):
    # Class count
    # count_class_1, count_class_0 = X_train[target].value_counts()

    # print('0 count: {0}, 1 count: {1}'.format(count_class_0, count_class_1))
    # Divide by class
    df_accept_response = df_[df_[target] == 0]
    df_directly_rejected = df_[df_[target] == 3]
    df_rest_of_PRs = df_[df_[target] != 3]
    print(df_rest_of_PRs.shape)
    print('Directly accepted samples shape: {}'.format(df_accept_response.shape))
    print('Directly rejected samples shape: {}'.format(df_directly_rejected.shape))
    df_directly_rejected_over = df_directly_rejected.sample(df_accept_response.shape[0], replace=True)
    print(df_directly_rejected_over.shape)
    df_test_over = pd.concat([df_directly_rejected_over, df_rest_of_PRs], axis=0)

    # df_reject_response = df_test_over[df_test_over[target] == 2]
    # df_rest_of_PRs = df_test_over[df_test_over[target] != 2]
    # df_rejected_response_over = df_reject_response.sample(df_accept_response.shape[0], replace=True)
    # print(df_rejected_response_over.shape)
    # df_test_over = pd.concat([df_rejected_response_over, df_rest_of_PRs], axis=0)
    #
    # df_directly_accepted = df_test_over[df_test_over[target] == 0]
    # df_rest_of_PRs = df_test_over[df_test_over[target] != 0]
    # df_rejected_response_over = df_directly_accepted.sample(df_accept_response.shape[0], replace=True)
    # print(df_rejected_response_over.shape)
    # df_test_over = pd.concat([df_rejected_response_over, df_rest_of_PRs], axis=0)

    print('Random over-sampling:')
    print(df_test_over[target].value_counts())
    # df_test_over = df_test_over.sort_values(by=['PR_Date_Closed_At', 'PR_Time_Closed_At'], ascending=True)
    df_test_over = shuffle(df_test_over)
    y_df = df_test_over[target]
    print(df_test_over.shape)

    return df_test_over, y_df

# def get_under_sampled_dataset_imblearn():
#     # Divide by class
#     df_class_0 = X_train[X_train[target] == 0]
#     df_class_1 = X_train[X_train[target] == 1]
#     print(f'Negative samples shape: {df_class_0.shape}')
#     print(f'Positive samples shape: {df_class_1.shape}')
#
#     rus = RandomUnderSampler(return_indices=True)
#     X_rus, y_rus, id_rus = rus.fit_sample(X_train, y_train)
#     shuffle(X_rus)
#     print('Removed indexes:', id_rus)
#     y_rus = X_rus[target]
#     return X_rus, y_rus

# def get_over_sampled_dataset_imblearn():
#     # Divide by class
#     df_class_0 = X_train[X_train[target] == 0]
#     df_class_1 = X_train[X_train[target] == 1]
#     print(f'Negative samples shape: {df_class_0.shape}')
#     print(f'Positive samples shape: {df_class_1.shape}')
#
#     ros = RandomOverSampler()
#     X_ros, y_ros = ros.fit_sample(X_train, y_train)
#
#     print(X_ros.shape[0] - X_train.shape[0], 'new random picked points')
#     shuffle(X_ros)
#     X_ros = pd.DataFrame(X_ros, columns=list(df.columns))
#     print(X_ros.shape)
#     y_ros = X_ros[target]
#
#     print(y_ros.shape)
#
#     return X_ros, y_ros

# def get_tomeklinks_under_sampled_dataset():
#     tl = TomekLinks(return_indices=True, ratio='majority')
#     X_tl, y_tl, id_tl = tl.fit_sample(X_train, y_train)
#
#     print('Removed indexes:', id_tl)
#     shuffle(X_tl)
#     y_tl = X_tl[target]
#
#     return X_tl, y_tl


# def get_clusterCentriods_under_sampled_dataset(X,y):
#     cc = ClusterCentroids(ratio={0: 10})
#     X_cc, y_cc = cc.fit_sample(X, y)

def get_smote_under_sampled_dataset(X, y):

    smote = SMOTE(random_state=42)
    X_sm, y_sm = smote.fit_sample(X, y)
    print(X_sm.shape)
    print(y_sm.shape)
    X_sm = pd.DataFrame(X_sm, columns=predictors)
    #y_sm = pd.DataFrame(y_sm, columns=['label'])
    return X_sm, y_sm

# X_train, y_train = get_smote_under_sampled_dataset(X_train[predictors], y_train)


X_train = X_train[predictors]
X_test = X_test[predictors]


print("Total Train dataset size: {}".format(X_train[predictors].shape))
print("Total Test dataset size: {}".format(X_test[predictors].shape))

# print(df.columns)

# Scale the training dataset: StandardScaler
def scale_data_standardscaler(df_):
    scaler_train =StandardScaler()
    df_scaled = scaler_train.fit_transform(np.array(df_).astype('float64'))
    df_scaled = pd.DataFrame(df_scaled, columns=predictors)

    return df_scaled

def extract_metric_from_report(report):
    report = list(report.split("\n"))
    report = report[-2].split(' ')
    # print(report)
    mylist = []
    for i in range(len(report)):
        if report[i] != '':
            mylist.append(report[i])

    return mylist[3], mylist[4], mylist[5]

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

def train_XGB_feature_importance(clf, x_train, y_train):
    clf = clf.fit(x_train, y_train, verbose=11)
    # importance_type = ['weight', 'gain', 'cover', 'total_gain', 'total_cover']
    f_gain = clf.get_booster().get_score(importance_type='gain')
    importance = sorted(f_gain.items(), key=operator.itemgetter(1))
    print(importance)
    df = pd.DataFrame(importance, columns=['feature', 'fscore'])
    df.to_csv('Results/accept/features_fscore_OS_2.csv', encoding='utf-8', index=True)


def train_XGB_model(clf, x_train, y_train, x_test, name=None):
    clf = clf.fit(x_train, y_train, verbose=11)
    # Save the model
    # with open('Results/accept/models/xgb_accept_FS.pickle.dat', 'wb') as f:
    #     pickle.dump(clf, f)

    # Load the model
    # with open('response_xgb_16.pickle.dat', 'rb') as f:
    #     load_xgb = pickle.load(f)

    # fig, ax = plt.subplots(figsize=(6, 10))
    # plot_importance(clf, importance_type='gain', ax=ax)
    # plt.savefig('plots/XGB_3_labels_2.png', bbox_inches='tight')
    # feature_importances = pd.DataFrame(clf.feature_importances_, index=X_train.columns)
    # feature_importances.to_csv('Saved_Models/3_labels/features_selected_2.csv', encoding='utf-8', index=True)
    # print('XGBoost important features')
    # print(feature_importances)

    # Barplot
    # plt.bar(range(len(clf.feature_importances_)), clf.feature_importances_)
    # plt.show()
    y_pred_train = clf.predict(x_train)
    y_predprob_train = clf.predict_proba(x_train)[:, 1]

    y_pred = clf.predict(x_test)
    y_predprob = clf.predict_proba(x_test)[:, 1]

    return y_pred_train, y_predprob_train, y_pred, y_predprob


def train_SVM_model(clf, x_train, y_train, x_test, name=None):
    clf.fit(x_train, y_train)
    svm = CalibratedClassifierCV(base_estimator=clf, cv='prefit')
    svm.fit(x_train, y_train)

    # with open('Saved_Models/3_labels/'+name+'.pickle.dat', 'wb') as f:
    #     pickle.dump(clf, f)
    # train
    y_pred_train = svm.predict(x_train)
    y_predprob_train = svm.predict_proba(x_train)[:, 1]
    # test
    y_pred = svm.predict(x_test)
    y_predprob = svm.predict_proba(x_test)[:, 1]

    return y_pred_train, y_predprob_train, y_pred, y_predprob


def train_RF_LR_model(clf, x_train, y_train, x_test, name=None):
    clf.fit(x_train, y_train)
    # with open('Results/accept/models/'+name+'.pickle.dat', 'wb') as f:
    #     pickle.dump(clf, f)
    # train
    y_pred_train = clf.predict(x_train)
    y_predprob_train = clf.predict_proba(x_train)[:, 1]
    # test
    y_pred = clf.predict(x_test)
    y_predprob = clf.predict_proba(x_test)[:, 1]

    return y_pred_train, y_predprob_train, y_pred, y_predprob

def train_XGB_model_feature_selection(clf, x_train, y_train, x_test):
    model = clf.fit(x_train, y_train, verbose=11)

    y_pred = model.predict(x_test)
    # y_predprob = model.predict_proba(x_test)[:, 1]
    predictions = [round(value) for value in y_pred]
    accuracy = metrics.accuracy_score(y_test, predictions)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))

    thresholds = sorted(model.feature_importances_, reverse=True)
    for thresh in thresholds:
        # select features using threshold
        selection = SelectFromModel(model, threshold=thresh, prefit=True)
        select_X_train = selection.transform(x_train)
        # train model
        selection_model = xgb.XGBClassifier(**params)
        selection_model.fit(select_X_train, y_train)
        # eval model
        select_X_test = selection.transform(x_test)
        y_pred = selection_model.predict(select_X_test)
        predictions = [round(value) for value in y_pred]
        accuracy = metrics.accuracy_score(y_test, predictions)
        print("Thresh=%.3f, n=%d, Accuracy: %.2f%%" % (thresh, select_X_train.shape[1], accuracy * 100.0))


def train_XGB_model_feature_selection_2(x_train, y_train, x_test, y_test):
    df = pd.read_csv("Results/accept/features_fscore_2.csv", sep=",")
    df = df.sort_values(by=['fscore'], ascending=False)
    thresholds = [200, 100, 50, 30, 25, 20, 15, 13, 10, 8, 5, 3, 2]
    results = pd.DataFrame(columns=['Model', 'AUC', 'neg_precision', 'pos_precision', 'neg_recall', 'pos_recall',
                                    'neg_f_score', 'pos_f_score', 'Precision', 'Recall', 'F-Score', 'Test_Accuracy',
                                    'Train_Accuracy'])

    for thresh in thresholds:
        # select features using threshold
        features_set = df.feature[df.fscore>=thresh]
        print('Threshold {}'.format(thresh))
        print('Total features {} used'.format(len(features_set)))

        model = xgb.XGBClassifier(max_depth=11, min_child_weight=9)
        model.fit(x_train[list(features_set)], y_train)
        # eval model
        y_pred_train = model.predict(x_train[list(features_set)])
        y_pred = model.predict(x_test[list(features_set)])
        y_predprob = model.predict_proba(x_test[list(features_set)])[:, 1]

        # Print model report:
        print(metrics.classification_report(y_test, y_pred, digits=2))
        precision, recall, fscore, support = score(y_test, y_pred)

        # Print model report:
        print("\nModel Report")
        print("Train Accuracy : %.4g" % metrics.accuracy_score(y_train, y_pred_train))
        print("Test Accuracy : %.4g" % metrics.accuracy_score(y_test, y_pred))
        print('Train error: {:.3f}'.format(1 - metrics.accuracy_score(y_train, y_pred_train)))
        print('Test error: {:.3f}'.format(1 - metrics.accuracy_score(y_test, y_pred)))
        print("AUC Score : %f" % metrics.roc_auc_score(y_test, y_predprob))
        print("Recall : %f" % metrics.recall_score(y_test, y_pred))
        print("Precision : %f" % metrics.precision_score(y_test, y_pred))
        print("F-measure : %f" % metrics.f1_score(y_test, y_pred))
        c_matrix = metrics.confusion_matrix(y_test, y_pred)
        print('========Confusion Matrix==========')
        print("          Rejected    Accepted")
        print('Rejected     {}      {}'.format(c_matrix[0][0], c_matrix[0][1]))
        print('Accepted     {}      {}'.format(c_matrix[1][0], c_matrix[1][1]))
        results = results.append(
            {'Model': thresh, 'AUC': metrics.roc_auc_score(y_test, y_predprob),
             'neg_precision': precision[0], 'pos_precision': precision[1],
             'neg_recall': recall[0], 'pos_recall': recall[1],
             'neg_f_score': fscore[0], 'pos_f_score': fscore[1],
             'Precision': metrics.precision_score(y_test, y_pred),
             'Recall': metrics.recall_score(y_test, y_pred),
             'F-Score': metrics.f1_score(y_test, y_pred),
             'Test_Accuracy': metrics.accuracy_score(y_test, y_pred),
             'Train_Accuracy': metrics.accuracy_score(y_train, y_pred_train)},
            ignore_index=True)

    results.to_csv('Results/accept/feature_selection_results_2.csv', sep=',', encoding='utf-8', index=False)


def start_train_models():
    results = pd.DataFrame(columns=['Model', 'AUC', 'neg_precision', 'pos_precision', 'neg_recall', 'pos_recall',
                                    'neg_f_score', 'pos_f_score', 'Precision', 'Recall', 'F-Score', 'Test_Accuracy',
                                    'Train_Accuracy'])

    classifiers = get_classifiers()
    X_train_scaled = scale_data_standardscaler(X_train[predictors])
    X_test_scaled = scale_data_standardscaler(X_test[predictors])

    for name, value in classifiers.items():
            clf = value
            print('Classifier: ', name)
            if name == 'XGBoost':
                y_pred_train, y_predprob_train, y_pred, y_predprob = train_XGB_model(clf, X_train, y_train, X_test, name)
                # train_XGB_model_feature_selection(clf, X_train[predictors], y_train, X_test[predictors])
            elif name == 'LinearSVC':
                y_pred_train, y_predprob_train, y_pred, y_predprob = train_SVM_model(clf, X_train_scaled, y_train, X_test_scaled, name)
            elif name == 'LogisticRegression':
                y_pred_train, y_predprob_train, y_pred, y_predprob = train_RF_LR_model(clf, X_train_scaled, y_train, X_test_scaled, name)
            else:
                y_pred_train, y_predprob_train, y_pred, y_predprob = train_RF_LR_model(clf, X_train, y_train, X_test, name)

            # Print model report:
            print(metrics.classification_report(y_test, y_pred, digits=2))
            precision, recall, fscore, support = score(y_test, y_pred)

            # Print model report:
            print("\nModel Report")
            print("Train Accuracy : %.4g" % metrics.accuracy_score(y_train, y_pred_train))
            print("Test Accuracy : %.4g" % metrics.accuracy_score(y_test, y_pred))
            print('Train error: {:.3f}'.format(1 - metrics.accuracy_score(y_train, y_pred_train)))
            print('Test error: {:.3f}'.format(1 - metrics.accuracy_score(y_test, y_pred)))
            print("AUC Score : %f" % metrics.roc_auc_score(y_test, y_predprob))
            print("Recall : %f" % metrics.recall_score(y_test, y_pred))
            print("Precision : %f" % metrics.precision_score(y_test, y_pred))
            print("F-measure : %f" % metrics.f1_score(y_test, y_pred))
            c_matrix = metrics.confusion_matrix(y_test, y_pred)
            print('========Confusion Matrix==========')
            print("          Rejected    Accepted")
            print('Rejected     {}      {}'.format(c_matrix[0][0], c_matrix[0][1]))
            print('Accepted     {}      {}'.format(c_matrix[1][0], c_matrix[1][1]))
            results = results.append(
                {'Model': name, 'AUC': metrics.roc_auc_score(y_test, y_predprob),
                 'neg_precision': precision[0], 'pos_precision': precision[1],
                 'neg_recall': recall[0], 'pos_recall': recall[1],
                 'neg_f_score': fscore[0], 'pos_f_score': fscore[1],
                 'Precision': metrics.precision_score(y_test, y_pred),
                 'Recall': metrics.recall_score(y_test, y_pred),
                 'F-Score': metrics.f1_score(y_test, y_pred),
                 'Test_Accuracy': metrics.accuracy_score(y_test, y_pred),
                 'Train_Accuracy': metrics.accuracy_score(y_train, y_pred_train)},
                ignore_index=True)

    # results.to_csv('Results/accept/contributor_accept_1.csv', sep=',', encoding='utf-8', index=False)


def start_10_fold_validation(df_):
    results = pd.DataFrame(columns=['Model', 'AUC', 'neg_precision', 'pos_precision', 'neg_recall', 'pos_recall',
                                    'neg_f_score', 'pos_f_score', 'Precision', 'Recall', 'F-Score', 'Test_Accuracy',
                                    'Train_Accuracy'])
    df = df_.sort_values(by=['PR_Date_Closed_At', 'PR_Time_Closed_At'], ascending=True)
    df_split = np.array_split(df, 11)
    print(df.shape)
    for index in range(len(df_split) - 1):
        train = pd.DataFrame()
        for i in range(index + 1):
            train = train.append(df_split[i])

        # print(f"Train dataset shape: {train.shape}")
        test = df_split[index + 1]
        # print(f"Test dataset shape: {test.shape}")

        X_train = train
        y_train = X_train[target]
        # X_train, y_train = get_smote_under_sampled_dataset(X_train[predictors], y_train)
        X_test = test
        y_test = X_test[target]

        X_train = X_train[predictors]
        X_test = X_test[predictors]

        X_train_scaled = scale_data_standardscaler(X_train)
        X_test_scaled = scale_data_standardscaler(X_test)


        classifiers = get_classifiers()

        for name, value in classifiers.items():
            clf = value
            print('Classifier: ', name)
            if name == 'XGBoost':
                y_pred_train, y_predprob_train, y_pred, y_predprob = train_XGB_model(clf, X_train, y_train, X_test,
                                                                                     name)
                # train_XGB_model_feature_selection(clf, X_train[predictors], y_train, X_test[predictors])
            elif name == 'LinearSVC':
                y_pred_train, y_predprob_train, y_pred, y_predprob = train_SVM_model(clf, X_train_scaled, y_train,
                                                                                     X_test_scaled, name)
            elif name == 'LogisticRegression':
                y_pred_train, y_predprob_train, y_pred, y_predprob = train_RF_LR_model(clf, X_train_scaled, y_train,
                                                                                       X_test_scaled, name)
            else:
                y_pred_train, y_predprob_train, y_pred, y_predprob = train_RF_LR_model(clf, X_train, y_train, X_test,
                                                                                       name)

            print(metrics.classification_report(y_test, y_pred, digits=2))
            precision, recall, fscore, support = score(y_test, y_pred)

            # Print model report:
            results = results.append(
                {'Model': name, 'AUC': metrics.roc_auc_score(y_test, y_predprob),
                 'neg_precision': precision[0], 'pos_precision': precision[1],
                 'neg_recall': recall[0], 'pos_recall': recall[1],
                 'neg_f_score': fscore[0], 'pos_f_score': fscore[1],
                 'Precision': metrics.precision_score(y_test, y_pred),
                 'Recall': metrics.recall_score(y_test, y_pred),
                 'F-Score': metrics.f1_score(y_test, y_pred),
                 'Test_Accuracy': metrics.accuracy_score(y_test, y_pred),
                 'Train_Accuracy': metrics.accuracy_score(y_train, y_pred_train)},
                ignore_index=True)
    avg_result = pd.DataFrame(columns=['Model', 'AUC', 'neg_precision', 'pos_precision', 'neg_recall', 'pos_recall',
                                    'neg_f_score', 'pos_f_score', 'Precision', 'Recall', 'F-Score', 'Test_Accuracy',
                                    'Train_Accuracy'])
    for name, value in classifiers.items():
        model_result = results.loc[results.Model == name]
        avg_result = avg_result.append(
            {'Model': name, 'AUC': model_result['AUC'].mean(),
             'neg_precision': model_result['neg_precision'].mean(),
             'pos_precision': model_result['pos_precision'].mean(),
             'neg_recall': model_result['neg_recall'].mean(),
             'pos_recall': model_result['pos_recall'].mean(),
             'neg_f_score': model_result['neg_f_score'].mean(),
             'pos_f_score': model_result['pos_f_score'].mean(),
             'Precision': model_result['Precision'].mean(),
             'Recall': model_result['Recall'].mean(),
             'F-Score': model_result['F-Score'].mean(),
             'Test_Accuracy': model_result['Test_Accuracy'].mean(),
             'Train_Accuracy': model_result['Train_Accuracy'].mean()},
            ignore_index=True)
    avg_result.to_csv('Results/accept/integrator_features_10_fold_avg.csv', sep=',', encoding='utf-8', index=False)
    results.to_csv('Results/accept/integrator_features_10_fold.csv', sep=',', encoding='utf-8', index=False)


def start_each_project_model(df):
    results = pd.DataFrame(columns=['Model', 'Project', 'AUC', 'neg_precision', 'pos_precision', 'neg_recall', 'pos_recall',
                                    'neg_f_score', 'pos_f_score', 'Precision', 'Recall', 'F-Score', 'Test_Accuracy',
                                    'Train_Accuracy'])
    classifiers = get_classifiers()

    for project in project_list:
        df_project = df.loc[df.Project_Name == project]
        print('Project {} is under processing'.format(project))

        X_test = df_project.loc[(df_project['PR_Date_Created_At'] >= start_date) & (df_project['PR_Date_Created_At'] <= end_date)]
        y_test = X_test[target]
        X_train = df_project.loc[(df_project['PR_Date_Created_At'] < start_date)]
        y_train = X_train[target]

        print("Total Train dataset size: {}".format(X_train[predictors].shape))
        print("Total Test dataset size: {}".format(X_test[predictors].shape))
        # X_train, y_train = get_smote_under_sampled_dataset(X_train[predictors], y_train)
        X_train = X_train[predictors]
        X_test = X_test[predictors]
        X_train_scaled = scale_data_standardscaler(X_train[predictors])
        X_test_scaled = scale_data_standardscaler(X_test[predictors])

        for name, value in classifiers.items():
            clf = value
            print('Classifier: ', name)
            if name == 'XGBoost':
                y_pred_train, y_predprob_train, y_pred, y_predprob = train_XGB_model(clf, X_train, y_train, X_test,
                                                                                     name)
                # train_XGB_model_feature_selection(clf, X_train[predictors], y_train, X_test[predictors])
            elif name == 'LinearSVC':
                y_pred_train, y_predprob_train, y_pred, y_predprob = train_SVM_model(clf, X_train_scaled, y_train,
                                                                                     X_test_scaled, name)
            elif name == 'LogisticRegression':
                y_pred_train, y_predprob_train, y_pred, y_predprob = train_RF_LR_model(clf, X_train_scaled, y_train,
                                                                                       X_test_scaled, name)
            else:
                y_pred_train, y_predprob_train, y_pred, y_predprob = train_RF_LR_model(clf, X_train, y_train, X_test,
                                                                                       name)

            # Print model report:
            print(metrics.classification_report(y_test, y_pred, digits=2))
            precision, recall, fscore, support = score(y_test, y_pred)

            # Print model report:
            print("\nModel Report")
            print("Train Accuracy : %.4g" % metrics.accuracy_score(y_train, y_pred_train))
            print("Test Accuracy : %.4g" % metrics.accuracy_score(y_test, y_pred))
            print('Train error: {:.3f}'.format(1 - metrics.accuracy_score(y_train, y_pred_train)))
            print('Test error: {:.3f}'.format(1 - metrics.accuracy_score(y_test, y_pred)))
            print("AUC Score : %f" % metrics.roc_auc_score(y_test, y_predprob))
            print("Recall : %f" % metrics.recall_score(y_test, y_pred))
            print("Precision : %f" % metrics.precision_score(y_test, y_pred))
            print("F-measure : %f" % metrics.f1_score(y_test, y_pred))
            c_matrix = metrics.confusion_matrix(y_test, y_pred)
            print('========Confusion Matrix==========')
            print("          Rejected    Accepted")
            print('Rejected     {}      {}'.format(c_matrix[0][0], c_matrix[0][1]))
            print('Accepted     {}      {}'.format(c_matrix[1][0], c_matrix[1][1]))
            results = results.append(
                {'Model': name, 'Project': project,
                 'AUC': metrics.roc_auc_score(y_test, y_predprob),
                 'neg_precision': precision[0], 'pos_precision': precision[1],
                 'neg_recall': recall[0], 'pos_recall': recall[1],
                 'neg_f_score': fscore[0], 'pos_f_score': fscore[1],
                 'Precision': metrics.precision_score(y_test, y_pred),
                 'Recall': metrics.recall_score(y_test, y_pred),
                 'F-Score': metrics.f1_score(y_test, y_pred),
                 'Test_Accuracy': metrics.accuracy_score(y_test, y_pred),
                 'Train_Accuracy': metrics.accuracy_score(y_train, y_pred_train)},
                ignore_index=True)

    results.to_csv('Results/accept/baseline_projects.csv', sep=',', encoding='utf-8', index=False)


parameters = {
    # 'RandomForest': {'max_depth': range(3, 15, 1),
    #                  'n_estimators': [200, 300, 400, 500], #'max_features': [5, 10, 20, 25, 30],
                        #'min_samples_leaf': [1, 2, 3, 4, 5]
                     #},
    # 'LinearSVC': {'loss': ['hinge', 'squared_hinge'], 'C': [0.001, 0.01, 0.1, 1, 10]},
    # 'LogisticRegression': {'C': [0.001, 0.01, 0.1, 1, 10]},
    'XGBoost': {'max_depth':range(3,10,1), 'min_child_weight':range(1,6,1),
                #'learning_rate':[i/100.0 for i in range(1,10)],
                # 'subsample':[i/10.0 for i in range(6,10)], 'colsample_bytree':[i/10.0 for i in range(6,10)],
                # 'reg_alpha':[1e-5, 1e-2, 0.1, 1, 100]
                } # max_depth, min_child_weight , gamma, subsample these parameters are used to control overfitting
}


def model_optimizer():
    classifiers = get_classifiers()
    df_PO = pd.DataFrame(columns=['Classifier','Best_Score', 'Best_Params'])
    for name, value in classifiers.items():
        clf = value
        print('Classifer: ', name)

        grid = GridSearchCV(estimator=clf, param_grid=parameters[name], cv=3, scoring='roc_auc', verbose=3)
        grid_result = grid.fit(X_train_scaled, y_train)

        clf = grid_result.best_estimator_
        clf.fit(X_train[predictors], y_train)

        y_pred_train = clf.predict(X_train[predictors])
        y_pred = clf.predict(X_test[predictors])
        y_predprob = clf.predict_proba(X_test[predictors])[:, 1]

        print("\nModel Report")
        print("Train Accuracy : %.4g" % metrics.accuracy_score(y_train, y_pred_train))
        print("Test Accuracy : %.4g" % metrics.accuracy_score(y_test, y_pred))
        print('Train error: {:.3f}'.format(1 - metrics.accuracy_score(y_train, y_pred_train)))
        print('Test error: {:.3f}'.format(1 - metrics.accuracy_score(y_test, y_pred)))
        print("AUC Score : %f" % metrics.roc_auc_score(y_test, y_predprob))
        print("Recall : %f" % metrics.recall_score(y_test, y_pred))
        print("Precision : %f" % metrics.precision_score(y_test, y_pred))
        print("F-measure : %f" % metrics.f1_score(y_test, y_pred))
        c_matrix = metrics.confusion_matrix(y_test, y_pred)
        print('========Confusion Matrix==========')
        print("          Rejected    Accepted")
        print('Rejected     {}      {}'.format(c_matrix[0][0], c_matrix[0][1]))
        print('Accepted     {}      {}'.format(c_matrix[1][0], c_matrix[1][1]))



        # print results
        # print(f'Best Accuracy for {grid_result.best_score_:.4} using {grid_result.best_params_}')
        means = grid_result.cv_results_['mean_test_score']
        stds = grid_result.cv_results_['std_test_score']
        params = grid_result.cv_results_['params']
        for mean, stdev, param in zip(means, stds, params):
            # print(f'mean={mean:.4}, std={stdev:.4} using {param}')
            print()

        df_PO = df_PO.append({'Classifier':name, 'Best_Score': grid.best_score_, 'Best_Params': grid.best_params_}, ignore_index=True)
    df_PO.to_csv('Results/Optimization/RF_4.csv', sep=',', encoding='utf-8', index=False)

    #Best Accuracy for 0.8538 using {'max_depth': 15, 'min_samples_leaf': 1, 'n_estimators': 400} RF
    # Best Accuracy for 0.7692 using {'eta': 0.01, 'max_depth': 9, 'min_child_weight': 1}
    # Best Accuracy for 0.791 using {'learning_rate': 0.09}
    # Best Accuracy for 0.9071 using {'max_features': 'sqrt', 'min_samples_leaf': 10, 'n_estimators': 500} RF
    # Best Accuracy for 0.9083 using {'max_features': 'auto', 'min_samples_leaf': 2, 'n_estimators': 500} RF
    # Best Accuracy for 0.9111 using {'max_features': 30, 'min_samples_leaf': 4, 'n_estimators': 500}


def model_optimier_2():
    for para in range(2, 9, 1):
        # para = para/11
        print("Max_depth value {}".format(para))
        clf = RandomForestClassifier(n_jobs=4, bootstrap=False, class_weight='balanced', max_depth=17,
                                     random_state=42, min_samples_split=7, min_samples_leaf=para)
        # n_estimators = para, min_samples_leaf = 3, min_samples_split = 7, max_features = 'sqrt',
        # print(xgb.XGBClassifier())
        # clf = xgb.XGBClassifier(max_depth=11, min_child_weight=9, n_estimators=para)

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        y_predprob = clf.predict_proba(X_test)[:, 1]

        y_pred_train = clf.predict(X_train)
        y_predprob_train = clf.predict_proba(X_train)[:, 1]

        print(metrics.classification_report(y_test, y_pred, digits=2))

        # precision, recall, fscore, support = score(y_test, y_pred)

        print("\nModel Report")
        print("Train Accuracy : %.4g" % metrics.accuracy_score(y_train, y_pred_train))
        print('Train error: {:.3f}'.format(1 - metrics.accuracy_score(y_train, y_pred_train)))
        print("Test Accuracy : %.4g" % metrics.accuracy_score(y_test, y_pred))
        print('Test error: {:.3f}'.format(1 - metrics.accuracy_score(y_test, y_pred)))
        print("AUC Score : %f" % metrics.roc_auc_score(y_test, y_predprob))
        print("Recall : %f" % metrics.recall_score(y_test, y_pred))
        print("Precision : %f" % metrics.precision_score(y_test, y_pred))
        print("F-measure : %f" % metrics.f1_score(y_test, y_pred))
        c_matrix = metrics.confusion_matrix(y_test, y_pred)
        print('========Confusion Matrix==========')
        print("          Rejected    Accepted")
        print('Rejected     {}      {}'.format(c_matrix[0][0], c_matrix[0][1]))
        print('Accepted     {}      {}'.format(c_matrix[1][0], c_matrix[1][1]))


def feature_selection_LR():

    from sklearn.feature_selection import RFE

    rfe_selector = RFE(estimator=RandomForestClassifier(), n_features_to_select=30, step=5, verbose=5)
    rfe_selector.fit(X_train_scaled, y_train)

    y_pred = rfe_selector.predict(X_test_scaled)
    y_predprob = rfe_selector.predict_proba(X_test_scaled)[:, 1]


    rfe_support = rfe_selector.get_support()
    rfe_feature = X_train[predictors].loc[:,rfe_support].columns.tolist()
    print(str(len(rfe_feature)), 'selected features')
    print('RFE features')
    print(rfe_feature)
    # Print model report:
    print("\nModel Report")
    #print("Train Accuracy : %.4g" % metrics.accuracy_score(y_train, y_pred_train))
    print("Test Accuracy : %.4g" % metrics.accuracy_score(y_test, y_pred))
    #print('Train error: {:.3f}'.format(1 - metrics.accuracy_score(y_train, y_pred_train)))
    print('Test error: {:.3f}'.format(1 - metrics.accuracy_score(y_test, y_pred)))
    print("AUC Score : %f" % metrics.roc_auc_score(y_test, y_predprob))
    print("Recall : %f" % metrics.recall_score(y_test, y_pred))
    print("Precision : %f" % metrics.precision_score(y_test, y_pred))
    print("F-measure : %f" % metrics.f1_score(y_test, y_pred))
    c_matrix = metrics.confusion_matrix(y_test, y_pred)
    print('========Confusion Matrix==========')
    print("          Rejected    Accepted")
    print('Rejected     {}      {}'.format(c_matrix[0][0], c_matrix[0][1]))
    print('Accepted     {}      {}'.format(c_matrix[1][0], c_matrix[1][1]))


def calcuate_average_of_10_folds(df):
    # df = pd.read_csv('Results/results_10_fold_3.csv')
    avg_result = pd.DataFrame(columns=['Model', 'Precision', 'Recall', 'F-measure', 'Test_Accuracy', 'Train_Accuracy'])
    classifiers = get_classifiers()
    for name, value in classifiers.items():
        model_result = df.loc[df.Model == name]
        avg_result = avg_result.append(
            {'Model': name,
             'Precision': model_result['Precision'].mean(),
             'Recall': model_result['Recall'].mean(),
             'F-measure': model_result['F-measure'].mean(),
             'Test_Accuracy': model_result['Test_Accuracy'].mean(),
             'Train_Accuracy': model_result['Train_Accuracy'].mean()},
            ignore_index=True)
    avg_result.to_csv('Results/results_10_fold_avg_3.csv', sep=',', encoding='utf-8', index=False)


def calcuate_average_of_10_folds_1():
    df = pd.read_csv('Results/accept/results_projects_2.csv')
    avg_result = pd.DataFrame(columns=['Model', 'AUC', 'neg_precision', 'pos_precision', 'neg_recall', 'pos_recall',
                                    'neg_f_score', 'pos_f_score', 'Precision', 'Recall', 'F-Score', 'Test_Accuracy',
                                    'Train_Accuracy'])
    classifiers = get_classifiers()
    for name, value in classifiers.items():
        model_result = df.loc[df.Model == name]
        avg_result = avg_result.append(
            {'Model': name, 'AUC': model_result['AUC'].mean(),
             'neg_precision': model_result['neg_precision'].mean(),
             'pos_precision': model_result['pos_precision'].mean(),
             'neg_recall': model_result['neg_recall'].mean(),
             'pos_recall': model_result['pos_recall'].mean(),
             'neg_f_score': model_result['neg_f_score'].mean(),
             'pos_f_score': model_result['pos_f_score'].mean(),
             'Precision': model_result['Precision'].mean(),
             'Recall': model_result['Recall'].mean(),
             'F-Score': model_result['F-Score'].mean(),
             'Test_Accuracy': model_result['Test_Accuracy'].mean(),
             'Train_Accuracy': model_result['Train_Accuracy'].mean()},
            ignore_index=True)
    avg_result.to_csv('Results/accept/results_project_avg_2.csv', sep=',', encoding='utf-8', index=False)


def extract_selected_features():
    df_f = pd.read_csv('Results/accept/features_fscore_2.csv')
    df_f = df_f.loc[df_f.fscore >= 15]
    print(df_f.sort_values(by=['fscore']))
    print(list(df_f.feature))
    print(df_f.shape)


def draw_features_barplot():
    df = pd.read_csv("Results/accept/features_fscore_2.csv", sep=",")
    df = df.sort_values(by=['fscore'], ascending=False)
    df = df[df.fscore >= 15]
    print(len(df.feature))
    print(list(df.feature))
    fig, ax = plt.subplots(figsize=(8, 3))
    df['fscore_log'] = np.log(df['fscore'])
    sns.set(style="whitegrid")
    sns.barplot(x="feature", y="fscore_log", data=df, ax=ax, palette="GnBu_d") #palette="Blues_d" GnBu_d ch:2.5,-.2,dark=.3
    ax.xaxis.set_tick_params(labelsize=9)
    ax.set_xlabel('Features')
    ax.set_ylabel('Average Gain (log-scaled)')
    plt.xticks(rotation=90)
    plt.savefig('Results/accept/plots/accept_SF_1.png', bbox_inches='tight')
    plt.show()


def train_baseline_model():
    results = pd.DataFrame(columns=['Model', 'AUC', 'neg_precision', 'pos_precision', 'neg_recall', 'pos_recall',
                                    'neg_f_score', 'pos_f_score', 'Precision', 'Recall', 'F-Score', 'Test_Accuracy',
                                    'Train_Accuracy'])

    clf = RandomForestClassifier(n_jobs=4, n_estimators=200, bootstrap=False, class_weight='balanced',
                                 max_depth=11, max_features='sqrt', min_samples_leaf=2) #max_depth=11, max_features='sqrt', min_samples_leaf=2
    y_pred_train, y_predprob_train, y_pred, y_predprob = train_RF_LR_model(clf, X_train, y_train, X_test, 'accept_baseline')

    # Print model report:
    print(metrics.classification_report(y_test, y_pred, digits=2))
    precision, recall, fscore, support = score(y_test, y_pred)

    # Print model report:
    print("\nModel Report")
    print("Train Accuracy : %.4g" % metrics.accuracy_score(y_train, y_pred_train))
    print("Test Accuracy : %.4g" % metrics.accuracy_score(y_test, y_pred))
    print('Train error: {:.3f}'.format(1 - metrics.accuracy_score(y_train, y_pred_train)))
    print('Test error: {:.3f}'.format(1 - metrics.accuracy_score(y_test, y_pred)))
    print("AUC Score : %f" % metrics.roc_auc_score(y_test, y_predprob))
    print("Recall : %f" % metrics.recall_score(y_test, y_pred))
    print("Precision : %f" % metrics.precision_score(y_test, y_pred))
    print("F-measure : %f" % metrics.f1_score(y_test, y_pred))
    c_matrix = metrics.confusion_matrix(y_test, y_pred)
    print('========Confusion Matrix==========')
    print("          Rejected    Accepted")
    print('Rejected     {}      {}'.format(c_matrix[0][0], c_matrix[0][1]))
    print('Accepted     {}      {}'.format(c_matrix[1][0], c_matrix[1][1]))
    results = results.append(
        {'Model': 'baseline', 'AUC': metrics.roc_auc_score(y_test, y_predprob),
         'neg_precision': precision[0], 'pos_precision': precision[1],
         'neg_recall': recall[0], 'pos_recall': recall[1],
         'neg_f_score': fscore[0], 'pos_f_score': fscore[1],
         'Precision': metrics.precision_score(y_test, y_pred),
         'Recall': metrics.recall_score(y_test, y_pred),
         'F-Score': metrics.f1_score(y_test, y_pred),
         'Test_Accuracy': metrics.accuracy_score(y_test, y_pred),
         'Train_Accuracy': metrics.accuracy_score(y_train, y_pred_train)},
        ignore_index=True)

    results.to_csv('Results/accept/result_baseline.csv', sep=',', encoding='utf-8', index=False)

if __name__ == '__main__':

    print('Processing')

    # X_train, y_train = get_over_sampled_dataset()

    # X_train, y_train = get_smote_under_sampled_dataset(X_train[predictors], y_train)

    # X_train, y_train = get_over_sampled_dataset(X_train[predictors_with_label])

    # train_XGB_feature_importance(xgb.XGBClassifier(max_depth=11, min_child_weight=9), X_train, y_train)

    # train_XGB_model_feature_selection_2(X_train, y_train, X_test, y_test)

    # start_train_models()

    start_10_fold_validation(df)

    # start_each_project_model(df)

    # model_optimizer()

    # model_optimier_2()

    # feature_selection_LR()

    # calcuate_average_of_10_folds_1()

    # extract_selected_features()

    # draw_features_barplot()

    # train_baseline_model()