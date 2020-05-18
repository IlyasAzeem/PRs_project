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

# import seaborn as sns
# sns.set(style="white")
# sns.set(style='whitegrid', color_codes=True)

df = pd.read_csv("E:\\Documents\\Research Work\\datasets\\3_multilabel.csv",
          sep=',', encoding='utf-8')

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
        'RandomForest': RandomForestClassifier(n_jobs=4, bootstrap=True, class_weight='balanced', n_estimators=500, max_depth = 15,
                                               random_state=42, oob_score=True, min_samples_split=7, min_samples_leaf=3),
        'LinearSVC': LinearSVC(max_iter=2000),
        'LogisticRegression': LogisticRegression(solver='lbfgs', multi_class='multinomial', max_iter=1200),
        'XGBoost': xgb.XGBClassifier(**params),  # {'max_depth': 9, 'min_child_weight': 5}
        'DT': DecisionTreeClassifier(max_depth=5), # max_depth=5
        'NaiveBayes': GaussianNB(),
        'KNN': KNeighborsClassifier(n_neighbors=7) # n_neighbors=7
    }

# print(RandomForestClassifier())


params = {
    'objective': 'multi:softprob',
    'num_class': 4,
    'eta': 0.08,
    'colsample_bytree': 0.886,
    'min_child_weight': 1.1,
    'max_depth': 9,
    'subsample': 0.886,
    'gamma': 0.1,
    'lambda': 10,
    'verbose_eval': True,
    'eval_metric': 'auc',
    'scale_pos_weight': 6,
    'seed': 201703,
    'missing': -1
}


def encode_labels(df1, column_name):
    encoder = LabelEncoder()
    df1[column_name] = [str(label) for label in df1[column_name]]
    encoder.fit(df1[column_name])
    one_hot_vector = encoder.transform(df1[column_name])
    return one_hot_vector


df['Language'] = encode_labels(df, 'Language')
df['Project_Domain'] = encode_labels(df, 'Project_Domain')

#Creating the dependent variable class
factor = pd.factorize(df['label'])
df.label = factor[0]
definitions = factor[1]
# print(df.label.head())
print(definitions)

# {0: 'directly_accepted', 1: 'response_required', 2: 'rejected',}

project_features = ['Project_Age', 'Team_Size', 'Stars', 'File_Touched_Average', 'Forks_Count', 'Watchers', 'Language',
                    'Project_Domain', 'Contributor_Num', 'Comments_Per_Closed_PR', 'Additions_Per_Week', 'Deletions_Per_Week',
                    'Merge_Latency', 'Comments_Per_Merged_PR', 'Churn_Average', 'Close_Latency', 'Project_Accept_Rate',
                    'Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Workload', 'Commits_Average',
                    'Open_Issues', 'PR_Time_Created_At', 'PR_Date_Closed_At', 'PR_Time_Closed_At', 'PR_Date_Created_At',
                    'Project_Name', 'label']
PR_features = ['Intra_Branch', 'Assignees_Count', 'Label_Count', 'Files_Changed', 'Contain_Fix_Bug', 'Wait_Time', 'Day',
               'Additions', 'Deletions', 'Commits_PR', 'first_response_time', 'first_response',
               'latency_after_first_response', 'conflict', 'X1_0', 'X1_1', 'X1_2', 'X1_3', 'X1_4', 'X1_5', 'X1_6', 'X1_7',
               'X1_8', 'X1_9', 'PR_Latency', 'title_words_count', 'body_words_count','Point_To_IssueOrPR', 'PR_Time_Created_At',
               'PR_Date_Closed_At', 'PR_Time_Closed_At', 'PR_Date_Created_At', 'Project_Name', 'label']
integrator_features = ['Participants_Count', 'Comments_Count', 'Last_Comment_Mention', 'Review_Comments_Count',
               'line_comments_count', 'comments_reviews_words_count', 'PR_Time_Created_At', 'PR_Date_Closed_At',
               'PR_Time_Closed_At', 'PR_Date_Created_At', 'Project_Name', 'label']

contributor_features = ['Followers', 'Closed_Num', 'Contributor', 'Public_Repos', 'Organization_Core_Member',
                        'Contributions', 'User_Accept_Rate', 'Accept_Num', 'Closed_Num_Rate', 'Prev_PRs', 'Following',
                        'PR_Time_Created_At', 'PR_Date_Closed_At', 'PR_Time_Closed_At', 'PR_Date_Created_At',
                        'Project_Name', 'label'
                        ]

# user_features = ['Participants_Count', 'Comments_Count', 'Last_Comment_Mention']
# issue_features = ['Point_To_IssueOrPR', 'Open_Issues']
# other_features = ['Workload', 'Commits_Average']
# integrator_features = ['Review_Comments_Count', 'line_comments_count', 'comments_reviews_words_count']

df['src_churn'] = df['Additions'] + df['Deletions']
df['num_comments'] = df['Review_Comments_Count'] + df['Comments_Count']

# df = df[['Closed_Num_Rate', 'Label_Count', 'num_comments', 'Following', 'Stars', 'Contributions', 'Merge_Latency', #'Rebaseable',
#           'Followers',  'Workload', 'Wednesday', 'PR_accept', 'Closed_Num', 'Public_Repos',
#           'Deletions_Per_Week', 'Contributor', 'File_Touched_Average', 'Forks_Count', 'Organization_Core_Member',
#           'Monday', 'Contain_Fix_Bug', 'src_churn', 'Team_Size', 'Last_Comment_Mention', 'Sunday',
#           'Thursday', 'Project_Age', 'Open_Issues', 'Intra_Branch', 'Saturday', 'Participants_Count',
#           'Comments_Per_Closed_PR', 'Watchers', 'Project_Accept_Rate', 'Point_To_IssueOrPR', 'Accept_Num', 'Close_Latency',
#           'Contributor_Num', 'Commits_Average', 'Assignees_Count', 'Friday', 'Commits_PR', 'Wait_Time', 'line_comments_count',
#           'Prev_PRs', 'Comments_Per_Merged_PR', 'Files_Changed', 'Day', 'Churn_Average', 'Language', 'Tuesday',
#           'Mergeable_State', 'Additions_Per_Week', 'User_Accept_Rate', 'X1_0', 'X1_1', 'X1_2', 'X1_3', 'X1_4', 'X1_5', 'X1_6',
#           'X1_7', 'X1_8', 'X1_9', 'PR_Latency', 'Project_Name', 'PR_Date_Created_At', 'PR_Time_Created_At', 'PR_Date_Closed_At',
#           'PR_Time_Closed_At', 'first_response_time', 'first_response', 'latency_after_first_response',
#           'title_words_count', 'body_words_count', 'comments_reviews_words_count',
#           'Project_Domain', 'label']]

# Selected Features
# df['src_churn'] = df['Additions'] + df['Deletions']
# df['num_comments'] = df['Review_Comments_Count'] + df['Comments_Count']

df = df[['num_comments', 'Contributor', 'Participants_Count', 'line_comments_count', 'Deletions_Per_Week', 'Additions_Per_Week',
         'Project_Accept_Rate', 'Mergeable_State', 'User_Accept_Rate', 'first_response', 'Project_Domain', 'latency_after_first_response',
         'comments_reviews_words_count', 'Wait_Time', 'Team_Size', 'Stars', 'Language', 'Assignees_Count', 'Sunday', 'Contributor_Num',
         'Watchers', 'Last_Comment_Mention', 'Contributions', 'Saturday', 'Wednesday', 'Label_Count', 'Commits_PR', 'PR_Latency',
         'Comments_Per_Merged_PR', 'Organization_Core_Member', 'Comments_Per_Closed_PR', 'PR_Time_Created_At', 'PR_Date_Closed_At',
        'PR_Time_Closed_At', 'PR_Date_Created_At',
         'Project_Name', 'label', 'PR_accept']]

# df = df[integrator_features]

# Previous work features
df = df[['src_churn', 'Commits_PR', 'Files_Changed', 'num_comments', 'Participants_Count', 'conflict', 'Team_Size',
         ]]

"""
    Total Train dataset size: (198076)
    Balance of the dataset
    Number of accepted pull requests (146407)
    Number of unaccepted pull requests (51669)

    Total Test dataset size: (22730)
    Balance of the dataset
    Number of accepted pull requests (18487)
    Number of unaccepted pull requests (4243)

"""

df = df.sort_values(by=['PR_Date_Closed_At', 'PR_Time_Closed_At'], ascending=True)

target = 'label'
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

predictors = [x for x in df.columns if x not in [target, 'PR_accept', 'PR_Date_Created_At', 'PR_Time_Created_At', 'PR_Date_Closed_At',
                                                 'PR_Time_Closed_At', 'Project_Name']]

predictors_with_label = [x for x in df.columns if x not in ['PR_accept', 'PR_Date_Created_At', 'PR_Time_Created_At', 'Project_Name']]

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

X_train, y_train = get_smote_under_sampled_dataset(X_train, y_train)

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

def train_XGB_feature_importance(clf, x_train, y_train, x_test, y_test):
    X_train_scaled = scale_data_standardscaler(X_train[predictors])
    X_test_scaled = scale_data_standardscaler(X_test[predictors])

    dtrain = clf.DMatrix(x_train, label=y_train)
    dtest = clf.DMatrix(x_test, label=y_test)
    clf = clf.train(params, dtrain, num_boost_round=50)
    clf.dump_model('Saved_Models/3_labels/xgb_raw.txt')
    # y_pred_train = clf.predict(dtrain)
    y_pred_test = clf.predict(dtest)

    y_pred_test = np.argmax(y_pred_test, axis=1)
    # y_test_labels = np.argmax(y_test, axis=1)

    reversefactor = dict(zip(range(3), definitions))
    y_test_1 = np.vectorize(reversefactor.get)(y_test)
    y_pred = np.vectorize(reversefactor.get)(y_pred_test)

    print(pd.crosstab(y_test_1, y_pred, rownames=['Actual PRs'], colnames=['Predicted PRs']))

    print(metrics.classification_report(y_test_1, y_pred, digits=3))

    importance = clf.get_score(importance_type='gain')
    importance = sorted(importance.items(), key=operator.itemgetter(1))
    print(importance)

    df = pd.DataFrame(importance, columns=['feature', 'fscore'])
    df.to_csv('Saved_Models/3_labels/features_fscore.csv', encoding='utf-8', index=True)
    df['fscore'] = df['fscore'] / df['fscore'].sum()

    plt.figure()
    df.plot()
    df.plot(kind='barh', x='feature', y='fscore', legend=False, figsize=(6, 10))
    plt.title('XGBoost Feature Importance')
    plt.xlabel('relative importance')
    plt.gcf().savefig('feature_importance_xgb.png')


def train_XGB_model(clf, x_train, y_train, x_test, name=None):
    clf = clf.fit(x_train, y_train, verbose=11)
    # importance_type = ['weight', 'gain', 'cover', 'total_gain', 'total_cover']
    # f_gain = clf.get_booster().get_score(importance_type='gain')
    # importance = sorted(f_gain.items(), key=operator.itemgetter(1))
    # print(importance)
    # df = pd.DataFrame(importance, columns=['feature', 'fscore'])
    # df.to_csv('Saved_Models/3_labels/features_fscore_2.csv', encoding='utf-8', index=True)

    # f_weight = clf.get_booster().get_score(importance_type='weight')
    # print(f_gain)
    # print(weight)

    # Save the model
    # with open('Saved_Models/3_labels/xgb_selected_features.pickle.dat', 'wb') as f:
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

    with open('Saved_Models/3_labels/'+name+'.pickle.dat', 'wb') as f:
        pickle.dump(clf, f)
    # train
    y_pred_train = svm.predict(x_train)
    y_predprob_train = svm.predict_proba(x_train)[:, 1]
    # test
    y_pred = svm.predict(x_test)
    y_predprob = svm.predict_proba(x_test)[:, 1]

    return y_pred_train, y_predprob_train, y_pred, y_predprob

def train_RF_LR_model(clf, x_train, y_train, x_test, name=None):
    clf.fit(x_train, y_train)
    with open('Saved_Models/3_labels/'+name+'.pickle.dat', 'wb') as f:
        pickle.dump(clf, f)
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


def XGB_features_ranking(x_train, y_train):

    for i in range(50):
        print("processing model number {}".format(i))
        model = xgb.XGBClassifier(**params)
        model.fit(x_train, y_train)
        # importance_type = ['weight', 'gain', 'cover', 'total_gain', 'total_cover']
        f_gain = model.get_booster().get_score(importance_type='gain')
        importance = sorted(f_gain.items(), key=operator.itemgetter(1))
        print(importance)
        df = pd.DataFrame(importance, columns=['feature', 'fscore'])
        df.to_csv('Results/features_selection/features_'+str(i)+'.csv', encoding='utf-8', index=True)


def train_XGB_model_feature_selection_2(x_train, y_train, x_test, y_test):
    df = pd.read_csv("../Models/Saved_Models/3_labels/features_fscore_2.csv", sep=",")
    df = df.sort_values(by=['fscore'], ascending=False)
    thresholds = [200, 100, 50, 30, 25, 20, 15, 10, 5, 3, 2]
    results = pd.DataFrame(columns=['Model', 'P_RR', 'P_DA', 'P_R', 'R_RR', 'R_DA', 'R_R', 'f1_RR', 'f1_DA',
                                    'f1_R', 'Avg_Pre', 'Avg_Recall', 'Avg_f1_Score',
                                    'Test_Accuracy', 'Train_Accuracy'])

    for thresh in thresholds:
        # select features using threshold
        features_set = df.feature[df.fscore>=thresh]
        print(thresh)
        print(len(features_set))

        model = xgb.XGBClassifier(**params)
        model.fit(x_train[list(features_set)], y_train)
        # eval model
        y_pred_train = model.predict(x_train[list(features_set)])
        y_pred_test = model.predict(x_test[list(features_set)])
        # Print model report:
        print("\nModel Report")
        print("Train Accuracy : %.4g" % metrics.accuracy_score(y_train, y_pred_train))
        print("Test Accuracy : %.4g" % metrics.accuracy_score(y_test, y_pred_test))
        print('Train error: {:.3f}'.format(1 - metrics.accuracy_score(y_train, y_pred_train)))
        print('Test error: {:.3f}'.format(1 - metrics.accuracy_score(y_test, y_pred_test)))

        test_accuracy = metrics.accuracy_score(y_test, y_pred_test)
        train_accuracy = metrics.accuracy_score(y_train, y_pred_train)

        precision, recall, fscore, support = score(y_test, y_pred_test)

        reversefactor = dict(zip(range(3), definitions))
        y_test_1 = np.vectorize(reversefactor.get)(y_test)
        y_pred = np.vectorize(reversefactor.get)(y_pred_test)

        print(pd.crosstab(y_test_1, y_pred, rownames=['Actual PRs'], colnames=['Predicted PRs']))

        print(metrics.classification_report(y_test_1, y_pred, digits=3))

        precision_avg, recall_avg, fscore_avg = extract_metric_from_report(
            metrics.classification_report(y_test_1, y_pred, digits=3))
        # AAR, DA, DR, RAR = extract_each_class_metric_from_report(metrics.classification_report(y_test_1, y_pred, digits=4))

        results = results.append(
            {'Model': thresh,
             'P_RR': precision[1], 'P_DA': precision[0], 'P_R': precision[2], 'R_RR': recall[1],
             'R_DA': recall[0], 'R_R': recall[2], 'f1_RR': fscore[1], 'f1_DA': fscore[0], 'f1_R': fscore[2],
             'Avg_Pre': precision_avg, 'Avg_Recall': recall_avg, 'Avg_f1_Score': fscore_avg,
             'Test_Accuracy': test_accuracy,
             'Train_Accuracy': train_accuracy},
            ignore_index=True)

    results.to_csv('Results/feature_selection_results_1.csv', sep=',', encoding='utf-8', index=False)


def start_train_models():
    # results = pd.DataFrame(columns=['Model', 'Precision', 'Recall', 'F-measure', 'Test_Accuracy', 'Train_Accuracy'])
    results = pd.DataFrame(columns=['Model', 'P_RR', 'P_DA','P_R', 'R_RR', 'R_DA','R_R', 'f1_RR', 'f1_DA',
                                    'f1_R', 'Avg_Pre', 'Avg_Recall', 'Avg_f1_Score',
                                    'Test_Accuracy', 'Train_Accuracy'])
    classifiers = get_classifiers()

    X_train_scaled = scale_data_standardscaler(X_train[predictors])
    X_test_scaled = scale_data_standardscaler(X_test[predictors])


    for name, value in classifiers.items():
            clf = value
            print('Classifier: ', name)
            if name == 'XGBoost':
                y_pred_train, y_predprob_train, y_pred_test, y_predprob = train_XGB_model(clf, X_train, y_train, X_test, name)
                # train_XGB_model_feature_selection(clf, X_train[predictors], y_train, X_test[predictors])
            elif name == 'LinearSVC':
                y_pred_train, y_predprob_train, y_pred_test, y_predprob = train_SVM_model(clf, X_train_scaled, y_train, X_test_scaled, name)
            elif name == 'LogisticRegression':
                y_pred_train, y_predprob_train, y_pred_test, y_predprob = train_RF_LR_model(clf, X_train_scaled, y_train, X_test_scaled, name)
            else:
                y_pred_train, y_predprob_train, y_pred_test, y_predprob = train_RF_LR_model(clf, X_train, y_train, X_test, name)

            # Print model report:
            print("\nModel Report")
            print("Train Accuracy : %.4g" % metrics.accuracy_score(y_train, y_pred_train))
            print("Test Accuracy : %.4g" % metrics.accuracy_score(y_test, y_pred_test))
            print('Train error: {:.3f}'.format(1 - metrics.accuracy_score(y_train, y_pred_train)))
            print('Test error: {:.3f}'.format(1 - metrics.accuracy_score(y_test, y_pred_test)))

            test_accuracy = metrics.accuracy_score(y_test, y_pred_test)
            train_accuracy = metrics.accuracy_score(y_train, y_pred_train)

            precision, recall, fscore, support = score(y_test, y_pred_test)

            reversefactor = dict(zip(range(3), definitions))
            y_test_1 = np.vectorize(reversefactor.get)(y_test)
            y_pred = np.vectorize(reversefactor.get)(y_pred_test)

            print(pd.crosstab(y_test_1, y_pred, rownames=['Actual PRs'], colnames=['Predicted PRs']))

            print(metrics.classification_report(y_test_1, y_pred, digits=3))

            precision_avg, recall_avg, fscore_avg = extract_metric_from_report(
                metrics.classification_report(y_test_1, y_pred, digits=3))
            # AAR, DA, DR, RAR = extract_each_class_metric_from_report(metrics.classification_report(y_test_1, y_pred, digits=4))

            results = results.append(
                {'Model': name,
                 'P_RR': precision[1], 'P_DA': precision[0],'P_R': precision[2], 'R_RR': recall[1],
                 'R_DA': recall[0], 'R_R': recall[2], 'f1_RR': fscore[1], 'f1_DA': fscore[0],'f1_R': fscore[2],
                 'Avg_Pre': precision_avg, 'Avg_Recall': recall_avg, 'Avg_f1_Score': fscore_avg,
                 'Test_Accuracy': test_accuracy,
                 'Train_Accuracy': train_accuracy},
                ignore_index=True)


            # results = results.append(
            #     {'Model': name,
            #      'Precision': precision,
            #      'Recall': recall,
            #      'F-measure': fscore,
            #      'Test_Accuracy': test_accuracy,
            #      'Train_Accuracy': train_accuracy},
            #     ignore_index=True)

    # results.to_csv('Results/results_3_label_2.csv', sep=',', encoding='utf-8', index=False)


def start_10_fold_validation(df_):
    # results = pd.DataFrame(
    #     columns=['Model', 'Precision', 'Recall', 'F-measure', 'Test_Accuracy', 'Train_Accuracy'])
    results = pd.DataFrame(columns=['Model', 'P_RR', 'P_DA', 'P_R', 'R_RR', 'R_DA', 'R_R', 'f1_RR', 'f1_DA',
                                    'f1_R', 'Avg_Pre', 'Avg_Recall', 'Avg_f1_Score',
                                    'Test_Accuracy', 'Train_Accuracy'])
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
        X_test = test
        y_test = X_test[target]

        X_train = X_train[predictors]
        X_test = X_test[predictors]

        X_train_scaled = scale_data_standardscaler(X_train)
        X_test_scaled = scale_data_standardscaler(X_test)


        classifiers = get_classifiers()

        for name, value in classifiers.items():
            clf = value
            print('Classifer: ', name)
            if name == 'XGBoost':
                y_pred_train, y_predprob_train, y_pred, y_predprob = train_XGB_model(clf, X_train, y_train,
                                                                                     X_test)
            elif name == 'LinearSVC':
                y_pred_train, y_predprob_train, y_pred, y_predprob = train_SVM_model(clf, X_train_scaled, y_train,
                                                                                     X_test_scaled)
            elif name == 'LogisticRegression':
                y_pred_train, y_predprob_train, y_pred, y_predprob = train_RF_LR_model(clf, X_train_scaled, y_train,
                                                                                       X_test_scaled)
            else:
                y_pred_train, y_predprob_train, y_pred, y_predprob = train_RF_LR_model(clf, X_train,
                                                                                       y_train, X_test)

            # Print model report:
            print("\nModel Report")
            print("Train Accuracy : %.4g" % metrics.accuracy_score(y_train, y_pred_train))
            print("Test Accuracy : %.4g" % metrics.accuracy_score(y_test, y_pred))
            print('Train error: {:.3f}'.format(1 - metrics.accuracy_score(y_train, y_pred_train)))
            print('Test error: {:.3f}'.format(1 - metrics.accuracy_score(y_test, y_pred)))


            test_accuracy = metrics.accuracy_score(y_test, y_pred)
            train_accuracy = metrics.accuracy_score(y_train, y_pred_train)

            precision, recall, fscore, support = score(y_test, y_pred)

            reversefactor = dict(zip(range(3), definitions))
            y_test_1 = np.vectorize(reversefactor.get)(y_test)
            y_pred_test = np.vectorize(reversefactor.get)(y_pred)

            print(pd.crosstab(y_test_1, y_pred_test, rownames=['Actual PRs'], colnames=['Predicted PRs']))

            print(metrics.classification_report(y_test_1, y_pred_test, digits=3))

            precision_avg, recall_avg, fscore_avg = extract_metric_from_report(
                metrics.classification_report(y_test_1, y_pred_test, digits=3))
            # AAR, DA, DR, RAR = extract_each_class_metric_from_report(
            #     metrics.classification_report(y_test_1, y_pred_test, digits=3))

            results = results.append(
                {'Model': name,
                 'P_RR': precision[1], 'P_DA': precision[0], 'P_R': precision[2], 'R_RR': recall[1],
                 'R_DA': recall[0], 'R_R': recall[2], 'f1_RR': fscore[1], 'f1_DA': fscore[0], 'f1_R': fscore[2],
                 'Avg_Pre': precision_avg, 'Avg_Recall': recall_avg, 'Avg_f1_Score': fscore_avg,
                 'Test_Accuracy': test_accuracy,
                 'Train_Accuracy': train_accuracy},
                ignore_index=True)

            # results = results.append(
            #     {'Model': name,
            #      'Precision': precision,
            #      'Recall': recall,
            #      'F-measure': fscore,
            #      'Test_Accuracy': test_accuracy,
            #      'Train_Accuracy': train_accuracy},
            #     ignore_index=True)

    # avg_result = pd.DataFrame(columns=['Model', 'Precision', 'Recall', 'F-measure', 'Test_Accuracy', 'Train_Accuracy'])
    # for name, value in classifiers.items():
    #     model_result = results.loc[results.Model == name]
    #     avg_result = avg_result.append(
    #         {'Model': name,
    #          'Precision': model_result['Precision'].mean(),
    #          'Recall': model_result['Recall'].mean(),
    #          'F-measure': model_result['F-measure'].mean(),
    #          'Test_Accuracy': model_result['Test_Accuracy'].mean(),
    #          'Train_Accuracy': model_result['Train_Accuracy'].mean()},
    #         ignore_index=True)
    # avg_result.to_csv('Results/results_10_fold_1.csv', sep=',', encoding='utf-8', index=False)
    results.to_csv('Results/results_3_label_10F_2.csv', sep=',', encoding='utf-8', index=False)
    # calcuate_average_of_10_folds_1(results)



def start_each_project_model(df):
    results = pd.DataFrame(columns=['Model', 'Project', 'P_AAR', 'P_DA', 'P_DR', 'P_RAR', 'R_AAR', 'R_DA', 'R_DR', 'R_RAR',
                                    'f1_AAR', 'f1_DA', 'f1_DR', 'f1_RAR', 'Avg_Pre', 'Avg_Recall', 'Avg_f1_Score',
                                    'Test_Accuracy', 'Train_Accuracy'])
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
        X_train, y_train = get_smote_under_sampled_dataset(X_train[predictors], y_train)
        X_train = X_train[predictors]
        X_test = X_test[predictors]
        X_train_scaled = scale_data_standardscaler(X_train[predictors])
        X_test_scaled = scale_data_standardscaler(X_test[predictors])

        for name, value in classifiers.items():
            clf = value
            print('Classifier: ', name)
            if name == 'XGBoost':
                y_pred_train, y_predprob_train, y_pred_test, y_predprob = train_XGB_model(clf, X_train, y_train, X_test)
                # train_XGB_model_feature_selection(clf, X_train[predictors], y_train, X_test[predictors])
            elif name == 'LinearSVC':
                y_pred_train, y_predprob_train, y_pred_test, y_predprob = train_SVM_model(clf, X_train_scaled, y_train,
                                                                                          X_test_scaled)
            elif name == 'LogisticRegression':
                y_pred_train, y_predprob_train, y_pred_test, y_predprob = train_RF_LR_model(clf, X_train_scaled,
                                                                                            y_train, X_test_scaled)
            else:
                y_pred_train, y_predprob_train, y_pred_test, y_predprob = train_RF_LR_model(clf, X_train, y_train,
                                                                                            X_test)

            # Print model report:
            print("\nModel Report")
            print("Train Accuracy : %.4g" % metrics.accuracy_score(y_train, y_pred_train))
            print("Test Accuracy : %.4g" % metrics.accuracy_score(y_test, y_pred_test))
            print('Train error: {:.3f}'.format(1 - metrics.accuracy_score(y_train, y_pred_train)))
            print('Test error: {:.3f}'.format(1 - metrics.accuracy_score(y_test, y_pred_test)))

            test_accuracy = metrics.accuracy_score(y_test, y_pred_test)
            train_accuracy = metrics.accuracy_score(y_train, y_pred_train)

            precision, recall, fscore, support = score(y_test, y_pred_test)

            reversefactor = dict(zip(range(4), definitions))
            y_test_1 = np.vectorize(reversefactor.get)(y_test)
            y_pred = np.vectorize(reversefactor.get)(y_pred_test)

            print(pd.crosstab(y_test_1, y_pred, rownames=['Actual PRs'], colnames=['Predicted PRs']))

            print(metrics.classification_report(y_test_1, y_pred, digits=4))

            precision_avg, recall_avg, fscore_avg = extract_metric_from_report(
                metrics.classification_report(y_test_1, y_pred, digits=4))
            # AAR, DA, DR, RAR = extract_each_class_metric_from_report(metrics.classification_report(y_test_1, y_pred, digits=4))
            try:
                results = results.append(
                    {'Model': name, 'Project': project,
                     'P_AAR': precision[1], 'P_DA': precision[0], 'P_DR': precision[3], 'P_RAR': precision[2],
                     'R_AAR': recall[1],
                     'R_DA': recall[0], 'R_DR': recall[3], 'R_RAR': recall[2], 'f1_AAR': fscore[1], 'f1_DA': fscore[0],
                     'f1_DR': fscore[3],
                     'f1_RAR': fscore[2],
                     'Avg_Pre': precision_avg, 'Avg_Recall': recall_avg, 'Avg_f1_Score': fscore_avg,
                     'Test_Accuracy': test_accuracy,
                     'Train_Accuracy': train_accuracy},
                    ignore_index=True)
            except IndexError:
                continue

    results.to_csv('Results/results_projects_default.csv', sep=',', encoding='utf-8', index=False)


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
    for para in range(10, 17, 1):
        print("Max_depth value {}".format(para))
        # clf = RandomForestClassifier(n_jobs=4, bootstrap=False, class_weight='balanced', max_depth=17,
        #                              n_estimators=para, min_samples_leaf=3, min_samples_split=7,
        #                              max_features='sqrt', random_state=42)
        clf = xgb.XGBClassifier(max_depth=para)

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        # y_predprob = clf.predict_proba(X_test)[:, 1]

        y_pred_train = clf.predict(X_train)
        # y_predprob_train = clf.predict_proba(X_train)[:, 1]

        reversefactor = dict(zip(range(4), definitions))
        y_test_1 = np.vectorize(reversefactor.get)(y_test)
        y_pred_test = np.vectorize(reversefactor.get)(y_pred)

        print(metrics.classification_report(y_test_1, y_pred_test, digits=4))

        print("\nModel Report")
        print("Train Accuracy : %.4g" % metrics.accuracy_score(y_train, y_pred_train))
        print('Train error: {:.3f}'.format(1 - metrics.accuracy_score(y_train, y_pred_train)))
        print("Test Accuracy : %.4g" % metrics.accuracy_score(y_test, y_pred))
        print('Test error: {:.3f}'.format(1 - metrics.accuracy_score(y_test, y_pred)))
        # print("AUC Score : %f" % metrics.roc_auc_score(y_test, y_predprob))
        # print("Recall : %f" % metrics.recall_score(y_test, y_pred))
        # print("Precision : %f" % metrics.precision_score(y_test, y_pred))
        # print("F-measure : %f" % metrics.f1_score(y_test, y_pred))
        # c_matrix = metrics.confusion_matrix(y_test, y_pred)
        # print('========Confusion Matrix==========')
        # print("          Rejected    Accepted")
        # print('Rejected     {}      {}'.format(c_matrix[0][0], c_matrix[0][1]))
        # print('Accepted     {}      {}'.format(c_matrix[1][0], c_matrix[1][1]))



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
    df = pd.read_csv('Results/results_3_label_10F_2.csv')
    avg_result = pd.DataFrame(columns=['Model', 'P_RR', 'P_DA', 'P_R', 'R_RR', 'R_DA', 'R_R', 'f1_RR', 'f1_DA',
                                    'f1_R', 'Avg_Pre', 'Avg_Recall', 'Avg_f1_Score',
                                    'Test_Accuracy', 'Train_Accuracy'])
    classifiers = get_classifiers()
    for name, value in classifiers.items():
        model_result = df.loc[df.Model == name]
        avg_result = avg_result.append(
            {'Model': name,
             'P_RR': model_result['P_RR'].mean(), 'P_DA': model_result['P_DA'].mean(), 'P_R': model_result['P_R'].mean(),
             'R_RR': model_result['R_RR'].mean(), 'R_DA': model_result['R_DA'].mean(), 'R_R': model_result['R_R'].mean(),
             'f1_RR': model_result['f1_RR'].mean(), 'f1_DA': model_result['f1_DA'].mean(), 'f1_R': model_result['f1_R'].mean(),
             'Avg_Pre': model_result['Avg_Pre'].mean(), 'Avg_Recall': model_result['Avg_Recall'].mean(),
             'Avg_f1_Score': model_result['Avg_f1_Score'].mean(),
             'Test_Accuracy': model_result['Test_Accuracy'].mean(),
             'Train_Accuracy': model_result['Train_Accuracy'].mean()},
            ignore_index=True)
    avg_result.to_csv('Results/results_3_label_10F_avg_2.csv', sep=',', encoding='utf-8', index=False)



if __name__ == '__main__':

    print('Processing')

    # X_train, y_train = get_over_sampled_dataset()

    # X_train, y_train = get_smote_under_sampled_dataset(X_train[predictors], y_train)

    # X_train, y_train = get_over_sampled_dataset(X_train[predictors_with_label])

    # train_XGB_feature_importance(xgb, X_train, y_train, X_test, y_test)

    # XGB_features_ranking(X_train, y_train)

    # train_XGB_model_feature_selection_2(X_train, y_train, X_test, y_test)

    start_train_models()

    # start_10_fold_validation(df)

    # start_each_project_model(df)

    # model_optimizer()

    # model_optimier_2()

    # feature_selection_LR()

    # calcuate_average_of_10_folds_1()