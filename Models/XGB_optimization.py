import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
# import matplotlib.pyplot as plt
# plt.rc("font", size=14)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC
from sklearn import metrics
from statistics import mean
import xgboost as xgb
from xgboost import plot_importance
import pickle
from sklearn.utils import shuffle
from imblearn.under_sampling import RandomUnderSampler, TomekLinks, ClusterCentroids
from imblearn.over_sampling import RandomOverSampler, SMOTE

# import seaborn as sns
# sns.set(style="white")
# sns.set(style='whitegrid', color_codes=True)

df = pd.read_csv("/home/ppf/Ilyas_dataset/accept_21_features.csv", sep=",", encoding="ISO-8859-1")

# 'cmssw'
project_list = ['react', 'django', 'nixpkgs', 'scikit-learn', 'yii2', 'cdnjs', 'terraform', 'cmssw', 'salt',
                'tensorflow', 'pandas', 'symfony', 'moby', 'rails', 'rust', 'kubernetes'
                ]

# Remove some of the PRs with negative latency
ID = df.Pull_Request_ID[df.latency_after_first_response < 0]
df = df.loc[~df.Pull_Request_ID.isin(ID)]

df = df[(df.Project_Name != 'angular.js') & (df.Project_Name != 'githubschool') & (df.Project_Name != 'curriculum')
        & (df.Project_Name != 'opencv') & (df.Project_Name != 'laravel')]

print(df.shape)
print(df.Project_Name.unique())

scoring = ['precision', 'recall', 'f1', 'roc_auc', 'accuracy']


# def get_classifiers():
#     return {
#         'RandomForest': RandomForestClassifier(n_jobs=4, bootstrap=True, class_weight='balanced', n_estimators=500,
#                                                max_depth=15,
#                                                random_state=42, oob_score=True, min_samples_split=7,
#                                                min_samples_leaf=3),
#         # max_depth=15, min_samples_leaf=1, n_estimators=400 Best Accuracy for 0.9111 using {'max_features': 30, 'min_samples_leaf': 4,
#         # 'n_estimators': 500}
#         'LinearSVC': LinearSVC(max_iter=2000),
#         'LogisticRegression': LogisticRegression(solver='lbfgs', n_jobs=4, multi_class='auto', max_iter=1200),
#         'XGBoost': xgb.XGBClassifier(**params),  # {'max_depth': 3, 'min_child_weight': 5}
#     }


# print(RandomForestClassifier())


params = {
    'objective': 'binary:logistic',
    'eta': 0.08,
    'colsample_bytree': 0.886,
    'min_child_weight': 1.1,
    'max_depth': 7,
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

df = df[['Comments_Count', 'Following', 'Project_Accept_Rate', 'Public_Repos', 'Additions', 'Saturday', 'Team_Size',
         'Deletions_Per_Week', 'Prev_PRs', 'Commits_PR', 'Files_Changed', 'File_Touched_Average', 'Day',
         'Organization_Core_Member', 'Deletions', 'Assignees_Count', 'Label_Count', 'Additions_Per_Week', 'Project_Age',
         'Accept_Num', 'Stars', 'Workload', 'Comments_Per_Merged_PR', 'Language', 'Last_Comment_Mention',
         'Project_Domain',
         'Project_Size', 'Merge_Latency', 'Commits_Average', 'Open_Issues', 'Tuesday', 'Watchers', 'Wednesday',
         'Closed_Num', 'Participants_Count', 'Thursday', 'Review_Comments_Count', 'Close_Latency', 'Contributor',
         'Forks_Count',
         'Contributions', 'Friday', 'Monday', 'Intra_Branch', 'Point_To_IssueOrPR', 'Contributor_Num',
         'Sunday', 'Comments_Per_Closed_PR', 'Followers', 'Wait_Time', 'Closed_Num_Rate', 'Contain_Fix_Bug',
         'User_Accept_Rate',
         'first_response', 'latency_after_first_response', 'conflict', 'title_words_count', 'body_words_count',
         'comments_reviews_words_count', 'Churn_Average', 'X1_0', 'X1_1', 'X1_2', 'X1_3', 'X1_4', 'X1_5', 'X1_6',
         'X1_7', 'X1_8', 'X1_9', 'PR_Latency', 'Project_Name', 'PR_Date_Created_At', 'PR_Time_Create_At',
         'PR_Date_Closed_At',
         'PR_Time_Closed_At', 'PR_accept']]

# df = df[['Project_Age', 'Project_Accept_Rate', 'Language', 'Watchers', 'Stars', 'Team_Size', 'Additions_Per_Week',
#      'Deletions_Per_Week', 'Comments_Per_Merged_PR', 'Contributor_Num', 'Churn_Average', 'Sunday', 'Monday',
#      'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Close_Latency', 'Comments_Per_Closed_PR', 'Forks_Count',
#      'File_Touched_Average', 'Merge_Latency', 'Rebaseable', 'Intra_Branch', #'Project_Domain',#'Mergeable',
#      'Additions', 'Deletions', 'Day', 'Wait_Time', 'Contain_Fix_Bug', 'PR_Latency', 'Files_Changed',
#      'Label_Count', 'Assignees_Count', 'Workload', 'Review_Comments_Count', 'Comments_Count',  'Commits_PR',
#      'Commits_Average', 'Contributor', 'Followers', 'Closed_Num', 'Public_Repos', 'Organization_Core_Member',
#      'Accept_Num', 'User_Accept_Rate', 'Contributions', 'Closed_Num_Rate', 'Following', 'Prev_PRs',
#      'Last_Comment_Mention', 'Point_To_IssueOrPR', 'Open_Issues',
#      'first_response', 'latency_after_first_response', 'X1_0', 'X1_1', 'X1_2', 'X1_3', 'X1_4', 'X1_5', 'X1_6', 'X1_7', 'X1_8', 'X1_9',
#       'PR_accept', 'PR_Date_Created_At', 'PR_Time_Create_At', 'PR_Date_Closed_At', 'PR_Time_Closed_At', 'Project_Name']]

# df = df[['Project_Age', 'Project_Accept_Rate', 'Language', 'Watchers', 'Stars', 'Team_Size', 'Additions_Per_Week',
#          'Deletions_Per_Week', 'Comments_Per_Merged_PR', 'Churn_Average', 'Close_Latency', 'Comments_Per_Closed_PR',
#          'Forks_Count', 'File_Touched_Average', 'Merge_Latency', 'Rebaseable', 'Additions', 'Deletions',
#          # 'Project_Domain',
#          'Wait_Time', 'PR_Latency', 'Files_Changed', 'Label_Count', 'Workload',
#          'Commits_Average', 'Contributor', 'Followers', 'Closed_Num', 'Public_Repos',
#          'Accept_Num', 'User_Accept_Rate', 'Contributions', 'Closed_Num_Rate', 'Prev_PRs',
#          'Open_Issues', 'first_response', 'latency_after_first_response', 'X1_0', 'X1_1', 'X1_2', 'X1_3', 'X1_4',
#          'X1_5',
#          'X1_6', 'X1_7', 'X1_8', 'X1_9',
#          'PR_accept', 'PR_Date_Created_At', 'PR_Time_Create_At', 'PR_Date_Closed_At', 'PR_Time_Closed_At',
#          'Project_Name']]

# df = df[['Project_Age', 'Project_Accept_Rate', 'Additions', 'Deletions',  'Commits_PR', 'Contain_Fix_Bug', 'Files_Changed',
#         'Organization_Core_Member', 'Intra_Branch', #'PR_age',
#         'User_Accept_Rate', 'Review_Comments_Count', 'Comments_Count', 'Last_Comment_Mention', 'PR_accept', 'PR_Date_Created_At',
#         'PR_Time_Create_At', 'PR_Date_Closed_At', 'PR_Time_Closed_At', 'Project_Name']]

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
# df = shuffle(df)

target = 'PR_accept'
start_date = '2017-09-01'
end_date = '2018-02-28'

X_test = df.loc[(df['PR_Date_Created_At'] >= start_date) & (df['PR_Date_Created_At'] <= end_date)]
y_test = X_test[target]
X_train = df.loc[(df['PR_Date_Created_At'] < start_date)]
y_train = X_train[target]

predictors = [x for x in df.columns if x not in [target, 'PR_Date_Created_At', 'PR_Time_Create_At', 'PR_Date_Closed_At',
                                                 'PR_Time_Closed_At', 'Project_Name']]


#
# predictors_with_target = [x for x in df.columns if x not in ['PR_Date_Created_At', 'PR_Time_Create_At', 'PR_Date_Closed_At',
#                                                  'PR_Time_Closed_At', 'Project_Name']]
#
#
X_train = X_train[predictors]
X_test = X_test[predictors]

# X = df
# y = df[target]
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, stratify=y, random_state=143)


def get_under_sampled_dataset():
    # Class count
    count_class_1, count_class_0 = X_train[target].value_counts()

    print(f'0 count: {count_class_0}, 1 count: {count_class_1}')
    # Divide by class
    df_class_0 = X_train[X_train[target] == 0]
    df_class_1 = X_train[X_train[target] == 1]
    print(f'Negative samples shape: {df_class_0.shape}')
    print(f'Positive samples shape: {df_class_1.shape}')

    df_class_1_under = df_class_1.sample(count_class_0)
    df_test_under = pd.concat([df_class_1_under, df_class_0], axis=0)

    print('Random under-sampling:')
    print(df_test_under[target].value_counts())
    shuffle(df_test_under)
    y_df = df_test_under[target]

    return df_test_under, y_df


def get_over_sampled_dataset():
    # Class count
    count_class_1, count_class_0 = X_train[target].value_counts()

    print(f'0 count: {count_class_0}, 1 count: {count_class_1}')
    # Divide by class
    df_class_0 = X_train[X_train[target] == 0]
    df_class_1 = X_train[X_train[target] == 1]
    print(f'Negative samples shape: {df_class_0.shape}')
    print(f'Positive samples shape: {df_class_1.shape}')

    df_class_0_over = df_class_0.sample(count_class_1, replace=True)
    df_test_over = pd.concat([df_class_0_over, df_class_1], axis=0)

    print('Random over-sampling:')
    print(df_test_over[target].value_counts())
    # shuffle(df_test_over)
    y_df = df_test_over[target]

    return df_test_over, y_df


def get_under_sampled_dataset_imblearn():
    # Divide by class
    df_class_0 = X_train[X_train[target] == 0]
    df_class_1 = X_train[X_train[target] == 1]
    print(f'Negative samples shape: {df_class_0.shape}')
    print(f'Positive samples shape: {df_class_1.shape}')

    rus = RandomUnderSampler(return_indices=True)
    X_rus, y_rus, id_rus = rus.fit_sample(X_train, y_train)
    shuffle(X_rus)
    print('Removed indexes:', id_rus)
    y_rus = X_rus[target]
    return X_rus, y_rus


def get_over_sampled_dataset_imblearn():
    # Divide by class
    df_class_0 = X_train[X_train[target] == 0]
    df_class_1 = X_train[X_train[target] == 1]
    print(f'Negative samples shape: {df_class_0.shape}')
    print(f'Positive samples shape: {df_class_1.shape}')

    ros = RandomOverSampler()
    X_ros, y_ros = ros.fit_sample(X_train, y_train)

    print(X_ros.shape[0] - X_train.shape[0], 'new random picked points')
    shuffle(X_ros)
    X_ros = pd.DataFrame(X_ros, columns=list(df.columns))
    print(X_ros.shape)
    y_ros = X_ros[target]

    print(y_ros.shape)

    return X_ros, y_ros


def get_tomeklinks_under_sampled_dataset():
    tl = TomekLinks(return_indices=True, ratio='majority')
    X_tl, y_tl, id_tl = tl.fit_sample(X_train, y_train)

    print('Removed indexes:', id_tl)
    shuffle(X_tl)
    y_tl = X_tl[target]

    return X_tl, y_tl


def get_clusterCentriods_under_sampled_dataset(X, y):
    cc = ClusterCentroids(ratio={0: 10})
    X_cc, y_cc = cc.fit_sample(X, y)


def get_smote_under_sampled_dataset(X, y):
    smote = SMOTE(ratio='minority')
    X_sm, y_sm = smote.fit_sample(X, y)

    X_sm = pd.DataFrame(X_sm, columns=predictors_with_target)
    shuffle(X_sm)
    y_sm = X_sm[target]

    return X_sm, y_sm


# X_train, y_train = get_over_sampled_dataset()

# X_train, y_train = get_smote_under_sampled_dataset(X_train[predictors_with_target], y_train)


# print("Total Train dataset size: {}".format(X_train[predictors].shape))
# print("Total Test dataset size: {}".format(X_test[predictors].shape))


# Scale the training dataset: StandardScaler
def scale_data_standardscaler(df_):
    scaler_train = StandardScaler()
    df_scaled = scaler_train.fit_transform(np.array(df_).astype('float64'))
    df_scaled = pd.DataFrame(df_scaled, columns=predictors)

    return df_scaled


# X_train_scaled = scale_data_standardscaler(X_train[predictors])
# X_test_scaled = scale_data_standardscaler(X_test[predictors])


def train_XGB_model(clf, x_train, y_train, x_test):
    clf = clf.fit(x_train, y_train, verbose=11)

    # Fit the algorithm on the data
    # clf.fit(x_train, y_train, eval_metric='auc')

    # Save the model
    # with open('response_xgb_16.pickle.dat', 'wb') as f:
    #     pickle.dump(alg, f)

    # Load the model
    # with open('response_xgb_16.pickle.dat', 'rb') as f:
    #     load_xgb = pickle.load(f)

    y_pred_train = clf.predict(x_train)
    y_predprob_train = clf.predict_proba(x_train)[:, 1]

    y_pred = clf.predict(x_test)
    y_predprob = clf.predict_proba(x_test)[:, 1]

    return y_pred_train, y_predprob_train, y_pred, y_predprob


def train_SVM_model(clf, x_train, y_train, x_test):
    clf.fit(x_train, y_train)
    svm = CalibratedClassifierCV(base_estimator=clf, cv='prefit')
    svm.fit(x_train, y_train)
    # train
    y_pred_train = svm.predict(x_train)
    y_predprob_train = svm.predict_proba(x_train)[:, 1]
    # test
    y_pred = svm.predict(x_test)
    y_predprob = svm.predict_proba(x_test)[:, 1]

    return y_pred_train, y_predprob_train, y_pred, y_predprob


def train_RF_LR_model(clf, x_train, y_train, x_test):
    clf.fit(x_train, y_train)
    # train
    y_pred_train = clf.predict(x_train)
    y_predprob_train = clf.predict_proba(x_train)[:, 1]
    # test
    y_pred = clf.predict(x_test)
    y_predprob = clf.predict_proba(x_test)[:, 1]

    return y_pred_train, y_predprob_train, y_pred, y_predprob


def start_train_models():
    results = pd.DataFrame(
        columns=['Model', 'AUC', 'Precision', 'Recall', 'F-measure', 'Test_Accuracy', 'Train_Accuracy'])
    classifiers = get_classifiers()

    for name, value in classifiers.items():
        clf = value
        print('Classifer: ', name)
        if name == 'XGBoost':
            y_pred_train, y_predprob_train, y_pred, y_predprob = train_XGB_model(clf, X_train[predictors], y_train,
                                                                                 X_test[predictors])
        elif name == 'LinearSVC':
            y_pred_train, y_predprob_train, y_pred, y_predprob = train_SVM_model(clf, X_train_scaled, y_train,
                                                                                 X_test_scaled)
        elif name == 'LogisticRegression':
            y_pred_train, y_predprob_train, y_pred, y_predprob = train_RF_LR_model(clf, X_train_scaled, y_train,
                                                                                   X_test_scaled)
        else:
            y_pred_train, y_predprob_train, y_pred, y_predprob = train_RF_LR_model(clf, X_train[predictors], y_train,
                                                                                   X_test[predictors])

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
             'Precision': metrics.precision_score(y_test, y_pred),
             'Recall': metrics.recall_score(y_test, y_pred),
             'F-measure': metrics.f1_score(y_test, y_pred),
             'Test_Accuracy': metrics.accuracy_score(y_test, y_pred),
             'Train_Accuracy': metrics.accuracy_score(y_train, y_pred_train)},
            ignore_index=True)

    # results.to_csv('Results/accept_OS_N2.csv', sep=',', encoding='utf-8', index=False)


# start_train_models()


def start_10_fold_validation(df_):
    results = pd.DataFrame(
        columns=['Model', 'AUC', 'Precision', 'Recall', 'F-measure', 'Test_Accuracy', 'Train_Accuracy'])
    df = df_.sort_values(by=['PR_Date_Closed_At', 'PR_Time_Closed_At'], ascending=True)
    df_split = np.array_split(df, 10)
    print(df.shape)
    for index in range(len(df_split) - 1):
        train = pd.DataFrame()
        for i in range(index + 1):
            train = train.append(df_split[i])

        print(f"Train dataset shape: {train.shape}")
        test = df_split[index + 1]
        print(f"Test dataset shape: {test.shape}")

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
                 'Precision': metrics.precision_score(y_test, y_pred),
                 'Recall': metrics.recall_score(y_test, y_pred),
                 'F-measure': metrics.f1_score(y_test, y_pred),
                 'Test_Accuracy': metrics.accuracy_score(y_test, y_pred),
                 'Train_Accuracy': metrics.accuracy_score(y_train, y_pred_train)},
                ignore_index=True)

    avg_result = pd.DataFrame(
        columns=['Model', 'AUC', 'Precision', 'Recall', 'F-measure', 'Test_Accuracy', 'Train_Accuracy'])
    for name, value in classifiers.items():
        model_result = results.loc[results.Model == name]
        avg_result = avg_result.append(
            {'Model': name, 'AUC': model_result['AUC'].mean(),
             'Precision': model_result['Precision'].mean(),
             'Recall': model_result['Recall'].mean(),
             'F-measure': model_result['F-measure'].mean(),
             'Test_Accuracy': model_result['Test_Accuracy'].mean(),
             'Train_Accuracy': model_result['Train_Accuracy'].mean()},
            ignore_index=True)
    avg_result.to_csv('Results/accept_10_fold_4_avg.csv', sep=',', encoding='utf-8', index=False)
    results.to_csv('Results/accept_10_fold_4.csv', sep=',', encoding='utf-8', index=False)


# start_10_fold_validation(df)

parameters = {
    # 'RandomForest': {'max_depth': range(3, 15, 1),
    #                  'n_estimators': [200, 300, 400, 500], #'max_features': [5, 10, 20, 25, 30],
    # 'min_samples_leaf': [1, 2, 3, 4, 5]
    # },
    # 'LinearSVC': {'loss': ['hinge', 'squared_hinge'], 'C': [0.001, 0.01, 0.1, 1, 10]},
    # 'LogisticRegression': {'C': [0.001, 0.01, 0.1, 1, 10]},
    'XGBoost': {'max_depth': range(3, 10, 1), 'min_child_weight': range(1, 6, 1),
                # 'learning_rate':[i/100.0 for i in range(1,10)],
                # 'subsample':[i/10.0 for i in range(6,10)], 'colsample_bytree':[i/10.0 for i in range(6,10)],
                # 'reg_alpha':[1e-5, 1e-2, 0.1, 1, 100]
                }  # max_depth, min_child_weight , gamma, subsample these parameters are used to control overfitting
}


def model_optimizer():
    classifiers = get_classifiers()
    df_PO = pd.DataFrame(columns=['Classifier', 'Best_Score', 'Best_Params'])
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
        print(f'Best Accuracy for {grid_result.best_score_:.4} using {grid_result.best_params_}')
        means = grid_result.cv_results_['mean_test_score']
        stds = grid_result.cv_results_['std_test_score']
        params = grid_result.cv_results_['params']
        for mean, stdev, param in zip(means, stds, params):
            print(f'mean={mean:.4}, std={stdev:.4} using {param}')

        df_PO = df_PO.append({'Classifier': name, 'Best_Score': grid.best_score_, 'Best_Params': grid.best_params_},
                             ignore_index=True)
    df_PO.to_csv('Results/Optimization/RF_4.csv', sep=',', encoding='utf-8', index=False)

    # Best Accuracy for 0.8538 using {'max_depth': 15, 'min_samples_leaf': 1, 'n_estimators': 400} RF
    # Best Accuracy for 0.7692 using {'eta': 0.01, 'max_depth': 9, 'min_child_weight': 1}
    # Best Accuracy for 0.791 using {'learning_rate': 0.09}
    # Best Accuracy for 0.9071 using {'max_features': 'sqrt', 'min_samples_leaf': 10, 'n_estimators': 500} RF
    # Best Accuracy for 0.9083 using {'max_features': 'auto', 'min_samples_leaf': 2, 'n_estimators': 500} RF
    # Best Accuracy for 0.9111 using {'max_features': 30, 'min_samples_leaf': 4, 'n_estimators': 500}


def model_optimier_2():
    for para in range(9, 25, 1):
        print("Max_depth value {}".format(para))
        clf = xgb.XGBClassifier(max_depth=para)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        y_predprob = clf.predict_proba(X_test)[:, 1]

        y_pred_train = clf.predict(X_train)
        y_predprob_train = clf.predict_proba(X_train)[:, 1]

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


# model_optimizer()

model_optimier_2()


def feature_selection_LR():
    from sklearn.feature_selection import RFE

    rfe_selector = RFE(estimator=RandomForestClassifier(), n_features_to_select=30, step=5, verbose=5)
    rfe_selector.fit(X_train_scaled, y_train)

    y_pred = rfe_selector.predict(X_test_scaled)
    y_predprob = rfe_selector.predict_proba(X_test_scaled)[:, 1]

    rfe_support = rfe_selector.get_support()
    rfe_feature = X_train[predictors].loc[:, rfe_support].columns.tolist()
    print(str(len(rfe_feature)), 'selected features')
    print('RFE features')
    print(rfe_feature)
    # Print model report:
    print("\nModel Report")
    # print("Train Accuracy : %.4g" % metrics.accuracy_score(y_train, y_pred_train))
    print("Test Accuracy : %.4g" % metrics.accuracy_score(y_test, y_pred))
    # print('Train error: {:.3f}'.format(1 - metrics.accuracy_score(y_train, y_pred_train)))
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

# feature_selection_LR()


