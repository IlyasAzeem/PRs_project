import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
plt.rc("font", size=14)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn import metrics
from statistics import mean
import xgboost as xgb
import pickle
from sklearn.utils import shuffle
import seaborn as sns
sns.set(style="white")
sns.set(style='whitegrid', color_codes=True)

df = pd.read_csv("/home/ppf/Ilyas_dataset/accept_21_features.csv", sep=",", encoding="ISO-8859-1")


# 'cmssw'
project_list = ['react', 'django', 'nixpkgs', 'scikit-learn', 'yii2', 'cdnjs', 'terraform', 'cmssw', 'salt', 'tensorflow', 'pandas',
                'symfony', 'moby', 'rails', 'rust', 'kubernetes'#, 'angular.js', 'laravel', 'curriculum', 'opencv', 'githubschool'
                 ]

# Handle missing values
# df.loc[(df.Forks_Count == 0) & (df.Project_Name == 'cdnjs'), 'Forks_Count'] = 3384
# df.loc[(df.Forks_Count == 0) & (df.Project_Name == 'terraform'), 'Forks_Count'] = 3659
# df.loc[(df.Forks_Count == 0) & (df.Project_Name == 'kubernetes'), 'Forks_Count'] = 11639

print(df.Timeline[df['PR_Latency']==0])


# Remove some of the PRs with negative latency
ID = df.Pull_Request_ID[df.latency_after_first_response < 0]
df = df.loc[~df.Pull_Request_ID.isin(ID)]


scoring = ['precision', 'recall', 'f1', 'roc_auc', 'accuracy']
#results = pd.DataFrame(columns=['Model', 'AUC', 'Precision', 'Recall', 'F-measure', 'Test_Accuracy', 'Train_Accuracy'])


def get_classifiers():
    return {
        'Randomforest': RandomForestClassifier(n_jobs=4, bootstrap=False, class_weight='balanced'), # n_jobs=4, bootstrap=False, class_weight='balanced', max_depth=15, min_samples_leaf=1, n_estimators=400
        'LinearSVC': LinearSVC(max_iter=2500),
        'LogisticRegression': LogisticRegression(solver='lbfgs', n_jobs=4, multi_class='auto', max_iter=1200),
        'XGBoost':xgb.XGBClassifier(**params),
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
    encoder = preprocessing.LabelEncoder()
    df1[column_name] = [str(label) for label in df1[column_name]]
    encoder.fit(df1[column_name])
    one_hot_vector = encoder.transform(df1[column_name])
    return one_hot_vector


# print(df.columns)
# print(df.shape)

df['Language'] = encode_labels(df, 'Language')
# df_proj['Maintenance_And_Evolution_Category'] = encode_labels(df_proj, 'Maintenance_And_Evolution_Category')
# df_proj['Title_Body_Sentiment'] = encode_labels(df_proj, 'Title_Body_Sentiment')
df['Project_Domain'] = encode_labels(df, 'Project_Domain')

# df = df[['Project_Age', 'Project_Accept_Rate', 'Language', 'Watchers', 'Stars', 'Team_Size', 'Additions_Per_Week',
#      'Deletions_Per_Week', 'Comments_Per_Merged_PR', 'Contributor_Num', 'Churn_Average', 'Sunday', 'Monday',
#      'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Close_Latency', 'Comments_Per_Closed_PR', 'Forks_Count',
#      'File_Touched_Average', 'Merge_Latency', 'Rebaseable', 'Intra_Branch', #'Project_Domain',#'Mergeable',
#      'Additions', 'Deletions', 'Day', 'Wait_Time', 'Contain_Fix_Bug', 'PR_Latency', 'Files_Changed',
#      'Label_Count', 'Assignees_Count', 'Workload', 'Review_Comments_Count', 'Comments_Count',  'Commits_PR',
#      'Commits_Average', 'Contributor', 'Followers', 'Closed_Num', 'Public_Repos', 'Organization_Core_Member',
#      'Accept_Num', 'User_Accept_Rate', 'Contributions', 'Closed_Num_Rate', 'Following', 'Prev_PRs',
#      'Last_Comment_Mention', 'Point_To_IssueOrPR', 'Open_Issues', 'Participants_Count',
#      'first_response', 'latency_after_first_response', 'X1_0', 'X1_1', 'X1_2', 'X1_3', 'X1_4', 'X1_5', 'X1_6', 'X1_7', 'X1_8', 'X1_9',
#       'PR_accept', 'PR_Date_Created_At', 'PR_Time_Create_At', 'PR_Date_Closed_At', 'PR_Time_Closed_At', 'Project_Name']]

df = df[['Comments_Count', 'Following', 'Project_Accept_Rate', 'Public_Repos', 'Additions', 'Saturday', 'Team_Size',
       'Deletions_Per_Week', 'Prev_PRs', 'Commits_PR', 'Files_Changed', 'File_Touched_Average', 'Day',
       'Organization_Core_Member',  'Deletions', 'Assignees_Count', 'Label_Count', 'Additions_Per_Week', 'Project_Age',
        'Accept_Num', 'Stars', 'Workload', 'Comments_Per_Merged_PR', 'Language', 'Last_Comment_Mention', 'Project_Domain',
        'Project_Size', 'Merge_Latency', 'Rebaseable', 'Commits_Average','Open_Issues', 'Tuesday', 'Watchers', 'Wednesday',
        'Closed_Num', 'Participants_Count', 'Thursday', 'Review_Comments_Count', 'Close_Latency', 'Contributor', 'Forks_Count',
         'Contributions', 'Friday', 'Monday', 'Intra_Branch', 'Point_To_IssueOrPR', 'Mergeable_State', 'Contributor_Num',
       'Sunday', 'Comments_Per_Closed_PR', 'Followers', 'Wait_Time', 'Closed_Num_Rate', 'Contain_Fix_Bug', 'User_Accept_Rate',
        'first_response', 'latency_after_first_response', 'conflict', 'title_words_count', 'body_words_count',
        'comments_reviews_words_count', 'Churn_Average', 'X1_0', 'X1_1', 'X1_2', 'X1_3', 'X1_4', 'X1_5', 'X1_6',
       'X1_7', 'X1_8', 'X1_9', 'PR_Latency', 'Project_Name', 'PR_Date_Created_At', 'PR_Time_Create_At', 'PR_Date_Closed_At',
       'PR_Time_Closed_At', 'Pull_Request_ID', 'PR_accept']]




# shuffle the data
df = shuffle(df)

target = 'PR_accept'

predictors = [x for x in df.columns if x not in [target, 'PR_Date_Created_At', 'PR_Time_Create_At', 'PR_Date_Closed_At',
                                                 'PR_Time_Closed_At', 'Project_Name']]

X = df
y= df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, stratify=y, random_state=143)

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
    shuffle(df_test_over)
    y_df = df_test_over[target]

    return df_test_over, y_df


# X_train, y_train = get_over_sampled_dataset()


print("Total Train dataset size: {}".format(X_train.shape))
print("Total Test dataset size: {}".format(X_test.shape))


# Scale the training dataset: StandardScaler
def scale_data_standardscaler(df_):
    scaler_train =StandardScaler()
    df_scaled = scaler_train.fit_transform(np.array(df_).astype('float64'))
    df_scaled = pd.DataFrame(df_scaled, columns=predictors)

    return df_scaled

X_train_scaled = scale_data_standardscaler(X_train[predictors])
X_test_scaled = scale_data_standardscaler(X_test[predictors])
#X_val_scaled = scale_data_standardscaler(X_val[predictors])


def train_XGB_model(alg, x_train, y_train, x_test, cv_folds=3, early_stopping_rounds=50):

    xgb_param = alg.get_xgb_params()
    xgtrain = xgb.DMatrix(x_train, label=y_train)
    cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds, stratified=True,
                      metrics='auc', early_stopping_rounds=early_stopping_rounds)
    alg.set_params(n_estimators=cvresult.shape[0])

    # Fit the algorithm on the data
    alg.fit(x_train, y_train, eval_metric='auc')

    # Save the model
    # with open('response_xgb_16.pickle.dat', 'wb') as f:
    #     pickle.dump(alg, f)

    # Load the model
    # with open('response_xgb_16.pickle.dat', 'rb') as f:
    #     load_xgb = pickle.load(f)

    y_pred_train = alg.predict(x_train)
    y_predprob_train = alg.predict_proba(x_train)[:, 1]

    y_pred = alg.predict(x_test)
    y_predprob = alg.predict_proba(x_test)[:, 1]

    return y_pred_train, y_predprob_train, y_pred, y_predprob


def train_SVM_model(clf, x_train, y_train, x_test, cv_folds=3):
    svm = CalibratedClassifierCV(clf, cv=cv_folds)
    svm.fit(x_train, y_train)
    # train
    y_pred_train = svm.predict(x_train)
    y_predprob_train = svm.predict_proba(x_train)[:, 1]
    # test
    y_pred = svm.predict(x_test)
    y_predprob = svm.predict_proba(x_test)[:, 1]

    return y_pred_train, y_predprob_train, y_pred, y_predprob

def train_RF_LR_model(clf, x_train, y_train, x_test, cv_folds=3):
    cv_results = cross_validate(clf, x_train, y_train, cv=cv_folds)
    print('Validation scores')
    print("Test accuracy: {}".format(cv_results['test_score'].mean()))
    print("Train accuracy: {}".format(cv_results['train_score'].mean()))
    clf.fit(x_train, y_train)
    # train
    y_pred_train = clf.predict(x_train)
    y_predprob_train = clf.predict_proba(x_train)[:, 1]
    # test
    y_pred = clf.predict(x_test)
    y_predprob = clf.predict_proba(x_test)[:, 1]

    # Save the model
    # with open('prioritizer.pickle.dat', 'wb') as f:
    #     pickle.dump(clf, f)

    # Load the model
    # with open('response_xgb_16.pickle.dat', 'rb') as f:
    #     load_xgb = pickle.load(f)

    return y_pred_train, y_predprob_train, y_pred, y_predprob




def start_train_models():
    results = pd.DataFrame(
        columns=['Model', 'AUC', 'Precision', 'Recall', 'F-measure', 'Test_Accuracy', 'Train_Accuracy'])
    classifiers = get_classifiers()

    for name, value in classifiers.items():
            clf = value
            print('Classifer: ', name)
            if name == 'XGBoost':
                y_pred_train, y_predprob_train, y_pred, y_predprob = train_XGB_model(clf, X_train_scaled, y_train, X_test_scaled, cv_folds=10)
            elif name == 'LinearSVC':
                y_pred_train, y_predprob_train, y_pred, y_predprob = train_SVM_model(clf, X_train_scaled, y_train, X_test_scaled, cv_folds=10)
            elif name == 'LogisticRegression':
                y_pred_train, y_predprob_train, y_pred, y_predprob = train_RF_LR_model(clf, X_train_scaled, y_train, X_test_scaled, cv_folds=10)
            else:
                y_pred_train, y_predprob_train, y_pred, y_predprob = train_RF_LR_model(clf, X_train_scaled, y_train, X_test_scaled, cv_folds=10)

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
            results= results.append(
                {'Model': name, 'AUC': metrics.roc_auc_score(y_test, y_predprob),
                 'Precision': metrics.precision_score(y_test, y_pred),
                 'Recall': metrics.recall_score(y_test, y_pred),
                 'F-measure': metrics.f1_score(y_test, y_pred),
                 'Test_Accuracy': metrics.accuracy_score(y_test, y_pred),
                 'Train_Accuracy': metrics.accuracy_score(y_train, y_pred_train)},
                ignore_index=True)


    results.to_csv('Results/accept_10_fold_3.csv', sep=',', encoding='utf-8', index=False)


# start_train_models()

