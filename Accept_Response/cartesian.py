import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
plt.rc("font", size=14)
from datetime import datetime
import pickle
import math
from os import listdir
from os.path import isfile, join
from sklearn.utils import shuffle
from sklearn import metrics
import os.path
from sklearn.dummy import DummyClassifier
from sklearn.metrics import precision_recall_fscore_support as score


# df = pd.read_csv("E:\\Documents\\Research Work\\datasets\\test_dataset_1.csv", sep=',', encoding='utf-8')
df = pd.read_csv("E:\\Documents\\Research Work\\datasets\\response_21_features_2.csv", sep=',', encoding='utf-8')

df = df[(df.Project_Name != 'githubschool') & (df.Project_Name != 'curriculum')]

# print(df.shape)
# print('Accept label')
# print(df.PR_accept.value_counts())
# print('Response label')
# print(df.PR_response.value_counts())

project_list = ['react', 'django', 'nixpkgs', 'scikit-learn', 'yii2', 'cdnjs', 'terraform', 'cmssw', 'salt',
                'tensorflow', 'pandas', 'symfony', 'moby', 'rails', 'rust', 'kubernetes', 'angular.js', 'laravel',
                'opencv']

folder_path = 'Results/CART/'

# One hot encoding
def encode_labels(df1, column_name):
    encoder = preprocessing.LabelEncoder()
    df1[column_name] = [str(label) for label in df1[column_name]]
    encoder.fit(df1[column_name])
    one_hot_vector = encoder.transform(df1[column_name])
    return one_hot_vector


df['Language'] = encode_labels(df, 'Language')
df['Project_Domain'] = encode_labels(df, 'Project_Domain')


df['src_churn'] = df['Additions'] + df['Deletions']
df['num_comments'] = df['Review_Comments_Count'] + df['Comments_Count']


predictors_a = ['Contributor', 'Project_Accept_Rate', 'User_Accept_Rate', 'Deletions_Per_Week', 'Wait_Time', 'Stars',
                'Additions_Per_Week', 'Saturday', 'latency_after_first_response', 'Assignees_Count', 'Wednesday',
                'comments_reviews_words_count', 'PR_Latency', 'Project_Domain', 'Team_Size', 'Language', 'Commits_PR',
                'Label_Count', 'Contributions', 'num_comments', 'Last_Comment_Mention', 'Comments_Per_Merged_PR',
                'Contributor_Num', 'Watchers', 'first_response', 'Files_Changed', 'Comments_Per_Closed_PR', 'Monday',
                'line_comments_count', 'Open_Issues', 'Sunday']

predictors_r = ['Day', 'Comments_Per_Closed_PR', 'Team_Size', 'Deletions_Per_Week', 'Wait_Time', 'Sunday',
                'Contributor_Num', 'Assignees_Count', 'first_response_time', 'num_comments', 'Label_Count',
                'PR_age']


predictors_prioritizer = ['PR_age', 'Project_Accept_Rate', 'Additions', 'Deletions',  'Commits_PR', 'Contain_Fix_Bug',
                          'Files_Changed', 'Organization_Core_Member', 'Intra_Branch',  'User_Accept_Rate',
                          'Review_Comments_Count', 'Comments_Count', 'Last_Comment_Mention',
                          ]

predictors_gousios = ['src_churn', 'Commits_PR', 'Files_Changed', 'num_comments', 'Participants_Count', 'conflict',
                      'Team_Size', 'Project_Size', 'File_Touched_Average', 'Commits_Average', 'Prev_PRs',
                      'User_Accept_Rate']

start_date = '2017-09-01'
end_date = '2018-02-28'

target = 'label'

X_test = df.loc[(df['PR_Date_Created_At'] >= start_date) & (df['PR_Date_Created_At'] <= end_date)]

X_test = X_test.loc[(X_test.Comments_Count == 0) & (X_test.Review_Comments_Count == 0) &
                              (X_test.first_response == 0) & (X_test.PR_accept == 1)]



def Two_Models_CART(df_test_PR, folder_path, file_name):
    pd.options.mode.chained_assignment = None
    with open('Results/accept/models/xgb_accept_FS.pickle.dat', 'rb') as f:
        accept_model = pickle.load(f)
        pred_accept = accept_model.predict(df_test_PR[predictors_a])
        pred_accept_prob = accept_model.predict_proba(df_test_PR[predictors_a])[:,1]
        df_test_PR['Result_Accept'] = pred_accept
        df_test_PR['Result_Accept_Prob'] = pred_accept_prob

    # print(df_test_PR[['Pull_Request_ID', 'PR_accept', 'Result_Accept']].head(10))

    with open('Results/response/models/xgb_response_FS.pickle.dat', 'rb') as f:
        response_model = pickle.load(f)
        pred_response = response_model.predict(df_test_PR[predictors_r])
        pred_response_prob = response_model.predict_proba(df_test_PR[predictors_r])[:,1]
        df_test_PR['Result_Response'] = pred_response
        df_test_PR['Result_Response_Prob'] = pred_response_prob

    # print(df_test_PR[['Pull_Request_ID', 'PR_response', 'Result_Response']].head(10))

    df_test_PR['Score'] = df_test_PR['Result_Accept_Prob'].apply(np.exp) + df_test_PR['Result_Response_Prob'].apply(np.exp)
    # result = [math.exp(y_pred_accept[index]) + math.exp(y_pred_response[index]) for index in range(len(y_pred_accept))]
    # result = [math.exp(y_pred_accept_prob[index]) + math.exp(y_pred_response_prob[index]) for index in range(len(y_pred_accept))]
    # result =[(y_pred_accept[index] + y_pred_response[index])/2 for index in range(len(y_pred_accept))]
    # result = {'result': result}
    # result = pd.DataFrame(result)
    # df_test_PR['Score_2'] = result
    print(df_test_PR[['Pull_Request_ID', 'Result_Accept', 'Result_Response', 'Score']].head(10))
    df_test_PR.sort_values(by=['Score'], ascending=False).to_csv(
        folder_path+file_name+'.csv', sep=',', encoding='utf-8', index=False)
    return df_test_PR.sort_values(by=['Score'], ascending=False)


def Accept_Response_Model(df_test_PR, folder_path):
    pd.options.mode.chained_assignment = None
    with open('../Accept/Models_04_06_19/Saved_Models/Models_04_06_19/accept_XGB.pickle.dat', 'rb') as f:
        accept_model = pickle.load(f)
        y_pred_accept = accept_model.predict(df_test_PR[predictors_a])
        # y_pred_accept = accept_model.predict_proba(df_test_PR[predictors_a])[:,1]
        df_test_PR['PR_accept_or_not'] = y_pred_accept
    predictors_r = ['Project_Age', 'Project_Accept_Rate', 'Language', 'Watchers', 'Stars', 'Team_Size',
                    'Additions_Per_Week',
                    'Deletions_Per_Week', 'Comments_Per_Merged_PR', 'Contributor_Num', 'Churn_Average', 'Sunday',
                    'Monday',
                    'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Close_Latency', 'Comments_Per_Closed_PR',
                    'Forks_Count', 'File_Touched_Average', 'Merge_Latency', 'Rebaseable', 'Intra_Branch',
                    'Project_Domain',
                    'Additions', 'Deletions', 'Day', 'Commits_PR', 'Wait_Time', 'Contain_Fix_Bug', 'PR_Latency',
                    'Files_Changed',
                    'Label_Count', 'Assignees_Count', 'Workload', 'PR_age', 'PR_accept_or_not','Commits_Average', 'Contributor',
                    'Followers',
                    'Closed_Num', 'Public_Repos', 'Organization_Core_Member', 'Accept_Num', 'User_Accept_Rate',
                    'Contributions',
                    'Closed_Num_Rate', 'Following', 'Prev_PRs', 'Review_Comments_Count', 'Participants_Count',
                    'Comments_Count',
                    'Last_Comment_Mention', 'Point_To_IssueOrPR', 'Open_Issues', 'first_response',
                    'latency_after_first_response',
                    'X1_0', 'X1_1', 'X1_2', 'X1_3', 'X1_4', 'X1_5', 'X1_6', 'X1_7', 'X1_8', 'X1_9']
    # print(df_test_PR[['Pull_Request_ID', 'PR_accept', 'Result_Accept']].head(10))

    with open('../Response/Models_04_06_19/Saved_Models/Models_DS_4/XGB_accept_response.pickle.dat', 'rb') as f:
        response_model = pickle.load(f)
        # y_pred_response = response_model.predict(df_test_PR[predictors_r])
        y_pred_response = response_model.predict_proba(df_test_PR[predictors_r])[:,1]
        df_test_PR['Score'] = y_pred_response

    # print(df_test_PR[['Pull_Request_ID', 'PR_response', 'Result_Response']].head(10))

    print(df_test_PR[['Pull_Request_ID', 'PR_accept', 'PR_response', 'PR_accept_or_not', 'Score']].head(10))
    # print(result['result'])
    df_test_PR.to_csv(
        algo_folder_path+'accept_resp_results.csv', sep=',', encoding='utf-8', index=False)
    return df_test_PR


def Gousios_Model(df_test_PR, folder_path):
    pd.options.mode.chained_assignment = None
    with open('Results/accept/models/accept_baseline.pickle.dat', 'rb') as f:
        accept_model = pickle.load(f)
        # y_pred_accept = accept_model.predict(df_test_PR[predictors_a])
        y_pred_accept_prob = accept_model.predict_proba(df_test_PR[predictors_gousios])[:,1]
        df_test_PR['accept_score'] = y_pred_accept_prob

    print(df_test_PR[['Pull_Request_ID', 'PR_accept', 'accept_score']].head(10))
    # df_test_PR[['Pull_Request_ID', 'accept_score', 'PR_accept', 'PR_response']].sort_values(
    #     by=['accept_score'], ascending=False).to_csv(folder_path+'Gousios_results.csv', sep=',', encoding='utf-8', index=False)
    return df_test_PR


def PRioritizer_Model(df_test_PR, folder_path):
    pd.options.mode.chained_assignment = None
    with open('Results/response/models/response_baseline.pickle.dat', 'rb') as f:
        response_model = pickle.load(f)
        y_pred_response = response_model.predict(df_test_PR[predictors_prioritizer])
        # y_pred_response_prob = response_model.predict_proba(df_test_PR[predictors_prioritizer])[:,1]
        df_test_PR['response_score'] = y_pred_response

    print(df_test_PR[['Pull_Request_ID', 'PR_response', 'response_score']].head(10))

    # print(df_test_PR['Score'].head(100))
    # df_test_PR[['Pull_Request_ID', 'response_score', 'PR_accept', 'PR_response']].sort_values(by=['response_score'],
    #                                                                                           ascending=False).to_csv(
    #     folder_path+'prioritizer_results.csv', sep=',', encoding='utf-8', index=False)
    return df_test_PR


def get_top_n_MAP_all(df, folder_path, file_name):
    months_tested = {'2017-09-01': '2017-09-30', '2017-10-01': '2017-10-31',
                     '2017-11-01': '2017-11-30',
                     '2017-12-01': '2017-12-31',
                     '2018-01-01': '2018-01-31', '2018-02-01': '2018-02-28',  #'2018-03-01':'2018-03-31'
                     }
    top_k_list = [5, 10, 20]
    df = df.set_index('Project_Name')
    df_algo_results = pd.DataFrame(columns=['Project', 'MAP_5'])
    for top_k in top_k_list:
        print("Now calculating MAP for {}".format(top_k))
        df_MAP = pd.DataFrame(columns=['MAP_'+str(top_k)])
        for project in project_list:
            total_result = []
            df_project = df.loc[[project]]
            for date_start, date_end in months_tested.items():
                df_month = df_project.loc[
                    (df_project['PR_Date_Created_At'] >= date_start) & (df_project['PR_Date_Created_At'] <= date_end)]
                print(project)
                # print(date_start, date_end)
                # total_result = []
                last_date = datetime.strptime(date_end, "%Y-%m-%d").day
                # print(last_date)
                for i in range(1, last_date+1):
                    MAP_accept_response = 0
                    positive_count_accept_response = 0
                    counter = 0
                    for index, row in df_month.iterrows():
                        if datetime.strptime(row['PR_Date_Created_At'], "%Y-%m-%d").day == i:
                            counter += 1
                            if counter >top_k: break
                            if row['PR_accept'] == 1 and row['PR_response'] == 1:
                            #if row['PR_response'] == 1: # For baseline model
                                positive_count_accept_response += 1
                                MAP_accept_response += positive_count_accept_response/(counter)

                    total_result.append(MAP_accept_response/positive_count_accept_response if positive_count_accept_response !=0 else 0)
                    MAP_day = MAP_accept_response/positive_count_accept_response if positive_count_accept_response !=0 else 0
                # print('Project {0} in month {1} have MAP {2:.3f}'.format(project, date_start.split('-')[0]+'-'+date_start.split('-')[1],
                #                                                          MAP_day))
            if top_k > 5:
                df_MAP = df_MAP.append({'MAP_'+str(top_k): np.mean(total_result)}, ignore_index=True)
                print(df_MAP.shape)
            else:
                df_algo_results = df_algo_results.append(
                    {'Project': project, 'MAP_' + str(top_k): np.mean(total_result)}, ignore_index=True)
                print(df_algo_results.shape)
        if top_k > 5:
            df_algo_results = pd.concat([df_algo_results, df_MAP['MAP_'+str(top_k)]], axis=1)
    print(df_algo_results.head())
    df_algo_results.sort_values(by=['MAP_5', 'MAP_10', 'MAP_20'], ascending=False).to_csv(
        folder_path+file_name+'.csv', sep=',', encoding='utf-8', index=False)


def get_top_n_recall_all(df, folder_path, file_name):
    months_tested = {'2017-09-01': '2017-09-30', '2017-10-01': '2017-10-31',
                     '2017-11-01': '2017-11-30',
                     '2017-12-01': '2017-12-31',
                     '2018-01-01': '2018-01-31', '2018-02-01': '2018-02-28',  #'2018-03-01':'2018-03-31'
                     }
    df = df.set_index('Project_Name')
    top_k_list = [5, 10, 20]
    df_algo_results = pd.DataFrame(columns=['Project', 'AR_5'])
    for top_k in top_k_list:
        print("Now calculating AR for {}".format(top_k))
        df_AR = pd.DataFrame(columns=['AR_' + str(top_k)])
        for project in project_list:
            total_result = []
            df_project = df.loc[[project]]
            for date_start, date_end in months_tested.items():
                df_month = df_project.loc[
                    (df_project['PR_Date_Created_At'] >= date_start) & (df_project['PR_Date_Created_At'] <= date_end)]
                print(project)
                # total_result = []
                # print(df_month.shape)
                # print(date_start, "-", date_end)
                last_date = datetime.strptime(date_end, "%Y-%m-%d").day
                # print(last_date)
                for i in range(1, last_date+1):
                    top_recall_num = 0
                    total_recall_num = 0
                    counter = 0
                    for index, row in df_month.iterrows():
                        if datetime.strptime(row['PR_Date_Created_At'], "%Y-%m-%d").day == i:
                            counter += 1
                            if row['PR_accept'] == 1 and row['PR_response'] == 1:
                            # if row['PR_response'] == 1: # for baseline model
                                if counter <= top_k:
                                    top_recall_num += 1
                                total_recall_num += 1
                    total_result.append(top_recall_num/total_recall_num if total_recall_num !=0 else 0)
                    AR_day = top_recall_num/total_recall_num if total_recall_num !=0 else 0
                print('Project {} in month {} have average recall {}'.format(
                    project, date_start.split('-')[0] + '-' + date_start.split('-')[1], AR_day))

            if top_k > 5:
                df_AR = df_AR.append({'AR_' + str(top_k): np.mean(total_result)}, ignore_index=True)
            else:
                df_algo_results = df_algo_results.append(
                    {'Project': project, 'AR_' + str(top_k): np.mean(total_result)}, ignore_index=True)
        if top_k > 5:
            df_algo_results = pd.concat([df_algo_results, df_AR['AR_' + str(top_k)]], axis=1)
    # print(df_algo_results.head())
    df_algo_results.sort_values(by=['AR_5', 'AR_10', 'AR_20'], ascending=False).to_csv(
        folder_path+file_name+'.csv', sep=',', encoding='utf-8', index=False)


def get_top_n_MAP(df, folder_path):
    months_tested = {'2017-09-01': '2017-09-30', '2017-10-01': '2017-10-31',
                     '2017-11-01': '2017-11-30',
                     '2017-12-01': '2017-12-31',
                     '2018-01-01': '2018-01-31', #'2018-02-01': '2018-02-28',  #'2018-03-01':'2018-03-31'
                     }
    top_k_list = [5, 10, 20]
    df = df.set_index('Project_Name')
    df_algo_results = pd.DataFrame(columns=['Project', 'Year-Month', 'MAP_5'])
    for top_k in top_k_list:
        print("Now calculating MAP for {}".format(top_k))
        df_MAP = pd.DataFrame(columns=['MAP_'+str(top_k)])
        for project in project_list:
            # total_result = []
            df_project = df.loc[[project]]
            for date_start, date_end in months_tested.items():
                df_month = df_project.loc[
                    (df_project['PR_Date_Created_At'] >= date_start) & (df_project['PR_Date_Created_At'] <= date_end)]
                print(project)
                # print(date_start, date_end)
                total_result = []
                last_date = datetime.strptime(date_end, "%Y-%m-%d").day
                print(last_date)
                for i in range(1, last_date+1):
                    MAP_accept_response = 0
                    positive_count_accept_response = 0
                    counter = 0
                    for index, row in df_month.iterrows():
                        if datetime.strptime(row['PR_Date_Created_At'], "%Y-%m-%d").day == i:
                            counter += 1
                            if counter >top_k: break
                            if row['PR_accept'] == 1 and row['PR_response'] == 1:
                            #if row['PR_response'] == 1: # For baseline model
                                positive_count_accept_response += 1
                                MAP_accept_response += positive_count_accept_response/(counter)

                    total_result.append(MAP_accept_response/positive_count_accept_response if positive_count_accept_response !=0 else 0)
                    MAP_day = MAP_accept_response/positive_count_accept_response if positive_count_accept_response !=0 else 0
                print('Project {0} in month {1} have MAP {2:.3f}'.format(project, date_start.split('-')[0]+'-'+date_start.split('-')[1],
                                                                         MAP_day))
                if top_k > 5:
                    df_MAP = df_MAP.append({'MAP_'+str(top_k): np.mean(total_result)}, ignore_index=True)
                else:
                    df_algo_results = df_algo_results.append(
                        {'Project': project, 'Year-Month': date_start.split('-')[0] + '-' + date_start.split('-')[1],
                         'MAP_' + str(top_k): np.mean(total_result)}, ignore_index=True)
        if top_k > 5:
            df_algo_results = pd.concat([df_algo_results, df_MAP['MAP_'+str(top_k)]], axis=1)
    # print(df_algo_results.head())
    df_algo_results.to_csv(folder_path+'MAP_results.csv', sep=',', encoding='utf-8', index=False)
    calculate_average_map_for_months(df_algo_results, folder_path)

def get_top_n_recall(df, folder_path):
    months_tested = {'2017-09-01': '2017-09-30', '2017-10-01': '2017-10-31',
                     '2017-11-01': '2017-11-30',
                     '2017-12-01': '2017-12-31',
                     '2018-01-01': '2018-01-31', #'2018-02-01': '2018-02-28',  #'2018-03-01':'2018-03-31'
                     }
    df = df.set_index('Project_Name')
    top_k_list = [5, 10, 20]
    df_algo_results = pd.DataFrame(columns=['Project', 'Year-Month', 'AR_5'])
    for top_k in top_k_list:
        print("Now calculating AR for {}".format(top_k))
        df_AR = pd.DataFrame(columns=['AR_' + str(top_k)])
        for project in project_list:
            df_project = df.loc[[project]]
            for date_start, date_end in months_tested.items():
                df_month = df_project.loc[
                    (df_project['PR_Date_Created_At'] >= date_start) & (df_project['PR_Date_Created_At'] <= date_end)]
                print(project)
                total_result = []
                # print(df_month.shape)
                # print(date_start, "-", date_end)
                last_date = datetime.strptime(date_end, "%Y-%m-%d").day
                # print(last_date)
                for i in range(1, last_date+1):
                    top_recall_num = 0
                    total_recall_num = 0
                    counter = 0
                    for index, row in df_month.iterrows():
                        if datetime.strptime(row['PR_Date_Created_At'], "%Y-%m-%d").day == i:
                            counter += 1
                            if row['PR_accept'] == 1 and row['PR_response'] == 1:
                            # if row['PR_response'] == 1: # for baseline model
                                if counter <= top_k:
                                    top_recall_num += 1
                                total_recall_num += 1
                    total_result.append(top_recall_num/total_recall_num if total_recall_num !=0 else 0)
                print('Project {} in month {} have average recall {}'.format(
                    project, date_start.split('-')[0] + '-' + date_start.split('-')[1],np.mean(total_result)))

                if top_k > 5:
                    df_AR = df_AR.append({'AR_' + str(top_k): np.mean(total_result)}, ignore_index=True)
                else:
                    df_algo_results = df_algo_results.append(
                        {'Project': project, 'Year-Month': date_start.split('-')[0] + '-' + date_start.split('-')[1],
                         'AR_' + str(top_k): np.mean(total_result)}, ignore_index=True)
        if top_k > 5:
            df_algo_results = pd.concat([df_algo_results, df_AR['AR_' + str(top_k)]], axis=1)
    # print(df_algo_results.head())
    df_algo_results.to_csv(folder_path+'AR_results.csv', sep=',', encoding='utf-8', index=False)
    calculate_average_recall_for_months(df_algo_results, folder_path)

def calculate_average_recall_for_months(df_recall, folder_path):
    df_recall = df_recall.set_index('Project')
    df_avg_Recall = pd.DataFrame(columns=['Project', 'Top-5', 'Top-10', 'Top-20'])
    for project in project_list:
        df_proj = df_recall.loc[project]
        df_avg_Recall = df_avg_Recall.append({'Project': project, 'Top-5': df_proj['AR_5'].mean(), 'Top-10': df_proj['AR_10'].mean(),
                              'Top-20': df_proj['AR_20'].mean()}, ignore_index=True)
    print(df_avg_Recall.sort_values(by=['Top-5', 'Top-10', 'Top-20'], ascending=False))
    df_avg_Recall.sort_values(by=['Top-5', 'Top-10', 'Top-20'], ascending=False).to_csv(folder_path+'Average_Recall_Months.csv',
                                                                                        index=False, sep=',', encoding='utf-8')

def calculate_average_map_for_months(df_map, folder_path):
    df_map = df_map.set_index('Project')
    df_avg_MAP = pd.DataFrame(columns=['Project', 'Top-5', 'Top-10', 'Top-20'])
    for project in project_list:
        df_proj = df_map.loc[project]
        df_avg_MAP = df_avg_MAP.append({'Project': project, 'Top-5': df_proj['MAP_5'].mean(), 'Top-10': df_proj['MAP_10'].mean(),
                              'Top-20': df_proj['MAP_20'].mean()}, ignore_index=True)
    print(df_avg_MAP.sort_values(by=['Top-5', 'Top-10', 'Top-20'], ascending=False))
    df_avg_MAP.sort_values(by=['Top-5', 'Top-10', 'Top-20'], ascending=False).to_csv(folder_path+'Average_MAP_Months.csv',
                                                                                        index=False, sep=',', encoding='utf-8')

def get_dynamic_MAP(df, folder_path):
    months_tested = {'2017-09-01': '2017-09-30', '2017-10-01': '2017-10-31',
                     '2017-11-01': '2017-11-30',
                     '2017-12-01': '2017-12-31',
                     '2018-01-01': '2018-01-31', #'2018-02-01': '2018-02-28',  #'2018-03-01':'2018-03-31'
                     }

    df = df.set_index('Project_Name')
    day_wise_result = pd.DataFrame(columns=['Project', 'Year-Month', 'Day', 'MAP'])
    month_wise_result = pd.DataFrame(columns=['Project', 'Year-Month', 'MAP'])
    for project in project_list:
        df_project = df.loc[[project]]
        for date_start, date_end in months_tested.items():
            df_month = df_project.loc[
                (df_project['PR_Date_Created_At'] >= date_start) & (df_project['PR_Date_Created_At'] <= date_end)]
            print(project)
            # print(date_start, date_end)
            total_result = []
            last_date = datetime.strptime(date_end, "%Y-%m-%d").day
            print(last_date)
            for i in range(1, last_date+1):
                MAP_accept_response = 0
                positive_count_accept_response = 0
                counter = 0
                top_k = 0
                for index, row in df_month.iterrows():
                    if datetime.strptime(row['PR_Date_Created_At'], "%Y-%m-%d").day == i:
                        if row['PR_accept'] == 1 and row['PR_response'] == 1:
                            top_k += 1
                for index, row in df_month.iterrows():
                    if datetime.strptime(row['PR_Date_Created_At'], "%Y-%m-%d").day == i:
                        counter += 1
                        if counter > top_k: break
                        if row['PR_accept'] == 1 and row['PR_response'] == 1:
                        #if row['PR_response'] == 1: # For baseline model
                            positive_count_accept_response += 1
                            MAP_accept_response += positive_count_accept_response/(counter)

                total_result.append(MAP_accept_response/positive_count_accept_response if positive_count_accept_response !=0 else 0)
                day_MAP = MAP_accept_response/positive_count_accept_response if positive_count_accept_response !=0 else 0
                day_wise_result = day_wise_result.append({'Project':project,
                                                          'Year-Month':date_start.split('-')[0]+'-'+date_start.split('-')[1],
                                                          'Day':i, 'MAP': day_MAP}, ignore_index=True)
                print('Project {0} in month {1} and day {2} have MAP {3:.3f}'.format(project, date_start.split('-')[0] + '-' +
                                                                         date_start.split('-')[1], i, day_MAP))
            print('Project {0} in month {1} have MAP {2:.3f}'.format(project, date_start.split('-')[0]+'-'+date_start.split('-')[1],
                                                                     np.mean(total_result)))
            month_wise_result = month_wise_result.append({'Project':project, 'Year-Month':date_start.split('-')[0]+'-'+date_start.split('-')[1],
                                                          'MAP':np.mean(total_result)}, ignore_index=True)
    # print(df_algo_results.head())
    day_wise_result.to_csv(folder_path+'MAP_day_results_1.csv', sep=',', encoding='utf-8', index=False)
    month_wise_result.to_csv(folder_path + 'MAP_month_results_1.csv', sep=',', encoding='utf-8', index=False)
    calculate_average_dynamic_map_for_months(month_wise_result, folder_path)

def get_dynamic_recall(df, folder_path):
    months_tested = {'2017-09-01': '2017-09-30', '2017-10-01': '2017-10-31',
                     '2017-11-01': '2017-11-30',
                     '2017-12-01': '2017-12-31',
                     '2018-01-01': '2018-01-31', #'2018-02-01': '2018-02-28',  #'2018-03-01':'2018-03-31'
                     }
    df = df.set_index('Project_Name')
    day_wise_result = pd.DataFrame(columns=['Project', 'Year-Month', 'Day', 'AR'])
    month_wise_result = pd.DataFrame(columns=['Project', 'Year-Month', 'AR'])
    for project in project_list:
        df_project = df.loc[[project]]
        for date_start, date_end in months_tested.items():
            df_month = df_project.loc[
                (df_project['PR_Date_Created_At'] >= date_start) & (df_project['PR_Date_Created_At'] <= date_end)]
            print(project)
            total_result = []
            last_date = datetime.strptime(date_end, "%Y-%m-%d").day
            for i in range(1, last_date+1):
                top_recall_num = 0
                total_recall_num = 0
                counter = 0
                top_k = 0
                for index, row in df_month.iterrows():
                    if datetime.strptime(row['PR_Date_Created_At'], "%Y-%m-%d").day == i:
                        if row['PR_accept'] == 1 and row['PR_response'] == 1:
                            top_k+=1
                for index, row in df_month.iterrows():
                    if datetime.strptime(row['PR_Date_Created_At'], "%Y-%m-%d").day == i:
                        counter += 1
                        if row['PR_accept'] == 1 and row['PR_response'] == 1:
                            if counter <= top_k:
                                top_recall_num += 1
                            total_recall_num += 1
                total_result.append(top_recall_num/total_recall_num if total_recall_num !=0 else 0)
                day_AR = top_recall_num/total_recall_num if total_recall_num !=0 else 0
                day_wise_result = day_wise_result.append({'Project': project,
                                                      'Year-Month': date_start.split('-')[0] + '-' +
                                                                    date_start.split('-')[1],
                                                      'Day': i, 'AR': day_AR}, ignore_index=True)
                print(
                'Project {0} in month {1} and day {2} have AR {3:.3f}'.format(project, date_start.split('-')[0] + '-' +
                                                                               date_start.split('-')[1], i, day_AR))
            print('Project {0} in month {1} have AR {2:.3f}'.format(project, date_start.split('-')[0] + '-' + date_start.split('-')[1],
                                                               np.mean(total_result)))
            month_wise_result = month_wise_result.append({'Project': project, 'Year-Month': date_start.split('-')[0] + '-' + date_start.split('-')[1],
             'AR': np.mean(total_result)}, ignore_index=True)
        # print(df_algo_results.head())
    day_wise_result.to_csv(folder_path + 'AR_day_results.csv', sep=',', encoding='utf-8', index=False)
    month_wise_result.to_csv(folder_path + 'AR_month_results.csv', sep=',', encoding='utf-8', index=False)
    calculate_average_dynamic_recall_for_months(month_wise_result, folder_path)

def calculate_average_dynamic_recall_for_months(df_recall, folder_path):
    df_recall = df_recall.set_index('Project')
    df_avg_Recall = pd.DataFrame(columns=['Project', 'Average_AR'])
    for project in project_list:
        df_proj = df_recall.loc[project]
        df_avg_Recall = df_avg_Recall.append({'Project': project, 'Average_AR':df_proj['AR'].mean()}, ignore_index=True)

    print(df_avg_Recall.sort_values(by=['Average_AR'], ascending=False))

    df_avg_Recall.sort_values(by=['Average_AR'], ascending=False).to_csv(folder_path+'AR_Months_Dynamic.csv',
                                                                                        index=False, sep=',', encoding='utf-8')

def calculate_average_dynamic_map_for_months(df_map, folder_path):
    df_map = df_map.set_index('Project')
    df_avg_MAP = pd.DataFrame(columns=['Project', 'Average_MAP'])
    for project in project_list:
        df_proj = df_map.loc[project]
        df_avg_MAP = df_avg_MAP.append({'Project': project, 'Average_MAP':df_proj['MAP'].mean()}, ignore_index=True)


    print(df_avg_MAP.sort_values(by=['Average_MAP'], ascending=False))

    df_avg_MAP.sort_values(by=['Average_MAP'], ascending=False).to_csv(folder_path+'Average_MAP_Months_Dynamic.csv',
                                                                                        index=False, sep=',', encoding='utf-8')

def calculate_median_of_TT_PRs_for_project(df_project):
    months_tested = {'2017-09-01': '2017-09-30', '2017-10-01': '2017-10-31',
                     '2017-11-01': '2017-11-30',
                     '2017-12-01': '2017-12-31',
                     '2018-01-01': '2018-01-31',  # '2018-02-01': '2018-02-28',  #'2018-03-01':'2018-03-31'
                     }

    relevant_PRs = []
    for date_start, date_end in months_tested.items():
        df_month = df_project.loc[
            (df_project['PR_Date_Created_At'] >= date_start) & (df_project['PR_Date_Created_At'] <= date_end)]
        last_date = datetime.strptime(date_end, "%Y-%m-%d").day
        for i in range(1, last_date + 1):
            per_day_count_relevant_PRs = 0
            for index, row in df_month.iterrows():
                if datetime.strptime(row['PR_Date_Created_At'], "%Y-%m-%d").day == i:
                    if row['PR_accept'] == 1 and row['PR_response'] == 1:
                        per_day_count_relevant_PRs += 1
            relevant_PRs.append(per_day_count_relevant_PRs)
    print("Median of relevant PRs: {0}".format(np.median(relevant_PRs)))

    return np.median(relevant_PRs)

def get_dynamic_MAP_top_median(df, folder_path):
    months_tested = {'2017-09-01': '2017-09-30', '2017-10-01': '2017-10-31',
                     '2017-11-01': '2017-11-30',
                     '2017-12-01': '2017-12-31',
                     '2018-01-01': '2018-01-31', #'2018-02-01': '2018-02-28',  #'2018-03-01':'2018-03-31'
                     }

    df = df.set_index('Project_Name')
    day_wise_result = pd.DataFrame(columns=['Project', 'Year-Month', 'Day', 'MAP'])
    month_wise_result = pd.DataFrame(columns=['Project', 'Year-Month', 'MAP'])
    for project in project_list:
        df_project = df.loc[[project]]
        top_k = calculate_median_of_TT_PRs_for_project(df_project)
        for date_start, date_end in months_tested.items():
            df_month = df_project.loc[
                (df_project['PR_Date_Created_At'] >= date_start) & (df_project['PR_Date_Created_At'] <= date_end)]
            print(project)
            # print(date_start, date_end)
            total_result = []
            last_date = datetime.strptime(date_end, "%Y-%m-%d").day
            print(last_date)
            for i in range(1, last_date+1):
                MAP_accept_response = 0
                positive_count_accept_response = 0
                counter = 0
                for index, row in df_month.iterrows():
                    if datetime.strptime(row['PR_Date_Created_At'], "%Y-%m-%d").day == i:
                        counter += 1
                        if counter >top_k: break
                        if row['PR_accept'] == 1 and row['PR_response'] == 1:
                        #if row['PR_response'] == 1: # For baseline model
                            positive_count_accept_response += 1
                            MAP_accept_response += positive_count_accept_response/(counter)

                total_result.append(MAP_accept_response/positive_count_accept_response if positive_count_accept_response !=0 else 0)
                day_MAP = MAP_accept_response/positive_count_accept_response if positive_count_accept_response !=0 else 0
                day_wise_result = day_wise_result.append({'Project':project,
                                                          'Year-Month':date_start.split('-')[0]+'-'+date_start.split('-')[1],
                                                          'Day':i, 'MAP': day_MAP}, ignore_index=True)
                print('Project {0} in month {1} and day {2} have MAP {3:.3f}'.format(project, date_start.split('-')[0] + '-' +
                                                                         date_start.split('-')[1], i, np.mean(total_result)))
            print('Project {0} in month {1} have MAP {2:.3f}'.format(project, date_start.split('-')[0]+'-'+date_start.split('-')[1],
                                                                     np.mean(total_result)))
            month_wise_result = month_wise_result.append({'Project':project, 'Year-Month':date_start.split('-')[0]+'-'+date_start.split('-')[1],
                                                          'MAP':np.mean(total_result)}, ignore_index=True)
    # print(df_algo_results.head())
    day_wise_result.to_csv(folder_path+'MAP_day_median.csv', sep=',', encoding='utf-8', index=False)
    month_wise_result.to_csv(folder_path + 'MAP_month_median.csv', sep=',', encoding='utf-8', index=False)
    # calculate_average_map_for_months(df_algo_results, folder_path)


def calculate_priority_score_for_each_project(df, for_which_model, folder_path):
    priority_factors = ['Pull_Request_ID', 'Title', 'url', 'Additions', 'Deletions', 'Commits_PR', 'Contain_Fix_Bug', 'Files_Changed',
                        'Organization_Core_Member', 'PR_Latency', 'Review_Comments_Count', 'Comments_Count', 'first_response',
                        'latency_after_first_response', 'Score']
    df_project_wise_PR = pd.DataFrame()
    for project in project_list:
        df_proj = df[df.Project_Name == project]
        if for_which_model == 'PR_algo':
            df_result = Execute_Two_Models_Algorithm(df_proj, folder_path)
        elif for_which_model == 'baseline':
            df_result = Execute_Baseline_Model(df_proj, folder_path)
        elif for_which_model == 'accept_response':
            df_result = Execute_Accept_Model_Algorithm(df_proj, folder_path)
        else:
            df_result = Execute_Response_Model_Algorithm(df_proj, folder_path)

        df_project_wise_PR = df_project_wise_PR.append(df_result[priority_factors].sort_values(by=['Score'], ascending=False))

    df_project_wise_PR.to_csv(folder_path + '20_jaeger_PRs.csv', sep=',', encoding='utf-8', index=False)

def Randomly_select_projects_PRs(df_, number_of_PRs):
    df_Rand = pd.DataFrame()
    # df = df.set_index('Project_Name')
    for project in project_list:
        df_proj = df_[df_['Project_Name'] == project]
        df_Rand = df_Rand.append(df_proj.sample(n=number_of_PRs, replace=True))
        # df_Rand = df_Rand.append(df_proj.groupby('PR_response', group_keys=False).apply(lambda x: x.sample(min(len(x), 5))))

    df_Rand.to_csv('/home/ppf/Ilyas_dataset/randomly_selected_PRs_10_stratified.csv', sep=',', encoding='utf-8', index=False)


def calculate_priority_score_for_whole_ds(df, for_which_model, folder_path, count):
    priority_factors = ['Pull_Request_ID', 'Score', 'PR_accept', 'PR_response' ,'Additions', 'Deletions', 'Commits_PR', 'Contain_Fix_Bug', 'Files_Changed',
                        'Organization_Core_Member', 'PR_Latency', 'Review_Comments_Count', 'Comments_Count', 'first_response',
                        'latency_after_first_response'
                        ]
    if for_which_model == 'PR_algo':
        df_result = Execute_Two_Models_Algorithm(df, folder_path)
    elif for_which_model == 'accept_response':
        df_result = Accept_Response_Model(df, folder_path)
    elif for_which_model == 'accept_response':
        df_result = Execute_Accept_Model_Algorithm(df, folder_path)
    else:
        df_result = Execute_Response_Model_Algorithm(df, folder_path)

    df_result = df_result[priority_factors].sort_values(by=['Score'], ascending=False)

    df_result.to_csv(folder_path + 'algo_results_'+str(count)+'.csv', sep=',', encoding='utf-8', index=False)


def Randomly_select_PRs(df_, number_of_PRs, total_samples):
    df_total = pd.DataFrame()
    # for i in range(total_samples):
    #     df_sample = df_[(df_['PR_accept'] == i) & (df_['PR_response'] == i)].sample(n=number_of_PRs, replace=True)
    #     df_total = df_total.append(df_sample)
    sample_1 = df_[(df_['PR_accept'] == 0) & (df_['PR_response'] == 0)].sample(n=number_of_PRs, replace=True)
    sample_2 = df_[(df_['PR_accept'] == 1) & (df_['PR_response'] == 1)].sample(n=number_of_PRs, replace=True)
    sample_3 = df_[(df_['PR_accept'] == 1) & (df_['PR_response'] == 0)].sample(n=number_of_PRs, replace=True)
    sample_4 = df_[(df_['PR_accept'] == 0) & (df_['PR_response'] == 1)].sample(n=number_of_PRs, replace=True)
    df_total = df_total.append(sample_1)
    df_total = df_total.append(sample_2)
    df_total = df_total.append(sample_3)
    df_total = df_total.append(sample_4)
    print(df_total.shape)
    print(df_total.PR_accept.value_counts())
    print(df_total.PR_response.value_counts())

    df_total.to_csv('Results/CART/400_samples.csv', sep=',', encoding='utf-8', index=False)


def select_columns_from_random_samples(folder_path):
    index = 0
    for file in listdir(folder_path):
        if os.path.isfile(folder_path + file):
            df = pd.read_csv(folder_path + file, encoding='utf-8')
            df_cart = df.sort_values(by=['Score'], ascending=False)  # for CART
            df_gousios = df.sort_values(by=['accept_score'], ascending=False)
            df_prioritizer = df.sort_values(by=['response_score'], ascending=False)

            df_cart[['Pull_Request_ID', 'PR_accept', 'PR_response', 'Score', 'accept_score', 'response_score']].to_csv(
                'Results/CART/10_samples_results/cart/sample_'+str(index)+'.csv',
                                                     sep=',', encoding='utf-8', index=False)
            # df_gousios[
            #     ['Pull_Request_ID', 'PR_accept', 'accept_score']].to_csv(
            #     'Results/CART/10_samples_results/gousios/sample_' + str(index) + '.csv',
            #     sep=',', encoding='utf-8', index=False)
            # df_prioritizer[
            #     ['Pull_Request_ID', 'PR_response', 'response_score']].to_csv(
            #     'Results/CART/10_samples_results/prioritizer/sample_' + str(index) + '.csv',
            #     sep=',', encoding='utf-8', index=False)
            index+=1


def precision_recall_curve():
    recoms = [0, 1, 0, 1, 0, 1, 1]  # N = 7
    NUM_ACTUAL_ADDED_ACCT = 5
    precs = []
    recalls = []

    for indx, rec in enumerate(recoms):
        print(indx+1)
        precs.append(sum(recoms[:indx + 1]) / (indx + 1))
        recalls.append(sum(recoms[:indx + 1]) / NUM_ACTUAL_ADDED_ACCT)

    print(precs)
    print(recalls)

    # fig, ax = plt.subplots()
    # ax.plot(recalls, precs, markersize=10, marker="o")
    # ax.set_xlabel("Recall")
    # ax.set_ylabel("Precision")
    # ax.set_title("P(i) vs. r(i) for Increasing $i$ for AP@7")
    # plt.show()


def get_top_n_MAP_for_samples(df, folder_path, file_name):
    df = shuffle(df)
    df_split = np.array_split(df, 10)
    top_k_list = [5, 10, 20, 30]
    models_list = ['MCM', 'FIFO', 'SSF']
    df_algo_results = pd.DataFrame(columns=['Model','Sample', 'MAP_5'])
    for top_k in top_k_list:
        print("Now calculating MAP for {}".format(top_k))
        df_MAP = pd.DataFrame(columns=['MAP_'+str(top_k)])
        for i in range(len(df_split)):
            for model in models_list:
                if model == 'MCM':
                    sample_df = df_split[i].sort_values(by=['Score'], ascending=True) # for MCM
                elif model == 'FIFO':
                    sample_df = df_split[i].sort_values(by=['PR_Date_Created_At', 'PR_Time_Created_At'], ascending=True) # for FIFO
                else:
                    sample_df = df_split[i]  # for SSF
                    sample_df['src_churn'] = sample_df['Additions'] + sample_df['Deletions']
                    sample_df = sample_df.sort_values(by=['src_churn', 'Files_Changed'], ascending=True)
                MAP_accept_response = 0
                positive_count_accept_response = 0
                counter = 0
                for index, row in sample_df.iterrows():
                    counter += 1
                    if counter >top_k: break
                    if row['label'] == 0 or row['label'] == 1:
                        positive_count_accept_response += 1
                        MAP_accept_response += positive_count_accept_response/(counter)
                result = MAP_accept_response / positive_count_accept_response if positive_count_accept_response != 0 else 0
                print(result)
                if top_k > 5:
                    df_MAP = df_MAP.append({'MAP_'+str(top_k): result}, ignore_index=True)
                    print(df_MAP.shape)
                else:
                    df_algo_results = df_algo_results.append(
                        {'Model':model,'Sample': 'Sample_'+str(i), 'MAP_' + str(top_k): result}, ignore_index=True)
                    print(df_algo_results.shape)
        if top_k > 5:
            df_algo_results = pd.concat([df_algo_results, df_MAP['MAP_'+str(top_k)]], axis=1)
    print(df_algo_results.head())
    df_algo_results.to_csv(folder_path+file_name+'.csv', sep=',', encoding='utf-8', index=False)


def get_top_n_recall_for_samples(df, folder_path, file_name):
    df = shuffle(df)
    df_split = np.array_split(df, 10)
    top_k_list = [5, 10, 20, 30]
    models_list = ['MCM', 'FIFO', 'SSF']
    df_algo_results = pd.DataFrame(columns=['Model', 'Sample', 'AR_5'])
    for top_k in top_k_list:
        print("Now calculating AR for {}".format(top_k))
        df_AR = pd.DataFrame(columns=['AR_' + str(top_k)])
        for i in range(len(df_split)):
            # print(df_split[i].shape)
            for model in models_list:
                if model == 'MCM':
                    sample_df = df_split[i].sort_values(by=['Score'], ascending=True) # for MCM
                elif model == 'FIFO':
                    sample_df = df_split[i].sort_values(by=['PR_Date_Created_At', 'PR_Time_Created_At'], ascending=True) # for FIFO
                else:
                    sample_df = df_split[i]  # for SSF
                    sample_df['src_churn'] = sample_df['Additions'] + sample_df['Deletions']
                    sample_df = sample_df.sort_values(by=['src_churn', 'Files_Changed'], ascending=True)
                top_recall_num = 0
                total_recall_num = 0
                counter = 0
                for index, row in sample_df.iterrows():
                    counter += 1
                    if row['label'] == 0 or row['label'] == 1:
                        if counter <= top_k:
                            top_recall_num += 1
                        total_recall_num += 1
                result = top_recall_num/total_recall_num if total_recall_num !=0 else 0
                print(result)
                if top_k > 5:
                    df_AR = df_AR.append({'AR_' + str(top_k): result}, ignore_index=True)
                else:
                    df_algo_results = df_algo_results.append(
                        {'Model':model,'Sample': 'Sample_'+str(i), 'AR_' + str(top_k): result}, ignore_index=True)
        if top_k > 5:
            df_algo_results = pd.concat([df_algo_results, df_AR['AR_' + str(top_k)]], axis=1)
    print(df_algo_results.head())
    df_algo_results.to_csv(folder_path+file_name+'.csv', sep=',', encoding='utf-8', index=False)

def split_samples(df, folds):
    df = shuffle(df)
    df_split = np.array_split(df, folds)
    for index in range(len(df_split)):
        df_sub_sample = df_split[index]
        df_sub_sample.to_csv('Results/CART/10_samples/sample_'+str(index)+'.csv', encoding='utf-8', index=None)


def get_top_n_MAP_for_samples_list(model, folder_path, result_folder_path, file_name):
    top_k_list = [5, 10, 20, 30]
    df_algo_results = pd.DataFrame(columns=['Model','Sample', 'MAP_5'])
    count = 0
    for top_k in top_k_list:
        print("Now calculating MAP for {}".format(top_k))
        df_MAP = pd.DataFrame(columns=['MAP_'+str(top_k)])
        for file in listdir(folder_path):
            if os.path.isfile(folder_path+file):
                df = pd.read_csv(folder_path + file, encoding='utf-8')
                # print(df.head())
                if model == 'CART':
                    sample_df = df.sort_values(by=['Score'], ascending=False) # for CART
                elif model == 'Gousios':
                    sample_df = df.sort_values(by=['accept_score'], ascending=False) # for FIFO
                elif model == 'PRioritizer':
                    sample_df = df.sort_values(by=['response_score'], ascending=False)
                else:
                    # for SSF
                    df['src_churn'] = df['Additions'] + df['Deletions']
                    sample_df = df.sort_values(by=['src_churn', 'Files_Changed'], ascending=True)
                MAP_accept_response = 0
                positive_count_accept_response = 0
                counter = 0
                for index, row in sample_df.iterrows():
                    counter += 1
                    if counter >top_k: break
                    if row['PR_accept'] == 1 and row['PR_response'] == 1:
                        positive_count_accept_response += 1
                        MAP_accept_response += positive_count_accept_response/(counter)
                result = MAP_accept_response / positive_count_accept_response if positive_count_accept_response != 0 else 0
                print(result)
                if top_k > 5:
                    df_MAP = df_MAP.append({'MAP_'+str(top_k): result}, ignore_index=True)
                    print(df_MAP.shape)
                else:
                    df_algo_results = df_algo_results.append(
                        {'Model':model,'Sample': 'Sample_'+str(count), 'MAP_' + str(top_k): result}, ignore_index=True)
                    count += 1
        if top_k > 5:
            df_algo_results = pd.concat([df_algo_results, df_MAP['MAP_'+str(top_k)]], axis=1)
    print(df_algo_results.head())
    df_algo_results.to_csv(result_folder_path+file_name+'.csv', sep=',', encoding='utf-8', index=False)


def get_top_n_recall_for_samples_list(model, folder_path, result_folder_path, file_name):
    top_k_list = [5, 10, 20, 30]
    count = 0
    df_algo_results = pd.DataFrame(columns=['Model', 'Sample', 'AR_5'])
    for top_k in top_k_list:
        print("Now calculating AR for {}".format(top_k))
        df_AR = pd.DataFrame(columns=['AR_' + str(top_k)])
        for file in listdir(folder_path):
            if os.path.isfile(folder_path+file):
                df = pd.read_csv(folder_path + file, encoding='utf-8')
                # print(df.head())
                if model == 'CART':
                    sample_df = df.sort_values(by=['Score'], ascending=False)  # for CART
                elif model == 'Gousios':
                    sample_df = df.sort_values(by=['accept_score'], ascending=False)  # for FIFO
                elif model == 'PRioritizer':
                    sample_df = df.sort_values(by=['response_score'], ascending=False)
                else:
                    # for SSF
                    df['src_churn'] = df['Additions'] + df['Deletions']
                    sample_df = df.sort_values(by=['src_churn', 'Files_Changed'], ascending=True)
                top_recall_num = 0
                total_recall_num = 0
                counter = 0
                for index, row in sample_df.iterrows():
                    counter += 1
                    if row['PR_accept'] == 1 and row['PR_response'] == 1:
                        if counter <= top_k:
                            top_recall_num += 1
                        total_recall_num += 1
                result = top_recall_num/total_recall_num if total_recall_num !=0 else 0
                print(result)
                if top_k > 5:
                    df_AR = df_AR.append({'AR_' + str(top_k): result}, ignore_index=True)
                else:
                    df_algo_results = df_algo_results.append(
                        {'Model':model,'Sample': 'Sample_'+str(count), 'AR_' + str(top_k): result}, ignore_index=True)
                    count += 1
        if top_k > 5:
            df_algo_results = pd.concat([df_algo_results, df_AR['AR_' + str(top_k)]], axis=1)
    print(df_algo_results.head())
    df_algo_results.to_csv(result_folder_path+file_name+'.csv', sep=',', encoding='utf-8', index=False)

def select_top_20_PRs_prioritized_CART():
    folder_path = 'Random_samples/CART/3_labels/all_columns/'
    result_folder_path = 'Random_samples/CART/3_labels/all_columns/selected/'
    for file in listdir(folder_path):
        if os.path.isfile(folder_path + file):
            df = pd.read_csv(folder_path + file, encoding='utf-8')
            print(file)

            df[['Pull_Request_ID', 'label', 'Score', 'Title', 'Body']].to_csv(result_folder_path + file,
                                                                              sep=',', encoding='utf-8', index=False)


def baseline_classifer(X, y):
    dummy_clf = DummyClassifier(strategy="most_frequent")
    dummy_clf.fit(X, y)

    y_pred = dummy_clf.predict(X)

    print(metrics.classification_report(y, y_pred, digits=3))

    print(dummy_clf.score(X, y))

def extract_metric_from_report(report):
    report = list(report.split("\n"))
    report = report[-2].split(' ')
    # print(report)
    mylist = []
    for i in range(len(report)):
        if report[i] != '':
            mylist.append(report[i])

    return mylist[3], mylist[4], mylist[5]

def calculate_results_samples(folder_path):
    results = pd.DataFrame(columns=['Sample', 'P_RR', 'P_DA', 'P_R', 'R_RR', 'R_DA', 'R_R', 'f1_RR', 'f1_DA',
                                    'f1_R', 'Avg_Pre', 'Avg_Recall', 'Avg_f1_Score',
                                    'Test_Accuracy'])
    for file in listdir(folder_path):
        if os.path.isfile(folder_path + file):
            print(file.split('.')[0])
            df = pd.read_csv(folder_path + file, encoding='utf-8')
            y_test = df['label']
            y_pred = df['Score']
            # Print model report:
            print("\nModel Report")
            print("Test Accuracy : %.4g" % metrics.accuracy_score(y_test, y_pred))
            print('Test error: {:.3f}'.format(1 - metrics.accuracy_score(y_test, y_pred)))

            test_accuracy = metrics.accuracy_score(y_test, y_pred)
            precision, recall, fscore, support = score(y_test, y_pred)

            print(pd.crosstab(y_test, y_pred, rownames=['Actual PRs'], colnames=['Predicted PRs']))

            print(metrics.classification_report(y_test, y_pred, digits=3))

            precision_avg, recall_avg, fscore_avg = extract_metric_from_report(
                metrics.classification_report(y_test, y_pred, digits=3))
            results = results.append(
                {'Sample': file.split('.')[0],
                 'P_RR': precision[1], 'P_DA': precision[0], 'P_R': precision[2], 'R_RR': recall[1],
                 'R_DA': recall[0], 'R_R': recall[2], 'f1_RR': fscore[1], 'f1_DA': fscore[0], 'f1_R': fscore[2],
                 'Avg_Pre': precision_avg, 'Avg_Recall': recall_avg, 'Avg_f1_Score': fscore_avg,
                 'Test_Accuracy': test_accuracy},
                ignore_index=True)

        results.to_csv('Random_samples/381_samples_results.csv', sep=',', encoding='utf-8', index=False)

def All_Model_Results(df_test_PR, folder_path):
    pd.options.mode.chained_assignment = None
    results = pd.DataFrame(columns=['Project', 'Model', 'P_RR', 'P_DA', 'P_R', 'R_RR', 'R_DA', 'R_R', 'f1_RR', 'f1_DA',
                                    'f1_R', 'Avg_Pre', 'Avg_Recall', 'Avg_f1_Score',
                                    'Test_Accuracy'])
    models_list = ['xgb_selected_features', 'DT', 'KNN', 'LogisticRegression', 'LinearSVC', 'NaiveBayes',
                   'RandomForest']
    for model in models_list:
        with open('../Models/Saved_Models/3_labels/'+model+'.pickle.dat', 'rb') as f:
            xgb_model = pickle.load(f)
            y_pred_accept = xgb_model.predict(df_test_PR[predictors])
            df_test_PR['Score'] = y_pred_accept
            # print(df_test_PR[['Pull_Request_ID', 'label', 'Score']].head(10))
            for project in project_list:
                print(project)
                df_proj = df_test_PR.loc[df_test_PR.Project_Name == project]
                y_test = df_proj['label']
                y_pred = df_proj['Score']
                # Print model report:
                print("\nModel Report")
                print("Test Accuracy : %.4g" % metrics.accuracy_score(y_test, y_pred))
                print('Test error: {:.3f}'.format(1 - metrics.accuracy_score(y_test, y_pred)))

                test_accuracy = metrics.accuracy_score(y_test, y_pred)
                precision, recall, fscore, support = score(y_test, y_pred)

                print(pd.crosstab(y_test, y_pred, rownames=['Actual PRs'], colnames=['Predicted PRs']))

                print(metrics.classification_report(y_test, y_pred, digits=3))

                precision_avg, recall_avg, fscore_avg = extract_metric_from_report(
                    metrics.classification_report(y_test, y_pred, digits=3))
                try:
                    results = results.append(
                        {'Project': project, 'Model': model,
                         'P_RR': precision[1], 'P_DA': precision[0], 'P_R': precision[2], 'R_RR': recall[1],
                         'R_DA': recall[0], 'R_R': recall[2], 'f1_RR': fscore[1], 'f1_DA': fscore[0], 'f1_R': fscore[2],
                         'Avg_Pre': precision_avg, 'Avg_Recall': recall_avg, 'Avg_f1_Score': fscore_avg,
                         'Test_Accuracy': test_accuracy},
                        ignore_index=True)
                except IndexError:
                    continue

    results.to_csv('../Models/Results/project_wise_results.csv', sep=',', encoding='utf-8', index=False)



if __name__ == '__main__':

    print('Processing')
    print(X_test.shape)
    # df_test_1 = pd.read_csv('Results/CART/3_labels/test_DS_results.csv')
    # acc = metrics.accuracy_score(df_test_1['label'], df_test_1['Score'])
    # print(acc)

    # test_model_on_random_samples(df_rand_sample)
    # split_samples(df_rand_sample, 10)

    # Gousios and PRioritizer baseline model
    # Gousios_Model(X_test, folder_path)
    result = PRioritizer_Model(X_test, folder_path)
    print(metrics.classification_report(result['PR_response'], result['response_score'], digits=2))

    # CART model
    # df_ = Two_Models_CART(X_test, folder_path, 'cart_results')
    # get_top_n_MAP_all(df_, folder_path, 'MAP_cart_results')
    # get_top_n_recall_all(df_, folder_path, 'AR_cart_results')
    # print(df_[['Pull_Request_ID', 'label']].head(100))

    ## Select random sample

    # cart_result = pd.read_csv('Results/CART/cart_results.csv')
    # print(cart_result.head())
    # Randomly_select_PRs(cart_result, 100, 2)

    ## Calaculate Score for Random samples
    # df_rand_sample = pd.read_csv('Results/CART/382_samples.csv')
    # df_rand_sample = shuffle(df_rand_sample)
    # df_ = Two_Models_CART(df_rand_sample, 'Results/CART/', '382_sample_results')
    # get_top_n_MAP_all(df_, folder_path, 'MAP_382_results')
    # get_top_n_recall_all(df_, folder_path, 'AR_382_results')
    # df_rand_sample = pd.read_csv('Random_samples/382_sample_results.csv')
    # acc = metrics.accuracy_score(df_rand_sample['label'], df_rand_sample['Score'])
    # print(acc)

    ## Split the ramdom sample to 10 parts
    # df = pd.read_csv('Results/CART/400_samples_all_results.csv')
    # df_gousios = pd.read_csv('Results/CART/Gousios_results.csv')
    # df_prioritizer = pd.read_csv('Results/CART/prioritizer_results.csv')
    # print(df.shape)
    # result = pd.merge(df, df_gousios[['Pull_Request_ID', 'accept_score']], on='Pull_Request_ID', how='left')
    # print(result.shape)
    # result = pd.merge(result, df_prioritizer[['Pull_Request_ID', 'response_score']], on='Pull_Request_ID', how='left')
    # print(result.shape)
    # print(df[['Pull_Request_ID', 'accept_score', 'response_score']].tail(20))
    # df_G = Gousios_Model(df, folder_path)
    # print(df_G.shape)
    # df_P = PRioritizer_Model(df_G, folder_path)
    # print(df_P.shape)
    # df_P.to_csv(folder_path+'400_samples_all_results.csv', sep=',')
    # print(df[['Pull_Request_ID', 'PR_accept', 'Result_Accept', 'PR_response', 'Result_Response']].tail(20))
    # print('Accept label')
    # print(df.PR_accept.value_counts())
    # print('Response label')
    # print(df.PR_response.value_counts())
    # print(metrics.classification_report(df['PR_accept'], df['Result_Accept'], digits=2))
    # print(metrics.classification_report(df['PR_response'], df['Result_Response'], digits=2))
    # split_samples(df, 10)
    # get_top_n_MAP_for_samples_list('CART', 'Results/CART/10_samples/', 'Results/CART/10_samples_results/', 'Cart_MAP_results')
    # get_top_n_recall_for_samples_list('CART', 'Results/CART/10_samples/', 'Results/CART/10_samples_results/', 'Cart_AR_results')
    # get_top_n_MAP_for_samples_list('Gousios', 'Results/CART/10_samples/', 'Results/CART/10_samples_results/', 'Gousios_MAP_results')
    # get_top_n_recall_for_samples_list('Gousios', 'Results/CART/10_samples/', 'Results/CART/10_samples_results/', 'Gousios_AR_results')
    # get_top_n_MAP_for_samples_list('PRioritizer', 'Results/CART/10_samples/', 'Results/CART/10_samples_results/', 'Prioritizer_MAP_results')
    # get_top_n_recall_for_samples_list('PRioritizer', 'Results/CART/10_samples/', 'Results/CART/10_samples_results/', 'Prioritizer_AR_results')

    # get_top_n_MAP_for_samples_list('Accept', 'Random_samples/Accept/', 'Random_samples/Accept/Results/', 'MAP_results')
    # get_top_n_recall_for_samples_list('Accept', 'Random_samples/Accept/', 'Random_samples/Accept/Results/', 'AR_results')

    # select_columns_from_random_samples('Results/CART/10_samples/')

    # model_on_random_samples('Random_samples/10_parts/')
    # X = X_test[predictors_for_baseline]
    # y = X_test['PR_accept']
    # baseline_classifer(X, y)

    # calcuate_results_samples('Random_samples/10_parts/')
    # select_top_20_PRs_prioritized_CART()

    # All_Model_Results(X_test, '')

    # Execute_Baseline_Model(X_test, 'Results/Baseline/')

    # folder_path = 'Random_samples/10_parts/'
    # count = 0
    # for file in listdir(folder_path):
    #     if os.path.isfile(folder_path + file):
    #         print(file.split('.')[0])
    #         df = pd.read_csv(folder_path + file)
    #         result = Execute_Baseline_Model(df, '')
    #         result[['Pull_Request_ID', 'PR_accept', 'label', 'Score']].to_csv('Random_samples/Accept/'+file, sep=',',
    #                                                                         index=False, encoding='utf-8')








