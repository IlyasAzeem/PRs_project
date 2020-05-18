import pandas as pd
from imblearn.over_sampling import SMOTE
from collections import Counter

# df_response = pd.read_csv("E:\\Research Work\\Sentiment Analysis Project\\Dataset\\Full_Dataset\\dataset_06_12_19\\response_21_features.csv", sep=',')
df_accept = pd.read_csv("E:\\Research Work\\Sentiment Analysis Project\\Dataset\\Full_Dataset\\dataset_06_12_19\\accept_21_features_2.csv", sep=',')

df = pd.read_csv("E:\\Research Work\\Sentiment Analysis Project\\Dataset\\Full_Dataset\\dataset_06_12_19\\accept_multilabel_1.csv", sep=',')

project_list = ['react', 'angular.js', 'django', 'nixpkgs', 'scikit-learn', 'yii2', 'githubschool', 'cdnjs', 'terraform',
                'laravel', 'cmssw', 'salt', 'curriculum', 'opencv', 'tensorflow', 'pandas', 'symfony', 'moby', 'rails',
                'rust', 'kubernetes']

columns_list = ['Following', 'Intra_Branch', 'Public_Repos', 'Mergeable_State', 'User_Accept_Rate', 'Review_Comments_Count',
                'url', 'Project_Age', 'Files_Changed', 'Sunday', 'File_Touched_Average', 'Friday', 'Private_Repos', 'Monday',
                'Timeline', 'Contain_Fix_Bug', 'Comments_Per_Closed_PR', 'Additions_Per_Week', 'Project_Accept_Rate',
                'Team_Size', 'Saturday', 'Workload', 'Deletions', 'Closed_Num', 'Wait_Time', 'Review_Comments_Embedding',
                'Wednesday', 'Contributor', 'Assignees_Count', 'Close_Latency', 'Last_Comment_Mention', 'Additions',
                'Language', 'PR_accept', 'Merge_Latency', 'Comments_Embedding', 'Watchers', 'Contributor_Num', 'Followers',
                'Deletions_Per_Week', 'Label_Count', 'Stars', 'Title', 'Participants_Count', 'Organization_Core_Member',
                'Tuesday', 'Commits_PR', 'Day', 'Body', 'Churn_Average', 'Prev_PRs', 'Contributions', 'Mergeable',
                'Point_To_IssueOrPR', 'Open_Issues', 'Rebaseable', 'Accept_Num', 'Closed_Num_Rate', 'Commits_Average',
                'Thursday', 'Comments_Count', 'Comments_Per_Merged_PR', 'Forks_Count', 'X1_0', 'X1_1', 'X1_2', 'X1_3', 'X1_4',
                'X1_5', 'X1_6', 'X1_7', 'X1_8', 'X1_9', 'X2_0', 'X2_1', 'X2_2', 'X2_3', 'X2_4', 'X2_5', 'X2_6', 'X2_7', 'X2_8',
                'X2_9', 'PR_Latency', 'Project_Name', 'PR_Date_Created_At', 'PR_Time_Created_At', 'PR_Date_Closed_At',
                'PR_Time_Closed_At', 'first_response_time', 'first_response', 'latency_after_first_response', 'wait_time_up',
                'PR_response', 'PR_age', 'conflict', 'title_words_count', 'body_words_count', 'comments_reviews_words_count',
                'Pull_Request_ID', 'Project_Domain', 'Project_Size']

# df_accept = df_accept[(df_accept.Project_Name != 'githubschool') & (df_accept.Project_Name != 'curriculum')]
# df = df[(df.Project_Name != 'githubschool') & (df.Project_Name != 'curriculum')]

# print(list(df_response.Project_Name.unique()))
# print(list(df_response.columns))

def create_multi_label_dataset(df_accept):
    print("Size of the whole dataset: {}".format(df_accept.shape))
    merged_PRs = df_accept.loc[df_accept.PR_accept == 1]
    unmerged_PRs = df_accept.loc[df_accept.PR_accept == 0]

    print("Size of merged PRs {}".format(merged_PRs.shape[0]))
    print("Size of unmerged PRs {}".format(unmerged_PRs.shape[0]))

    df_directly_accepted = merged_PRs.loc[(merged_PRs.Comments_Count == 0) & (merged_PRs.Review_Comments_Count == 0) & (merged_PRs.first_response == 0)]
    df_directly_rejected = unmerged_PRs.loc[(unmerged_PRs.Comments_Count == 0) & (unmerged_PRs.Review_Comments_Count == 0) & (unmerged_PRs.first_response == 0)]

    df_accepted_line_comments = merged_PRs.loc[(merged_PRs.Comments_Count == 0) & (merged_PRs.Review_Comments_Count == 0) & (merged_PRs.first_response == 1)]
    df_rejected_line_comments = unmerged_PRs.loc[(unmerged_PRs.Comments_Count == 0) & (unmerged_PRs.Review_Comments_Count == 0)  & (unmerged_PRs.first_response == 1)]
    print(df_directly_accepted.shape)
    print("Percentage of accepted PRs without a single comment {}".format((df_directly_accepted.shape[0]*100)/df_accept.shape[0]))
    print("Percentage of rejected PRs without a single comment {}".format((df_directly_rejected.shape[0]*100)/df_accept.shape[0]))

    print("Number of Accepted PRs without comments: {}".format(df_directly_accepted.shape[0]))
    print("Number of Rejected PRs without comments: {}".format(df_directly_rejected.shape[0]))

    # print("Percentage of accepted PRs without a single comment {}".format((df_accepted_line_comments.shape[0]*100)/df_accept.shape[0]))
    # print("Percentage of rejected PRs without a single comment {}".format((df_rejected_line_comments.shape[0]*100)/df_accept.shape[0]))

    print("Number of Accepted PRs with line-comments: {}".format(df_accepted_line_comments.shape[0]))
    print("Number of Rejected PRs with linecomments: {}".format(df_rejected_line_comments.shape[0]))

    df_accepted_after = merged_PRs.loc[(merged_PRs.Comments_Count >= 1) | (merged_PRs.Review_Comments_Count >=1)]
    df_rejected_after = unmerged_PRs.loc[(unmerged_PRs.Comments_Count >= 1) | (unmerged_PRs.Review_Comments_Count >= 1)]

    print("Percentage of accepted PRs after discussion {}".format((df_accepted_after.shape[0]*100)/df_accept.shape[0]))
    print("Percentage of rejected PRs after discussion {}".format((df_rejected_after.shape[0]*100)/df_accept.shape[0]))

    print("Number of Accepted PRs after discussion: {}".format(df_accepted_after.shape[0]))
    print("Number of Rejected PRs after discussion: {}".format(df_rejected_after.shape[0]))

    pd.options.mode.chained_assignment = None
    df = pd.DataFrame()

    df_directly_accepted['label'] = 'directly_accepted'
    df_directly_rejected['label'] = 'directly_rejected'
    df_accepted_line_comments['label'] = 'accepted_response'
    df_rejected_line_comments['label'] = 'rejected_response'
    df_accepted_after['label'] = 'accepted_response'
    df_rejected_after['label'] = 'rejected_response'

    df = df.append(df_directly_accepted)
    df = df.append(df_directly_rejected)
    df = df.append(df_accepted_line_comments)
    df = df.append(df_rejected_line_comments)
    df = df.append(df_accepted_after)
    df = df.append(df_rejected_after)

    print(df.shape)
    # print(df['label'].head(5))

    # df.to_csv(
    #     "E:\\Research Work\\Sentiment Analysis Project\\Dataset\\Full_Dataset\\dataset_06_12_19\\accept_multilabel.csv",
    #     sep=',', encoding='utf-8', index=False)


def create_multi_label_dataset_updated(df_accept):
    df_accept = df_accept[(df_accept.Project_Name != 'githubschool') & (df_accept.Project_Name != 'curriculum')]
    print("Size of the whole dataset: {}".format(df_accept.shape))
    merged_PRs = df_accept.loc[df_accept.PR_accept == 1]
    unmerged_PRs = df_accept.loc[df_accept.PR_accept == 0]

    print("Size of merged PRs {}".format(merged_PRs.shape[0]))
    print("Size of unmerged PRs {}".format(unmerged_PRs.shape[0]))

    df_directly_accepted = merged_PRs.loc[(merged_PRs.Comments_Count == 0) & (merged_PRs.Review_Comments_Count == 0) &
                                          (merged_PRs.line_comments_count == 0)]
    # df_directly_rejected = unmerged_PRs.loc[(unmerged_PRs.Comments_Count == 0) & (unmerged_PRs.Review_Comments_Count == 0)
    #                                         & (unmerged_PRs.line_comments_count == 0)]
    print("Check response of directly accepted PRs {}".format(df_directly_accepted.loc[df_directly_accepted.first_response == 0].shape[0]))
    # print("Check response of directly rejected PRs {}".format(
    #     df_directly_rejected.loc[df_directly_rejected.first_response == 0].shape[0]))
    print("Percentage of accepted PRs without a single comment {}".format((df_directly_accepted.shape[0]*100)/df_accept.shape[0]))
    # print("Percentage of rejected PRs without a single comment {}".format((df_directly_rejected.shape[0]*100)/df_accept.shape[0]))

    print("Number of Accepted PRs without comments: {}".format(df_directly_accepted.shape[0]))
    # print("Number of Rejected PRs without comments: {}".format(df_directly_rejected.shape[0]))


    df_accepted_after = merged_PRs.loc[(merged_PRs.Comments_Count >= 1) | (merged_PRs.Review_Comments_Count >=1) |
                                       (merged_PRs.line_comments_count >=1)]
    # df_rejected_after = unmerged_PRs.loc[(unmerged_PRs.Comments_Count >= 1) | (unmerged_PRs.Review_Comments_Count >= 1) |
    #                                      (unmerged_PRs.line_comments_count >=1)]

    print("Percentage of accepted PRs after discussion {}".format((df_accepted_after.shape[0]*100)/df_accept.shape[0]))
    print("Percentage of rejected PRs {}".format((unmerged_PRs.shape[0]*100)/df_accept.shape[0]))

    print("Number of Accepted PRs after discussion: {}".format(df_accepted_after.shape[0]))
    # print("Number of Rejected PRs after discussion: {}".format(df_rejected_after.shape[0]))

    pd.options.mode.chained_assignment = None
    df = pd.DataFrame()

    df_directly_accepted['label'] = 'directly_accepted'
    # df_directly_rejected['label'] = 'directly_rejected'
    df_accepted_after['label'] = 'response_required'
    unmerged_PRs['label'] = 'rejected'

    df = df.append(df_directly_accepted)
    df = df.append(df_accepted_after)
    df = df.append(unmerged_PRs)
    # df = df.append(df_directly_rejected)


    print(df.shape)
    df.loc[(df.label == 'directly_accepted') & (df.first_response == 1), 'label'] = 'response_required'
    print(df.loc[df.label == 'directly_accepted'].shape[0])

    df.to_csv(
        "E:\\Research Work\\Sentiment Analysis Project\\Dataset\\Full_Dataset\\dataset_06_12_19\\3_multilabel.csv",
        sep=',', encoding='utf-8', index=False)
    # df_directly_accepted[['Pull_Request_ID', 'Timeline']].loc[df_directly_accepted.first_response == 1].to_csv(
    #     "Dataset_Samples\\two_prs.csv",
    #     sep=',', encoding='utf-8', index=False)

# create_multi_label_dataset_updated(df_accept)

# df_directly_accepted = df.loc[df.label == 'directly_accepted']
# df_directly_rejected = df.loc[df.label == 'directly_rejected']
# df_accepted_after = df.loc[df.label == 'accepted_response']
# df_rejected_after = df.loc[df.label == 'rejected_response']
#
# print(df_directly_accepted.shape)
# print(df_directly_rejected.shape)
# print(df_accepted_after.shape)
# print(df_rejected_after.shape)

def select_sample_for_reading(df, num_of_PRs, file_name):
    df = df.sample(num_of_PRs)
    df['Reason'] = ''
    df[['Pull_Request_ID', 'Title', 'Body', 'Reason']].to_csv('Dataset_Samples/'+file_name+'.csv', index=False, encoding='utf-8', sep=',')


# select_sample_for_reading(df_directly_rejected, 358, 'directly_rejected')
# select_sample_for_reading(df_directly_accepted.loc[df_directly_accepted.first_response == 0], 381, 'directly_accepted')

# check the contributors for directly accepted and rejected PRs
def analyze_directly_AR_PRs(df):
    print(df[['User_Accept_Rate', 'Prev_PRs', 'Followers']].quantile([0.95, 0.90, 0.80]))
    print(df[['User_Accept_Rate', 'Prev_PRs', 'Followers']].describe())




# analyze_directly_AR_PRs(df_directly_accepted)

# analyze_directly_AR_PRs(df_directly_rejected)

# df_1 = pd.read_csv('Dataset_Samples\\Accepted\\accepted_categories.csv', encoding='utf-8')
# print(df_1.head())


# df_1[300:].to_csv('Dataset_Samples/Accepted/ilyas.csv', index=False, encoding='utf-8', sep=',')

from scipy.stats import kruskal
from scipy.stats import mannwhitneyu
def calcualte_kruskal_test(df1, df2, df3, df4, df5, df6, df7):
    print('Results of kruskal test')
    stat, p = kruskal(df1, df2, df3, df4, df5, df6, df7)
    print('Statistics=%.3f, p=%.3f' % (stat, p))
    # interpret
    alpha = 0.05
    if p > alpha:
        print('Same distributions (fail to reject H0)')
    else:
        print('Different distributions (reject H0)')


def calculate_mannwhitneyu(value1, value2):
    print('Results of Unpaired Mann Whiteny Test')
    stat, p = mannwhitneyu(value1, value2)
    print('Statistics= {0}, p= {1}'.format(stat, p))
    # interpret
    alpha = 0.05
    if p > alpha:
        print('Same distribution (fail to reject H0)')
    else:
        print('Different distribution (reject H0)')

def cliffsDelta(lst1, lst2, **dull):

    """Returns delta and true if there are more than 'dull' differences"""
    if not dull:
        dull = {'small': 0.147, 'medium': 0.33, 'large': 0.474} # effect sizes from (Hess and Kromrey, 2004)
    m, n = len(lst1), len(lst2)
    lst2 = sorted(lst2)
    j = more = less = 0
    for repeats, x in runs(sorted(lst1)):
        while j <= (n - 1) and lst2[j] < x:
            j += 1
        more += j*repeats
        while j <= (n - 1) and lst2[j] == x:
            j += 1
        less += (n - j)*repeats
    d = (more - less) / (m*n)
    size = lookup_size(d, dull)
    print("Delta Effect Size is {}".format(size))
    print("Delta Effect value is {}".format(d))
    return d, size

def lookup_size(delta: float, dull: dict) -> str:
    """
    :type delta: float
    :type dull: dict, a dictionary of small, medium, large thresholds.
    """
    delta = abs(delta)
    if delta < dull['small']:
        return 'negligible'
    if dull['small'] <= delta < dull['medium']:
        return 'small'
    if dull['medium'] <= delta < dull['large']:
        return 'medium'
    if delta >= dull['large']:
        return 'large'


def runs(lst):
    """Iterator, chunks repeated values"""
    for j, two in enumerate(lst):
        if j == 0:
            one, i = two, 0
        if one != two:
            yield j - i, one
            i = j
        one = two
    yield j - i + 1, two

# df_2 = pd.read_csv("Results/results_10_fold_avg_3.csv")

models_list = {
    'DT' : [0.8017265289418413,0.8977030751087224,0.7824018225919535,0.9485634065139689,0.9825935225647962,0.4169972035934769,
            0.8684871677556781,0.9378001397832281,0.5386492129455778,0.8083,0.8093,0.7877,0.8092547559407981,0.7974554684912455],
    'XGBoost' : [0.8701428650061835,0.9325933921306262,0.8061198256443085,0.9296742626705988,0.9775246802863243,0.658201896016054,
                 0.8986787785121371,0.9539644356030033,0.723478555909052,0.8595,0.8619999999999999,0.8576,0.8620625625047404,0.9240307055009612],
    'RandomForest' : [0.8616848395771861,0.9124909202217066,0.745655379605953,0.9050092869403688,0.9867280157417853,0.6311657033202129,
              0.8825412197646919,0.9475338809345402,0.6827383471088317,0.8364,0.8405999999999999,0.8362,0.8406999171320235,0.9012057125518792],
    'NaiveBayes' : [0.7113148481001954,0.3570926943645599,0.6008610896973245,0.7174395865751337,0.912255124513373,0.20396438255876168,
                    0.7136606026285681,0.5060705572727751,0.2998844558119226,0.6478,0.5995000000000001,0.5809,0.5994549340772111,0.5218983203963552],
    'KNN' : [0.73209371455766,0.48046707147731366,0.4816202109798716,0.6982224797770809,0.466380150056222,0.5233649931146376,0.711908323289503,
             0.4625098639076219,0.4999529735250065,0.6396000000000001,0.6321,0.6322000000000001,0.6321321023285813,0.7583315787394046],
    'LinearSVC' : [0.7588605723675672,0.25234739822499813,0.571456718743137,0.8421113984752505,0.14811867636029075,0.5600343932577497,
                   0.7947508870126446,0.12839721119034922,0.5639556117410346,0.6566,0.7007000000000001,0.6666,0.7007509719352027,0.6960437158592057],
    'LR' : [0.8240110861001343,0.7216962340551639,0.6423201287481592,0.8196603500551888,0.9049180426366059,0.5308868140267607,0.8200910927484231,
            0.7362326258195615,0.5786169636486463,0.7651000000000001,0.7562,0.7502,0.7561194977857515,0.7953126280616432],
}

print(models_list['XGBoost'][11])

feature_models_list = {
    'All' : [0.86582,0.8678799999999999,0.8637599999999999,0.8678821873905213],
    'Integrator' : [0.7376,0.7879,0.7433,0.78789553173463],
    'Pull_Request' : [0.7196,0.7307,0.6966,0.730725310347368],
    'Contributor' : [0.6837,0.7046,0.6397,0.704645080487715],
    'Project' : [0.5038,0.5596,0.5287,0.5595830110141083],
}


calcualte_kruskal_test(models_list['DT'][11], models_list['XGBoost'][11], models_list['RandomForest'][11], models_list['NaiveBayes'][11],
                       models_list['KNN'][11], models_list['LinearSVC'][11], models_list['LR'][11])
# calcualte_kruskal_test(feature_models_list['All'], feature_models_list['Integrator'], feature_models_list['Pull_Request'],
#                        feature_models_list['Contributor'], feature_models_list['Project'])

# for m in models_list:
#     print(m)
#     # print(models_list[m])
#     if m == 'XGBoost':
#         continue
#     else:
#         calculate_mannwhitneyu(models_list['XGBoost'], models_list[m])
#         d, size = cliffsDelta(models_list['XGBoost'], models_list[m])

# calculate_mannwhitneyu([0.86582,0.8678799999999999,0.8637599999999999,0.8678821873905213],
#                        [0.7376, 0.7879, 0.7433, 0.78789553173463])
# d, size = cliffsDelta([0.86582,0.8678799999999999,0.8637599999999999,0.8678821873905213],
#                        [0.7376, 0.7879, 0.7433, 0.78789553173463])


def extract_selected_features():

    df_f = pd.read_csv('Saved_Models/3_labels/features_selected.csv')

    df_f = df_f.loc[df_f.threshold > 0.008]
    print(df_f.sort_values(by=['threshold']))
    print(list(df_f.features))
    print(df_f.shape)

# extract_selected_features()

def analyze_directly_accepted_prs():
    print('Analysis')
    df = pd.read_csv('Dataset_Samples/Accepted/all.csv', encoding='windows-1252')
    print(df.head())


# analyze_directly_accepted_prs()

# df = pd.read_csv('E:\Research Work\PRs_project\Prioritization\Results\MAP_AR\MAP_samples.csv', encoding='utf-8')
# df = pd.read_csv('E:\Research Work\PRs_project\Prioritization\Results\MAP_AR\AR_samples.csv', encoding='utf-8')

# For top-10 MAP
# FIFO = [0.397222222, 0.7438492059999999, 0.855555556, 0.773809524, 0.6083333329999999, 0.532738095, 0.621164021,
#         0.642857143, 0.48611111100000004, 0.37777777799999995]
# SBM = [0.111111111, 0.797222222, 0.35, 0.242063492, 0.370555556, 0.36103174600000004, 0.54047619, 0.26666666699999997,
#        0.444285714, 0.2]
# MCM = [0.582275132, 0.78859127, 0.6438492060000001, 0.880257937, 0.543849206, 0.797222222, 0.8552579370000001,
#        0.8338789679999999, 0.625925926, 0.581944444]
# For top-20 MAP
MCM = [0.6158507089999999, 0.7899486440000001, 0.699250602, 0.8416750909999999, 0.608735951, 0.795666007, 0.847794707,
       0.793102064, 0.640599051, 0.613753671]
FIFO = [0.48370628600000004, 0.678705505, 0.7323674440000001, 0.7129016840000001, 0.516488296, 0.582889311,
        0.6359732370000001, 0.3764324, 0.463796109, 0.358217469]
SBM = [0.257897835, 0.794593315, 0.42918414899999996, 0.346996676, 0.407608252, 0.440099206, 0.533593824,
       0.28052503100000004, 0.527287289, 0.325732763]

# For top-20 AR
# SBM = [0.41176470600000004, 0.541666667, 0.434782609, 0.409090909, 0.5, 0.526315789, 0.545454545, 0.46153846200000004,
#        0.6315789470000001, 0.6]
# FIFO = [0.588235294, 0.45833333299999995, 0.565217391, 0.590909091, 0.5625, 0.6315789470000001, 0.545454545,
#         0.46153846200000004, 0.42105263200000004, 0.333333333]
# MCM = [0.764705882, 0.666666667, 0.695652174, 0.772727273, 0.875, 0.842105263, 0.772727273, 0.923076923,
#        0.6315789470000001, 0.7333333329999999]


# calculate_mannwhitneyu(MCM, SBM)
# d, size = cliffsDelta(MCM, SBM)
#
# calculate_mannwhitneyu(MCM, FIFO)
# d, size = cliffsDelta(MCM, FIFO)

# print(list(df['Top_20'].loc[df.Model == 'MCM']))


# df_mc = pd.read_csv("E:\\Research Work\\Sentiment Analysis Project\\Dataset\\Full_Dataset\\dataset_06_12_19\\accept_multilabel_2.csv",
#           sep=',', encoding='utf-8')
#
# df_mc = df_mc[(df_mc.Project_Name != 'githubschool') & (df_mc.Project_Name != 'curriculum')]
#
# print(df_mc.shape)
# print(df_mc['label'].value_counts())
#
# factor = pd.factorize(df_mc['label'])
# df_mc.label = factor[0]
# definitions = factor[1]
# # print(df.label.head())
# # print(definitions)
#
# df_mc = df_mc[['Closed_Num_Rate', 'Label_Count','Following', 'Contributions', 'Merge_Latency', 'Followers', 'Comments_Count',
#          'Workload', 'Closed_Num', 'Public_Repos', 'Contributor', 'File_Touched_Average', 'Forks_Count', 'Additions',
#          'Project_Age', 'Open_Issues', 'Participants_Count', 'Comments_Per_Closed_PR',  'Review_Comments_Count', #'src_churn',
#          'Project_Accept_Rate', 'Accept_Num', 'Close_Latency', 'Commits_Average', 'Commits_PR', 'Wait_Time', #'num_comments',
#          'line_comments_count', 'Prev_PRs', 'Comments_Per_Merged_PR', 'Files_Changed', 'Churn_Average', 'Deletions',
#          'User_Accept_Rate', 'X1_0', 'X1_1', 'X1_2', 'X1_3', 'X1_4', 'X1_5', 'X1_6', 'X1_7', 'X1_8', 'first_response_time',
#          'latency_after_first_response', 'title_words_count', 'comments_reviews_words_count', 'PR_Time_Created_At', 'PR_Date_Closed_At',
#           'PR_Time_Closed_At', 'PR_Date_Created_At', 'label']]
#
# start_date = '2017-09-01'
# end_date = '2018-02-28'
# X_test = df_mc.loc[(df_mc['PR_Date_Created_At'] >= start_date) & (df_mc['PR_Date_Created_At'] <= end_date)]
# y_test = X_test['label']
# X_train = df_mc.loc[(df_mc['PR_Date_Created_At'] < start_date)]
# y_train = X_train['label']
#
# predictors = [x for x in df_mc.columns if x not in ['label', 'PR_Date_Created_At', 'PR_Time_Created_At', 'PR_Date_Closed_At',
#                                                  'PR_Time_Closed_At']]
#
#
# def get_smote_under_sampled_dataset(X, y):
#
#     smote = SMOTE(random_state=42)
#     X_sm, y_sm = smote.fit_sample(X, y)
#     print(X_sm.shape)
#     print(y_sm.shape)
#     X_sm = pd.DataFrame(X_sm, columns=predictors)
#     y_sm = pd.DataFrame(y_sm, columns=['label'])
#
#     return X_sm, y_sm
#
# X, y = get_smote_under_sampled_dataset(X_train[predictors], y_train)
#
# print(X.shape)
# print(y['label'].value_counts())
# # print(Counter(y))

def create_data_for_Scott_test():
    df = pd.read_csv('Results/results_10_fold_5.csv')
    print(list(df.Model.unique()))
    model_list = ['RandomForest', 'LinearSVC', 'LogisticRegression', 'KNN', 'DT', 'XGBoost', 'NaiveBayes']
    # df_models = pd.DataFrame()
    dict = {}
    for model in model_list:
        fscore = df.Avg_f1_Score[df.Model == model]
        print(model)
        dict[model] = list(pd.to_numeric(fscore))
        # print(fscore)
        # df_models = df_models.append({'model': model, 'f_score': fscore}, ignore_index=True)
    df_models = pd.DataFrame(dict)
    print(df_models.head(10))
    df_models.to_csv('Results/scott_fscore.csv', sep=',', encoding='utf-8', index=False)



# create_data_for_Scott_test()