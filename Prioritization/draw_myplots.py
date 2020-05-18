import pandas as pd
import matplotlib.pyplot as plt
plt.rc("font", size=8)
import seaborn as sns
sns.set(style="white")
sns.set(style='whitegrid', color_codes=True)
import numpy as np
# import openpyxl

from scipy.stats import mannwhitneyu

# print(sns.__version__)

df_MAP_all = pd.read_csv("../Accept_Response/Results/CART/10_samples_results/All_MAP_results.csv", sep=",")
df_AR_all = pd.read_csv("../Accept_Response/Results/CART/10_samples_results/All_AR_results.csv", sep=",")

# df_MAP_all = pd.read_csv("Results/MAP_AR/3_labels/MAP_test_all_models.csv", sep=",")
# df_AR_all = pd.read_csv("Results/MAP_AR/3_labels/AR_test_all_models.csv", sep=",")


df_accept = pd.read_csv("../Accept_Response/R analysis/accept_projects_all.csv", sep=",")
df_accept_d = pd.read_csv("../Accept_Response/Results/accept/results_projects_OS_1.csv", sep=",")

df_response = pd.read_csv("../Accept_Response/R analysis/response_projects_all.csv", sep=",")
df_response_d = pd.read_csv("../Accept_Response/Results/response/results_projects_OS_1.csv", sep=",")

df_accept = df_accept[(df_accept['Model'] == 'XGBoost') | (df_accept['Model'] == 'baseline')]
df_response = df_response[(df_response['Model'] == 'XGBoost') | (df_response['Model'] == 'baseline')]

# print(df_accept.head())

# Model Project AUC  Precision  Recall  F-measure  Accuracy

classifiers = ['RandomForest', 'LinearSVC', 'LogisticRegression', 'XGBoost']
project_list = ['react', 'django', 'nixpkgs', 'scikit-learn', 'yii2', 'cdnjs', 'terraform', 'cmssw', 'salt',
                'tensorflow', 'pandas', 'symfony', 'moby', 'rails', 'rust', 'kubernetes', 'angular.js', 'laravel',
                'opencv']

# print((df.loc[(df['Project'] == 'react') & (df['Model'] == 'random forest')]))

# def draw_lineplot_AUC_10():
#     df_avg_accept = pd.DataFrame(
#         columns=['Project', 'Model', 'Avg_AUC', 'Avg_Precision', 'Avg_Recall', 'Avg_F-measure', 'Avg_Accuracy'])
#     df_avg_response = pd.DataFrame(
#         columns=['Project', 'Model', 'Avg_AUC', 'Avg_Precision', 'Avg_Recall', 'Avg_F-measure', 'Avg_Accuracy'])
#
#     for clf in classifiers:
#         for project in project_list:
#             print(clf, project)
#             # print((df.loc[(df['Project'] == project) & (df['Model'] == clf)]).mean().to_frame())
#             # print(clf, project, df['AUC'].mean())
#             df_p = (df_accept.loc[(df_accept['Project'] == project) & (df_accept['Model'] == clf)])
#             df_avg_accept = df_avg_accept.append({'Project': project, 'Model': clf, 'Avg_AUC': df_p['AUC'].mean(), 'Avg_Precision': df_p['Precision'].mean(),
#                            'Avg_Recall': df_p['Recall'].mean(), 'Avg_F-measure': df_p['F-measure'].mean(), 'Avg_Accuracy': df_p['Accuracy'].mean()},
#                           ignore_index=True)
#             print(df_avg_accept)
#             df_p1 = (df_response.loc[(df_accept['Project'] == project) & (df_response['Model'] == clf)])
#             df_avg_response = df_avg_response.append(
#                 {'Project': project, 'Model': clf, 'Avg_AUC': df_p1['AUC'].mean(), 'Avg_Precision': df_p1['Precision'].mean(),
#                  'Avg_Recall': df_p1['Recall'].mean(), 'Avg_F-measure': df_p1['F-measure'].mean(),
#                  'Avg_Accuracy': df_p1['Accuracy'].mean()},
#                 ignore_index=True)
#             print(df_avg_response)
#
#     fig, ax = plt.subplots(figsize=(12,6), ncols=2, nrows=1)
#
#     sns.lineplot(x=df_avg_accept['Project'], y=df_avg_accept['Avg_AUC'], hue='Model', data=df_avg_accept, palette='hls', style='Model',
#                  dashes=False, markers=True, ax=ax[0])
#     ax[0].set_xlabel("Projects")
#     ax[0].set_ylabel("Average AUC Accept")
#     # ax[0].set_title('Something here')
#     plt.sca(ax[0])
#     plt.xticks(rotation=45)
#     plt.legend(['RF', 'SVM', 'LR', 'XGB'])
#
#     sns.lineplot(x=df_avg_response['Project'], y=df_avg_response['Avg_AUC'], hue='Model', data=df_avg_response, palette='hls', style='Model',
#                  dashes=False, markers=True, ax=ax[1])
#
#     ax[1].set_xlabel("Projects")
#     ax[1].set_ylabel("Average AUC Response")
#     # ax[1].set_title('Something here')
#     plt.sca(ax[1])
#     plt.legend(['RF', 'SVM', 'LR', 'XGB'])
#     plt.xticks(rotation=45)
#
#     # plt.savefig('auc_projects_test_train.png')
#     plt.show()



def draw_lineplot_for_metric(df_A, df_R, x_value, y_value, ylabel1, ylabel2, hue_value, img_name):

    fig, ax = plt.subplots(figsize=(12,5), ncols=2, nrows=1)
    # line_labels = ['RF', 'SVM', 'LR', 'XGB', 'Baseline']
    line_labels = ['XGB', 'Baseline']
    sns.lineplot(x=df_A[x_value], y=df_A[y_value], hue=hue_value, data=df_A, palette='hls', style='Model', #legend=False,
                 dashes=False, markers=True, ax=ax[0])
    ax[0].set_xlabel("Projects")
    ax[0].set_ylabel(ylabel1)
    yticks_labels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    ax[0].set_yticks(yticks_labels)
    # ax[0].set_title('Something here')
    plt.sca(ax[0])
    plt.xticks(rotation=45)
    plt.legend(['XGB', 'Gousios et al'])

    sns.lineplot(x=df_R[x_value], y=df_R[y_value], hue=hue_value, data=df_R, palette='hls', style='Model', #legend=False,
                 dashes=False, markers=True, ax=ax[1])
    ax[1].set_xlabel("Projects")
    ax[1].set_ylabel(ylabel2)
    yticks_labels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    ax[1].set_yticks(yticks_labels)
    # ax[1].set_title('Something here')
    plt.sca(ax[1])
    plt.legend(['XGB', 'PRioritizer'])
    plt.xticks(rotation=45)
    # fig.legend(line_labels, loc = 9, ncol=4)
    # fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    # plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    plt.tight_layout()
    plt.savefig('plots/accept_response/'+img_name+'.png', bbox_inches='tight')
    # plt.show()

# draw_lineplot_for_metric(df_MAP_all, df_AR_all, 'Sample', y_value, 'MAP', 'AR',
#                              'Model', 'comparison_'+y_value)

# y_value = 'Recall'
# y_value = 'Precision'
# y_value = 'AUC'
# y_value = 'F-Score'
# value_list = ['Recall', 'Precision', 'AUC', 'F-Score', 'Test_Accuracy']
# for y_value in value_list:
#     # print(y_value)
#     # draw_lineplot_for_metric(df_accept, df_accept_d, 'Project', y_value, y_value+' AP OS', y_value+' OS', 'Model',
#     #                          y_value+'_projects_3')
#     # draw_lineplot_for_metric(df_accept, df_accept_d, 'Project', y_value, y_value, y_value + ' OS', 'Model',
#     #                          'projects_4_'+y_value)
#     draw_lineplot_for_metric(df_accept, df_response, 'Project', y_value, y_value+' Acceptance', y_value+' Response',
#                              'Model', 'comparison_'+y_value)

# def draw_single_lineplot_for_metric(df, x_value, y_value, xlabel, ylabel, img_name):
#
#     fig, ax = plt.subplots(figsize=(10,6), ncols=1, nrows=1)
#
#     sns.lineplot(x=df[x_value], y=df[y_value], hue='Model', data=df, palette='hls', style='Model',
#                  dashes=False, markers=True, ax=ax)
#     ax.set_xlabel(xlabel)
#     ax.set_ylabel(ylabel)
#     # ax[0].set_title('Something here')
#     # plt.sca(ax[0])
#     plt.xticks(rotation=45)
#     plt.legend(['RF', 'SVM', 'LR', 'XGB'])
#
#     # plt.savefig(img_name+'.png', bbox_inches='tight')
#     plt.show()


# def draw_barplot(df, x_value, y_value, xlabel, ylabel, img_name):
#     fig, ax = plt.subplots(figsize=(10,6), ncols=1, nrows=1)
#
#     sns.barplot(x=df[x_value], y=df[y_value], hue='Model', data=df, palette='hls', ax=ax)
#     ax.set_xlabel(xlabel)
#     ax.set_ylabel(ylabel)
#     # ax[0].set_title('Something here')
#     # plt.sca(ax[0])
#     plt.xticks(rotation=45)
#     plt.legend(['RF', 'SVM', 'LR', 'XGB'])
#
#     # plt.savefig(img_name+'.png', bbox_inches='tight')
#     plt.show()

# draw_barplot(df_accept_language, 'Language', 'AUC', 'Projects', 'AUC Average', 'myimage')

# def calculate_average_all_metrics(df_A, df_R):
#     df_avg_accept = pd.DataFrame(
#         columns=['Model', 'Avg_AUC', 'Avg_Precision', 'Avg_Recall', 'Avg_F-measure', 'Avg_Accuracy'])
#     df_avg_response = pd.DataFrame(
#         columns=['Model', 'Avg_AUC', 'Avg_Precision', 'Avg_Recall', 'Avg_F-measure', 'Avg_Accuracy'])
#     for clf in classifiers:
#         df_a = df_accept.loc[df_accept['Model'] == clf]
#         df_avg_accept = df_avg_accept.append(
#             {'Model': clf, 'Avg_AUC': df_a['AUC'].mean(), 'Avg_Precision': df_a['Precision'].mean(),
#              'Avg_Recall': df_a['Recall'].mean(), 'Avg_F-measure': df_a['F-measure'].mean(),
#              'Avg_Accuracy': df_a['Accuracy'].mean()},
#             ignore_index=True)
#         df_r = df_response.loc[df_response['Model'] == clf]
#         df_avg_response = df_avg_response.append(
#             {'Model': clf, 'Avg_AUC': df_r['AUC'].mean(), 'Avg_Precision': df_r['Precision'].mean(),
#              'Avg_Recall': df_r['Recall'].mean(), 'Avg_F-measure': df_r['F-measure'].mean(),
#              'Avg_Accuracy': df_r['Accuracy'].mean()},
#             ignore_index=True)
#         # print(df_avg)
#     print(df_avg_accept)
#     print(df_avg_response)
#     df_avg_accept.to_csv('Calculation_Results/accept_avg_metrics.csv', sep=',', encoding='utf-8')
#     df_avg_response.to_csv('Calculation_Results/response_avg_metrics.csv', sep=',', encoding='utf-8')


def draw_scattorplot_MAP_AR_Together(df_map, df_Ar, x_value, y_value, ylabel1, ylabel2, hue_value, img_name):

    fig, ax = plt.subplots(figsize=(12,5), ncols=2, nrows=1)
    # markers = {"CARTESIAN": "s", "FIFO": "X", "SBM": "+"}

    sns.scatterplot(x=df_map[x_value], y=df_map[y_value], hue=hue_value, data=df_map, palette='hls', style=hue_value, #legend=False,
                   ax=ax[0], s=100)
    ax[0].set_xlabel("Samples")
    ax[0].set_ylabel(ylabel1)
    yticks_labels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    ax[0].set_yticks(yticks_labels)
    plt.sca(ax[0])
    plt.xticks(rotation=45)
    # plt.legend(['CARTESIAN', 'FIFO', 'SBM'])

    sns.scatterplot(x=df_Ar[x_value], y=df_Ar[y_value], hue=hue_value, data=df_Ar, palette='hls', style=hue_value, #legend=False,
                  ax=ax[1], s=100)

    ax[1].set_xlabel("Samples")
    ax[1].set_ylabel(ylabel2)
    yticks_labels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    ax[1].set_yticks(yticks_labels)
    plt.sca(ax[1])
    # plt.legend(['CARTESIAN', 'FIFO', 'SBM'])
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('plots/3_labels/'+img_name+'.png', bbox_inches='tight')
    plt.show()


def draw_lineplot_MAP_AR_Together(df_map, df_Ar, x_value, y_value, ylabel1, ylabel2, hue_value, img_name):

    fig, ax = plt.subplots(figsize=(12,5), ncols=2, nrows=1)

    sns.lineplot(x=df_map[x_value], y=df_map[y_value], hue=hue_value, data=df_map, palette='hls', style=hue_value, legend=False,
                 dashes=False, markers=True, ax=ax[0])
    ax[0].set_xlabel("Samples")
    ax[0].set_ylabel(ylabel1)
    yticks_labels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    ax[0].set_yticks(yticks_labels)
    plt.sca(ax[0])
    plt.xticks(rotation=45)
    plt.legend(['AR-Prioritizer', 'Gousios model', 'PRioritizer'])

    sns.lineplot(x=df_Ar[x_value], y=df_Ar[y_value], hue=hue_value, data=df_Ar, palette='hls', style=hue_value, legend=False,
                 dashes=False, markers=True, ax=ax[1])

    ax[1].set_xlabel("Samples")
    ax[1].set_ylabel(ylabel2)
    yticks_labels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    ax[1].set_yticks(yticks_labels)
    plt.sca(ax[1])
    plt.legend(['AR-Prioritizer', 'Gousios model', 'PRioritizer'])
    plt.xticks(rotation=45)
    # line_labels = ['RF', 'SVM', 'LR', 'XGB']
    plt.tight_layout()
    plt.savefig('plots/CART/'+img_name+'.png', bbox_inches='tight')
    # plt.show()

# top_k = 'Top_30'
# top_label = 'Top@30'

top_k_list = {'Top_5': 'Top@5', 'Top_10': 'Top@10', 'Top_20': 'Top@20', 'Top_30': 'Top@30'}

for k in top_k_list:
    # print(top_k_list[k])
    draw_lineplot_MAP_AR_Together(df_MAP_all, df_AR_all, 'Sample', k, 'Mean Average Precision ('+top_k_list[k]+')',
                              'Average Recall ('+top_k_list[k]+')', 'Model', 'MAP_AR_'+k)

# draw_scattorplot_MAP_AR_Together(df_MAP_all, df_AR_all, 'Sample', top_k, 'Mean Average Precision ('+top_label+')',
#                                  'Average Recall (' + top_label + ')', 'Model', '4_models_' + top_k)




def draw_single_lineplot_MAP_AR(df_, x_value, y_value, xlabel, ylabel, img_name):

    fig, ax = plt.subplots(figsize=(6,5), ncols=1, nrows=1)

    sns.lineplot(x=df_[x_value], y=df_[y_value], data=df_, hue='Models', palette='hls', style='Models',
                 dashes=False, markers=True, ax=ax)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    yticks_labels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    ax.set_yticks(yticks_labels)
    plt.xticks(rotation=45)

    plt.savefig('plots/old/'+img_name+'.png', bbox_inches='tight')
    plt.show()

# draw_single_lineplot_MAP_AR(df_MAP_all, 'Project', 'Top-10', 'Projects', 'Mean Average Precision (Top@10)', 'map_top10')




# def test_cart_output_sum_avg_exp():
#     df_avg = pd.read_csv(
#         '/home/ppf/PycharmProjects/PRs_Prioritization/Models/PR_Algorithm/Results/PR_algo/Response_test_dataset/XGB/test_results/algo_results_avg.csv',
#                          encoding='utf-8', sep=',')
#     df_exp = pd.read_csv(
#         '/home/ppf/PycharmProjects/PRs_Prioritization/Models/PR_Algorithm/Results/PR_algo/Response_test_dataset/XGB/test_results/algo_results_exp.csv',
#         encoding='utf-8', sep=',')
#     df_sum = pd.read_csv(
#         '/home/ppf/PycharmProjects/PRs_Prioritization/Models/PR_Algorithm/Results/PR_algo/Response_test_dataset/XGB/test_results/algo_results_sum.csv',
#         encoding='utf-8', sep=',')
#
#     df_avg['type'] = 'avg'
#     df_exp['type'] = 'exp'
#     df_sum['type'] = 'sum'
#
#     # df_all = pd.DataFrame()
#     # df_all = df_all.append(df_sum[['Pull_Request_ID', 'Score', 'type']])
#     # df_all = df_all.append(df_exp[['Pull_Request_ID', 'Score', 'type']])
#     # df_all = df_all.append(df_avg[['Pull_Request_ID', 'Score', 'type']])
#     # print(df_all.head())
#     # print(df_all.shape)
#
#     # sns.lineplot(x=range(0, len(df_avg['Pull_Request_ID'])), y=df_avg['Score'])
#     sns.lineplot(x=range(0,len(df_exp['Pull_Request_ID'])), y=df_exp['Score'])
#     # sns.lineplot(x=range(0, len(df_sum['Pull_Request_ID'])), y=df_sum['Score'])
#     plt.savefig('plots/exp.png', bbox_inches='tight')
#     plt.show()


# test_cart_output_sum_avg_exp()

# def draw_latency_lineplot_MAP_AR(df_, x_value, y_value, xlabel, ylabel, img_name):
#
#     fig, ax = plt.subplots(figsize=(6,5), ncols=1, nrows=1)
#
#     sns.lineplot(x=x_value, y=y_value, data=df_, palette='hls',
#                  dashes=False, markers=True, ax=ax)
#
#     sns.lineplot(x=range(0,len(df_latency_1['Pull_Request_ID'])), y=df_latency_1['first_response']/24/7, data=df_, palette='hls',
#                  dashes=False, markers=True, ax=ax)
#
#     ax.set_xlabel(xlabel)
#     ax.set_ylabel(ylabel)
#     xticks_labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
#     ax.set_xticks(xticks_labels)
#     # yticks_labels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
#     # ax.set_yticks(yticks_labels)
#     # plt.xticks(rotation=45)
#     # plt.savefig('plots/'+img_name+'.png', bbox_inches='tight')
#     plt.show()

# draw_latency_lineplot_MAP_AR(df_latency, range(0,len(df_latency['Pull_Request_ID'])), df_latency['first_response']/24/7,
#                              'PRs', 'PR latency', 'xyz')


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



def calculate_cliff_delta(value_1, value_2):

    d, size = cliffsDelta(value_1, value_2)

    print("Delta Effect Size is {}".format(size))
    print("Delta Effect value is {'%.2f'% 0}".format(d))


# df_cart_map = df_MAP_all.loc[df_MAP_all['Model'] == 'CARTESIAN']
# df_fifo_map = df_MAP_all.loc[df_MAP_all['Model'] == 'FIFO']
# df_ssf_map = df_MAP_all.loc[df_MAP_all['Model'] == 'SSF']
#
# df_cart_AR = df_AR_all.loc[df_AR_all['Model'] == 'CARTESIAN']
# df_fifo_AR = df_AR_all.loc[df_AR_all['Model'] == 'FIFO']
# df_ssf_AR = df_AR_all.loc[df_AR_all['Model'] == 'SSF']

# calculate_cliff_delta(df_cart_AR['Top_10'], df_fifo_AR['Top_10'])


# print(df_cart_map[['Top_5', 'Top_10', 'Top_20']].mean())
# print(df_cart_AR[['Top_5', 'Top_10', 'Top_20']].mean())
#
# print(df_fifo_map[['Top_5', 'Top_10', 'Top_20']].mean())
# print(df_fifo_AR[['Top_5', 'Top_10', 'Top_20']].mean())
#
# print(df_ssf_map[['Top_5', 'Top_10', 'Top_20']].mean())
# print(df_ssf_AR[['Top_5', 'Top_10', 'Top_20']].mean())


"""
CART vs FIFO MAP
Delta Effect Size is large
Delta Effect value is 0.58
CART vs SSF MAP
Delta Effect Size is large
Delta Effect value is 0.60

CART vs FIFO AR
Delta Effect Size is large
Delta Effect value is 0.80
CART vs SSF AR
Delta Effect Size is large
Delta Effect value is 0.77

"""

def calculate_mannwhitneyu(value1, value2):
    print('\nResults of Unpaired Mann Whiteny Test')
    stat, p = mannwhitneyu(value1, value2)
    print('Statistics= {"%.3f" % 0}, p= {"%.9f" % 1}'.format(stat, p))
    # interpret
    alpha = 0.05
    if p > alpha:
        print('Same distribution (fail to reject H0)')
    else:
        print('Different distribution (reject H0)')


# calculate_mannwhitneyu(df_cart_AR['Top_10'], df_ssf_AR['Top_10'])


"""
CART vs FIFO MAP
Results of Unpaired Mann Whiteny Test
Statistics= 54.000, p= 0.002801649
Different distribution (reject H0)
CART vs SSF MAP
Results of Unpaired Mann Whiteny Test
Statistics= 51.000, p= 0.001968273
Different distribution (reject H0)

CART vs FIFO AR
Results of Unpaired Mann Whiteny Test
Statistics= 26.000, p= 0.000065273
Different distribution (reject H0)
CART vs SSF AR
Results of Unpaired Mann Whiteny Test
Statistics= 29.000, p= 0.000102668
Different distribution (reject H0)
"""

def draw_features_barplot():
    df = pd.read_csv("../Models/Saved_Models/3_labels/features_fscore_2.csv", sep=",")
    df = df.sort_values(by=['fscore'], ascending=False)
    df = df[df.fscore >= 10]
    print(len(df.feature))
    print(list(df.feature))
    fig, ax = plt.subplots(figsize=(8, 3))
    df['fscore_log'] = np.log(df['fscore'])
    sns.set(style="whitegrid")
    tips = sns.load_dataset("tips")
    sns.barplot(x="feature", y="fscore_log", data=df, ax=ax, palette="GnBu_d") #palette="Blues_d" GnBu_d ch:2.5,-.2,dark=.3
    ax.xaxis.set_tick_params(labelsize=9)
    ax.set_xlabel('Features')
    ax.set_ylabel('Average Gain (log-scaled)')
    plt.xticks(rotation=90)
    plt.savefig('plots/features_fscore_5.png', bbox_inches='tight')
    plt.show()



# draw_features_barplot()


def draw_sample_results_barplot():
    df = pd.read_csv("Random_samples/381_samples_results.csv", sep=",")

    fig, ax = plt.subplots(figsize=(5, 3))
    sns.set(style="whitegrid")
    # tips = sns.load_dataset("tips")
    sns.barplot(x="Sample", y="Avg_f1_Score", data=df, ax=ax) #palette="Blues_d"
    ax.xaxis.set_tick_params(labelsize=9)
    ax.set_xlabel('Test Samples')
    ax.set_ylabel('F-Score')
    yticks_labels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    ax.set_yticks(yticks_labels)
    plt.xticks(rotation=45)
    plt.savefig('plots/sample_fscore.png', bbox_inches='tight')
    plt.show()

# draw_sample_results_barplot()