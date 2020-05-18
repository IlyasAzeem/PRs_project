import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV, train_test_split

from tensorflow import keras

import time
from sklearn.utils import shuffle
from keras.utils import to_categorical
from imblearn.over_sampling import SMOTE




df = pd.read_csv("E:\\Research Work\\Sentiment Analysis Project\\Dataset\\Full_Dataset\\dataset_06_12_19\\3_multilabel.csv",
          sep=',', encoding='utf-8')

# 'cmssw'
project_list = ['react', 'django', 'nixpkgs', 'scikit-learn', 'yii2', 'cdnjs', 'terraform', 'cmssw', 'salt', 'tensorflow', 'pandas',
                'symfony', 'moby', 'rails', 'rust', 'kubernetes', 'angular.js', 'laravel', 'opencv',
                ]

# Remove some of the PRs with negative latency
# ID = df.Pull_Request_ID[df.latency_after_first_response < 0]
# df = df.loc[~df.Pull_Request_ID.isin(ID)]

df = df[(df.Project_Name != 'githubschool') & (df.Project_Name != 'curriculum')]

print(df.shape)


scoring = ['precision', 'recall', 'f1', 'roc_auc', 'accuracy']
results = pd.DataFrame(columns=['Model', 'AUC', 'Precision', 'Recall', 'F-measure', 'Accuracy'])

def encode_labels(df1, column_name):
    encoder = LabelEncoder()
    df1[column_name] = [str(label) for label in df1[column_name]]
    encoder.fit(df1[column_name])
    one_hot_vector = encoder.transform(df1[column_name])
    return one_hot_vector


#Creating the dependent variable class
# factor = pd.factorize(df['label'])
# df.label = factor[0]
# definitions = factor[1]
# print(df.label.head())
# print(definitions)

df['Language'] = encode_labels(df, 'Language')
df['Project_Domain'] = encode_labels(df, 'Project_Domain')
df['label'] = encode_labels(df, 'label')
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

df = df[['num_comments', 'Contributor', 'Participants_Count', 'line_comments_count', 'Deletions_Per_Week', 'Additions_Per_Week',
         'Project_Accept_Rate', 'Mergeable_State', 'User_Accept_Rate', 'first_response', 'Project_Domain', 'latency_after_first_response',
         'comments_reviews_words_count', 'Wait_Time', 'Team_Size', 'Stars', 'Language', 'Assignees_Count', 'Sunday', 'Contributor_Num',
         'Watchers', 'Last_Comment_Mention', 'Contributions', 'Saturday', 'Wednesday', 'Label_Count', 'Commits_PR', 'PR_Latency',
         'Comments_Per_Merged_PR', 'Organization_Core_Member', 'Comments_Per_Closed_PR', 'PR_Time_Created_At', 'PR_Date_Closed_At',
        'PR_Time_Closed_At', 'PR_Date_Created_At',
         'Project_Name', 'label']]

target = 'label'
start_date = '2017-09-01'
end_date = '2018-02-28'

df = df.sort_values(by=['PR_Date_Closed_At', 'PR_Time_Closed_At'], ascending=True)

X_test = df.loc[(df['PR_Date_Created_At'] >= start_date) & (df['PR_Date_Created_At'] <= end_date)]
y_test = X_test[target]
X_train = df.loc[(df['PR_Date_Created_At'] < start_date)]
y_train = X_train[target]

predictors = [x for x in df.columns if x not in [target, 'PR_accept', 'PR_Date_Created_At', 'PR_Time_Created_At', 'PR_Date_Closed_At',
                                                 'PR_Time_Closed_At', 'Project_Name']]


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
#     # shuffle(df_test_under)
#
#     y_df = df_test_under[target]
#
#     return df_test_under, y_df
#
#
# def get_over_sampled_dataset():
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
#     df_class_0_over = df_class_0.sample(count_class_1, replace=True)
#     df_test_over = pd.concat([df_class_0_over, df_class_1], axis=0)
#
#     print('Random over-sampling:')
#     print(df_test_over[target].value_counts())
#     shuffle(df_test_over)
#
#     y_df = df_test_over[target]
#
#     return df_test_over, y_df


# X_train, y_train = get_over_sampled_dataset()
# X_train, y_train = get_under_sampled_dataset()

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
y_train = to_categorical(y_train)
X_test = X_test[predictors]
y_test = to_categorical(y_test)

print("Total Train dataset size: {}".format(X_train.shape))
print("Total Test dataset size: {}".format(X_test.shape))
# print("Total validation dataset size: {}".format(X_val.shape))


# Scale the training dataset: StandardScaler

def scale_data_standardscaler(df_):
    scaler_train =StandardScaler()
    df_scaled = scaler_train.fit_transform(np.array(df_).astype('float64'))
    df_scaled = pd.DataFrame(df_scaled, columns=predictors)

    return df_scaled

X_train_scaled = scale_data_standardscaler(X_train[predictors])
X_test_scaled = scale_data_standardscaler(X_test[predictors])
#X_val_scaled = scale_data_standardscaler(X_val[predictors])

# METRICS = [
#       keras.metrics.TruePositives(name='tp'),
#       keras.metrics.FalsePositives(name='fp'),
#       keras.metrics.TrueNegatives(name='tn'),
#       keras.metrics.FalseNegatives(name='fn'),
#       keras.metrics.BinaryAccuracy(name='accuracy'),
#       keras.metrics.Precision(name='precision'),
#       keras.metrics.Recall(name='recall'),
#       # keras.metrics.AUC(name='auc'),
#     ]


def check_data_distribution(pos_df, neg_df):
    pos_df = scale_data_standardscaler(pos_df)
    neg_df = scale_data_standardscaler(neg_df)
    sns.jointplot(pos_df['Project_Age'], pos_df['PR_Latency'],
                  kind='hex', xlim=(-5, 5), ylim=(-5, 5))
    plt.suptitle("Positive distribution")

    sns.jointplot(neg_df['Project_Age'], neg_df['PR_Latency'],
                  kind='hex', xlim=(-5, 5), ylim=(-5, 5))
    _ = plt.suptitle("Negative distribution")
    plt.show()

# check_data_distribution(df_pos[predictors], df_neg[predictors])

def get_class_weights(pos_, neg_, total_):
    # Scaling by total/2 helps keep the loss to a similar magnitude.
    # The sum of the weights of all examples stays the same.
    weight_for_0 = (1 / neg_) * (total_) / 2.0
    weight_for_1 = (1 / pos_) * (total_) / 2.0

    class_weight = {0: weight_for_0, 1: weight_for_1}

    print('Weight for class 0: {:.2f}'.format(weight_for_0))
    print('Weight for class 1: {:.2f}'.format(weight_for_1))

    return class_weight

def get_initial_bias(pos_, neg_):
    initial_bias = np.log([pos_ / neg_])

    return initial_bias


input_neurons = len(predictors)

def get_optimizer():
    return {
        'rms': keras.optimizers.RMSprop(),
        'sgd': keras.optimizers.SGD(),
        'adam': keras.optimizers.Adam()
    }


def get_activator():
    return {
        'relu': keras.activations.relu(),
        'sigmoid': keras.activations.sigmoid(),
        'tanh': keras.activations.tanh()
    }


def draw_loss_plot(model_history, img_name):
    # Plot the loss function
    fig, ax = plt.subplots(1, 1, figsize=(10,6))
    ax.plot(np.sqrt(model_history.history['loss']), 'r', label='train')
    ax.plot(np.sqrt(model_history.history['val_loss']), 'b' ,label='val')
    ax.set_xlabel(r'Epoch', fontsize=20)
    ax.set_ylabel(r'Loss', fontsize=20)
    ax.legend()
    ax.tick_params(labelsize=20)
    plt.savefig('plots/' + img_name + '_loss.png')
    plt.show()

def draw_accuracy_plot(model_history, img_name):
    # Plot the accuracy
    fig, ax = plt.subplots(1, 1, figsize=(10,6))
    ax.plot(np.sqrt(model_history.history['accuracy']), 'r', label='train')
    ax.plot(np.sqrt(model_history.history['val_accuracy']), 'b' ,label='val')
    ax.set_xlabel(r'Epoch', fontsize=20)
    ax.set_ylabel(r'Accuracy', fontsize=20)
    ax.legend()
    ax.tick_params(labelsize=20)
    plt.savefig('plots/'+img_name+'_acc.png')
    plt.show()
# print(input_neurons)

def model_layer_and_neuron_units_optimizer(output_bias=None):
    dense_layers = [2, 3, 4, 5, 6]
    layer_sizes = [32, 40, 64, 128, 256, 512]

    if output_bias is not None:
        output_bias = tf.keras.initializers.Constant(output_bias)

    for dense_layer in dense_layers:
        for layer_size in layer_sizes:
            NAME = "Accept-{}-nodes-{}-dense-{}".format(layer_size, dense_layer, int(time.time()))
            print(NAME)

            model = keras.models.Sequential()
            model.add(keras.layers.Dense(layer_size, kernel_initializer='normal', kernel_regularizer=keras.regularizers.l2(0.001),
                                         input_dim=input_neurons, activation='relu'))
            model.add(keras.layers.Dropout(0.2))

            for l in range(dense_layer-1):
                model.add(keras.layers.Dense(layer_size, kernel_initializer='normal', kernel_regularizer=keras.regularizers.l2(0.001),
                                             activation='relu'))
                model.add(keras.layers.Dropout(0.2))

            model.add(keras.layers.Dense(1, kernel_initializer='normal', activation='sigmoid', bias_initializer=output_bias))

            tensorboard = keras.callbacks.TensorBoard(log_dir="logs/dropout/2/{}".format(NAME))

            model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

            model_history = model.fit(X_train_scaled, y_train, epochs=100, batch_size=512, verbose=0,
                            validation_data=(X_val_scaled, y_val), callbacks=[tensorboard])

            val_loss, val_acc = model.evaluate(X_val_scaled, y_val)
            test_loss, test_acc = model.evaluate(X_test_scaled, y_test)
            train_loss, train_acc = model.evaluate(X_train_scaled, y_train)

            print('Train accuracy: {}'.format(train_acc))
            print('Validation accuracy: {}'.format(val_acc))
            print('Test accuracy: {}'.format(test_acc))
            print('Train loss: {}'.format(train_loss))
            print('Validation loss: {}'.format(val_loss))
            print('Test loss: {}'.format(test_loss))

            y_pred = model.predict(X_test_scaled)
            y_pred = (y_pred > 0.5)

            y_predprob = model.predict_proba(X_test_scaled)

            print("\nModel Report")
            print("Accuracy : %.4g" % metrics.accuracy_score(y_test, y_pred))
            print("AUC Score : %f" % metrics.roc_auc_score(y_test, y_predprob))
            print("Recall : %f" % metrics.recall_score(y_test, y_pred))
            print("Precision : %f" % metrics.precision_score(y_test, y_pred))
            print("F-measure : %f" % metrics.f1_score(y_test, y_pred))
            c_matrix = metrics.confusion_matrix(y_test, y_pred)
            print('========Confusion Matrix==========')
            print("          Rejected    Accepted")
            print('Rejected     {}      {}'.format(c_matrix[0][0], c_matrix[0][1]))
            print('Accepted     {}      {}'.format(c_matrix[1][0], c_matrix[1][1]))
            # draw_accuracy_plot(model_history, NAME)
            # draw_loss_plot(model_history, NAME)


# model_layer_and_neuron_units_optimizer()

def create_model(opt='adam', init_mode='uniform', activator='softmax', output_bias=None):
    layer_size = 32
    NAME = "Accept-{}-nodes-{}-dense-{}".format(layer_size, 3, int(time.time()))
    # create model
    if output_bias is not None:
        output_bias = tf.keras.initializers.Constant(output_bias)
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(layer_size, kernel_regularizer=keras.regularizers.l2(0.001),
                                 input_dim=input_neurons, activation='relu'))
    model.add(keras.layers.Dense(64, kernel_regularizer=keras.regularizers.l2(0.001),
                                 activation='relu'))
    model.add(keras.layers.Dense(64, kernel_regularizer=keras.regularizers.l2(0.001),
                                 activation='relu'))
    model.add(keras.layers.Dense(64, kernel_regularizer=keras.regularizers.l2(0.001),
                           activation='relu'))
    # model.add(
    #     keras.layers.Dense(layer_size, kernel_regularizer=keras.regularizers.l2(0.001),
    #                        activation='relu'))
    # model.add(
    #     keras.layers.Dense(layer_size, kernel_regularizer=keras.regularizers.l2(0.001),
    #                        activation='relu'))
    # model.add(Dense(layer_size, activation='relu'))
    model.add(keras.layers.Dense(3, activation=activator, bias_initializer=output_bias))
    # compile model
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['acc'])
    return model, NAME


def run_model():
    classifier, model_name = create_model()
    model_history = classifier.fit(X_train_scaled, y_train, epochs=100, batch_size=256, verbose=1,
                              validation_data=(X_test_scaled, y_test))

    # val_loss, val_acc = classifier.evaluate(X_val_scaled, y_val)
    test_loss, test_acc = classifier.evaluate(X_test_scaled, y_test)

    train_loss, train_acc = classifier.evaluate(X_train_scaled, y_train)
    print('Train accuracy: {}'.format(train_acc))
    # print('Validation accuracy: {}'.format(val_acc))
    print('Test accuracy: {}'.format(test_acc))
    print('Train loss: {}'.format(train_loss))
    # print('Validation loss: {}'.format(val_loss))
    print('Test loss: {}'.format(test_loss))
    y_pred = classifier.predict(X_test_scaled)

    predictions = np.argmax(y_pred, axis=1)
    y_test_labels = np.argmax(y_test, axis=1)

    # print(pd.crosstab(y_test, y_pred, rownames=['Actual PRs'], colnames=['Predicted PRs']))

    print(metrics.classification_report(y_test_labels, predictions, digits=3))

    # precision, recall, fscore = extract_metric_from_report(
    #     metrics.classification_report(y_test_1, y_pred_1, digits=4))

    # print("\nModel Report")
    # print("Accuracy : %.4g" % metrics.accuracy_score(y_test, y_pred))
    # print("AUC Score : %f" % metrics.roc_auc_score(y_test, y_predprob))
    # print("Recall : %f" % metrics.recall_score(y_test, y_pred))
    # print("Precision : %f" % metrics.precision_score(y_test, y_pred))
    # print("F-measure : %f" % metrics.f1_score(y_test, y_pred))
    # c_matrix = metrics.confusion_matrix(y_test, y_pred)
    # print('========Confusion Matrix==========')
    # print("          Rejected    Accepted")
    # print('Rejected     {}      {}'.format(c_matrix[0][0], c_matrix[0][1]))
    # print('Accepted     {}      {}'.format(c_matrix[1][0], c_matrix[1][1]))

    # print(model_history.history.keys())
    # draw_accuracy_plot(model_history, model_name)
    # draw_loss_plot(model_history, model_name)


run_model()



# def check_model_with_optimizers():
#
#     for name, value in get_optimizer().items():
#         NAME = 'DL_accept_40x40-256_{}_{}'.format(name, int(time.time()))
#         classifier = create_model(opt=value)
#         print(f"Optimizer: {name}")
#         # tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME))
#         model_history = classifier.fit(X_train_scaled, y_train, epochs=200, batch_size=256, verbose=0,
#                         validation_data=(X_test_scaled, y_test))
#
#         val_loss, val_acc = classifier.evaluate(X_test_scaled, y_test)
#
#         print('Model error: {}'.format(val_loss))
#         print('Model accuracy: {}'.format(val_acc))
#
#         y_pred = classifier.predict(X_test_scaled)
#         y_pred = (y_pred > 0.5)
#
#         y_predprob = classifier.predict_proba(X_test_scaled)
#
#         print("\nModel Report")
#         print("Accuracy : %.4g" % metrics.accuracy_score(y_test, y_pred))
#         print("AUC Score : %f" % metrics.roc_auc_score(y_test, y_predprob))
#         print("Recall : %f" % metrics.recall_score(y_test, y_pred))
#         print("Precision : %f" % metrics.precision_score(y_test, y_pred))
#         print("F-measure : %f" % metrics.f1_score(y_test, y_pred))
#         c_matrix = metrics.confusion_matrix(y_test, y_pred)
#         print('========Confusion Matrix==========')
#         print("          Rejected    Accepted")
#         print('Rejected     {}      {}'.format(c_matrix[0][0], c_matrix[0][1]))
#         print('Accepted     {}      {}'.format(c_matrix[1][0], c_matrix[1][1]))
#         draw_accuracy_plot(model_history)
#         draw_loss_plot(model_history)



# def check_model_with_activators():
#
#     for name, value in get_activator().items():
#         NAME = 'DL_accept_40x40-256_{}_{}'.format(name, int(time.time()))
#         classifier = create_model(activator=value)
#         print(f"Optimizer: {name}")
#         # tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME))
#         model_history = classifier.fit(X_train_scaled, y_train, epochs=200, batch_size=256, verbose=0,
#                         validation_data=(X_test_scaled, y_test))
#
#         val_loss, val_acc = classifier.evaluate(X_test_scaled, y_test)
#
#         print('Model error: {}'.format(val_loss))
#         print('Model accuracy: {}'.format(val_acc))
#
#         y_pred = classifier.predict(X_test_scaled)
#         y_pred = (y_pred > 0.5)
#
#         y_predprob = classifier.predict_proba(X_test_scaled)
#
#         print("\nModel Report")
#         print("Accuracy : %.4g" % metrics.accuracy_score(y_test, y_pred))
#         print("AUC Score : %f" % metrics.roc_auc_score(y_test, y_predprob))
#         print("Recall : %f" % metrics.recall_score(y_test, y_pred))
#         print("Precision : %f" % metrics.precision_score(y_test, y_pred))
#         print("F-measure : %f" % metrics.f1_score(y_test, y_pred))
#         c_matrix = metrics.confusion_matrix(y_test, y_pred)
#         print('========Confusion Matrix==========')
#         print("          Rejected    Accepted")
#         print('Rejected     {}      {}'.format(c_matrix[0][0], c_matrix[0][1]))
#         print('Accepted     {}      {}'.format(c_matrix[1][0], c_matrix[1][1]))
#         draw_accuracy_plot(model_history)
#         draw_loss_plot(model_history)



# def model_optimizer_1():
#     epochs = 10
#     batch_size = 256
#     model_CV = keras.wrappers.scikit_learn.KerasClassifier(build_fn=create_model, epochs=epochs,
#                                batch_size=batch_size, verbose=1)
#     # define the grid search parameters
#     init_mode = ['uniform', 'lecun_uniform', 'normal', 'zero',
#                  'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform']
#
#     param_grid = dict(init_mode=init_mode)
#     grid = GridSearchCV(estimator=model_CV, param_grid=param_grid, n_jobs=-1, cv=3)
#     grid_result = grid.fit(X_train_scaled, y_train)
#
#     # print results
#     print(f'Best Accuracy for {grid_result.best_score_} using {grid_result.best_params_}')
#     means = grid_result.cv_results_['mean_test_score']
#     stds = grid_result.cv_results_['std_test_score']
#     params = grid_result.cv_results_['params']
#     for mean, stdev, param in zip(means, stds, params):
#         print(f' mean={mean:.4}, std={stdev:.4} using {param}')


# model_optimization()


# def model_optimizer_2():
#
#     model_CV = keras.wrappers.scikit_learn.KerasClassifier(build_fn=create_model, verbose=1)
#
#     # define grid search parameters
#     init_mode = ['normal', 'uniform']
#     batches = [64, 128]
#     epochs = [50, 100]
#
#     # grid search for initializer, batch size and number of epochs
#     param_grid = dict(epochs = epochs, batch_size = batches, init_mode = init_mode)
#     grid = GridSearchCV(estimator=model_CV, param_grid=param_grid, cv=10)
#     grid_result = grid.fit(X_train_scaled, y_train)
#     # print results
#     print(f'Best Accuracy for {grid_result.best_score_:.4} using {grid_result.best_params_}')
#     means = grid_result.cv_results_['mean_test_score']
#     stds = grid_result.cv_results_['std_test_score']
#     params = grid_result.cv_results_['params']
#     for mean, stdev, param in zip(means, stds, params):
#         print(f'mean={mean:.4}, std={stdev:.4} using {param}')

# model_optimizer_2()

# Best Accuracy for 0.7961 using {'batch_size': 128, 'epochs': 20, 'init_mode': 'uniform'}
# Best Accuracy for 0.8448 using {'batch_size': 128, 'epochs': 50, 'init_mode': 'normal'}

# Best Accuracy for 0.7969 using {'batch_size': 512, 'epochs': 100, 'init_mode': 'uniform'}

