# Geoffrey So 9/30/2016
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
import tensorflow as tf
import logging
logging.getLogger().setLevel(logging.INFO)
import xgboost
from sklearn import cross_validation
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix

# read data
names = np.array(['sex','length','diameter','height','whole_weight',
                  'shucked_weight','viscera_weight','shell_weight','rings'])
df=pd.read_csv('abalone.data',header=None,names=names)

# change sex into one hot, remove from original, combine one hot with df
def convert_to_onehot(df):
    # assume data types are all consistent in each column
    value = df.values[0]
    is_number = np.array([])
    for item in value:
         is_number = np.append(is_number, isinstance(item, int)
                               or isinstance(item, float))
    # convert non-numeric to one hot
    non_number = names[is_number==0]
    to_onehot = df[non_number]
    one_hot = pd.get_dummies(to_onehot)
    # set temp dataframe to delete categorical columns from df
    temp = df.drop(df[non_number], axis=1, inplace=False)
    # combine one hot with temp 
    # assume no duplicate rows or have to use inner or outer etc
    return pd.concat([one_hot, temp], axis=1)

def convert_to_age_group(target):
    '''
    0: young
    1: medium
    2: old
    '''
    age_group = []
    offset = 1.5
    for ring_count in target:
        if ring_count < 9-offset:
            age_group.append(0)
        elif ring_count < 11-offset:
            age_group.append(1)
        else:
            age_group.append(2)
    return np.array(age_group).astype(np.int32)

data = convert_to_onehot(df)

# define answer
#age = data['rings'].values + 1.5
#age = np.asarray(age, dtype="|S6")
age = data['rings'].values


# remove rings (age) from df
data.drop('rings', axis=1, inplace=True)
data = data.values

# define index that separate train from test
data_length = len(data)
test_size = int(len(data) * 0.1)

# randomize the order of the data
np.random.seed(123456)
random_index = np.arange(data_length)
np.random.shuffle(random_index)
age = age[random_index]
data = data[random_index]


# do cross validation by using different 10% of data as test set
all_accuracies = np.array([])
for i in range(10):
    train_set = np.concatenate((data[:i*test_size], data[(i+1)*test_size:]),
                               axis=0)
    train_target = np.concatenate((age[:i*test_size], age[(i+1)*test_size:]),
                                  axis=0)
    test_set = data[i * test_size:(i+1) * test_size]
    test_target = age[i * test_size:(i+1) * test_size]

    # convert data to young:0, medium:1, old:2
    train_age_target = convert_to_age_group(train_target)
    test_age_target = convert_to_age_group(test_target)

    # Random Forest
    rf = RandomForestClassifier(n_estimators=1000)
    rf.fit(train_set, train_age_target)
    # do prediction
    RF_predictions = rf.predict(test_set)
    RF_accuracy = accuracy_score(test_age_target, RF_predictions)

    # XGB
    model = xgboost.XGBClassifier(learning_rate=0.1,
                                  n_estimators=1000)
    model.fit(train_set, train_age_target)
    XGB_predictions = model.predict(test_set)
    #XGB_predictions = [round(value) for value in y_pred]
    # evaluate predictions
    XGB_accuracy = accuracy_score(test_age_target, XGB_predictions)

    # DNN
    # Specify that all features have real-value data
    feature_columns = [tf.contrib.layers.real_valued_column("", dimension=10)]
    # Build 3 layer DNN with 10, 20, 10 units respectively.
    classifier = tf.contrib.learn.DNNClassifier(
        feature_columns=feature_columns,
        hidden_units=[10,20],
        n_classes=3,
        optimizer=tf.train.ProximalAdagradOptimizer(learning_rate=0.1)
        )
    classifier.fit(x=train_set,
                   y=train_age_target,
                   steps=1000)

    DNN_predictions = classifier.predict(test_set)
    DNN_accuracy = accuracy_score(test_age_target, DNN_predictions)
    
    current_accuracies = np.array([RF_accuracy,XGB_accuracy,DNN_accuracy])
    print('CURRENT ACCURACIES (RF,XGB,DDN)', current_accuracies)
    all_accuracies = np.concatenate([all_accuracies,current_accuracies])

# For the last i=9th cross validation, plot confusion matrices
RF_confusion_matrix = confusion_matrix(test_age_target,RF_prediction)
plot_confusion_matrix(RF_confusion_matrix,
                      classes=['young','medium','old'],
                      title='RF Confusion matrix, without normalization')

XGB_confusion_matrix = confusion_matrix(test_age_target,XGB_prediction)
plot_confusion_matrix(RF_confusion_matrix,
                      classes=['young','medium','old'],
                      title='XGB Confusion matrix, without normalization')

DNN_confusion_matrix = confusion_matrix(test_age_target,DNN_prediction)
plot_confusion_matrix(RF_confusion_matrix,
                      classes=['young','medium','old'],
                      title='DNN Confusion matrix, without normalization')
    
all_accuracies.resize(10,3)
print('average accuracy', np.mean(all_accuracies,axis=0))
