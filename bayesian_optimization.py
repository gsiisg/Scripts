# Geoffrey So 9/30/2016
from __future__ import print_function
from __future__ import division
from sklearn.datasets import make_classification
from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import RandomForestClassifier as RFC
import numpy as np
import pandas as pd
from collections import Counter

import tensorflow as tf
import logging
logging.getLogger().setLevel(logging.INFO)
#import xgboost
from sklearn import cross_validation
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from bayes_opt import BayesianOptimization

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

train_age_target = convert_to_age_group(age)

'''
def svccv(C, gamma):
    return cross_val_score(SVC(C=C, gamma=gamma, random_state=2),
                           data, train_age_target, cv=10).mean()

svcBO = BayesianOptimization(svccv, {'C': (0.001, 1000),
                                     'gamma': (0.0001, 0.1)})
svcBO.explore({'C': [0.001, 0.01, 0.1, 1.0], 'gamma': [0.001, 0.01, 0.1, 1.0]})
svcBO.maximize(init_points=10, n_iter=40)
print('SVC: %f' % svcBO.res['max']['max_val'])
'''
feature_columns = [tf.contrib.layers.real_valued_column("", dimension=10)]

def dnncv(h1, h2, h3, h4,
          learning_rate, dropout):
    return cross_val_score(
        tf.contrib.learn.DNNClassifier(
        feature_columns=feature_columns,
        hidden_units=[h1, h2, h3, h4],
        n_classes=3,
        optimizer=tf.train.ProximalAdagradOptimizer(
            learning_rate=learning_rate),
        dropout=dropout    
        )
        ),
    data, train_age_target, cv=10, n_jobs=-1).mean()

dnnBO = BayesianOptimization(dnncv, {'h1': (10, 100),
                                     'h2': (10, 100),
                                     'h3': (10, 100),
                                     'h4': (10, 100),
                                     'learning_rate': (1e-4,1e-1),
                                     'dropout':(0.1,0.9),
})

dnnBO.explore({'h1': [20, 50],
               'h2': [20, 50],
               'h3': [20, 50],
               'h4': [20, 50],
               'learning_rate': [1e-3,1e-2],
               'dropout': [.3,.6],
})

dnnBO.maximize(init_points=12, n_iter=40)

print('DNN: %f' % dnnBO.res['max']['max_val']) 

'''
def rfccv(n_estimators, min_samples_split, max_features):
    return cross_val_score(RFC(n_estimators=int(n_estimators),
                               min_samples_split=int(min_samples_split),
                               max_features=min(max_features, 0.999),
                               random_state=2),
                           data, train_age_target, cv=10, n_jobs=-1).mean()


#rf = RandomForestClassifier(n_estimators=1000)
#cv = cross_val_score(rf, data, train_age_target, cv=10)

rfcBO = BayesianOptimization(rfccv, {'n_estimators': (10, 1000),
                                    'min_samples_split': (2,100),
                                    'max_features': (.1, .999)})
rfcBO.maximize(init_points=10, n_iter=20)

print('RFC: %f' % rfcBO.res['max']['max_val'])
'''

#print(cv)
#print(cv.mean())
#print('latest')

