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
from sklearn.model_selection import cross_val_score

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

rf = RandomForestClassifier(n_estimators=1000)
cross_val_score(rf, data, age, cv=5)

rf.fit(train_set, train_age_target)


