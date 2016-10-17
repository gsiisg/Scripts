# Geoffrey So 9/30/2016

import numpy as np
import pandas as pd
from collections import Counter
from sklearn.ensemble import RandomForestClassifier

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

# do cross validation by using different 10% of data as test set
all_std = []
for i in range(10):
    train_set = np.concatenate((data[:i*test_size], data[(i+1)*test_size:]),
                               axis=0)
    train_target = np.concatenate((age[:i*test_size], age[(i+1)*test_size:]),
                                  axis=0)
    test_set = data[i * test_size:(i+1) * test_size]
    test_target = age[i * test_size:(i+1) * test_size]

    # train model

    def convert_to_age_group(target):
        age_group = []
        for ring_count in target:
            if ring_count < 9:
                age_group.append(0)
            elif ring_count < 11:
                age_group.append(1)
            else:
                age_group.append(2)
        return np.array(age_group).astype(np.int32)

    train_age_target = convert_to_age_group(train_target)
    test_age_target = convert_to_age_group(test_target)

    rf = RandomForestClassifier(n_estimators=1000)
    #rf.fit(train_set, train_target)
    rf.fit(train_set, train_age_target)

    
    # do prediction
    predicted = rf.predict(test_set)

    accuracy = (predicted==test_age_target).sum()/len(test_age_target)
    print('accuracy', accuracy)

    error = np.asarray(predicted,dtype=float) - \
            np.asarray(test_target,dtype=float)

    std = np.std(error)
    print('iteration ', i, 'has error ', std)
    all_std.append(std)

print('average std: ',np.mean(all_std))
