# Geoffrey So 9/30/2016

import numpy as np
import pandas as pd
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
import tensorflow as tf
import numpy as np
import logging
logging.getLogger().setLevel(logging.INFO)

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
age = data['rings'].values
#age = np.asarray(age, dtype="|S6")

# remove rings (age) from df
data.drop('rings', axis=1, inplace=True)
data = data.values

# define index that separate train from test
data_length = len(data)
test_size = int(len(data) * 0.1)

i=0
train_set = np.concatenate((data[:i*test_size], data[(i+1)*test_size:]),
                               axis=0)
train_target = np.concatenate((age[:i*test_size], age[(i+1)*test_size:]),
                                  axis=0)
test_set = data[i * test_size:(i+1) * test_size]
test_target = age[i * test_size:(i+1) * test_size]

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

# Data sets

# Specify that all features have real-value data
feature_columns = [tf.contrib.layers.real_valued_column("", dimension=10)]

# Build 3 layer DNN with 10, 20, 10 units respectively.
#classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,
#                                            hidden_units=[10,20],
#                                                      n_classes=3)
classifier = tf.contrib.learn.DNNClassifier(
    n_classes=3,
    feature_columns=feature_columns,
    hidden_units=[10, 20],
    optimizer=tf.train.ProximalAdagradOptimizer(
    learning_rate=0.1,
    l1_regularization_strength=0.001
    ))


classifier.fit(x=train_set,
               y=train_age_target,
               steps=1000)

accuracy_score = classifier.evaluate(x=test_set,
                                     y=test_age_target)["accuracy"]
print('Accuracy: {0:f}'.format(accuracy_score))


'''
classifier = tf.contrib.learn.DNNClassifier(
    n_classes=3,
    feature_columns=feature_columns,
    hidden_units=[100, 200, 100],
    optimizer=tf.train.AdagradOptimizer(
      learning_rate=1e-6
    ))

classifier = tf.contrib.learn.DNNClassifier(
    n_classes=3,
    feature_columns=feature_columns,
    hidden_units=[10, 20],
    optimizer=tf.train.ProximalAdagradOptimizer(
    learning_rate=0.01,
    l1_regularization_strength=0.001
    ))


# Fit model
classifier.fit(x=train_set.astype(np.float32),
               y=train_target.astype(np.int32),
               steps=100)

classifier.fit(x=train_set,
               y=train_target,
               steps=100)

classifier.fit(x=train_set.astype(np.float32),
               y=train_target.astype(np.int32),
               steps=10)

# Evaluate accuracy.
accuracy_score = classifier.evaluate(x=test_set,
                                     y=test_target)["accuracy"]
print('Accuracy: {0:f}'.format(accuracy_score))



# Classify two new flower samples.
new_samples = np.array(
    [[6.4, 3.2, 4.5, 1.5], [5.8, 3.1, 5.0, 1.7]], dtype=float)
y = classifier.predict(new_samples)
print('Predictions: {}'.format(str(y)))



education = tf.contrib.layers.sparse_column_with_hash_bucket(column_name="education",
                                           hash_bucket_size=1000)
occupation = tf.contrib.layers.sparse_column_with_hash_bucket(column_name="occupation",
                                            hash_bucket_size=1000)

education_emb = tf.contrib.layers.embedding_column(sparse_id_column=education, dimension=16,
                                 combiner="sum")
occupation_emb = tf.contrib.layers.embedding_column(sparse_id_column=occupation, dimension=16,
                                 combiner="sum")
'''
