# Geoffrey So 10/14/2016

#---- imports ----------------------------
import pandas as pd
import numpy as np
from bayes_opt import BayesianOptimization
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.svm import SVC
import xgboost

#---- function definitions ---------------



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

def my_cross_val_score(classifier, data, target, kfolds=10):
    cumulative_accuracy = np.array([])
    for i in range(kfolds):
        train_set = np.concatenate((data[:i*test_size],
                                    data[(i+1)*test_size:]),
                                    axis=0)
        train_target = np.concatenate((target[:i*test_size],
                                       target[(i+1)*test_size:]),
                                       axis=0)
        test_set = data[i * test_size:(i+1) * test_size]
        test_target = target[i * test_size:(i+1) * test_size]

        # do prediction on test set
        prediction = classifier.fit(train_set, train_age_target)
        accuracy = accuracy_score(test_target, prediction)
        cumulative_accuracy = np.append(cumulative_accuracy,accuracy)
    return cumulative_accuracy

#---- start program -----------------------

# read data
names = np.array(['sex','length','diameter','height','whole_weight',
                  'shucked_weight','viscera_weight','shell_weight','rings'])
df=pd.read_csv('abalone.data',header=None,names=names)

# convert data to one hot vectors
data = convert_to_onehot(df)

# define answer age target
age = data['rings'].values

# remove rings (age) from df
data.drop('rings', axis=1, inplace=True)
data = data.values

# define index that separate train from test
cv_fold = 10
data_length = len(data)
#test_size = int(len(data) * 1.0/cv_fold)

# randomize the order of the data
np.random.seed(123456)
random_index = np.arange(data_length)
np.random.shuffle(random_index)
age = age[random_index]
data = data[random_index]

target = convert_to_age_group(age)

#---- SVM ---------------------------
def svccv(C, gamma):
    return cross_val_score(SVC(C=C, gamma=gamma, random_state=2),
                           data, target, cv=10).mean()

svcBO = BayesianOptimization(svccv, {'C': (0.001, 1000),
                                     'gamma': (0.0001, 0.1)})
svcBO.explore({'C': [0.001, 0.01, 0.1, 1.0], 'gamma': [0.001, 0.01, 0.1, 1.0]})
svcBO.maximize(init_points=10, n_iter=20)
print('SVC: %f' % svcBO.res['max']['max_val'])

#---- XGB ----------------------------
def xgbcv(learning_rate, n_estimators):
    return cross_val_score(xgboost.XGBClassifier(
                           learning_rate=learning_rate,
                           n_estimators=n_estimators),
                           data, target, cv=10).mean()

xgbBO = BayesianOptimization(xgbcv, {'learning_rate': (1e-4, 1e-1),
                                     'n_estimators': (100, 2000)})
xgbBO.explore({'learning_rate': [1e-4, 5e-2, 1e-1],
               'n_estimators': [100,800,2000]})
xgbBO.maximize(init_points=10, n_iter=20)
print('SVC: %f' % svcBO.res['max']['max_val'])





        
model.fit(train_set, train_age_target)



'''
DNN = MLPClassifier(solver='adam', alpha=1e-5, batch_size='auto',
                    hidden_layer_sizes=(20, 20), random_state=1,
                    learning_rate='adaptive',learning_rate_init=1e-3)
cv = cross_val_score(DNN, data, target, cv=cv_fold)

print(cv.mean())
'''
