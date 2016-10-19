# Geoffrey So 10/18/2016

#---- imports ----------------------------
import pandas as pd
import numpy as np
from bayes_opt import BayesianOptimization
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.linear_model import LogisticRegression as LR
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier as DTC
import xgboost
from imblearn.over_sampling import RandomOverSampler as ROS
from imblearn.under_sampling import RandomUnderSampler as RUS

#---- function definitions ---------------



def convert_to_onehot(df):
    # change categorical into one hot,
    # remove from original,
    # combine one hot with df
    # only convert first column to one hot
    to_onehot = df.columns[0]
    one_hot = pd.get_dummies(df[to_onehot])
    # set temp dataframe to delete categorical columns from df
    temp = df.drop(to_onehot, axis=1, inplace=False)
    # combine one hot with temp 
    # assume no duplicate rows or have to use inner or outer etc
    return pd.concat([one_hot, temp], axis=1)

#---- start program -----------------------

df1=pd.read_csv('data_mining_test_1.csv')
# 4 entries have clicks = 2, changing them to 1
#np.argwhere(target==2)
#Out[156]: 
#array([[ 180],
#       [ 209],
#       [2410],
#       [2485]])
target = (df1['Click'].values>0)*1
df1.drop('Click',axis=1,inplace=True)
data = convert_to_onehot(df1)
column_names = np.array(list(data.columns)).astype(str)
data = data.values

# randomize the order of the data
np.random.seed(123456)
data_length = len(data)
random_index = np.arange(data_length)
np.random.shuffle(random_index)
target = target[random_index]
data = data[random_index]

#---- tempRF -------------------
def rfcv(n_estimators):
    return cross_val_score(RFC(n_estimators=int(n_estimators),
                               max_features=30,
                               n_jobs=4,
                               class_weight='balanced'),
                           data, target, cv=10).mean()

rfBO = BayesianOptimization(rfcv, {'n_estimators': (100, 1000)})
rfBO.explore({'n_estimators': [100,1000]})

rfBO.maximize(init_points=10, n_iter=10)
print('SKDNN: %f' % rfBO.res['max']['max_val'])

#---- RF -------------------------------------
def rfcv(n_estimators, max_features):
    return cross_val_score(RFC(n_estimators=int(n_estimators),
                               max_features=int(max_features),
                               n_jobs=4),
                           data, target, cv=10).mean()

rfBO = BayesianOptimization(rfcv, {'n_estimators': (10, 1000),
                                   'max_features': (5,30)})
rfBO.explore({'n_estimators': [10,1000],
              'max_features': [5, 30]})
rfBO.maximize(init_points=10, n_iter=20)
print('SKDNN: %f' % rfBO.res['max']['max_val'])

#---- SVM ---------------------------
def svccv(C, gamma):
    return cross_val_score(SVC(C=10**C,
                               gamma=10**gamma,
                               random_state=None,
                               probability=True),
                           data, target, cv=10).mean()

svcBO = BayesianOptimization(svccv, {'C': (-5, 5),
                                     'gamma': (-5, 0)})
svcBO.explore({'C': [-5, 0, 5],
               'gamma': [-5, -2.5, 0]})
svcBO.maximize(init_points=10, n_iter=20)
print('SVC: %f' % svcBO.res['max']['max_val'])

#---- XGB ----------------------------
negpos = 1.0*(len(target)-target.sum())/target.sum()
def xgbcv(learning_rate, n_estimators):
    return cross_val_score(xgboost.XGBClassifier(
                           learning_rate=10**learning_rate,
                           n_estimators=int(n_estimators),
                           scale_pos_weight=negpos),
                           data, target, cv=10).mean()

xgbBO = BayesianOptimization(xgbcv, {'learning_rate': (-4, -1),
                                     'n_estimators': (100, 1000)})
xgbBO.explore({'learning_rate': [-4, -2.5, -1],
               'n_estimators': [100,500,1000]})
xgbBO.maximize(init_points=10, n_iter=20)
print('XGB: %f' % xgbBO.res['max']['max_val'])


#---- skDNN ---------------------------------
def skdnncv(h1, h2, learning_rate_init):
    return cross_val_score(MLPClassifier(solver='adam',
                                         alpha=1e-5,
                                         batch_size='auto',
                                         hidden_layer_sizes=(int(h1), int(h2)),
                                         learning_rate='adaptive',
                                         learning_rate_init=
                                         10**learning_rate_init),
                           data, target, cv=10).mean()
skdnnBO = BayesianOptimization(skdnncv, {'h1': (10, 100),
                                         'h2': (10, 100),
                                         'learning_rate_init': (-5,-1)})
skdnnBO.explore({'h1': [10,100],
                 'h2': [10,100],
                 'learning_rate_init': [-5, -1]})
skdnnBO.maximize(init_points=10, n_iter=20)
print('SKDNN: %f' % skdnnBO.res['max']['max_val'])


#---- set classifiers to be combined for voting -------------

RF = RFC(n_estimators = int(rfBO.res['max']['max_params']['n_estimators']),
         max_features = int(rfBO.res['max']['max_params']['max_features']))

SVM = SVC(C=10**svcBO.res['max']['max_params']['C'],
          gamma=10**svcBO.res['max']['max_params']['gamma'],
          random_state=None,
          probability=True)

XGB = xgboost.XGBClassifier(learning_rate =
                            10**xgbBO.res['max']['max_params']['learning_rate'],
                            n_estimators =
                            int(xgbBO.res['max']['max_params']['n_estimators']))

SKDNN = MLPClassifier(solver='adam',
                      alpha=1e-5,
                      batch_size='auto',
                      hidden_layer_sizes=(
                          int(skdnnBO.res['max']['max_params']['h1']),
                          int(skdnnBO.res['max']['max_params']['h2'])),
                      learning_rate='adaptive',
                      learning_rate_init =
                      10**skdnnBO.res['max']['max_params']['learning_rate_init'])




estimators = [('RF',RF),('SVM',SVM),('XGB',XGB),('SKDNN',SKDNN)]

hardVC = VotingClassifier(estimators=estimators,voting='hard')

# need to have SVM set probability=True for soft voting to work
softVC = VotingClassifier(estimators=estimators,voting='soft')

RFaccuracy, SVMaccuracy, XGBaccuracy, SKDNNaccuracy = (
    rfBO.res['max']['max_val'],
    svcBO.res['max']['max_val'],
    xgbBO.res['max']['max_val'],
    skdnnBO.res['max']['max_val'])

weightVC = VotingClassifier(estimators=[('RF',RF),
                                        ('SVM',SVM),
                                        ('XGB',XGB),
                                        ('SKDNN',SKDNN)],
                            voting='soft',
                            weights=[RFaccuracy,
                                     SVMaccuracy,
                                     XGBaccuracy,
                                     SKDNNaccuracy])

hardAccuracy = cross_val_score(hardVC, data, target, cv=10).mean()
softAccuracy = cross_val_score(softVC, data, target, cv=10).mean()
weightAccuracy = cross_val_score(weightVC, data, target, cv=10).mean()

print('RF:', RFaccuracy)
print('SVM:', SVMaccuracy)
print('XGB:', XGBaccuracy)
print('SKDNN:', SKDNNaccuracy)
print('hard:', hardAccuracy)
print('soft:', softAccuracy)
print('weight:', weightAccuracy)

#---- Confusion Matrix
cv = 10
test_size = int(data_length*1.0/cv)
i = 0
train_data = np.concatenate((data[:i*test_size],
                            data[(i+1)*test_size:]),
                            axis=0)
train_target = np.concatenate((target[:i*test_size],
                            target[(i+1)*test_size:]),
                            axis=0)
test_data = data[i * test_size:(i+1) * test_size]
test_target = target[i * test_size:(i+1) * test_size]

# train each model with 90% of data
# make prediction using 10% hold out test data
RF.fit(train_data, train_target)
RFprediction = RF.predict(test_data)

SVM.fit(train_data, train_target)
SVMprediction = SVM.predict(test_data)

XGB.fit(train_data, train_target)
XGBprediction = XGB.predict(test_data)

SKDNN.fit(train_data, train_target)
SKDNNprediction = SKDNN.predict(test_data)

#---- Precision/Recall --------------------
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
RFprecision = precision_score(test_target, RFprediction)
SVMprecision = precision_score(test_target, SVMprediction)
XGBprecision = precision_score(test_target, XGBprediction)
SKDNNprecision = precision_score(test_target, SKDNNprediction)
print('precision', RFprecision, SVMprecision, XGBprecision, SKDNNprecision)
RFrecall = recall_score(test_target, RFprediction)
SVMrecall = recall_score(test_target, SVMprediction)
XGBrecall = recall_score(test_target, XGBprediction)
SKDNNrecall = recall_score(test_target, SKDNNprediction)
print('recall', RFprecision, SVMprecision, XGBprecision, SKDNNprecision)



#---- Features Importance -----------------
RF.fit(data, target)
importances = RF.feature_importances_
std = np.std([tree.feature_importances_ for tree in RF.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(data.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))


# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(data.shape[1]), importances[indices],
       color="r", yerr=std[indices], align="center")
plt.xticks(range(data.shape[1]), column_names[indices], rotation='vertical')
plt.xlim([-1, data.shape[1]])
plt.show()
