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
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
import itertools
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
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

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def plot_roc_curve(fpr,tpr,roc_auc_value):
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc_value)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()
    
def my_cross_val_score(classifier, data, target, cv=10):
    cumulative_score = np.array([])
    test_size = int(1.0*len(target)/cv)
    for i in range(cv):
        train_data = np.concatenate((data[:i*test_size],
                                    data[(i+1)*test_size:]),
                                    axis=0)
        train_target = np.concatenate((target[:i*test_size],
                                       target[(i+1)*test_size:]),
                                       axis=0)
        test_data = data[i * test_size:(i+1) * test_size]
        test_target = target[i * test_size:(i+1) * test_size]
        
        # do prediction on test set
        classifier.fit(train_data, train_target)
        prediction = classifier.predict(test_data)
        #generate_metrics(classifier,train_data,train_target,test_data,test_target)
        precision = precision_score(test_target, prediction)
        recall = recall_score(test_target, prediction)
        print(test_target, prediction)
        print('precision,recall:',precision,recall)
        cumulative_score = np.append(cumulative_score,precision+recall)
    return cumulative_score

def generate_metrics(classifier, data, target, test_data, test_target):
    classifier.fit(data, target)
    prediction = classifier.predict(test_data)
    # probability
    prob = classifier.predict_proba(test_data)[:,0]
    confusionMatrix = confusion_matrix(test_target, prediction)
    precision = precision_score(test_target, prediction)
    recall = recall_score(test_target, prediction)
    roc_auc_value = roc_auc_score(test_target, prob)
    fpr, tpr, _ = roc_curve(test_target, prob)
    plot_roc_curve(fpr,tpr,roc_auc_value)
    print('precision, recall, roc_auc:', precision, recall, roc_auc)
    plot_confusion_matrix(confusionMatrix,
                          classes=['0','1'],
                          title='Confusion Matrix')
    
#---- start program -----------------------

df1=pd.read_csv('data_mining_test_1.csv')
# 4 entries have clicks = 2, changing them to 1
#np.argwhere(target==2)
#Out[156]: 
#array([[ 180],
#       [ 209],
#       [2410],
#       [2485]])
CTR = (df1['Click'].values/df1['Impression'].values)
target = (1.0*(df1['Click'].values>0)).astype(int)
df1.drop('Click',axis=1,inplace=True)
data = convert_to_onehot(df1)
column_names = np.array(list(data.columns)).astype(str)
# try getting rid of Campaign
#data = df1.drop('A',axis=1,inplace=False)
data = data.values

# randomize the order of the data
np.random.seed(12345)
data_length = len(data)
random_index = np.arange(data_length)
np.random.shuffle(random_index)
target = target[random_index]
data = data[random_index]

#---- set test set -------------
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

us_train_data, us_train_target = RUS().fit_sample(train_data, train_target)
os_train_data, os_train_target = ROS().fit_sample(train_data, train_target)

#us_data, us_target = RUS().fit_sample(data, target)
#os_data, os_target = ROS().fit_sample(data, target)

#---- set over sample -----------------

RF = RFC(n_estimators = 10000, n_jobs=4, random_state=9999, max_features=2)
generate_metrics(RF, os_train_data, os_train_target, test_data, test_target)
#generate_metrics(RF, os_train_data, os_train_target, test_data, test_target)
my_cross_val_score(RFC(n_estimators=10000, max_features=2, n_jobs=4),
                           us_train_data, us_train_target, cv=10).mean()


SVM = SVC(C=10,
          gamma=0.1,
          random_state=None,
          probability=True)
generate_metrics(SVM, os_train_data, os_train_target, test_data, test_target)

XGB = xgboost.XGBClassifier(learning_rate=1e-3,
                            n_estimators=10000,
                            seed=9999)
generate_metrics(XGB, os_train_data, os_train_target, test_data, test_target)


SKDNN = MLPClassifier(solver='adam',
                      alpha=1e-5,
                      batch_size='auto',
                      hidden_layer_sizes=(30,40,50,60),
                      learning_rate='adaptive',
                      learning_rate_init = 1e-2)
generate_metrics(SKDNN, os_train_data, os_train_target, test_data, test_target)

#--- XGB ensemble ----------------------
negpos = 1.0*(len(target)-target.sum())/target.sum()
def xgbcv(learning_rate, n_estimators):
    return my_cross_val_score(xgboost.XGBClassifier(learning_rate=
                                                    10**learning_rate,
                                                    n_estimators=int(n_estimators),
                                                    #scale_pos_weight=negpos
                                                    seed=9999
                                                    ),
                              os_train_data, os_train_target, cv=10).mean()

xgbBO = BayesianOptimization(xgbcv, {'learning_rate': (-4, -1),
                                     'n_estimators': (100, 1000)})
xgbBO.explore({'learning_rate': [-6, -3.5, -1],
               'n_estimators': [100,500,1000]})
xgbBO.maximize(init_points=10, n_iter=40)
print('XGB: %f' % xgbBO.res['max']['max_val'])
