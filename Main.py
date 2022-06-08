from xgboost import XGBClassifier
from sklearn.decomposition import PCA
# from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import RandomOverSampler
from sklearn.dummy import DummyClassifier
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from scipy.stats.mstats import winsorize
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")


def over_sample_train_test(x, y):
    ros = RandomOverSampler(random_state=0)
    ros.fit(x, y)
    x_res, y_res = ros.fit_resample(x, y)
    x_train, x_test, y_train, y_test = train_test_split(
        x_res, y_res, test_size=0.3, random_state=1)
    return x_train, x_test, y_train, y_test


def dummy_classifier(x_train, x_test, y_train, y_test):
    dclf = DummyClassifier()
    dclf.fit(x_train, y_train)
    score = dclf.score(x_test, y_test)
    print('ZeroR: ', score)


def knn_classifier(x_train, x_test, y_train, y_test):
    clf = KNeighborsClassifier(n_neighbors=2)
    clf.fit(x_train, y_train)
    print('Knn: ', clf.score(x_test, y_test))
    
def xgboost_classifier(x_train, x_test, y_train, y_test):
    xgb = XGBClassifier(use_label_encoder=False, random_state=42, eval_metric='mlogloss')
    xgb.fit(x_train, y_train)
    print('Xgb:   ', xgb.score(x_test, y_test))

def apply_model(model, x_train, x_test, y_train, y_test):
    model.fit(x_train, y_train)
    print('Test Score:   ', model.score(x_test, y_test))

df_train = pd.read_csv('Train.csv')

y = df_train.iloc[:, -1:]
x = df_train.iloc[:, :-1]
#x = df.drop('Class', axis=1)
#y = df.Class

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=1)

dummy_classifier(x_train, x_test, y_train, y_test)
knn_classifier(x_train, x_test, y_train, y_test)
xgboost_classifier(x_train, x_test, y_train, y_test)

print('-------------------------------')
y = df_train.iloc[:, -1:]
x = df_train.iloc[:, :-1]
x_train, x_test, y_train, y_test = over_sample_train_test(x, y)
dummy_classifier(x_train, x_test, y_train, y_test)
knn_classifier(x_train, x_test, y_train, y_test)
xgboost_classifier(x_train, x_test, y_train, y_test)

df_traincopy = df_train.copy()

for col in df_traincopy.columns[:3]:
    l=0.1 if col=='feature_2' else 0.05
    df_traincopy[col] = winsorize(df_traincopy[col],limits=l)
    
print('-------------------------------')
y = df_traincopy.iloc[:, -1:]
x = df_traincopy.iloc[:, :-1]
x_train, x_test, y_train, y_test = over_sample_train_test(x, y)
dummy_classifier(x_train, x_test, y_train, y_test)
knn_classifier(x_train, x_test, y_train, y_test)
xgboost_classifier(x_train, x_test, y_train, y_test)

# print('-------------------------------')
# oversample = SMOTE()
# x, y = oversample.fit_resample(x, y)
# x_train, x_test, y_train, y_test = train_test_split(
#     x, y, test_size=0.3, random_state=1)
# dummy_classifier(x_train, x_test, y_train, y_test)
# knn_classifier(x_train, x_test, y_train, y_test)
# apply_model(xgb, x_train, x_test, y_train, y_test)

print('-------------------------------')
df_trainpca = df_train.copy() 

pca = PCA(random_state=44)
pc = pca.fit_transform(df_trainpca.drop('Class',axis=1))
pc_x,pc_y='pc_x','pc_y'
df_trainpca[pc_x]=pc[:,0]
df_trainpca[pc_y]=pc[:,1]

x = df_trainpca[[pc_x,pc_y]]
y = df_trainpca[['Class']].values
x_train,x_test,y_train,y_test=over_sample_train_test(x,y)
dummy_classifier(x_train, x_test, y_train, y_test)
knn_classifier(x_train, x_test, y_train, y_test)
xgboost_classifier(x_train, x_test, y_train, y_test)