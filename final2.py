# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 21:50:23 2020

@author: tunej
"""
#cd D:\OneDrive - dongguk.edu\동국대학교\2020-2머신러닝과딥러닝\기말고사

# PCA전 random forest의 feature_importance 기반 0.1 이상 변수 기준!!!!
# PCA전 random forest의 feature_importance 기반 0.1 이상 변수 기준!!!!
# PCA전 random forest의 feature_importance 기반 0.1 이상 변수 기준!!!!


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df = pd.read_csv("data_pepTestCustomers.csv")
df.head()
#데이터 보기, null값이 있는지 확인
df.info()


#income, age 특성이 너무 크므로, StandardScaler를 이용하여 표준화해보았습니다.
from sklearn.preprocessing import StandardScaler
stdsc = StandardScaler()
df[['income','age']] = stdsc.fit_transform(df[['income','age']])
df[['income','age']] = stdsc.transform(df[['income','age']])

#income, age 특성이 너무 크므로, MinMaxScaler를 이용하여 0과 1사이의 숫자로 정규화해보았습니다.
from sklearn.preprocessing import MinMaxScaler
mms = MinMaxScaler()
df[['income','age']] = mms.fit_transform(df[['income','age']])


#지역은 크기, 순서와 상관없는 범주형 특성이므로, 원-핫 인코딩으로 변환합니다. 
df.region.value_counts()
df.region.replace({0: 'region_0', 1: 'region_1',2: 'region_2',3: 'region_3'},inplace=True)
region_=pd.get_dummies(df['region'],drop_first=True)
df = pd.concat([df, region_], axis=1)
df.drop(['region'],axis=1,inplace=True)
print(df)

#랜덤포레스트의 feature_importance 기반으로 0.1이상의 것만 저장
ndf = df[['children','income','age','pep']]
#데이터 나누기
from sklearn.model_selection import train_test_split
dfx = ndf.drop(['pep'],axis=1)
dfy = ndf['pep']
x_train, x_test, y_train, y_test = train_test_split(dfx,dfy,test_size=0.3,random_state=0)



#산점도 그려보기
import seaborn as sns
sns.pairplot(ndf, diag_kind='kde', hue="pep", palette='bright')

plt.show()


#레이블 저장
feat_labels=df.columns

#feature importance 그래프
def plot_feature_importances(model,dfx):
    n_features = dfx.shape[1]
    plt.barh(np.arange(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), dfx.columns)
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")
    plt.ylim(-1, n_features)


#디시젼트리
from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(max_depth=8, random_state=0)
tree.fit(x_train, y_train)
predicted = tree.predict(x_test); print(predicted)
print(f"score is {tree.score(x_test, y_test)}")
plot_feature_importances(tree,dfx)
#standardaized : score : 0.7
#normalized : score : 0.705


#랜덤포레스트
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=300,max_depth=8)
rf.fit(x_train,y_train)
predicted = rf.predict(x_test)
print(predicted)
print('score is %s'%(rf.score(x_test,y_test)))
plot_feature_importances(rf,dfx)
#standardaized : score : 0.75
#normalized : score : 0.772


#KNN
from sklearn.neighbors import KNeighborsClassifier
neighbor = KNeighborsClassifier(n_neighbors=3)
neighbor.fit(x_train, y_train)
predicted = neighbor.predict(x_test); #print(predicted)
print(f"score is {neighbor.score(x_test, y_test)}")
#standardaized : score : 0.694
#normalized : score : 0.76




#그리드 서치를 이용한 SVM
param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100],
              'gamma': [0.001, 0.01, 0.1, 1, 10, 100]}
print("Parameter grid:\n{}".format(param_grid))
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
grid_search = GridSearchCV(SVC(), param_grid, cv=5, return_train_score=True)
grid_search.fit(x_train, y_train)
print("Test set score: {:.2f}".format(grid_search.score(x_test, y_test)))
print("Best parameters: {}".format(grid_search.best_params_))
print("Best cross-validation score: {:.2f}".format(grid_search.best_score_))


#SVM
from sklearn.svm import SVC
svm = SVC(kernel='rbf',C=1, gamma=0.1,random_state=1).fit(x_train,y_train)
predicted = svm.predict(x_test)
print(predicted)
print('score is %s'%(svm.score(x_test,y_test)))


