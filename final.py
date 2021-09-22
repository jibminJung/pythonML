# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 21:50:23 2020

@author: tunej
"""
#cd D:\OneDrive - dongguk.edu\동국대학교\2020-2머신러닝과딥러닝\기말고사
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df = pd.read_csv("data_pepTestCustomers.csv")


#feature importance 그래프
def plot_feature_importances(model,dfx):
    n_features = dfx.shape[1]
    plt.barh(np.arange(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), dfx.columns)
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")
    plt.ylim(-1, n_features)



#디시젼트리 모델을 생성합니다.
from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(max_depth=12, random_state=0) #매개변수 값은 여러번 변경을 반복하여 최상이라고 생각되는 값을 넣었습니다.
tree.fit(x_train, y_train)
predicted = tree.predict(x_test)
print("테스트 점수 : {:.3f}".format(tree.score(x_test,y_test)))
#변수중요도를 나타내봅니다. 전처리, 변수생성시 참고합니다.
plot_feature_importances(tree,dfx)


#랜덤포레스트 모델을 생성합니다.
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100,max_depth=12) #매개변수 값은 여러번 변경을 반복하여 최상이라고 생각되는 값을 넣었습니다.
rf.fit(x_train,y_train)
predicted = rf.predict(x_test)
print("테스트 점수 : {:.3f}".format(rf.score(x_test,y_test)))
#변수중요도를 나타내봅니다. 전처리, 변수생성시 참고합니다.
plot_feature_importances(rf,dfx)


#KNN모델을 생성합니다. 랜덤포레스트를 기반으로 주요 특성을 선택하여 차원의 저주를 해결합니다.
from sklearn.feature_selection import SelectFromModel
from sklearn.neighbors import KNeighborsClassifier
#파이프라인을 사용해줍니다. 랜덤포레스트의 매개변수를 적절히 조절하고, 특성주요도의 평균에 해당하는 피쳐를 선택하여 적용합니다. 
from sklearn.pipeline import Pipeline
knn_s = Pipeline([('rf', SelectFromModel(RandomForestClassifier(n_estimators=100, random_state=42),
    threshold="mean")), ('knn', KNeighborsClassifier(n_neighbors=3))])
knn_s.fit(x_train, y_train)
predicted = knn_s.predict(x_test)
print("테스트 점수 : {:.3f}".format(knn_s.score(x_test,y_test)))



#로지스틱 회귀
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver='liblinear', multi_class='auto',C=1, random_state=1)
lr = lr.fit(x_train, y_train)
predicted = lr.predict(x_test)
print("테스트 점수 : {:.3f}".format(lr.score(x_test,y_test)))

#파이프라인을 사용해줍니다. 랜덤포레스트의 매개변수를 적절히 조절하고, 특성주요도의 평균에 해당하는 피쳐를 선택하여 적용합니다. 
from sklearn.pipeline import Pipeline
lr_s = Pipeline([('rf', SelectFromModel(RandomForestClassifier(n_estimators=100, random_state=42),
    threshold="mean")), ('lr', LogisticRegression(solver='liblinear', multi_class='auto',C=1, random_state=1))])
lr_s.fit(x_train, y_train)
predicted = lr_s.predict(x_test)
print("테스트 점수 : {:.3f}".format(lr_s.score(x_test,y_test)))


#그리드 서치를 이용한 SVM
param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100],
              'gamma': [0.001, 0.01, 0.1, 1, 10, 100],
              'kernel':['rbf','linear','poly','sigmoid'],
              'random_state':[0,1,48]}
print("Parameter grid:\n{}".format(param_grid))
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
grid_search = GridSearchCV(SVC(), param_grid, cv=5, return_train_score=True)
grid_search.fit(x_train, y_train)
print("Test set score: {:.2f}".format(grid_search.score(x_test, y_test)))
print("Best parameters: {}".format(grid_search.best_params_))
print("Best cross-validation score: {:.2f}".format(grid_search.best_score_))


#최적 파라미터로 생성한 SVM
svm = SVC(kernel='rbf',random_state=0,gamma=0.1,C=1)
svm = svm.fit(x_train, y_train)
predicted = svm.predict(x_test)
print("테스트 점수 : {:.3f}".format(svm.score(x_test,y_test)))

#파이프라인을 사용해줍니다. 랜덤포레스트의 매개변수를 적절히 조절하고, 특성주요도의 평균에 해당하는 피쳐를 선택하여 적용합니다. 
from sklearn.pipeline import Pipeline
svm_s = Pipeline([('rf', SelectFromModel(RandomForestClassifier(n_estimators=100, random_state=42),
    threshold="mean")), ('svm', SVC(kernel='rbf',random_state=0,gamma=0.1,C=1))])
svm_s.fit(x_train, y_train)
predicted = svm_s.predict(x_test)
print("테스트 점수 : {:.3f}".format(svm_s.score(x_test,y_test)))



#5개의 모델을 합쳐, 투표 분류기를 만듭니다.
#
from sklearn.ensemble import VotingClassifier
vc = VotingClassifier(estimators=[
        ('lr', lr), ('rf', rf), ('svm', svm),('knn',knn_s),('tree',tree)], voting='hard')
vc = vc.fit(x_train, y_train)
predicted = vc.predict(x_test)
print("테스트 점수 : {:.3f}".format(vc.score(x_test,y_test)))

from sklearn.model_selection import cross_val_score
scores = cross_val_score(vc, dfx, dfy, cv=5)
scores
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

#파이프라인을 사용해줍니다. 랜덤포레스트의 매개변수를 적절히 조절하고, 특성주요도의 평균에 해당하는 피쳐를 선택하여 적용합니다. 
from sklearn.pipeline import Pipeline
vc_s = Pipeline([('rf', SelectFromModel(RandomForestClassifier(n_estimators=100, random_state=42),
    threshold="mean")), ('vc', VotingClassifier(estimators=[
        ('lr', lr), ('rf', rf), ('svm', svm),('knn',knn_s),('tree',tree)], voting='hard'))])
vc_s.fit(x_train, y_train)
predicted = vc_s.predict(x_test)
print("테스트 점수 : {:.3f}".format(vc_s.score(x_test,y_test)))








