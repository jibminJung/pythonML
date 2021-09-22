# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 03:01:16 2020

@author: tunej
"""

#cd D:\OneDrive - dongguk.edu\동국대학교\2020-2머신러닝과딥러닝\기말고사
import pandas as pd
import numpy as np
#데이터를 불러옵니다.
df = pd.read_csv("data_pepTestCustomers.csv")

######################################################
#변수생성

#나이에 비해 소득이 어느정도인지 비율 income/age ratio를 만들어 보겠습니다.
df['IAratio'] = df['income']/df['age']

#소득을 자녀 수로 나눈 액수 income/children을 realincome으로 만들어 보겠습니다.
df['realincome'] = np.where(df['children']!=0,df['income']/df['children'],df['income'])

#married, children, car, mortgage가 있으면 지출이 많다는 가정으로 expense 생성
df['expense']= df['income']*(df['married']*0.1 +df['children']*0.1 +df['car']*0.1 +df['mortgage']*0.1)

#소득 수준을 본인 연령대의 인원들 중 어느 위치에 있는지 나타내는 변수를 만들었으나, 퍼포먼스가 좋지 않았습니다. 그래서 제거하고 밑에서 연속형 변수로 만들었습니다.
#나이별로 소득을 기준으로 고소득,중간소득,저소득 세 가지로 분류하는 변수를 만들어 보겠습니다.
#본인 나이 대에서 소득 수준이 어느정도인지 나타냅니다.
#판다스의 cut을 이용하여 만들겠습니다.
#incomelevel 열을 먼저 만들어줍니다.
# df['incomelevel']='nan'
# 나이별로 그룹을 선택하여 반복문으로 처리해줍니다.
#for age, group_data in df[['age', 'income']].groupby('age'):
    #먼저 넘파이의 히스토그램을 이용해서 소득을 세 구간으로 나누는 경계값을 구합니다. 그리고 저장해줍니다.
    # count, bin_dividers = np.histogram(group_data['income'],bins=3)
    # #저소득을 0 중간소득을 1 고소득을 2라고 합니다.
    # bin_names = [0,1,2]
    # #변수명을 incomelevel 로 만들고, 생성해줍니다.
    # group_data['incomelevel'] = pd.cut(
    #     x=group_data['income'],
    #     bins=bin_dividers,
    #     labels=bin_names,
    #     include_lowest=True)    
    # #나이별로 구한 소득층을 전체 데이터에 병합해줍니다.
    # df.update(group_data['incomelevel'])
    #group_data['incomerank'] = group_data.income.rank(pct = True) 
    #df.update(group_data['incomerank'])
    
#소득 분위에 따라 가입 여부가 다르다는 가정으로, 나이별 소득 분위를 나타내는 변수를 연속형 자료로 만들어보았습니다.
df['incomerank']='nan'
# 나이별로 그룹을 선택하여 반복문으로 처리해줍니다.
for age, group_data in df[['age', 'income']].groupby('age'):
    group_data['incomerank'] = group_data.income.rank(pct = True) 
    df.update(group_data['incomerank'])
    
#married와 children을 더해서 부양 의무를 가지고 있는 인원 수를 구해보았습니다.
df['family'] = df['married']+df['children']

######################################################
#전처리
#랜덤포레스트의 특성 중요도를 보면서, 주요도가 낮은 변수를 제거해보면서 모델을 학습했습니다.

#성별은 주요 변수가 아니기에 제거해보았더니, 성능이 개선되었습니다.
df.drop(['sex'],axis=1,inplace=True)

#region변수는 성능에 좋지 않은 영향을 미치는 것으로 보고 제거 했습니다.
#(제거됨)지역은 크기, 순서와 상관없는 범주형 특성이므로, 원-핫 인코딩으로 변환합니다. 
# df.region.value_counts()
# df.region.replace({0: 'region_0', 1: 'region_1',2: 'region_2',3: 'region_3'},inplace=True)
# region_=pd.get_dummies(df['region'],drop_first=True)
# df = pd.concat([df, region_], axis=1)
df.drop(['region'],axis=1,inplace=True)

#자녀 특성을 원-핫 인코딩으로 변환합니다. 
df.children.value_counts()
df.children.replace({0: 'children_0', 1: 'children_1',2: 'children_2',3: 'children_3'},inplace=True)
children_=pd.get_dummies(df['children'],drop_first=True)
df = pd.concat([df, children_], axis=1)
df.drop(['children'],axis=1,inplace=True)


#데이터 나누기 : 데이터를 x변수와 y변수로 나누어줍니다. x는 pep와 id를 제외한 변수이며 y는 pep변수입니다.
from sklearn.model_selection import train_test_split
dfx = df.drop(['pep','id'],axis=1)
dfy = df['pep']
x_train, x_test, y_train, y_test = train_test_split(dfx,dfy,test_size=0.3,random_state=0)

#income, age 특성이 너무 크므로, StandardScaler를 이용하여 표준화해보았습니다.
from sklearn.preprocessing import StandardScaler
stdsc = StandardScaler()
x_train = stdsc.fit_transform(x_train)
x_test = stdsc.transform(x_test)


######################################################
#모델 생성

#모델을 생성하기전에 feature importance를 그래프로 보기위해 함수를 만들어줍니다.
import matplotlib.pyplot as plt
def plot_feature_importances(model,dfx):
    n_features = dfx.shape[1]
    plt.barh(np.arange(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), dfx.columns)
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")
    plt.ylim(-1, n_features)


#랜덤포레스트 모델을 생성합니다.
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100,max_depth=12) #매개변수 값은 여러번 변경을 반복하여 최상이라고 생각되는 값을 넣었습니다.
rf.fit(x_train,y_train)
predicted = rf.predict(x_test)
print("테스트 점수 : {:.3f}".format(rf.score(x_test,y_test)))
#테스트 점수 : 0.874 ~ 0.894

#변수중요도를 나타내봅니다. 전처리, 변수생성시 참고합니다.
plot_feature_importances(rf,dfx)

