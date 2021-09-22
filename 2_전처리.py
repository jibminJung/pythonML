# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 00:03:54 2020

@author: tunej
"""
#데이터를 불러옵니다.
#cd D:\OneDrive - dongguk.edu\동국대학교\2020-2머신러닝과딥러닝\기말고사
import pandas as pd
import numpy as np
df = pd.read_csv("data_pepTestCustomers.csv")

##이전에 작업한 변수들입니다.
df['IAratio'] = df['income']/df['age']
df['realincome'] = np.where(df['children']!=0,df['income']/df['children'],df['income'])
df['expense']= df['income']*(df['married']*0.1 +df['children']*0.1 +df['car']*0.1 +df['mortgage']*0.1)
df['incomerank']='nan'
for age, group_data in df[['age', 'income']].groupby('age'):
    group_data['incomerank'] = group_data.income.rank(pct = True) 
    df.update(group_data['incomerank'])
df['family'] = df['married']+df['children']



#성별은 주요 변수가 아니기에 제거해보았더니, 성능이 개선되었습니다.
df.drop(['sex'],axis=1,inplace=True)

# 지역은 크기, 순서와 상관없는 범주형 특성이므로, 원-핫 인코딩으로 변환합니다. 
df.region.value_counts()
df.region.replace({0: 'region_0', 1: 'region_1',2: 'region_2',3: 'region_3'},inplace=True)
region_=pd.get_dummies(df['region'],drop_first=True)
df = pd.concat([df, region_], axis=1)
df.drop(['region'],axis=1,inplace=True)
#그러나 여기서 만든 region변수는 성능에 좋지 않은 영향을 미치는 것으로 보고 이후에 제거 했습니다.

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

print(dfx.columns)

print(dfx.head())

print(dfx.shape)

print(x_train)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)
