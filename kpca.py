# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 13:07:54 2020

@author: tunej
"""
df = pd.read_csv("data_pepTestCustomers.csv")

dfx = df.drop(['pep','id'],axis=1)
dfy = df['pep']

#income, age 특성이 너무 크므로, StandardScaler를 이용하여 표준화해보았습니다.
from sklearn.preprocessing import StandardScaler
stdsc = StandardScaler()
df[['income','age']] = stdsc.fit_transform(df[['income','age']])

#지역은 크기, 순서와 상관없는 범주형 특성이므로, 원-핫 인코딩으로 변환합니다. 
df.region.value_counts()
df.region.replace({0: 'region_0', 1: 'region_1',2: 'region_2',3: 'region_3'},inplace=True)
region_=pd.get_dummies(df['region'],drop_first=True)
df = pd.concat([df, region_], axis=1)
df.drop(['region'],axis=1,inplace=True)
print(df)

from sklearn.decomposition import KernelPCA
kpca = KernelPCA(n_components=2, kernel='rbf', gamma=1)
kpca.fit(dfx)
#print(kpca.explained_variance_ratio_)

dfx = kpca.fit_transform(dfx)

#kdfx = pd.DataFrame(kdfx)
#dfx = pd.concat([dfx, kdfx], axis=1)
x_train, x_test, y_train, y_test = train_test_split(dfx,dfy,test_size=0.3,random_state=0)
