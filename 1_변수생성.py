# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 00:55:40 2020

@author: tunej
"""
#cd D:\OneDrive - dongguk.edu\동국대학교\2020-2머신러닝과딥러닝\기말고사
import pandas as pd
import numpy as np
#데이터를 불러옵니다.
df = pd.read_csv("data_pepTestCustomers.csv")
df.head()
#널값 확인과 데이터 타입을 확인해봅니다.
df.info()
df.describe()

#나이에 비해 소득이 어느정도인지 비율 income/age ratio를 만들어 보겠습니다.
df['IAratio'] = df['income']/df['age']
print(df)

#소득을 자녀 수로 나눈 액수 income/children을 realincome으로 만들어 보겠습니다.
df['realincome'] = np.where(df['children']!=0,df['income']/df['children'],df['income'])
print(df)

#married, children, car, mortgage가 있으면 지출이 많다는 가정으로 expense 생성
df['expense']= df['income']*(df['married']*0.1 +df['children']*0.1 +df['car']*0.1 +df['mortgage']*0.1)
print(df)


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
print(df)
    
#married와 children을 더해서 부양 의무를 가지고 있는 인원 수를 구해보았습니다.
df['family'] = df['married']+df['children']
print(df)


