# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 16:45:43 2023

@author: q56104034
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# from sklearn.datasets import load_iris
import os
import glob
from datetime import date, datetime, time ,timedelta

data_dir = './data2/'
patients_dir = './patients2/'
df_name = glob.glob(data_dir+'*.csv')
patients_name = glob.glob(patients_dir+'*.csv')

# age_df['exists'] = age_df['exists'].map(lambda x: True if x else np.nan )


# df['性別'] = df['性別'].map(lambda x: 'male' if not x else np.nan)
# df['生日'] = df['生日'].map(lambda x: 'male' if not x else np.nan)
# df['門診日'] = df['male'].map(lambda x: 'male' if not x else np.nan)
# df['身高'] = df['male'].map(lambda x: 'male' if not x else np.nan)
# df['體重'] = df['male'].map(lambda x: 'male' if not x else np.nan)
# df['S'] = df['male'].map(lambda x: 'male' if not x else np.nan)
# df['O'] = df['male'].map(lambda x: 'male' if not x else np.nan)
# df['A+P'] = df['male'].map(lambda x: 'male' if not x else np.nan)


# count = 0

# for i in patients_name: #i = patients_name[0]
#     print('-'*5, i, '-' * 5)
#     df = pd.read_csv(i)
#     print(df['age'][0])
#     if df['age'][0] < 216:
#         count+=1
    
count = 0

for i in df_name: #i = df_name[0]
    print('-'*5, i, '-' * 5)
    df = pd.read_csv(i)
    # df.columns
    # df = df.dropna(axis= 'index', how='any')
    # df.reset_index(drop=True, inplace = True)
    df_id = df['病歷號'][0]
    new_df = pd.DataFrame(columns=['病歷號','性別','age','生日','門診日','身高','體重','用藥資訊','父身高','母身高','S','O','A+P','骨齡檢查單號'])
    
    for index, data in df.iterrows():
        count+=1
        if ( pd.isnull(data['性別']) or pd.isnull(data['生日']) or pd.isnull(data['門診日']) or pd.isnull(data['身高']) or pd.isnull(data['體重']) or pd.isnull(data['S'])):
            continue
        elif not df_id == data['病歷號'] :
           
            # print(new_df)
            new_df.to_csv("./patients2/" + str(df_id) + ".csv", encoding='utf_8_sig', index = False)

            new_df = pd.DataFrame(columns=['病歷號','性別','age','生日','門診日','身高','體重','用藥資訊','父身高','母身高','S','O','A+P','骨齡檢查單號'])
            df_id = data['病歷號']
            born = date(int(data['生日'].split('-',-1)[0]), int(data['生日'].split('-',-1)[1]), int(data['生日'].split('-',-1)[2]))
            Consult = date(int(data['門診日'].split('-',-1)[0]), int(data['門診日'].split('-',-1)[1]), int(data['門診日'].split('-',-1)[2]))
            
            data['age'] = round((Consult - born).days / 30)
            new_df = new_df.append(data, ignore_index=True)           

        else:
            born = date(int(data['生日'].split('-',-1)[0]), int(data['生日'].split('-',-1)[1]), int(data['生日'].split('-',-1)[2]))
            Consult = date(int(data['門診日'].split('-',-1)[0]), int(data['門診日'].split('-',-1)[1]), int(data['門診日'].split('-',-1)[2]))
            
            data['age'] = round((Consult - born).days / 30)
            new_df = new_df.append(data, ignore_index=True)
        
        # if data['age'] < 126: count+=1
            
        # df['病歷號'][j]
    
        new_df.to_csv("./patients2/" + str(df_id) + ".csv", encoding='utf_8_sig', index = False)



#%%


# for i in df_name: #i = './data/1.csv'
#     print('-'*5, i, '-' * 5)
#     df = pd.read_csv(i)
#     # df.columns
#     # df = df.dropna(axis= 'index', how='any')
#     # df.reset_index(drop=True, inplace = True)
#     df_id = df['病歷號'][0]
#     new_df = pd.DataFrame(columns=['病歷號','性別','age','生日','門診日','身高','體重','用藥資訊','父身高','母身高','S','O','A+P'])
    
#     for index, data in df.iterrows():
#         # print(index)
#         # print(data)
#         if ( pd.isnull(data['性別']) or pd.isnull(data['生日']) or pd.isnull(data['門診日']) or pd.isnull(data['身高']) or pd.isnull(data['體重']) or pd.isnull(data['S'])):
#             continue
#         elif not df_id == data['病歷號'] :
           
#             print(new_df)
#             # new_df.to_csv("./patients/" + str(df_id) + ".csv", encoding='utf_8_sig', index = False)

#             new_df = pd.DataFrame(columns=['病歷號','性別','age','生日','門診日','身高','體重','用藥資訊','父身高','母身高','S','O','A+P'])
#             df_id = data['病歷號']
#             born = date(int(data['生日'].split('/',-1)[0]), int(data['生日'].split('/',-1)[1]), int(data['生日'].split('/',-1)[2]))
#             Consult = date(int(data['門診日'].split('/',-1)[0]), int(data['門診日'].split('/',-1)[1]), int(data['門診日'].split('/',-1)[2]))
            
#             data['age'] = round((Consult - born).days / 30)
#             new_df = new_df.append(data, ignore_index=True)           
#             if new_df['age'][0] < 216:
#                 count+=1
#         else:
           
#             born = date(int(data['生日'].split('/',-1)[0]), int(data['生日'].split('/',-1)[1]), int(data['生日'].split('/',-1)[2]))
#             Consult = date(int(data['門診日'].split('/',-1)[0]), int(data['門診日'].split('/',-1)[1]), int(data['門診日'].split('/',-1)[2]))
            
#             data['age'] = round((Consult - born).days / 30)
#             new_df = new_df.append(data, ignore_index=True)
        
#         # if data['age'] < 126: count+=1
            
#         # df['病歷號'][j]
    
#     # new_df.to_csv("./patients/" + str(df_id) + ".csv", encoding='utf_8_sig', index = False)