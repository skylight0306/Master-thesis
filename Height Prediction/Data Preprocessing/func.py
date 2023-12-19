# -*- coding: utf-8 -*-
"""
Created on Mon May 22 15:38:43 2023

@author: doral
"""

import cv2
import numpy as np
import os
import pandas as pd


total_height_df = pd.read_csv('total_bone_height.csv')

patients_df = pd.read_csv('病例和骨齡單號.csv')

boneage_df = pd.read_csv('new1.csv')


age_df = pd.DataFrame(columns=['檢查單號','病歷號','病患生日','性別','age','檢查結束時間','醫令名稱','Exam','boneage'])

age_df['病歷號'] = patients_df['病歷號']
age_df['醫令名稱'] = patients_df['影像名稱']
age_df['檢查單號'] = patients_df['骨齡檢查單號']


for i in range(len(age_df)):
    print( str(i) + ' / ' + str(len(age_df)))
    for j in range(len(total_height_df)):
        if age_df['病歷號'][i] == total_height_df['病歷號'][j]:
            if total_height_df['性別'][j] == '1':
                age_df['性別'][i] = "M"
            elif total_height_df['性別'][j] == '2':
                age_df['性別'][i] = "F"
                
            age_df['病患生日'][i] = total_height_df['生日'][j]
            
            break
    
    # for k in range(len(boneage_df)):
        
    #     if age_df['檢查單號'][i] == boneage_df['單號'][k]:
    #         age_df['檢查結束時間'][i] = boneage_df['檢查畢時間'][k]
    #         age_df['Exam'][i] = boneage_df['報告內容'][k]
    #         age_df['boneage'][i] = boneage_df['boneage'][k]
        
        
        
        
age_df.to_csv("new_bone.csv", encoding='utf_8_sig', index = False)







#%%

total_height_df['boneage'] = -1

for i in range(len(total_height_df)):
    print( str(i) + ' / ' + str(len(total_height_df)))
    num = total_height_df['骨齡檢查單號'][i].split(',')[0]
    for j in range(len(boneage_df)):
        if num == boneage_df['單號'][j]:
            total_height_df['boneage'][i] = boneage_df['boneage'][j]
    
    
total_height_df['boneage'] = total_height_df['boneage'].map(lambda x: x if str(x)=='-1' else np.nan )

total_height_df = age_df.dropna(axis= 'index', how='any')
total_height_df.reset_index(drop=True, inplace = True)

    
total_height_df.to_csv("bone_height.csv", encoding='utf_8_sig', index = False)



#%%

newbone_df = pd.read_csv('new_bone.csv')
height_df = pd.read_csv('boneage_height.csv')



for i in range(len(age_df)):
    print( str(i) + ' / ' + str(len(age_df)))
    for j in range(len(total_height_df)):
        if age_df['病歷號'][i] == total_height_df['病歷號'][j]:
            
            if str(total_height_df['性別'][j]) == '1':
                age_df['性別'][i] = "M"
            elif str(total_height_df['性別'][j]) == '2':
                age_df['性別'][i] = "F"
                
            age_df['病患生日'][i] = total_height_df['生日'][j]
            
            break
    
    
age_df['age'] = age_df['boneage'] 

age_df = age_df.dropna(axis= 'index', how='any')
age_df.reset_index(drop=True, inplace = True)

age_df.to_csv("new_boneage.csv", encoding='utf_8_sig', index = False)
