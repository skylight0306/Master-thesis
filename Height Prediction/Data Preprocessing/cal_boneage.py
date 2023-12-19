# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 14:34:54 2023

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
import re
data_dir = './new1.csv'



df = pd.read_csv(data_dir)

# df['報告內容'] = df['報告內容'].map(lambda x: x.split('\n')[0] if (len(x.split('\n')) == 2 or len(x.split('\n')) == 1) else x.split('\n')[1])

for i in range(len(df['報告內容'])): #i=565
    word = df['報告內容'][i]
    arr = word.split('\n')

    for j in range(len(arr)): #j=2
        
        if  ( 'year' in arr[j] ) or ( 'month' in arr[j] ) or ( 'old' in arr[j] ):

            df['報告內容'][i] = arr[j]
            break
        
def is_number(s):    
    try:    # 如果能运⾏ float(s) 语句，返回 True（字符串 s 是浮点数）        
        float(s)        
        return True    
    except ValueError:  # ValueError 为 Python 的⼀种标准异常，表⽰"传⼊⽆效的参数"        
        pass  # 如果引发了 ValueError 这种异常，不做任何事情（pass：不做任何事情，⼀般⽤做占位语句）    
    try:        
        import unicodedata  # 处理 ASCII 码的包        
        unicodedata.numeric(s)  # 把⼀个表⽰数字的字符串转换为浮点数返回的函数        
        return True    
    except (TypeError, ValueError):        
        pass    
        return False

df['boneage'] = 'error'
# df.columns
# df = df.dropna(axis= 'index', how='any')
# df.reset_index(drop=True, inplace = True)


for i in range(len(df)): #i = 1
    
    age_word = df['報告內容'][i]
    print(i)
    print(age_word)
    # arr = re.split(' |-',age_word)
    arr = age_word.split(' ')
    
    
    
    boneage = 0
    idx = []
    for j in range(len(arr)) : #j=5
        
        # if arr[j].isdigit():
        #     idx.append(j)
        if is_number(arr[j]):
            idx.append(j)    
    if len(idx) == 1: # case1 : 14 years old. or 14 months.
        if "year" in arr[idx[0]+1] :
            boneage = float(arr[idx[0]]) * 12
        elif "month" in arr[idx[0]+1] :
            boneage = float(arr[idx[0]])
        else:
            df['boneage'][i] = "ERROR"
            continue
        
    elif len(idx) == 2: # case2 : 7 years and 10 months old.
        if "year" in arr[idx[0]+1] :
            boneage = float(arr[idx[0]]) * 12
        else:
            df['boneage'][i] = "ERROR"
            continue 
        if "month" in arr[idx[1]+1] :
            boneage += float(arr[idx[1]])      
        else:
            df['boneage'][i] = "ERROR"   
            continue
        
    elif len(idx) == 3: # case3 : 8 years and 10 months to 10 years old.
        first = 0
        second = 0
        #first year or first month
        if "year" in arr[idx[0]+1] :
            first = float(arr[idx[0]]) * 12
        elif "month" in arr[idx[0]+1] :
            first = float(arr[idx[0]])
        else:
            df['boneage'][i] = "ERROR"
            continue 
        
        #first month or second year
        if "year" in arr[idx[1]+1] :
            second = float(arr[idx[1]]) * 12
        elif "month" in arr[idx[1]+1] :
            first += float(arr[idx[1]])      
        else:
            df['boneage'][i] = "ERROR"   
            continue
        
        #second year or second month
        if "year" in arr[idx[2]+1] :
            second = float(arr[idx[2]]) * 12
        elif "month" in arr[idx[2]+1] :
            second += float(arr[idx[2]])
        else:
            df['boneage'][i] = "ERROR"
            continue   
        boneage = (float(first) + float(second)) / 2.0
        
    elif len(idx) == 4: # case4 : 8 years and 10 months to 10 years and 10 months old.
        first = 0
        second = 0
        #first year
        if "year" in arr[idx[0]+1] :
            first = float(arr[idx[0]]) * 12
        else:
            df['boneage'][i] = "ERROR"
            continue 
        
        #first month 
        if "month" in arr[idx[1]+1] :
            first += float(arr[idx[1]])      
        else:
            df['boneage'][i] = "ERROR"   
            continue
        
        #second year
        if "year" in arr[idx[2]+1] :
            second = float(arr[idx[2]]) * 12
        else:
            df['boneage'][i] = "ERROR"
            continue  

        #second year or second month
        if "month" in arr[idx[3]+1] :
            second += float(arr[idx[3]])
        else:
            df['boneage'][i] = "ERROR"
            continue 
        boneage = (float(first) + float(second)) / 2.0
        
    else: 
        if "-" in age_word :
            for j in range(len(arr)) : #j=0
                
                if "-" in arr[j]:
                    idx.append(j)
                      
            
            age_arr = arr[idx[0]].split('-')
            
            num=[]
            for j in range(len(age_arr)) : #j=5
                
                # if age_arr[j].isdigit():
                #     idx.append(j)
                if is_number(age_arr[j]):
                    num.append(j)                 
                    
            if len(num) == 1: # case5 : 12-years-old.
                if "year" in age_arr[1] :
                    boneage = float(age_arr[0]) * 12.0
                elif "month" in age_arr[1] :
                    boneage = float(age_arr[0])
                else :
                    df['boneage'][i] = "ERROR"
                    continue            
                
            elif len(num) == 2: # case6 : 12-13 years old.
            
                    
                if "year" in arr[idx[0]+1] :
                    boneage = (float(age_arr[0]) + float(age_arr[1])) * 6.0
                elif "month" in arr[idx[0]+1] :
                    boneage = (float(age_arr[0]) + float(age_arr[1])) / 2.0
                else:
                    df['boneage'][i] = "ERROR"
                    continue            
            else :
                df['boneage'][i] = "ERROR"
                continue 
            
        
    df['boneage'][i] = boneage



df.to_csv('new1.csv',encoding='utf_8_sig', index = False)









