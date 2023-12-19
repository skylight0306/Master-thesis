# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 15:56:14 2023

@author: doral
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# from sklearn.datasets import load_iris
import random
import os
import glob
from datetime import date, datetime, time ,timedelta





def isNumber(word):
    
    if '.' in word:    
        arr = word.split('.')
        if not len(arr) == 2: 
            return False
        else:
            if arr[0].isdigit() and arr[1].isdigit():
                return True
    else:
        if word.isdigit():
            return True
        else :
            return False
        
    


patients_dir = './patients2/'

df_name = glob.glob(patients_dir+'*.csv')



count = input("input the number of the patients : ")

for i in range(int(count),len(df_name)): #i=0


    print( str(i) + ' / ' + str(len(df_name)) + '\nCurrent patient : ' + str(df_name[i]) )
    print( '-' * 20)
    print('\n')
    df = pd.read_csv(df_name[i])
    if not ( pd.isnull(df['父身高'][0]) or pd.isnull(df['母身高'][0]) ) :
        continue
    
    
    got = False
    report = []
    for j in range(len(df)): #j=0
        if 'father' in df['S'][j] or 'mother' in df['S'][j] or 'dad' in df['S'][j] or 'mom' in df['S'][j] or 'cm' in df['S'][j]:
            report.append(df['S'][j])
    
    
    
    
    for j in range(len(report)): #j=0
        print( str(j+1) + ' / ' + str(len(report)) )
        print( '-' * 20)
        print('\n')
        print( report[j] )
        
        father = input("input the father height : (next message key '.') ( break key '0') ")
        if father == '.':
            father = ''
            mother = ''
            continue
        elif father == '0':
            father = ''
            mother = ''
            break
        
        mother = input("input the mother height : (next message key '.') ( break key '0') ")
        if mother == '.':
            father = ''
            mother = ''
            continue
        elif mother == '0':
            father = ''
            mother = ''
            break
              
        if isNumber(mother) and isNumber(father) :
            got = True
            break
    
    
    if got:
        df['父身高'] = float(father)
        df['母身高'] = float(mother)
        df.to_csv(df_name[i], encoding='utf_8_sig', index = False)
    
    


