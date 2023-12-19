# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 20:21:03 2023

@author: doral
"""

import time
import pyautogui
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import Select
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# from sklearn.datasets import load_iris
import random
import os
import glob
from datetime import date, datetime, time ,timedelta

age_df = pd.read_csv('all_order.csv')



options = Options()
options.chrome_executable_path='chromedriver.exe'
driver=webdriver.Chrome(options=options)


# age_df = age_df.dropna(axis= 'index', how='any')
# age_df.reset_index(drop=True, inplace = True)
driver.get('https://bonexpert.com/ahp/')
           
age_df['bonexpert'] = np.nan


count = input("input the number of the patients : ")



time.sleep(5)


for i in range(int(count),len(age_df)): #i=42
    print(str(i) + ' / ' + str(len(age_df)))
    if not pd.isnull(age_df['boneage'][i]) and age_df['gender'][i] == 0:
        
        if age_df['gender'][i] == 1:
            gender = 'Male'
        else:
            gender = 'Female'
        age = age_df['age'][i]/12.0
        boneage = age_df['boneage'][i]/12.0
        height = age_df['身高'][i]
        fheight = age_df['父身高'][i]
        mheight = age_df['母身高'][i]
        
        print( 'NO. '+ str(i))
        print( 'gender : '+ gender)
        print( 'age : '+ str(age) + ' years')
        print( 'boneage : '+ str(boneage) + ' years')
        print( 'height : ' + str(height) + ' cm')
        print( 'fheight : ' + str(fheight) + ' cm')
        print( 'mheight : ' + str(mheight) + ' cm')
        
    

        
        
        
        
        # url = 'https://bonexpert.com/ahp/#/?gdr=' + gender+'&eth=AsiChi&age=' + str(age)+'&ba='+ str(boneage) +'&h='+str(height)+'&fh='+str(fheight)+'&mh='+str(mheight)

        
        driver.find_element(By.ID,"boneage").send_keys(boneage)
        driver.find_element(By.ID,"age").send_keys(age)
        driver.find_element(By.ID,"height").send_keys(height)
        driver.find_element(By.ID,"fheight").send_keys(fheight)
        driver.find_element(By.ID,"mheight").send_keys(mheight)
        pyautogui.hotkey('tab')
        pyautogui.hotkey('tab') #female
        time.sleep(0.5)
        
        pyautogui.hotkey('ctrl','a')
        pyautogui.hotkey('ctrl','c')
        pyautogui.hotkey('alt','tab')
        time.sleep(0.1)
        pyautogui.hotkey('ctrl','a')
        pyautogui.hotkey('ctrl','v')
        time.sleep(0.1)
        pyautogui.hotkey('ctrl','s')
        
        time.sleep(1)
        pyautogui.hotkey('alt','tab')
        
        
        
        
        arr=[]
        f = open('test.txt')
        
        for line in f:
            arr.append(line)

        # AHP = str(arr[-4].split()[0]) #male
        
        AHP = str(arr[-8].split()[0])
        age_df['bonexpert'][i] = AHP
        
        print('predict height : ' + str(AHP) + 'cm' )
        print( '-' * 45)
        
        
        
        driver.find_element(By.ID,"boneage").clear()
        driver.find_element(By.ID,"age").clear()
        driver.find_element(By.ID,"height").clear()
        driver.find_element(By.ID,"fheight").clear()
        driver.find_element(By.ID,"mheight").clear()
        
    
age_df.to_csv("bonexpert.csv", encoding='utf_8_sig', index = False)
# age_df['gender']
# age_df['age']
# age_df['boneage']
# age_df['母身高']
# age_df['父身高']
