# -*- coding: utf-8 -*-
"""
Created on Sat Oct 22 22:57:19 2022

@author: q56104034
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




#%%
def mae(groundtruth, pred):  #groundtruth=y_train pred=train_pred
    
    err = 0.0
    for i in range(len(groundtruth)): #i=0
        
        err+=abs(float(groundtruth[i])-float(pred[i]))
    
    avg_err = err/float(len(groundtruth))
    
    
    
    return avg_err


seed = 1999
def setup_seed(seed):

     np.random.seed(seed)
     random.seed(seed)



setup_seed(seed)
#%%

# age_df = pd.read_csv('bonexpert.csv')

# res_df = pd.read_csv('predict.csv')


# age_df = age_df.dropna(axis= 'index', how='any')
# age_df.reset_index(drop=True, inplace = True)
# res_df['bonexpert'] = np.nan


# for i in range( len(age_df) ): #i = 0
#     print( str(i) +' / ' + str(len(age_df)))
#     for j in range(len(res_df)): #j =0
#         if age_df['病歷號'][i] == res_df['病歷號'][j] and age_df['pred_boneage'][i] == res_df['pred_boneage'][j] and age_df['門診日'][i] == res_df['門診日'][j]:
#             res_df['bonexpert'][j] = age_df['bonexpert'][i]


# res_df.to_csv("predict_bonexpert.csv", encoding='utf_8_sig', index = False)




#%%


# def is_number(s):    
#     try:    # 如果能运⾏ float(s) 语句，返回 True（字符串 s 是浮点数）        
#         float(s)        
#         return True    
#     except ValueError:  # ValueError 为 Python 的⼀种标准异常，表⽰"传⼊⽆效的参数"        
#         pass  # 如果引发了 ValueError 这种异常，不做任何事情（pass：不做任何事情，⼀般⽤做占位语句）    
#     try:        
#         import unicodedata  # 处理 ASCII 码的包        
#         unicodedata.numeric(s)  # 把⼀个表⽰数字的字符串转换为浮点数返回的函数        
#         return True    
#     except (TypeError, ValueError):        
#         pass    
#         return False
    
    
# age_df = pd.read_csv('bonexpert.csv')
# arr = []
# for i in range( len(age_df) ): #i = 13
#     print( str(i) +' / ' + str(len(age_df)))
#     if not pd.isnull(age_df['bonexpert'][i]):
#         if not is_number(age_df['bonexpert'][i]):
#             print(str(age_df['bonexpert'][i]))
#             arr.append(age_df['bonexpert'][i])
#             age_df['bonexpert'][i] = np.nan


# age_df.to_csv("bonexpert.csv", encoding='utf_8_sig', index = False)

#%%
# patients_dir = './patients2/'
# df_name = glob.glob(patients_dir+'*.csv')

# age_df = pd.DataFrame(columns=['病歷號','性別','age','生日','門診日','身高','體重','用藥資訊','父身高','母身高','骨齡檢查單號'])

# for i in df_name: #i = df_name[0]
#     print('-'*5, i, '-' * 5)
#     df = pd.read_csv(i)
#     # df.columns
#     # df = df.dropna(axis= 'index', how='any')
#     # df.reset_index(drop=True, inplace = True)
#     age_df = pd.concat([age_df,df],axis=0, ignore_index=True)






# age_df = pd.DataFrame(columns=['病歷號','性別','age','生日','門診日','身高','體重','用藥資訊','父身高','母身高','S','O','A+P'])

# for i in df_name: #i = df_name[0]
#     print('-'*5, i, '-' * 5)
#     df = pd.read_csv(i)
#     # df.columns
#     # df = df.dropna(axis= 'index', how='any')
#     # df.reset_index(drop=True, inplace = True)
#     age_df = pd.concat([age_df,df],axis=0, ignore_index=True)

    
# # age_df = pd.read_csv('boneage_file.csv')
# age_df=age_df.loc[:,['病歷號','性別','age','生日','門診日','身高','體重','用藥資訊','父身高','母身高']]

# age_df['gender'] = age_df['性別'].map(lambda x: 1.0 if x==1 else 0.0)
# # age_df.query('gender==1.0') Confirm how many are male
# # f = lambda x: 1 if x=="C" else (0 if x=="S" else 2)

# age_df['用藥資訊'] = age_df['用藥資訊'].fillna(str(0))
# age_df['用藥資訊'] = age_df['用藥資訊'].astype(str)

# age_df['用藥資訊'] = age_df['用藥資訊'].map(lambda x: 0 if not ',' in x else x)

# age_df['用藥資訊'] = age_df['用藥資訊'].astype(str)

# age_df['growth_hormone'] = age_df['用藥資訊'].map(lambda x: -1 if (( '907452' in x ) or ( '525953' in x ) or ( '526455' in x )) else ( 1 if (( '407756' in x ) or ( '407751' in x ) or ( '827954' in x ) or ( '407952' in x )) else 0 ))

# for i in range(len(age_df['growth_hormone'])):
#     print(age_df['growth_hormone'][i])
    
# df_data = pd.DataFrame(data= age_df,
#                       columns= ['病歷號', '性別', 'age', '生日', '門診日', '身高', '體重', '用藥資訊', '父身高', '母身高',
#                             'growth_hormone','骨齡檢查單號'])
# # #83523





# df_data = pd.DataFrame(data= age_df,
#                      columns= ['病歷號', '性別', 'age', '生日', '門診日', '身高', '體重', '用藥資訊', '父身高', '母身高',
#                             '骨齡檢查單號', 'growth_hormone'])

# # for i in range(len(age_df['用藥資訊'])):
# #     print(age_df['用藥資訊'][i])

# age_df.to_csv("total_height.csv", encoding='utf_8_sig', index = False)


# # age_df['weight'] = age_df['weight'].map(lambda x: np.nan if x=='.' else x)

# arr = []
# name = ''


# for i in df['病歷號']: 
   
#     if not i == name:
#         name = i
#         arr.append(i)




# df_data = df_data.dropna(axis= 'index', how='any')
# df_data.reset_index(drop=True, inplace = True)

# age_df.to_csv("病例和骨齡單號.csv", encoding='utf_8_sig', index = False)

# d_data = pd.DataFrame(data= df_data,
#                       columns= ['病歷號','骨齡檢查單號'])


# name = ''
# age_df = pd.DataFrame(columns=['病歷號','骨齡檢查單號'])


# for i in range(len(d_data)): #i=2
#     if not d_data['骨齡檢查單號'][i] == name :
        
#         name = d_data['骨齡檢查單號'][i]
#         data['骨齡檢查單號'] = d_data['骨齡檢查單號'][i][:-1]
#         data['病歷號'] = d_data['病歷號'][i]
#         age_df = age_df.append(data, ignore_index=True)
      




#%% with boneage

# age_df = pd.read_csv('nboneage_height.csv')
# # age_df['boneage'] = np.nan

# # # df = age_df['boneage']
# # # df = df.dropna(axis= 'index', how='any')
# # # df.reset_index(drop=True, inplace = True)



# bone_df = pd.read_csv('new_bone.csv')

# for i in range(len(bone_df)): #i=0
#     print(str(i) + ' / ' + str(len(bone_df)))
#     # print( bone_df['病歷號'][i])
    
#     for j in range(len(age_df)): #j=0
#         # print(age_df['病歷號'][j])
#         if bone_df['病歷號碼'][i] == age_df['病歷號'][j] :
            
           
#             bone_day = datetime.strptime(bone_df['檢查結束時間'][i], '%Y/%m/%d %H:%M')
#             height_day = datetime.strptime(age_df['門診日'][j], '%Y/%m/%d')
#             error = abs(bone_day-height_day).days
            
#             if error < 5:
#                 age_df['pred_boneage'][j] = bone_df['pred_boneage'][i]
#                 # print('found it !')
            
#             # born = date(int(data['生日'].split('-',-1)[0]), int(data['生日'].split('-',-1)[1]), int(data['生日'].split('-',-1)[2]))


# age_df.to_csv("boneage_height.csv", encoding='utf_8_sig', index = False)



# age_df.to_csv("height_boneage.csv", encoding='utf_8_sig', index = False)



# type(age_df['height'][3])
# age_df.fillna(0)
# age_df = age_df.dropna(axis= 'index', how='any')
# age_df.reset_index(drop=True, inplace = True)
# len(dir(age_df['height']))

# age_df['mature'] = [216-int(i) for i in age_df['true_age']]
# age_df['height'] = [216-float(i) for i in age_df['height']]
# age_df['state'] = [i-j for i, j in zip(age_df['true_age'], age_df['bone_age'])]
# dataset_size = len(age_df)


# print('Total:', len(age_df), 'data')




# age_df = pd.read_csv('new.csv')
# name=''
# count = 0
# for i in range(len(age_df)):
#     if name == age_df['病歷號'][i]:
#         count +=1
#         print(name)
#     name = age_df['病歷號'][i]


# # age_df = pd.read_csv('height.csv')

# df = pd.read_csv('../qoca/bone.csv')

# arr_2 = sorted(df['病歷號碼'])




# arr = sorted(age_df['病歷號碼'])


# arr = arr + arr_2
# arr = sorted(arr)


# name = ''
# arr = []
# count = 0
# for i in range(len(arr)):
#     if not arr[i] == name :
#         count+=1
#         name = arr[i]
#         arr.append(arr[i])



# age_df['病歷'] = only_arr

# arr.to_csv('病例.csv',encoding='utf-8')

# d ={'病歷':only_arr}
# df = pd.DataFrame(data=d)

# df.to_csv("病例.csv",encoding='utf_8_sig', index = False)




# data = []
# data2 = []


# df_name = glob.glob('*.csv')

# for i in range(1,len(df_name)): #i = df_name[0]


#     print('-'*5, i, '-' * 5)
#     df = pd.read_csv(df_name[i])
#     data2 = sorted(df['病歷號'])
    
#     data += data2

    
    
#     data = sorted(data)
# arr = data

# height = pd.read_csv('../qoca/bone.csv')
# height['病歷號碼']

# 424 patients exist
# for i in range(len(arr)):
    
#     exist = False
#     print(arr[i])
#     for j in range(len(height['病歷號碼'])):
        
#         if arr[i] == height['病歷號碼'][j] :
#             exist_arr.append(arr[i])
#             break
            
#     print(exist)


#%%


# new_df = pd.DataFrame(columns= ['病歷號','gender', 'age','身高', '體重','growth_hormone','parents_h','bmi'])


# age_df = pd.read_csv('all_order.csv')


# for i in range(len(age_df)): #i=1
#     print( str(i) +' / ' + str(len(age_df)))

#     if not name == age_df['病歷號'][i]:
#         bone_day = datetime.strptime(age_df['門診日'][i], '%Y/%m/%d')
#         new_df = new_df.append(age_df.iloc[i], ignore_index=True) 
#         name = age_df['病歷號'][i]
        
#     else:
        
#         new_day = datetime.strptime(age_df['門診日'][i], '%Y/%m/%d')
        
#         error = abs(bone_day-new_day).days
#         if not error == 0:
#             new_df = new_df.append(age_df.iloc[i], ignore_index=True) 
            
#         bone_day = new_day
        
        
        
#     age_df.to_csv("all_1year.csv", encoding='utf_8_sig', index = False)
        
#%%

# age_df = pd.read_csv('all_order.csv')

# # age_df['gender'] = age_df['gender'].map(lambda x: True if x else np.nan ) # gender

# age_df['age'] = age_df['age'].map(lambda x: x if x < 217 else np.nan )

# age_df = age_df.dropna(axis= 'index', how='any')
# age_df.reset_index(drop=True, inplace = True)



# age_df['after1yearH'] = np.nan
# name = ''
# count = 0

# for i in range(len(age_df)): # i=5
#     print( str(i) +' / ' + str(len(age_df)))
#     if not name == age_df['病歷號'][i]:
#         name = age_df['病歷號'][i]
#         arr = pd.DataFrame(columns=['idx','age'])
#         d = {
#             "idx": i,
#             "age": age_df['age'][i],
#         }

#         # if not pd.isnull(age_df['boneage'][i]):
#         arr = arr.append(d,ignore_index=True)
#     else:
#         for j in range(len(arr)): #j=0
#             if abs(age_df['age'][i] - arr['age'][j]) == 12:
#                 age_df['after1yearH'][arr['idx'][j]] = age_df['身高'][i]
                
#             elif abs(age_df['age'][i] - arr['age'][j]) == 13 :
#                 age_df['after1yearH'][arr['idx'][j]] = age_df['身高'][i]
                
#             elif abs(age_df['age'][i] - arr['age'][j]) == 11 :
#                 age_df['after1yearH'][arr['idx'][j]] = age_df['身高'][i]
                
#         d = {
#             "idx": i,
#             "age": age_df['age'][i],
#         }                
#         # if not pd.isnull(age_df['boneage'][i]):
#         arr = arr.append(d,ignore_index=True)
# age_df.to_csv("after1year.csv",encoding='utf_8_sig', index = False)
        


#%%



# age_df = pd.read_csv('all_order.csv')

# # age_df['gender'] = age_df['gender'].map(lambda x: True if x else np.nan ) # gender

# # age_df['age'] = age_df['age'].map(lambda x: x if x < 217 else np.nan )

# # age_df = age_df.dropna(axis= 'index', how='any')
# # age_df.reset_index(drop=True, inplace = True)



# age_df['after_month'] = np.nan
# age_df['after_H'] =np.nan


# new_df = pd.DataFrame(columns= ['病歷號','gender', 'age','身高', '體重','growth_hormone','parents_h','bmi','after_month','after_H'])

# name = ''
# count = 0

# for i in range(len(age_df)): # i=13
#     print( str(i) +' / ' + str(len(age_df)))
#     if not pd.isnull(age_df['boneage'][i]):
        
#         name = age_df['病歷號'][i]
        
        
#         for j in range(i+1,len(age_df)):
#             ori_d = age_df.iloc[i]
#             if not name == age_df['病歷號'][j]:
#                 break
            
#             d = age_df.iloc[j]
            
#             ori_d['after_month'] = d['age'] - ori_d['age']
#             ori_d['after_H'] = d['身高']
            
            
#             new_df = new_df.append(ori_d,ignore_index=True)
        
    

# new_df.to_csv("predict.csv",encoding='utf_8_sig', index = False)
        

#%%


# df_data = pd.DataFrame(data= age_df,
#                        columns= ['病歷號','gender','age','身高', '體重','growth_hormone','父身高','母身高','boneage','pred_boneage'])

# age_df = pd.read_csv('all_1year.csv')


# age_df = df_data


# name = ''
# new_df = pd.DataFrame(columns= ['病歷號','gender', 'age','身高', '體重','growth_hormone','parents_h','bmi'])
# arr = pd.DataFrame(columns=['idx','age'])

# age_df['nFAH'] = np.nan

# for i in range(len(age_df)): # i=0
#     print( str(i) +' / ' + str(len(age_df)))
#     if not name == age_df['病歷號'][i]:

#         if len(arr) > 1:
#             if age_df['gender'][arr['idx'][0]] == 1:
#                 male = True
#             else:
#                 male = False
                
#             if male and age_df['boneage'][arr['idx'][len(arr)-1]] > 191.9:

#                 for j in range(len(arr)-1):
#                     age_df['nFAH'][arr['idx'][j]] =  age_df['身高'][arr['idx'][len(arr)-1]]
#                     new_df = new_df.append(age_df.iloc[arr['idx'][j]], ignore_index=True)   
                
#             elif ( not male ) and age_df['boneage'][arr['idx'][len(arr)-1]] > 167.9:

#                 for j in range(len(arr)-1):
#                     age_df['nFAH'][arr['idx'][j]] =  age_df['身高'][arr['idx'][len(arr)-1]]
#                     new_df = new_df.append(age_df.iloc[arr['idx'][j]], ignore_index=True) 
        
        
#         name = age_df['病歷號'][i]
#         arr = pd.DataFrame(columns=['idx','age'])
#         d = {
#             "idx": i,
#             "age": age_df['age'][i],
#         }

#         # if not pd.isnull(age_df['boneage'][i]):
#         arr = arr.append(d,ignore_index=True)
#     else:

#         d = {
#             "idx": i,
#             "age": age_df['age'][i],
#         }                
#         # if not pd.isnull(age_df['boneage'][i]):
#         arr = arr.append(d,ignore_index=True)


#%%
'''
name = ''
arr = []
old_arr = []

for i in range(len(age_df)):
    if not name == age_df['病歷號'][i]:
        name = age_df['病歷號'][i]
        arr.append(age_df['病歷號'][i])


for i in range(len(old_arr)):
    for j in range(len(arr)):
        if arr[j] == old_arr[i]:
            break
        
        arr.append(old_arr[i])




name=''
bone_arr=[]
bone_df= pd.read_csv('../qoca/bone.csv')
bone_df['病歷號'] = sorted(bone_df['病歷號碼'])

for i in range(len(bone_df)):
    if not name == bone_df['病歷號'][i]:
        name = bone_df['病歷號'][i]
        bone_arr.append(bone_df['病歷號'][i])

for i in range(len(bone_arr)):
    print( str(i) +' / ' + str(len(bone_arr)))
    for j in range(len(arr)):
        if arr[j] == bone_arr[i]:
            break
        
        arr.append(bone_arr[i])
        break

d ={'病歷號':arr}
df = pd.DataFrame(data=d)
new_df.to_csv("test.csv",encoding='utf_8_sig', index = False)
'''


# df_data = pd.DataFrame(data= age_df,
#                      columns= ['gender','age','身高', '體重','growth_hormone','父身高','母身高','boneage'])





# old_df = pd.read_csv('old_height_boneage.csv')
# old_df = old_df.dropna(axis= 'index', how='any')
# old_df.reset_index(drop=True, inplace = True)

# for i in range(len(old_df)): #i=0
#     print(str(i) + ' / ' + str(len(old_df)))
#     # print( bone_df['病歷號'][i])
    
#     for j in range(len(age_df)): #j=0
#         # print(age_df['病歷號'][j])
#         if old_df['病歷號'][i] == age_df['病歷號'][j] :
            
            
#             bone_day = datetime.strptime(old_df['門診日'][i], '%Y/%m/%d')
#             height_day = datetime.strptime(age_df['門診日'][j], '%Y/%m/%d')
#             error = abs(bone_day-height_day).days
            
#             if error < 5:
#                 age_df['boneage'][j] = old_df['boneage'][i]
#                 # print('found it !')
            
#             # born = date(int(data['生日'].split('-',-1)[0]), int(data['生日'].split('-',-1)[1]), int(data['生日'].split('-',-1)[2]))



# age_df.to_csv("boneage_height.csv", encoding='utf_8_sig', index = False)
#%%


# age_df = pd.read_csv('after1year.csv')
# pre_height=0.0
# name=''

# new_df = pd.DataFrame(columns= ['病歷號','gender', 'age','身高', '體重','growth_hormone','parents_h','bmi'])


# for i in range(len(age_df)): #i=1
#     print( str(i) +' / ' + str(len(age_df)))

#     if not name == age_df['病歷號'][i]:
#         pre_height = age_df['身高'][i]
        
#         name = age_df['病歷號'][i]
        
#     else:
    
#        if pre_height - age_df['身高'][i] > 5.0:
#            new_df = new_df.append(age_df.iloc[i], ignore_index=True) 
    
#        pre_height = age_df['身高'][i]  

#     age_df.to_csv("all_1year.csv", encoding='utf_8_sig', index = False)

# new_df[0:50]
# new_df[50:]



#%%



# age_df = pd.read_csv('boneage_height.csv')
# age_df = pd.read_csv('predict.csv')
age_df = pd.read_csv('predict_bonexpert.csv')

# age_df['gender'] = age_df['gender'].map(lambda x: True if x else np.nan ) # gender
# a=(age_df['gender']==0)
# age_df[a]
age_df['age'] = age_df['age'].map(lambda x: x if x < 217 else np.nan )
# age_df['after_month'] = age_df['after_month'].map(lambda x: x if x < 24 else np.nan )


age_df = age_df.dropna(axis= 'index', how='any')
age_df.reset_index(drop=True, inplace = True)


# df_data = pd.DataFrame(data= age_df,
#                        columns= ['病歷號','gender','age','身高', '體重','growth_hormone','父身高','母身高','boneage','pred_boneage','after_month','after_H'])


df_data = pd.DataFrame(data= age_df,
                       columns= ['病歷號','gender','age','身高', '體重','growth_hormone','父身高','母身高','boneage','pred_boneage','after_month','after_H','bonexpert'])



# df_data = pd.DataFrame(data= age_df,
#                       columns= ['gender','age','身高', '體重','growth_hormone','父身高','母身高'])

#%%
from sklearn import preprocessing


df_data['pred_boneage'] = df_data['pred_boneage'].map(lambda x: 216.0 if x > 216.0 else x )
df_data['bmi'] = [j / ( (float(i)/100.0) ** 2 ) for i,j in zip(df_data['身高'],df_data['體重'])]
df_data['parents_h'] =  [(float(i) +float(j))/2.0 for i,j in zip(df_data['父身高'],df_data['母身高'])]
# df_data['bone_err'] = [float(i) - float(j) for i,j in zip(df_data['pred_boneage'],df_data['boneage'])]
df_data['grow'] =  [(float(i) -float(j)) for i,j in zip(df_data['after_H'],df_data['身高'])]
df_data['GH'] = df_data['growth_hormone'].map(lambda x: x if x==1 else 0 )
df_data['GnRHa'] = df_data['growth_hormone'].map(lambda x: 1.0 if x==-1.0 else 0.0)

# df_data = preprocessing.scale(df_data)



# for i in range(len(df_data)): #i=0
#     if abs(df_data['bone_err'][i]) > 40 :
#         print( str(df_data['病歷號'][i]) + " / " + str(df_data['pred_boneage'][i]) + " / " + str(df_data['boneage'][i]))
    






#%%

# df_data['height_category'] = pd.cut(df_data['age'], 10)
df_data['month_category'] = pd.cut(df_data['after_month'], 10)


def split_patients( data, test_size): #data = df_data
    
    # -----Model 1-----
    # train_df = pd.DataFrame(columns= ['病歷號','gender','age','身高','體重','parents_h'])
    # test_df = pd.DataFrame(columns= ['病歷號','gender','age','身高','體重','parents_h'])
 

    # -----Model 2-----
    # train_df = pd.DataFrame(columns= ['病歷號','gender','age','身高','體重','parents_h','bmi'])
    # test_df = pd.DataFrame(columns= ['病歷號','gender','age','身高','體重','parents_h','bmi'])
 
    # -----Model 3-----
    train_df = pd.DataFrame(columns= ['病歷號','gender', 'age','身高', '體重','growth_hormone','parents_h','bmi'])
    test_df = pd.DataFrame(columns= ['病歷號','gender', 'age','身高', '體重','growth_hormone','parents_h','bmi'])
    
    
    # -----Model 4-----    
    # train_df = pd.DataFrame(columns= ['病歷號','gender','age','身高', '體重','growth_hormone','parents_h','bmi','boneage'])
    # test_df = pd.DataFrame(columns= ['病歷號','gender','age','身高', '體重','growth_hormone','parents_h','bmi','boneage'])


    # -----Model 5-----    
    # train_df = pd.DataFrame(columns= ['病歷號','gender','age','身高', '體重','growth_hormone','parents_h','bmi','boneage','bone_err'])
    # test_df = pd.DataFrame(columns= ['病歷號','gender','age','身高', '體重','growth_hormone','parents_h','bmi','boneage','bone_err'])
  
    
    
    name_arr = []
    name = ''
    for i in range(len(data)):

        if not name == age_df['病歷號'][i]:
            name = age_df['病歷號'][i]
            name_arr.append(age_df['病歷號'][i])
            
    
    # train_size = int(len(name_arr) * (1.0-test_size))
    # test_size = len(name_arr) - train_size
    
    train_patients, test_patients = train_test_split(  name_arr, 
                                    test_size = test_size, 
                                    random_state=seed)
    
    for i in range(len(train_patients)): #i=0
        df = data.loc[(data['病歷號'] == train_patients[i] )]
        train_df = pd.concat([train_df,df])
    
    for i in range(len(test_patients)): #i=0
        df = data.loc[(data['病歷號'] == test_patients[i] )]
        test_df = pd.concat([test_df,df])  
    
    return train_df, test_df

#%%

from sklearn.model_selection import train_test_split


# 5-fold
# train_org_df, val_org_df = train_test_split(  df_data, 
#                                 test_size = 0.2, 
#                                 random_state=seed,
#                                 stratify = df_data['height_category'])

train_org_df, val_org_df = split_patients(df_data, 0.2)


'''
A_df, C_df = train_test_split(  A_df, 
                                test_size = 0.5, 
                                random_state=seed,
                                stratify = A_df['height_category'])

A_df, D_df = train_test_split(  A_df, 
                                test_size = 0.5, 
                                random_state=seed,
                                stratify = A_df['height_category'])

C_df, E_df = train_test_split(  C_df, 
                                test_size = 0.5, 
                                random_state=seed,
                                stratify = C_df['height_category'])


train_org_df = pd.concat([B_df,C_df,D_df,E_df])
val_org_df = A_df


HEIGHT
CASE1 :
train_df = B_df, C_df, D_df, E_df
val_df = A_df
XGBOOST

CASE2 :
train_df = A_df, C_df, D_df, E_df
val_df = B_df
XGBOOST

CASE3 :
train_df = A_df, B_df, D_df, E_df
val_df = C_df
XGBOOST

CASE4 :
train_df = A_df, B_df, C_df, E_df
val_df = D_df
XGBOOST

CASE5 :
train_df = A_df, B_df, C_df, D_df
val_df = E_df
XGBOOST
'''

#%%
# train_df = pd.DataFrame(data= train_org_df,
#                       columns= ['gender','age','身高', '體重','growth_hormone','parents_h','bmi','boneage','bone_err'])

# val_df = pd.DataFrame(data= val_org_df,
#                       columns= ['gender','age','身高', '體重','growth_hormone','parents_h','bmi','boneage','bone_err'])
 
# -----Model 1-----
# best_train_loss : 1.899443716562939, std : best_train_std3.3635303270722927, best_test_loss : 3.9485084528140524, best_test_std : 3.39248285029718, n_estimators : 30, max_depth : 24, learning_rate : 0.12
# train_df = pd.DataFrame(data= train_org_df, columns= ['gender','age','身高','體重','parents_h'])
# val_df = pd.DataFrame(data= val_org_df, columns= ['gender','age','身高','體重','parents_h'])


# -----Model 2-----
# best_train_loss : 0.03810514248280383, std : best_train_std0.5928989631968195, best_test_loss : 1.3402268350096354, best_test_std : 1.6876576518953856, n_estimators : 130, max_depth : 11, learning_rate : 0.1
# train_df = pd.DataFrame(data= train_org_df, columns= ['gender','age','身高','體重','parents_h','bmi'])
# val_df = pd.DataFrame(data= val_org_df, columns= ['gender','age','身高','體重','parents_h','bmi'])

# -----Model 3-----

 # best_train_loss : 0.05472486079266311, std : best_train_std0.5518555960895801, best_test_loss : 1.3274727156002604, best_test_std : 1.6684532223282837, n_estimators : 164, max_depth : 5, learning_rate : 0.14
# train_df = pd.DataFrame(data= train_org_df, columns= ['gender','age','身高', '體重','GH','GnRHa','parents_h','bmi'])
# val_df = pd.DataFrame(data= val_org_df, columns= ['gender','age','身高', '體重','GH','GnRHa','parents_h','bmi'])

# train_df = pd.DataFrame(data= train_org_df, columns= ['gender','age','身高', '體重','growth_hormone','parents_h','bmi'])
# val_df = pd.DataFrame(data= val_org_df, columns= ['gender','age','身高', '體重','growth_hormone','parents_h','bmi'])


# -----Model 4-----   
# 'best_train_loss : 0.06983022163991702, std : best_train_std0.5613530619678301, best_test_loss : 1.5507554033811304, best_test_std : 1.8890516459320312, n_estimators : 164, max_depth : 5, learning_rate : 0.12'
# train_df = pd.DataFrame(data= train_org_df, columns= ['gender','age','身高', '體重','GH','GnRHa','parents_h','bmi','boneage'])
# val_df = pd.DataFrame(data= val_org_df, columns= ['gender','age','身高', '體重','GH','GnRHa','parents_h','bmi','boneage'])


# best_train_loss : 0.04333060377770504, std : best_train_std0.4944083281322173, best_test_loss : 1.2005692274769946, best_test_std : 1.6260760637986515, n_estimators : 197, max_depth : 5, learning_rate : 0.14
# train_df = pd.DataFrame(data= train_org_df, columns= ['gender','boneage','身高', '體重','GH','GnRHa','parents_h','bmi'])
# val_df = pd.DataFrame(data= val_org_df, columns= ['gender','boneage','身高', '體重','GH','GnRHa','parents_h','bmi'])


# -----Model 5-----    
# train_df = pd.DataFrame(data= train_org_df, columns= ['gender','age','身高', '體重','GH','GnRHa','parents_h','bmi','pred_boneage'])
# val_df = pd.DataFrame(data= val_org_df, columns= ['gender','age','身高', '體重','GH','GnRHa','parents_h','bmi','pred_boneage'])

# best_train_loss : 0.024983622666229943, std : best_train_std0.47136146697832626, best_test_loss : 1.5287952065054615, best_test_std : 1.9045649665346267, n_estimators : 210, max_depth : 5, learning_rate : 0.14
# train_df = pd.DataFrame(data= train_org_df, columns= ['gender','pred_boneage','身高', '體重','GH','GnRHa','parents_h','bmi'])
# val_df = pd.DataFrame(data= val_org_df, columns= ['gender','pred_boneage','身高', '體重','GH','GnRHa','parents_h','bmi'])
#%%
# -----Model 1----- 
 # 'best_train_loss : 1.1678849902534103, std : 1.7819910628048496best_test_loss : 1.9038361508452535, best_test_std : 1.940963418011408, n_estimators : 30, max_depth : 5, learning_rate : 0.1'
# train_df = pd.DataFrame(data= train_org_df, columns= ['gender','age','身高', '體重','parents_h','after1yearH'])
# val_df = pd.DataFrame(data= val_org_df, columns= ['gender','age','身高', '體重','parents_h','after1yearH'])




# -----Model 2----- 
 # 'best_train_loss : 0.821474983755685, std : 1.6758095134971438best_test_loss : 1.870676202860858, best_test_std : 1.9172896684613683, n_estimators : 30, max_depth : 11, learning_rate : 0.1'
# train_df = pd.DataFrame(data= train_org_df, columns= ['gender','age','身高', '體重','GH','GnRHa','parents_h','after1yearH'])
# val_df = pd.DataFrame(data= val_org_df, columns= ['gender','age','身高', '體重','GH','GnRHa','parents_h','after1yearH'])



# -----Model 3-----    
 # 'best_train_loss : 0.06211825860948669, std : 0.7684586627524481best_test_loss : 1.8779583875162564, best_test_std : 1.9483217394334604, n_estimators : 90, max_depth : 17, learning_rate : 0.12'
# train_df = pd.DataFrame(data= train_org_df, columns= ['gender','age','身高', '體重','GH','GnRHa','parents_h','bmi','after1yearH'])
# val_df = pd.DataFrame(data= val_org_df, columns= ['gender','age','身高', '體重','GH','GnRHa','parents_h','bmi','after1yearH'])



# -----Model 4-----    
# new  'best_train_loss : 1.0271256599788812, std : 1.5272442264377344best_test_loss : 1.7555555555555562, best_test_std : 1.5356665018610884, n_estimators : 30, max_depth : 5, learning_rate : 0.1'
# train_df = pd.DataFrame(data= train_org_df, columns= ['gender','pred_boneage','身高', '體重','GH','GnRHa','parents_h','bmi','after1yearH'])
# val_df = pd.DataFrame(data= val_org_df, columns= ['gender','pred_boneage','身高', '體重','GH','GnRHa','parents_h','bmi','after1yearH'])

 # 'best_train_loss : 0.9395061728395059, std : 1.678526076241702best_test_loss : 1.8459037711313415, best_test_std : 1.9040404635849715, n_estimators : 30, max_depth : 11, learning_rate : 0.08'
# train_df = pd.DataFrame(data= train_org_df, columns= ['gender','pred_boneage','身高', '體重','GH','GnRHa','parents_h','after1yearH'])
# val_df = pd.DataFrame(data= val_org_df, columns= ['gender','pred_boneage','身高', '體重','GH','GnRHa','parents_h','after1yearH'])



# -----Model 5----- 


# new  'best_train_loss : 1.0271256599788812, std : 1.5272442264377344best_test_loss : 1.7555555555555562, best_test_std : 1.5356665018610884, n_estimators : 30, max_depth : 5, learning_rate : 0.1
# train_df = pd.DataFrame(data= train_org_df, columns= ['gender','boneage','身高', '體重','GH','GnRHa','parents_h','bmi','after1yearH'])
# val_df = pd.DataFrame(data= val_org_df, columns= ['gender','boneage','身高', '體重','GH','GnRHa','parents_h','bmi','after1yearH'])


# 'best_train_loss : 0.8662605588044177, std : 1.6669533436715782best_test_loss : 1.845123537061116, best_test_std : 1.858556360933723, n_estimators : 30, max_depth : 11, learning_rate : 0.1'
# train_df = pd.DataFrame(data= train_org_df, columns= ['gender','boneage','身高', '體重','GH','GnRHa','parents_h','after1yearH'])
# val_df = pd.DataFrame(data= val_org_df, columns= ['gender','boneage','身高', '體重','GH','GnRHa','parents_h','after1yearH'])



# -------

 # 'best_train_loss : 0.32571474983755666, std : 1.3567861921223878best_test_loss : 1.8936280884265257, best_test_std : 1.958947588277788, n_estimators : 30, max_depth : 17, learning_rate : 0.14'
# train_df = pd.DataFrame(data= train_org_df, columns= ['gender','age','pred_boneage','身高', '體重','GH','GnRHa','parents_h','bmi','after1yearH'])
# val_df = pd.DataFrame(data= val_org_df, columns= ['gender','age','pred_boneage','身高', '體重','GH','GnRHa','parents_h','bmi','after1yearH'])

#%%
# -----Model 1----- 
# 'train_loss: 0.77 ± 1.11 cm,    test_loss: 0.79 ± 0.73 cm / n_estimators : 5, max_depth : 20, learning_rate : 0.1'
# 'train_loss: 0.99 ± 1.09 cm,    test_loss: 1.02 ± 0.95 cm / n_estimators : 5, max_depth : 10, learning_rate : 0.1'
# 'train_loss: 1.07 ± 1.35 cm,    test_loss: 1.41 ± 1.34 cm / n_estimators : 10, max_depth : 20, learning_rate : 0.1'
# train_df = pd.DataFrame(data= train_org_df, columns= ['gender','age','身高', '體重','parents_h','after1yearH'])
# val_df = pd.DataFrame(data= val_org_df, columns= ['gender','age','身高', '體重','parents_h','after1yearH'])




# -----Model 2----- 
# train_df = pd.DataFrame(data= train_org_df, columns= ['gender','age','身高', '體重','GH','GnRHa','parents_h','after1yearH'])
# val_df = pd.DataFrame(data= val_org_df, columns= ['gender','age','身高', '體重','GH','GnRHa','parents_h','after1yearH'])



# -----Model 3-----    
# train_df = pd.DataFrame(data= train_org_df, columns= ['gender','age','身高', '體重','GH','GnRHa','parents_h','bmi','after1yearH'])
# val_df = pd.DataFrame(data= val_org_df, columns= ['gender','age','身高', '體重','GH','GnRHa','parents_h','bmi','after1yearH'])



# -----Model 4-----    
# 'train_loss: 0.78 ± 1.11 cm,    test_loss: 0.79 ± 0.73 cm / n_estimators : 5, max_depth : 10, learning_rate : 0.12'
# 'train_loss: 0.91 ± 1.09 cm,    test_loss: 1.0 ± 0.92 cm / n_estimators : 5, max_depth : 25, learning_rate : 0.1'
# 'train_loss: 0.95 ± 1.31 cm,    test_loss: 1.33 ± 1.26 cm / n_estimators : 10, max_depth : 25, learning_rate : 0.1'

# train_df = pd.DataFrame(data= train_org_df, columns= ['gender','age','pred_boneage','身高', '體重','GH','GnRHa','parents_h','after_month','grow'])
# val_df = pd.DataFrame(data= val_org_df, columns= ['gender','age','pred_boneage','身高', '體重','GH','GnRHa','parents_h','after_month','grow'])


# 'train_loss: 1.78 ± 1.92 cm,    test_loss: 2.74 ± 2.69 cm / n_estimators : 30, max_depth : 5, learning_rate : 0.1']

# train_df = pd.DataFrame(data= train_org_df, columns= ['gender','age','pred_boneage','身高', '體重','GH','GnRHa','parents_h','after_month','grow'])
# val_df = pd.DataFrame(data= val_org_df, columns= ['gender','age','pred_boneage','身高', '體重','GH','GnRHa','parents_h','after_month','grow'])
#%%










# train_df = pd.DataFrame(data= train_org_df, columns= ['gender','pred_boneage','身高','GH','GnRHa','parents_h','after_month','grow'])
# val_df = pd.DataFrame(data= val_org_df, columns= ['gender','pred_boneage','身高','GH','GnRHa','parents_h','after_month','grow'])






# ----
# 'train_loss: 0.72 ± 0.76 cm,    test_loss: 0.78 ± 0.68 cm / n_estimators : 5, max_depth : 25, learning_rate : 0.12'
# 'train_loss: 0.94 ± 0.94 cm,    test_loss: 1.0 ± 0.91 cm / n_estimators : 5, max_depth : 25, learning_rate : 0.08'


 # 'train_loss: 0.93 ± 1.25 cm,    test_loss: 1.35 ± 1.25 cm / n_estimators : 25, max_depth : 15, learning_rate : 0.08'
 
 
 # 'train_loss: 0.91 ± 1.47 cm,    test_loss: 2.39 ± 2.25 cm / n_estimators : 15, max_depth : 25, learning_rate : 0.14'
 # 'train_loss: 1.43 ± 1.75 cm,    test_loss: 2.38 ± 2.21 cm / n_estimators : 10, max_depth : 15, learning_rate : 0.1'
 # 'train_loss: 1.4 ± 1.69 cm,    test_loss: 2.37 ± 2.14 cm / n_estimators : 10, max_depth : 13, learning_rate : 0.12'
 # 'train_loss: 1.82 ± 1.83 cm,    test_loss: 2.34 ± 2.25 cm / n_estimators : 8, max_depth : 6, learning_rate : 0.14'
train_df = pd.DataFrame(data= train_org_df, columns= ['gender','pred_boneage','身高','GH','GnRHa','parents_h','after_month','grow'])
val_df = pd.DataFrame(data= val_org_df, columns= ['gender','pred_boneage','身高','GH','GnRHa','parents_h','after_month','grow'])
# train_df = pd.DataFrame(data= train_org_df, columns= ['gender','boneage','身高','GH','GnRHa','parents_h','after_month','grow'])
# val_df = pd.DataFrame(data= val_org_df, columns= ['gender','boneage','身高','GH','GnRHa','parents_h','after_month','grow'])

# train_df = pd.DataFrame(data= train_org_df, columns= ['gender','pred_boneage','身高','parents_h','after_month','grow'])
# val_df = pd.DataFrame(data= val_org_df, columns= ['gender','pred_boneage','身高','parents_h','after_month','grow'])

# doctor BAA
# 'train_loss: 0.82 ± 0.76 cm,    test_loss: 0.85 ± 0.75 cm / n_estimators : 28, max_depth : 5, learning_rate : 0.08'
# 'train_loss: 0.96 ± 0.99 cm,    test_loss: 1.06 ± 0.94 cm / n_estimators : 16, max_depth : 12, learning_rate : 0.08'
# 'train_loss: 1.27 ± 1.34 cm,    test_loss: 1.38 ± 1.28 cm / n_estimators : 8, max_depth : 24, learning_rate : 0.08'
# 'train_loss: 1.74 ± 2.06 cm,    test_loss: 2.5 ± 2.21 cm / n_estimators : 6, max_depth : 18, learning_rate : 0.08'
 




 # ----
 # 'train_loss: 0.77 ± 0.77 cm,    test_loss: 0.83 ± 0.74 cm / n_estimators : 16, max_depth : 8, learning_rate : 0.08'
 # 'train_loss: 0.82 ± 0.99 cm,    test_loss: 1.03 ± 0.94 cm / n_estimators : 16, max_depth : 18, learning_rate : 0.08
 # 'train_loss: 1.0 ± 1.28 cm,    test_loss: 1.34 ± 1.26 cm / n_estimators : 14, max_depth : 18, learning_rate : 0.1'
 # 'train_loss: 1.52 ± 1.67 cm,    test_loss: 2.53 ± 2.52 cm / n_estimators : 30, max_depth : 5, learning_rate : 0.14
# train_df = pd.DataFrame(data= train_org_df, columns= ['gender','age','pred_boneage','身高','GH','GnRHa','parents_h','after_month','grow'])
# val_df = pd.DataFrame(data= val_org_df, columns= ['gender','age','pred_boneage','身高','GH','GnRHa','parents_h','after_month','grow'])


 
 # ----
 # 'train_loss: 0.72 ± 0.78 cm,    test_loss: 0.84 ± 0.74 cm / n_estimators : 30, max_depth : 8, learning_rate : 0.08'
 # 'train_loss: 0.66 ± 0.96 cm,    test_loss: 1.03 ± 0.95 cm / n_estimators : 18, max_depth : 30, learning_rate : 0.08'
 # 24months
 # 'train_loss: 0.95 ± 1.27 cm,    test_loss: 1.33 ± 1.26 cm / n_estimators : 16, max_depth : 16, learning_rate : 0.1'
 # 'train_loss: 1.93 ± 2.17 cm,    test_loss: 2.41 ± 2.48 cm / n_estimators : 6, max_depth : 6, learning_rate : 0.14'
# train_df = pd.DataFrame(data= train_org_df, columns= ['gender','age','pred_boneage','身高', '體重','GH','GnRHa','parents_h','after_month','grow'])
# val_df = pd.DataFrame(data= val_org_df, columns= ['gender','age','pred_boneage','身高', '體重','GH','GnRHa','parents_h','after_month','grow'])











# -----Model 5----- 
# 'train_loss: 0.77 ± 1.11 cm,    test_loss: 0.79 ± 0.73 cm / n_estimators : 5, max_depth : 20, learning_rate : 0.1'
# 'train_loss: 0.99 ± 1.09 cm,    test_loss: 1.02 ± 0.95 cm / n_estimators : 5, max_depth : 10, learning_rate : 0.1'
# 'train_loss: 1.07 ± 1.35 cm,    test_loss: 1.41 ± 1.34 cm / n_estimators : 10, max_depth : 20, learning_rate : 0.1'

# train_df = pd.DataFrame(data= train_org_df, columns= ['gender','boneage','身高', '體重','GH','GnRHa','parents_h','after_month','grow'])
# val_df = pd.DataFrame(data= val_org_df, columns= ['gender','boneage','身高', '體重','GH','GnRHa','parents_h','after_month','grow'])

#%%

import statistics
def normalize_values(data,n): #data = X_train
    # 提取父母身高的欄位（假設該欄位在索引位置 0）
    parents_height = [row[n] for row in data]
    
    # 計算父母身高均值
    mean_height = statistics.mean(parents_height)
    
    # 計算父母身高的標準差
    std_deviation = statistics.stdev(parents_height)
    
    # 將每個值轉換為以正負標準差的方式呈現
    for row in data:
        row[n] = (row[n] - mean_height) / std_deviation
    
    
    
    return data






#%%
# train = train_org_df.drop(labels=['after1yearH'],axis=1).values
# test = val_org_df.drop(labels=['after1yearH'],axis=1).values


# X_train = train_df.drop(labels=['after1yearH'],axis=1).values
# y_train = train_df['after1yearH'].values

# X_test = val_df.drop(labels=['after1yearH'],axis=1).values
# y_test = val_df['after1yearH'].values

# X = np.concatenate([X_train, X_test])

train = train_org_df.drop(labels=['grow'],axis=1).values
test = val_org_df.drop(labels=['grow'],axis=1).values


X_train = train_df.drop(labels=['grow'],axis=1).values
y_train = train_df['grow'].values

X_test = val_df.drop(labels=['grow'],axis=1).values
y_test = val_df['grow'].values

# X = np.concatenate([X_train, X_test])


# for i in range(1,8):
#     if i == 4 or i == 5:
#         continue
#     X = normalize_values(X,i)






# X = preprocessing.scale(X)

# X_train = X[0:len(X_train)]
# X_test = X[len(X_train):]



# X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.3, random_state=42)

print('train shape:', X_train.shape)
print('test shape:', X_test.shape)

import xgboost as xgb
from xgboost import XGBClassifier



xgb.set_config(verbosity=0)
# from sklearn.preprocessing import LabelEncoder
# le = LabelEncoder()
# y_train = le.fit_transform(y_train)

# param['gpu_id'] = 0
# param['max_bin'] = 16
# param['tree_method'] = 'gpu_hist'

# 建立 XGBClassifier 模型
# xgboostModel = XGBClassifier(n_estimators=1000, learning_rate= 0.01, max_depth= 20,tree_method = 'gpu_hist', gpu_id=0)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()




def train_xgboost(X_train, y_train, X_test, y_test, n, depth, lr):
    
    
    xgboostModel = XGBClassifier(n_estimators=n, learning_rate= lr, max_depth= depth)
    # xgboostModel = XGBClassifier(n_estimators=n, learning_rate= lr, max_depth= depth)
    # y_train_encoder = le.fit_transform(y_train)
    
    xgboostModel.fit(X_train, y_train)
    
    # xgboostModel.fit(X_train, y_train)
    

    train_pred = xgboostModel.predict(X_train)
    test_pred = xgboostModel.predict(X_test)
    
    # train_pred = le.inverse_transform(train_pred)
    # test_pred = le.inverse_transform(test_pred)
    
    train_err = abs(train_pred-y_train)
    test_err = abs(test_pred-y_test)
    
    train_std = np.std(train_err)
    test_std = np.std(test_err)
    
    
    
    
    # print('訓練集: ',mae(y_train, train_pred))
    # print('測試集: ',mae(y_test, test_pred))



    # arr=[]
    '''
    for i in range(len(y_train)): #i=0
        
        err = abs(y_train[i]-train_pred[i])
        if ( err > 15):
            # print(str(err), " : ", train_pred[i])
            arr.append("train : " + str(y_train[i]) + "  pred : " + str(train_pred[i]) + "----" +str(err)+ " : "+ str(train[i]) )            
    '''  
    
    '''
    for i in range(len(y_test)): #i=0
        if test[i][12] < month:
            err = abs(y_test[i]-test_pred[i])
            if ( err > 10):
                # print(str(err), " : ", train_pred[i])
                arr.append("train : " + str(y_test[i]) + "  pred : " + str(test_pred[i]) + "----" +str(err)+ " : "+ str(test[i]) )            
    '''
    
    # return mae(y_train, train_pred), mae(y_test, test_pred), train_std, test_std

    return y_train, train_pred, y_test, test_pred


# val_org_df['yearold'] = [(float(i) + float(j) ) for i,j in zip(val_org_df['age'],val_org_df['after_month'])]

            # train_loss4, train_std4 = cal_nFAH(y_train, train_pred, train)
            # test_loss4, test_std4 = cal_nFAH(y_test, test_pred, test)
            
            
def cal_nFAH(y_test, test_pred, test):

    real_test=[]
    real_pred=[]

    # df=pd.DataFrame(columns= [''])
    # val_org_df.reset_index(drop=True, inplace = True)
    
    
    # for i in range(len(val_org_df)): #i=0
    #     if val_org_df['after_month'][i] < month:
    #         real_test.append(y_test[i])
    #         real_pred.append(test_pred[i])
    #         df = df.append(val_org_df.iloc[i],ignore_index=True)
    
    
    val_org_df['yearold'] = [(float(i) + float(j) ) for i,j in zip(val_org_df['age'],val_org_df['after_month'])]

    
    df = pd.DataFrame(columns=[])
    # tt=[]
    
    val_org_df.reset_index(drop=True, inplace = True)
    
    
    for i in range(len(val_org_df)): #i=0
        # if val_org_df['gender'][i] == 1 and val_org_df['yearold'][i] > 191.0:
        if val_org_df['gender'][i] == 1 and val_org_df['yearold'][i] > 191.0 and val_org_df['age'][i] > 0.0 and val_org_df['age'][i] < 109.0:
            real_test.append(y_test[i])
            real_pred.append(test_pred[i])
            df = df.append(val_org_df.iloc[i],ignore_index=True)
        # elif val_org_df['gender'][i] == 0 and val_org_df['yearold'][i] > 167.0:
        elif val_org_df['gender'][i] == 0 and val_org_df['yearold'][i] > 167.0 and val_org_df['age'][i] > 0.0 and val_org_df['age'][i] < 109.0:
            real_test.append(y_test[i])
            real_pred.append(test_pred[i])
            df = df.append(val_org_df.iloc[i],ignore_index=True)
            
            
    # bonexpert_pred = np.array(df['bonexpert']-df['身高'])
    
    # er=abs(np.array(real_test)-np.array(bonexpert_pred))
    # bonexpert_std = np.std(er)
    # mae(real_test,bonexpert_pred)
    
    test_err = abs(np.array(real_test)-np.array(real_pred))
    test_std = np.std(test_err)
    
    '''
    
    
    
    
    
    '''
    
    
    '''
                    our         bonexpert       doctor
    5~9         3.22 ± 2.16     3.32 ± 2.10     2.50 ± 1.66            50
    9~12        3.07 ± 2.47     2.92 ± 2.60     3.45 ± 2.56            445
    12~15       2.07 ± 2.14     3.91 ± 3.96     2.12 ± 1.94            666
    15~18       1.47 ± 1.54     4.12 ± 2.91     1.77 ± 1.59            227
    avg         2.33 ± 2.21     3.60 ± 3.86     2.50 ± 2.21            1388
    
    '''
    
    
    
    # for i in range(len(test_err)): #i=0
        
        
    #     if test_err[i] > 10.0:
    
    #         print( 'err : ' + str(test_err[i]) + ' /  ' + str(df['身高'][i] + real_pred[i]))
    #         print(df.iloc[i])
    #         print('-'*20)
 
    
    return  mae(real_test, real_pred), test_std
    

#%%

            
def cal_train_nFAH(y_test, test_pred, test):

    real_test=[]
    real_pred=[]

    # df=pd.DataFrame(columns= [''])
    # val_org_df.reset_index(drop=True, inplace = True)
    
    
    # for i in range(len(val_org_df)): #i=0
    #     if val_org_df['after_month'][i] < month:
    #         real_test.append(y_test[i])
    #         real_pred.append(test_pred[i])
    #         df = df.append(val_org_df.iloc[i],ignore_index=True)
    
    
    train_org_df['yearold'] = [(float(i) + float(j) ) for i,j in zip(train_org_df['age'],train_org_df['after_month'])]

    
    df = pd.DataFrame(columns=[])
    # tt=[]
    
    train_org_df.reset_index(drop=True, inplace = True)
    
    
    for i in range(len(train_org_df)): #i=0
    
        if train_org_df['gender'][i] == 1 and train_org_df['yearold'][i] > 191.0:
            real_test.append(y_train[i])
            real_pred.append(train_pred[i])
            # df = df.append(train_org_df.iloc[i],ignore_index=True)
            
        elif train_org_df['gender'][i] == 0 and train_org_df['yearold'][i] > 167.0:
            real_test.append(y_train[i])
            real_pred.append(train_pred[i])
            # df = df.append(train_org_df.iloc[i],ignore_index=True)

    
    # bonexpert_pred = np.array(df['bonexpert']-df['身高'])
    
    # er=abs(np.array(real_test)-np.array(bonexpert_pred))
    # bonexpert_std = np.std(er)
    # mae(real_test,bonexpert_pred)
    
    test_err = abs(np.array(real_test)-np.array(real_pred))
    test_std = np.std(test_err)
    
    # count = 0
    
    # for i in range(len(test_err)): #i=0
        
        
    #     if test_err[i] > 10.0:
    
    #         print( 'err : ' + str(test_err[i]) + ' /  ' + str(df['身高'][i] + real_pred[i]))
    #         print(df.iloc[i])
    #         print('-'*20)
            
    #         count+=1
        
    #     # if count > 20:
    #         # break
    
    return  mae(real_test, real_pred), test_std
    



#%%


# def cal_MAE(month, y_test, test_pred, val_org_df):
    # 6, y_test, test_pred, test
def cal_MAE(month, y_test, test_pred, test):
    real_test=[]
    real_pred=[]
    # df=pd.DataFrame(columns= [''])
    # val_org_df.reset_index(drop=True, inplace = True)
    

    # for i in range(len(val_org_df)): #i=0
    #     if val_org_df['after_month'][i] < month:
    #         real_test.append(y_test[i])
    #         real_pred.append(test_pred[i])
    #         df = df.append(val_org_df.iloc[i],ignore_index=True)
    
    
    
    
    # df = pd.DataFrame(columns=[])
    # tt=[]

    for i in range(len(test)): #i=0
        if test[i][12] < month:
            real_test.append(y_test[i])
            real_pred.append(test_pred[i])
            # df = df.append(val_org_df.iloc[i],ignore_index=True)
            
            
            
            
    test_err = abs(np.array(real_test)-np.array(real_pred))
    test_std = np.std(test_err)
    
    
    
    
    # print( "best_test_loss : ", mae(real_test, real_pred), "    std : ", test_std)  
    
    return  mae(real_test, real_pred), test_std








#%%




# Start Training ->  n_estimators : 180, max_depth : 10, learning_rate : 0.1
# train_loss :  0.010732383407090424     std :  0.19040736486614931
# test_loss :  0.9494895279575014     std :  1.725682716637792
# time = 5


n_estimators = [int(x) for x in np.linspace(start=5, stop=30, num=14)]
max_depth = [int(x) for x in np.linspace(5, 30, num=14)]
learning_rate=[round(float(x),2) for x in np.linspace(start=0.08, stop=0.14, num=4)]


# n_estimators = [int(x) for x in np.linspace(start=220, stop=220, num=1)]
# max_depth = [int(x) for x in np.linspace(10, 10, num=1)]
# learning_rate=[round(float(x),2) for x in np.linspace(start=0.1, stop=0.1, num=1)]





# learning_rate=0.1

best_train_loss = 9999 
best_test_loss = 9999
best_train_loss2 = 9999 
best_test_loss2 = 9999
best_train_loss3 = 9999 
best_test_loss3 = 9999
best_train_loss4 = 9999 
best_test_loss4 = 9999
best_train_std = 0
best_test_std = 0



record = []
record2 = []
record3 = []
record4 =[]
record_str=''

for i in range(len(n_estimators)):
    
    for j in range(len(max_depth)):
        
        for k in range(len(learning_rate)):
    
            print( "Start Training -> ",  "n_estimators : " + str(n_estimators[i]) + ", max_depth : " + str(max_depth[j]) + ", learning_rate : " + str(learning_rate[k]))
            # train_loss, test_loss, train_std, test_std = train_xgboost(X_train, y_train, X_test, y_test, n_estimators[i], max_depth[j], learning_rate[k])
            y_train, train_pred, y_test, test_pred = train_xgboost(X_train, y_train, X_test, y_test, n_estimators[i], max_depth[j], learning_rate[k])


            # 6 months

            train_loss, train_std = cal_MAE(7, y_train, train_pred, train)
            test_loss, test_std = cal_MAE(7, y_test, test_pred, test)
            record_str =  "train_loss: "+ str(round(train_loss,2))+ " ± "+ str(round(train_std,2)) + ' cm,    test_loss: '+ str(round(test_loss,2))+ " ± "+ str(round(test_std,2)) + ' cm'
            print('-'*10 +'6 months'+'-'*10)
            print( record_str)

            # 12 months

            train_loss2, train_std2 = cal_MAE(13, y_train, train_pred, train)
            test_loss2, test_std2 = cal_MAE(13, y_test, test_pred, test)
            record_str2 =  "train_loss: "+ str(round(train_loss2,2))+ " ± "+ str(round(train_std2,2)) + ' cm,    test_loss: '+ str(round(test_loss2,2))+ " ± "+ str(round(test_std2,2)) + ' cm'
            print('-'*10 +'12 months'+'-'*10)
            print( record_str2)

            # 24 months

            train_loss3, train_std3 = cal_MAE(25, y_train, train_pred, train)
            test_loss3, test_std3 = cal_MAE(25, y_test, test_pred, test)
            record_str3 =  "train_loss: "+ str(round(train_loss3,2))+ " ± "+ str(round(train_std3,2)) + ' cm,    test_loss: '+ str(round(test_loss3,2))+ " ± "+ str(round(test_std3,2)) + ' cm'
            print('-'*10 +'24 months'+'-'*10)
            print( record_str3)
            
            
            # nFAH

            train_loss4, train_std4 = cal_nFAH(y_train, train_pred, train)
            test_loss4, test_std4 = cal_nFAH(y_test, test_pred, test)
            record_str4 =  "train_loss: "+ str(round(train_loss4,2))+ " ± "+ str(round(train_std4,2)) + ' cm,    test_loss: '+ str(round(test_loss4,2))+ " ± "+ str(round(test_std4,2)) + ' cm'
            print('-'*10 +'nFAH'+'-'*10)
            print( record_str4)     
            
            if test_loss < best_test_loss:
                best_test_loss=test_loss
                record.append( record_str + " / n_estimators : " + str(n_estimators[i]) + ", max_depth : " + str(max_depth[j]) + ", learning_rate : " + str(learning_rate[k]))

            if test_loss2 < best_test_loss2:
                best_test_loss2=test_loss2
                record2.append( record_str2 + " / n_estimators : " + str(n_estimators[i]) + ", max_depth : " + str(max_depth[j]) + ", learning_rate : " + str(learning_rate[k]))

            if test_loss3 < best_test_loss3:
                best_test_loss3=test_loss3
                record3.append( record_str3 + " / n_estimators : " + str(n_estimators[i]) + ", max_depth : " + str(max_depth[j]) + ", learning_rate : " + str(learning_rate[k]))

            if test_loss4 < best_test_loss4:
                best_test_loss4=test_loss4
                record4.append( record_str4 + " / n_estimators : " + str(n_estimators[i]) + ", max_depth : " + str(max_depth[j]) + ", learning_rate : " + str(learning_rate[k]))






# model.save_model("model_sklearn.json")





#%%


df.columns


import matplotlib.pyplot as plt



age_range = list(df['age'] /12.0)
errors = np.array(real_pred) - np.array(real_test)
# errors = np.array(bonexpert_train_pred) - np.array(real_train)


boy_err=[]
boy_age=[]

girl_err=[]
girl_age=[]


for i in range(len(df)): #i=0
    # print(i)
    if df['gender'][i] == 1: # male
        boy_err.append(errors[i])
        boy_age.append(age_range[i])
    else:
        girl_err.append(errors[i])
        girl_age.append(age_range[i])

errors.min()


plt.figure(figsize=(10, 6))  # 設定圖片尺寸

# 設定圖表標題和軸標籤
plt.title("Height Prediction Error", fontsize=16)

# 繪製散點圖
plt.scatter(boy_age, boy_err, alpha = 0.2, color = 'blue', label='誤差')
plt.scatter(girl_age, girl_err, alpha = 0.2, color = 'red', label='誤差')
# 繪製曲線
# plt.plot(age_range, errors, color='red', linewidth=2, label='誤差曲線')
plt.ylim(-30,30)

plt.xlabel('Age (year)')
plt.ylabel('Height error (cm)')
# 設定刻度標籤的字型大小
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)

# 顯示圖表
plt.show()



#%%

import matplotlib.pyplot as plt


sortdf = df.sort_values(by=['after_month'])

df['real']=0.0
df['pred'] =0.0
df['abs_err'] = abs(df['real']-df['pred'])
# df['abs_err'] = abs(df['bonexpert']-df['身高'])
for i in range( len(df)):
    
    df['real'][i] = real_test[i]
    df['pred'][i] = real_pred[i]
    


sortdf.reset_index(drop=True, inplace = True)
sortdf['after_month']

avgdf = pd.DataFrame(columns=['avg_after_month','avg_err'])
data = {
  "avg_after_month" :0.0,
  "avg_err" : 0.0
}



for curr in range( int(sortdf['after_month'].max())): #curr=0
    
    avg = 0
    amount =0
    for i in range( len(sortdf) ): #i=4
    
        if curr == sortdf['after_month'][i]: #and sortdf['gender'][i] == 1:
            avg += sortdf['abs_err'][i]
            amount+=1
    
    if amount > 0:
        data['avg_after_month']=curr
        data['avg_err']=avg/amount
        
        avgdf = avgdf.append(data,ignore_index=True)
            
        

female_avgdf = avgdf
male_avgdf = avgdf


age_range = list(df['after_month'] /12.0)
errors = np.array(real_pred) - np.array(real_test)
# errors = np.array(bonexpert_train_pred) - np.array(real_train)


boy_err=[]
boy_age=[]

girl_err=[]
girl_age=[]



val_org_df.reset_index(drop=True, inplace = True)

for i in range(len(df)): #i=0
    # print(i)
    if df['gender'][i] == 1: # male
        boy_err.append(errors[i])
        boy_age.append(age_range[i])
    else:
        girl_err.append(errors[i])
        girl_age.append(age_range[i])

errors.min()


plt.figure(figsize=(10, 6))  # 設定圖片尺寸

# 設定圖表標題和軸標籤
plt.title("Height Prediction Error", fontsize=16)

# 繪製散點圖
plt.scatter(boy_age, boy_err, alpha = 0.2, color = 'blue', label='誤差')
plt.scatter(girl_age, girl_err, alpha = 0.2, color = 'red', label='誤差')
# 繪製曲線
# plt.plot(age_range, errors, color='red', linewidth=2, label='誤差曲線')
plt.ylim(-30,30)

plt.xlabel('Age (year)')
plt.ylabel('Height error (cm)')
# 設定刻度標籤的字型大小
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)

# 顯示圖表
plt.show()


# import matplotlib相關套件

import matplotlib.pyplot as plt

# import字型管理套件

from matplotlib.font_manager import FontProperties

 


female_age = female_avgdf['avg_after_month']



female_avg_err = female_avgdf['avg_err']



male_age = male_avgdf['avg_after_month']


male_avg_err = male_avgdf['avg_err']


age = avgdf['avg_after_month']
avg_err = avgdf['avg_err']

# 設定圖片大小為長15、寬10

plt.figure(figsize=(15,10),dpi=100,linewidth = 2)

# 把資料放進來並指定對應的X軸、Y軸的資料，用方形做標記(s-)，並指定線條顏色為紅色，使用label標記線條含意

# plt.plot(female_age,female_avg_err,'s-',color = 'r', label="female")

# 把資料放進來並指定對應的X軸、Y軸的資料 用圓形做標記(o-)，並指定線條顏色為綠色、使用label標記線條含意
plt.plot(age,avg_err,'s-',color = 'b')
# plt.plot(month,stock_foxconnn,'o-',color = 'g', label="FOXCONN")

plt.ylim(0,12)

# 設定圖片標題，以及指定字型設定，x代表與圖案最左側的距離，y代表與圖片的距離

# 设置刻度字体大小

plt.xticks(fontsize=20)

plt.yticks(fontsize=20)

# 標示x軸(labelpad代表與圖片的距離)

plt.xlabel("months", fontsize=30, labelpad = 15)

# 標示y軸(labelpad代表與圖片的距離)

plt.ylabel("Absolute Mean Height Error", fontsize=30, labelpad = 20)

# 顯示出線條標記位置

plt.legend(loc = "best", fontsize=20)

# 畫出圖片

plt.show()

avgdf.to_csv("bonemethod.csv", encoding='utf_8_sig', index = False)

#%%



xgboostModel = XGBClassifier(n_estimators=200, learning_rate= 0.1, max_depth= 26)

xgboostModel.fit(X_train, y_train)

train_pred = xgboostModel.predict(X_train)
test_pred = xgboostModel.predict(X_test)
print('訓練集: ',mae(y_train, train_pred))
print('測試集: ',mae(y_test, test_pred))

arr=[]
err = 0.0
for i in range(len(train_pred)): #i=0
    mae = abs(float(y_train[i])-float(train_pred[i]))
    if mae > 7:
        print(mae)
        arr.append(str(X_train[i]) + "mae:" + str(mae))
    
    err+=mae

avg_err = err/float(len(y_train))


arr=[]
err = 0.0
for i in range(len(test_pred)): #i=0
    mae = abs(float(y_test[i])-float(test_pred[i]))
    if mae > 29:
        print(mae)
        arr.append(str(X_test[i]) + "mae:" + str(mae))
    
    err+=mae

avg_err = err/float(len(y_test))

#%%

from xgboost import plot_importance
from xgboost import plot_tree
from matplotlib import pyplot 


plot_importance(xgboostModel)
print('特徵重要程度: ',xgboostModel.feature_importances_)

# 特徵重要程度:  [0.08360689 0.08984546 0.10506732 0.04025269 0.2645768  0.05280426  0.36384663]

pyplot.bar(['gender','pred_boneage','height', 'weight','GH','GnRHa','parents_h','after_month'], xgboostModel.feature_importances_) 
pyplot.show() 


xgboostModel.save_model("6-months Predict.json")



# 建立測試集的 DataFrme
df_test=pd.DataFrame(X_test, columns= ['gender','age', '體重','growth_hormone','父身高','母身高'])
df_test['身高'] = y_test
pred = xgboostModel.predict(X_test)
df_test['Predict'] = pred


sns.lmplot(x="mature", y="state", hue='height', data=df_test, fit_reg=False, legend=False)
plt.legend(title='target', loc='upper left', labels=['weight', 'mature', 'state'])
plt.show()

#%%

from xgboost import XGBClassifier
import numpy as np
new_xgb = XGBClassifier()

new_xgb.load_model("6-months Predict.json")


# ['gender','pred_boneage','身高','GH','GnRHa','parents_h','after_month']
a = np.array([[0, 99.90618134,122.5, 0.0, 0.0, 172.0, 5.0]])



predict_grow = new_xgb.predict(a)
predict_height = a[0][2] + predict_grow


