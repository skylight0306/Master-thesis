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

os.environ['CUDA_VISIBLE_DEVICES'] = "0"


#%%
def mae(groundtruth, pred):  #groundtruth=y_test
    
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


# for i in df_data['病歷號']: #
   
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

# age_df['boneage'] = np.nan

# # df = age_df['boneage']
# # df = df.dropna(axis= 'index', how='any')
# # df.reset_index(drop=True, inplace = True)



# bone_df = pd.read_csv('boneage_height.csv')

# for i in range(len(bone_df)): #i=0
#     print( bone_df['病歷號碼'][i])
    
#     for j in range(len(age_df)): #j=2658
#         # print(age_df['病歷號'][j])
#         if bone_df['病歷號碼'][i] == age_df['病歷號'][j] :
            
            
#             bone_day = datetime.strptime(bone_df['檢查結束時間'][i], '%Y/%m/%d %H:%M')
#             height_day = datetime.strptime(age_df['門診日'][j], '%Y/%m/%d')
#             error = abs(bone_day-height_day).days
            
#             if error < 5:
#                 age_df['boneage'][j] = bone_df['boneage'][i]
#                 print('found it !')
            
#             # born = date(int(data['生日'].split('-',-1)[0]), int(data['生日'].split('-',-1)[1]), int(data['生日'].split('-',-1)[2]))







# age_df.to_csv("height_boneage.csv", encoding='utf_8_sig', index = False)

#%%

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



#%%

# age_df = pd.read_csv('2014-2022清單_給學生.csv')

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
df.to_csv("病例.csv",encoding='utf_8_sig', index = False)
'''
#%%


# df_data = pd.DataFrame(data= age_df,
#                      columns= ['gender','age','身高', '體重','growth_hormone','父身高','母身高','boneage'])



# old_df = pd.read_csv('height.csv')

age_df = pd.read_csv('total_height.csv')

df_data = pd.DataFrame(data= age_df,
                      columns= ['gender','age','身高', '體重','growth_hormone','父身高','母身高'])


# age_df['身高'].max()
# age_df['身高'].min()
# age_df[age_df['身高']>200]['身高']
# a = age_df[age_df['身高']<50]
# k=a['身高']

# for i in k: #i=0
#     print(i)
# age_df['身高'].min()


df_data['height_category'] = pd.cut(df_data['身高'], 10)



from sklearn.model_selection import train_test_split


# 5-fold
A_df, B_df = train_test_split(  df_data, 
                                test_size = 0.2, 
                                random_state=seed,
                                stratify = df_data['height_category'])

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


train_df = pd.concat([B_df,C_df,D_df,E_df])
val_df = A_df

'''
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
train_df = pd.DataFrame(data= train_df,
                      columns= ['gender','age','身高', '體重','growth_hormone','父身高','母身高'])

val_df = pd.DataFrame(data= val_df,
                      columns= ['gender','age','身高', '體重','growth_hormone','父身高','母身高'])


# train_df = pd.DataFrame(data= train_df,
#                      columns= ['gender','age','身高', '體重','growth_hormone','父身高','母身高','boneage'])

# val_df = pd.DataFrame(data= val_df,
#                      columns= ['gender','age','身高', '體重','growth_hormone','父身高','母身高','boneage'])



X_train = train_df.drop(labels=['身高'],axis=1).values
y_train = train_df['身高'].values

X_test = val_df.drop(labels=['身高'],axis=1).values
y_test = val_df['身高'].values

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
    
    
    xgboostModel = XGBClassifier(n_estimators=n, learning_rate= lr, max_depth= depth, tree_method="gpu_hist",gpu_id='0')
    # xgboostModel = XGBClassifier(n_estimators=n, learning_rate= lr, max_depth= depth)
    y_train_encoder = le.fit_transform(y_train)
    
    xgboostModel.fit(X_train, y_train_encoder)
    
    
    

    train_pred = xgboostModel.predict(X_train)
    test_pred = xgboostModel.predict(X_test)
    
    train_pred = le.inverse_transform(train_pred)
    test_pred = le.inverse_transform(test_pred)
    # print('訓練集: ',mae(y_train, train_pred))
    # print('測試集: ',mae(y_test, test_pred))
    # arr=[]
    
    # for i in range(len(y_test)): #i=0
        
    #     err = abs(y_test[i]-test_pred[i])
    #     if ( err > 10):
    #         print(str(err), " : ", X_test[i])
    #         arr.append(str(y_test[i]) + "----" +str(err)+ " : "+ str(X_test[i]))
    arr=[]
    
    for i in range(len(y_train)): #i=0
        
        err = abs(y_train[i]-train_pred[i])
        if ( err > 10):
            # print(str(err), " : ", train_pred[i])
            arr.append("train : " + str(y_train[i]) + "  pred : " + str(train_pred[i]) + "----" +str(err)+ " : "+ str(X_train[i]))            
        # print(err)
    return mae(y_train, train_pred), mae(y_test, test_pred)


# time = 5
n_estimators = [int(x) for x in np.linspace(start=200, stop=500, num=10)]
max_depth = [int(x) for x in np.linspace(20, 100, num=10)]
learning_rate=[round(float(x),2) for x in np.linspace(start=0.12, stop=0.12, num=1)]
# learning_rate=0.1

best_train_loss = 9999 
best_test_loss = 9999
record = []

# Start Training -> n_estimators : 325, max_depth : 20, learning_rate : 0.1 train_loss : 0.9358185931715366 test_loss : 2.477616306954439
# best_train_loss : 0.1420793911970382, best_test_loss : 2.450326139088731, n_estimators : 200, max_depth : 26, learning_rate : 0.1
# 'best_train_loss : 0.13111362700699886, best_test_loss : 2.4044983197311574, n_estimators : 277, max_depth : 26, learning_rate : 0.14'
 # 'best_train_loss : 0.12923104628128937, best_test_loss : 2.278443804034583, n_estimators : 300, max_depth : 20, learning_rate : 0.1'
 # 'best_train_loss : 0.1322204213938411, best_test_loss : 2.2442939481268023, n_estimators : 266, max_depth : 10, learning_rate : 0.12'
for i in range(len(n_estimators)):
    
    for j in range(len(max_depth)):
        
        for k in range(len(learning_rate)):
    
            print( "Start Training -> ",  "n_estimators : " + str(n_estimators[i]) + ", max_depth : " + str(max_depth[j]) + ", learning_rate : " + str(learning_rate[k]))
            train_loss, test_loss = train_xgboost(X_train, y_train, X_test, y_test, n_estimators[i], max_depth[j], learning_rate[k])
            
            print( "train_loss : ", train_loss)
            print( "test_loss : ", test_loss)
            print( "best_train_loss : ", best_train_loss)
            print( "best_test_loss : ", best_test_loss)         
            if test_loss < best_test_loss:
                best_train_loss=train_loss
                best_test_loss=test_loss
                
                record.append("best_train_loss : " + str(train_loss) + ", best_test_loss : " +  str(test_loss)  + ", n_estimators : " + str(n_estimators[i]) + ", max_depth : " + str(max_depth[j]) + ", learning_rate : " + str(learning_rate[k]))


# model.save_model("model_sklearn.json")


# xgboostModel = XGBClassifier(n_estimators=1000, learning_rate= 0.01, max_depth= 20)

# # # 使用訓練資料訓練模型
# xgboostModel.fit(X_train, y_train)
# # # 使用訓練資料預測分類
# train_pred = xgboostModel.predict(X_train)
# test_pred = xgboostModel.predict(X_test)

# # # 預測成功的比例
# # # print('訓練集: ',xgboostModel.score(X_train,y_train))
# # # print('測試集: ',xgboostModel.score(X_test,y_test))

# print('訓練集: ',mae(y_train, train_pred))
# print('測試集: ',mae(y_test, test_pred))


# train_loss, test_loss = train_xgboost(X_train, y_train, X_test, y_test, 200, 26, 0.1)

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

plot_importance(xgboostModel)
print('特徵重要程度: ',xgboostModel.feature_importances_)








# 建立測試集的 DataFrme
df_test=pd.DataFrame(X_test, columns= ['gender','age', '體重','growth_hormone','父身高','母身高'])
df_test['身高'] = y_test
pred = xgboostModel.predict(X_test)
df_test['Predict'] = pred


sns.lmplot(x="mature", y="state", hue='height', data=df_test, fit_reg=False, legend=False)
plt.legend(title='target', loc='upper left', labels=['weight', 'mature', 'state'])
plt.show()

#%%



# avg_err = mae(y_test, pred)

# len(pred)






