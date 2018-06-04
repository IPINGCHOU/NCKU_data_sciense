# -*- coding: utf-8 -*-
"""
Created on Thu Mar 29 13:57:47 2018

@author: kami
"""

# KNN

#%%

#packages import
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from pandas import Series, DataFrame, Index

# data import

data=pd.read_csv("E:/WORKOUT/Statistic/data_sciense_intro/report1/train_game_date.csv", index_col = 0)
test_data=pd.read_csv("E:/WORKOUT/Statistic/data_sciense_intro/report1/test_game_date.csv", index_col = 0)

train = data.drop(columns = ['action_type'])
combine = pd.concat([train, test_data], ignore_index = True)

without_na_combine = combine[pd.notnull(combine['shot_made_flag'])]
with_na_combine = combine[pd.isnull(combine['shot_made_flag'])]
with_na_id = with_na_combine['shot_id'].reset_index()
with_na_id = with_na_id.drop(columns = ['index'])


X_without_na = without_na_combine.drop(columns = ['shot_made_flag','shot_id'])
X_without_na["playoffs"] = X_without_na["playoffs"].astype("category").cat.codes
X_without_na["period"] = X_without_na["period"].astype("category").cat.codes
X_without_na["season"] = X_without_na["season"].astype("category").cat.codes
X_without_na["shot_zone_area"] = X_without_na["shot_zone_area"].astype("category").cat.codes
X_without_na["opponent"] = X_without_na["opponent"].astype("category").cat.codes
Y_without_na = without_na_combine['shot_made_flag']

X_train_without_na, X_test_without_na, Y_train_without_na, Y_test_without_na = train_test_split(X_without_na, Y_without_na,
                                                                                                train_size = 0.8,
                                                                                                random_state = 123,
                                                                                                stratify = Y_without_na)




X_with_na = with_na_combine.drop(columns = ['shot_made_flag','shot_id'])
X_with_na["playoffs"] = X_with_na["playoffs"].astype("category").cat.codes
X_with_na["period"] = X_with_na["period"].astype("category").cat.codes
X_with_na["season"] = X_with_na["season"].astype("category").cat.codes
X_with_na["shot_zone_area"] = X_with_na["shot_zone_area"].astype("category").cat.codes
X_with_na["opponent"] = X_with_na["opponent"].astype("category").cat.codes
Y_with_na = with_na_combine['shot_made_flag']

X_without_na = np.array(X_without_na)
Y_without_na = np.array(Y_without_na)
X_with_na = np.array(X_with_na)
Y_with_na = np.array(Y_with_na)
#%%

# classifier settings
classifier = KNeighborsClassifier(n_neighbors = 200, weights = 'uniform',
                                  algorithm = 'auto')

classifier.fit(X_train_without_na, Y_train_without_na)
#%%
classifier.score(X_test_without_na , Y_test_without_na, sample_weight=None)
#%%
classifier.fit(X_without_na, Y_without_na)
pred = classifier.predict(X_with_na)
pred = pred.tolist()
pred = Series(pred)

#%%
na_fits = pd.concat([with_na_id, pred], axis=1)
na_fits.columns = ['id', 'pred']
na_fits = na_fits.astype('int')



#%%

data=pd.read_csv("E:/WORKOUT/Statistic/data_sciense_intro/report1/train.csv")
test_data=pd.read_csv("E:/WORKOUT/Statistic/data_sciense_intro/report1/test.csv")

for i in range(5000):
    current_id = na_fits['id'][i]
    current_pred = na_fits['pred'][i] 
    
    data_loc = data.loc[data['shot_id'] == current_id].index.tolist()
    test_data_loc = test_data.loc[test_data['shot_id'] == current_id].index.tolist()
    
    if len(data_loc) != 0:
        data['shot_made_flag'][data_loc] = current_pred
    elif len(test_data_loc) != 0:
        test_data['shot_made_flag'][test_data_loc] = current_pred
    else:
        print('something is wrong')
    
    
    if i % 10 == 0:
        now_work = "Current working:" + str(i)
        print(now_work)
    
#%%

data.to_csv("E:/WORKOUT/Statistic/data_sciense_intro/report1/train_KNN.csv", index = False)    
test_data.to_csv("E:/WORKOUT/Statistic/data_sciense_intro/report1/test_KNN.csv", index = False)
    
    
    
    
