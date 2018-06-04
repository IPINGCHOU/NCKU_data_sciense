
# coding: utf-8

# In[1]:


from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np
import datetime as dt

train=pd.read_csv("E:/WORKOUT/Statistic/data_sciense_intro/report1/train.csv")
test=pd.read_csv("E:/WORKOUT/Statistic/data_sciense_intro/report1/test.csv")
X_train = train.drop(columns=['action_type','shot_id'])
X_test = test.drop(columns=['shot_id'])
Y_train= np.array(train["action_type"])

X_train['game_date']=pd.to_datetime(X_train['game_date'])
epoch = dt.datetime(1970, 1, 1)
step = 0
temp =[]
for t in [(d - epoch).total_seconds() for d in X_train['game_date']]:
    temp.append('%.0f' % t)  
X_train.drop('game_date', axis = 1, inplace = True)
X_train['game_date'] = temp
X_train['game_date'] = X_train['game_date'].astype('float32')
mean_game_date= X_train['game_date'].mean()
std_game_date= X_train['game_date'].std()
X_train['game_date'] = (X_train['game_date'] - mean_game_date) / (std_game_date)


X_test['game_date']=pd.to_datetime(X_test['game_date'])
epoch = dt.datetime(1970, 1, 1)
step = 0
temp =[]
for t in [(d - epoch).total_seconds() for d in X_test['game_date']]:
    temp.append('%.0f' % t)  
X_test.drop('game_date', axis = 1, inplace = True)
X_test['game_date'] = temp
X_test['game_date'] = X_test['game_date'].astype('float32')
X_test['game_date'] = (X_test['game_date'] - mean_game_date) / (std_game_date)

data=X_train.append(X_test)
data["playoffs"]=data["playoffs"].astype("category").cat.codes
data["period"]=data["period"].astype("category").cat.codes
data["season"]=data["season"].astype("category").cat.codes
data["shot_made_flag"]=data["shot_made_flag"].astype("category").cat.codes
data["shot_zone_area"]=data["shot_zone_area"].astype("category").cat.codes
data["opponent"]=data["opponent"].astype("category").cat.codes

X_train=data[0:24557]
X_test=data[24557:30698]


# In[38]:

from sklearn.metrics import accuracy_score
rf = RandomForestClassifier(n_estimators = 2000, oob_score = True,
                            random_state = 10, min_samples_split = 80,
                            max_features = 3, min_samples_leaf = 10,
                            bootstrap = True, verbose = 1)
rf.fit(X_train, Y_train)

from sklearn.metrics import accuracy_score
predicted = rf.predict(X_train)
accuracy = accuracy_score(Y_train, predicted)
print('Out-of-bag score estimate: ')
print(rf.oob_score_)
print('Mean accuracy score: ')
print(accuracy)


# In[40]:


final_pred=rf.predict(X_test)
output=pd.DataFrame()
output[0]=list(test['shot_id'])
output[1]=final_pred
output.to_csv('E:/WORKOUT/Statistic/data_sciense_intro/report1/out/out_rf_test.csv', index = False, header = False)

# %% # cv transform

from sklearn.preprocessing import label_binarize
Y_train_trans = label_binarize(Y_train, classes=pd.unique(Y_train))

#%%  
param_grid = {
    'bootstrap': [True],
    'max_features': list(range(5,11,1)),
    'min_samples_leaf': list(range(10,60,10)),
    'min_samples_split': [30, 40 , 50, 60, 70, 80, 90, 100],
    'n_estimators': [100]
}
# Create a based model
rf = RandomForestClassifier()
# Instantiate the grid search model
grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 
                          cv = 10, verbose = 2)

grid_search.fit(X_train, Y_train_trans)
grid_search.grid_scores_,grid_search.best_params_, grid_search.best_score_




#%%
from operator import itemgetter

# Utility function to report best scores
def report(grid_scores, n_top):
    top_scores = sorted(grid_scores, key=itemgetter(1), reverse=True)[:n_top]
    for i, score in enumerate(top_scores):
        print("Model with rank: {0}".format(i + 1))
        print("Mean validation score: {0:.4f})".format(
              score.mean_validation_score,
              np.std(score.cv_validation_scores)))
        print("Parameters: {0}".format(score.parameters))
        print("")

report(grid_search.grid_scores_, 10)

