# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 19:20:33 2018

@author: kami
"""


# coding: utf-8

# In[18]:


#import data
import numpy as np
import pandas as pd
import tensorflow as tf
import datetime as dt
from sklearn.utils import shuffle

#  X_train  Y_train

data=pd.read_csv("E:/WORKOUT/Statistic/data_sciense_intro/report1/train.csv")
data = shuffle(data)
Y_train= np.array(data["action_type"])
test_data=pd.read_csv("E:/WORKOUT/Statistic/data_sciense_intro/report1/test.csv")
X_train = data.drop(columns=['action_type','shot_id'])
X_train["playoffs"]=X_train["playoffs"].astype("str")
X_train["period"]=X_train["period"].astype("str")
X_train["season"]=X_train["season"].astype("str")
X_train['shot_made_flag']=X_train['shot_made_flag'].astype("str")
X_train['loc_x'] = X_train['loc_x'].abs()
X_train['loc_y'] = X_train['loc_y'].abs()

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

X_train = pd.get_dummies(X_train, prefix=['period','playoffs', 'season', 'shot_made_flag',
                                          'shot_zone_area', 'opponent'])
X_train= X_train.astype('float32')

X_train =np.array(X_train)


#  X_test

X_test = test_data.drop(columns=['shot_id'])
X_test["playoffs"]=X_test["playoffs"].astype("str")
X_test["period"]=X_test["period"].astype("str")
X_test["season"]=X_test["season"].astype("str")
X_test['shot_made_flag']=X_test['shot_made_flag'].astype("str")
X_test['loc_x'] = X_test['loc_x'].abs()
X_test['loc_y'] = X_test['loc_y'].abs()

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

X_test = pd.get_dummies(X_test, prefix=['period','playoffs', 'season', 'shot_made_flag',
                                          'shot_zone_area', 'opponent'])
X_test.insert(loc=13, column='period_7', value=0)

X_test= X_test.astype('float32')

X_test =np.array(X_test)



''' Convert to one-hot encoding '''
from keras.utils import np_utils
Y_name ,Y_train_transform =np.unique(Y_train, return_inverse=True)
Y_train_transform = Y_train_transform.astype('int')
Y_train_transform = np_utils.to_categorical(Y_train_transform,57)




# In[19]:


#import packages
from keras.models import Sequential, Model, load_model
from keras.layers.core import Dense, Activation, Dropout
from keras.optimizers import Adam, RMSprop, Adagrad , adadelta, Nadam
from keras.regularizers import l1,l2
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers.core import Dropout
from keras.callbacks import Callback
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import Input, Concatenate
from keras.models import model_from_json

#Self-defined Callbacks
class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.loss = []
        self.acc = []
        self.val_loss = []
        self.val_acc = []
    def on_epoch_end(self, batch, logs={}):
        self.loss.append(logs.get('loss'))
        self.val_loss.append(logs.get('val_loss'))
        self.acc.append(logs.get('acc'))
        self.val_acc.append(logs.get('val_acc'))
        
loss_history = LossHistory()

#early stopping
earlyStopping=EarlyStopping(monitor= 'loss', patience= 3)

#model check point
checkpoint = ModelCheckpoint('E:/WORKOUT/Statistic/data_sciense_intro/report1/DNN_model/API_merge_sbs_bsb_KNN.h5',
                             monitor='val_acc', verbose = 1, save_best_only = True, mode = 'max')


# In[23]:

#kernel_regularizer=l2(0.001)

#DNN model
#left model
input_1 = Input(shape = (79,))
left = Dense(4096, )(input_1)
left = LeakyReLU(0.02)(left)
left = Dense(2048, )(left)
left = LeakyReLU(0.02)(left)
left = Dense(1024, )(left)
left = LeakyReLU(0.02)(left)
left = Dense(512, )(left)
left = LeakyReLU(0.02)(left)
left = Dense(256, )(left)
left = LeakyReLU(0.02)(left)
left = Dense(128, kernel_regularizer=l2(0.001))(left)
left = LeakyReLU(0.02)(left)
left = Dense(256, )(left)
left = LeakyReLU(0.02)(left)
left = Dense(512, )(left)
left = LeakyReLU(0.02)(left)
left = Dense(1024, )(left)
left = LeakyReLU(0.02)(left)
left = Dense(2048, )(left)
left = LeakyReLU(0.02)(left)
left = Dense(4096, )(left)
left = LeakyReLU(0.02)(left)
left = Dropout(0.4)(left)

#right model
input_2 = Input(shape = (79,))
right = Dense(4096, )(input_2)
right = LeakyReLU(0.02)(right)
right = Dense(2048, )(right)
right = LeakyReLU(0.02)(right)
right = Dense(1024, )(right)
right = LeakyReLU(0.02)(right)
right = Dense(512, )(right)
right = LeakyReLU(0.02)(right)
right = Dense(256, )(right)
right = LeakyReLU(0.02)(right)
right = Dense(128, kernel_regularizer=l2(0.001))(right)
right = LeakyReLU(0.02)(right)
right = Dense(256, )(right)
right = LeakyReLU(0.02)(right)
right = Dense(512, )(right)
right = LeakyReLU(0.02)(right)
right = Dense(1024, )(right)
right = LeakyReLU(0.02)(right)
right = Dense(2048, )(right)
right = LeakyReLU(0.02)(right)
right = Dense(4096, )(right)
right = LeakyReLU(0.02)(right)
right = Dropout(0.4)(right)

# bsbmodel - merge left and right
bsbmodel_merge = Concatenate()([left, right])
bsbmodel = Dense(4096, )(bsbmodel_merge)
bsbmodel = LeakyReLU(0.02)(bsbmodel)
bsbmodel = Dense(2048, )(bsbmodel)
bsbmodel = LeakyReLU(0.02)(bsbmodel)
bsbmodel = Dense(1024, )(bsbmodel)
bsbmodel = LeakyReLU(0.02)(bsbmodel)
bsbmodel = Dense(512, )(bsbmodel)
bsbmodel = LeakyReLU(0.02)(bsbmodel)
bsbmodel = Dense(256, )(bsbmodel)
bsbmodel = LeakyReLU(0.02)(bsbmodel)
bsbmodel = Dropout(0.4)(bsbmodel)

#DNN sbs model
input_3 = Input(shape = (79,))
sbsmodel = Dense(128, )(input_3)
sbsmodel = LeakyReLU(0.02)(sbsmodel)
sbsmodel = Dense(128, kernel_regularizer=l2(0.001))(sbsmodel)
sbsmodel = LeakyReLU(0.02)(sbsmodel)
sbsmodel = Dense(256, )(sbsmodel)
sbsmodel = LeakyReLU(0.02)(sbsmodel)
sbsmodel = Dense(256, kernel_regularizer=l2(0.001))(sbsmodel)
sbsmodel = LeakyReLU(0.02)(sbsmodel)

sbsmodel = Dense(512, )(sbsmodel)
sbsmodel = LeakyReLU(0.02)(sbsmodel)
sbsmodel = Dense(512, kernel_regularizer=l2(0.001))(sbsmodel)
sbsmodel = LeakyReLU(0.02)(sbsmodel)
sbsmodel = Dense(1024, )(sbsmodel)
sbsmodel = LeakyReLU(0.02)(sbsmodel)
sbsmodel = Dense(1024, kernel_regularizer=l2(0.001))(sbsmodel)
sbsmodel = LeakyReLU(0.02)(sbsmodel)

sbsmodel = Dense(2048, )(sbsmodel)
sbsmodel = LeakyReLU(0.02)(sbsmodel)
sbsmodel = Dense(2048, kernel_regularizer=l2(0.001))(sbsmodel)
sbsmodel = LeakyReLU(0.02)(sbsmodel)
sbsmodel = Dense(4096, )(sbsmodel)
sbsmodel = LeakyReLU(0.02)(sbsmodel)
sbsmodel = Dropout(0.4)(sbsmodel)
sbsmodel = Dense(4096, kernel_regularizer=l2(0.001))(sbsmodel)
sbsmodel = LeakyReLU(0.02)(sbsmodel)

sbsmodel = Dense(2048, )(sbsmodel)
sbsmodel = LeakyReLU(0.02)(sbsmodel)
sbsmodel = Dense(2048, kernel_regularizer=l2(0.001))(sbsmodel)
sbsmodel = LeakyReLU(0.02)(sbsmodel)
sbsmodel = Dense(1024, )(sbsmodel)
sbsmodel = LeakyReLU(0.02)(sbsmodel)
sbsmodel = Dense(1024, kernel_regularizer=l2(0.001))(sbsmodel)
sbsmodel = LeakyReLU(0.02)(sbsmodel)

sbsmodel = Dense(512, )(sbsmodel)
sbsmodel = LeakyReLU(0.02)(sbsmodel)
sbsmodel = Dense(512, kernel_regularizer=l2(0.001))(sbsmodel)
sbsmodel = LeakyReLU(0.02)(sbsmodel)
sbsmodel = Dense(256, )(sbsmodel)
sbsmodel = LeakyReLU(0.02)(sbsmodel)
sbsmodel = Dense(256, kernel_regularizer=l2(0.001))(sbsmodel)
sbsmodel = LeakyReLU(0.02)(sbsmodel)
sbsmodel = Dropout(0.4)(sbsmodel)

# big bsb model
input_3 = Input(shape = (79,))
bbsbmodel = Dense(2048, )(input_3)
bbsbmodel = LeakyReLU(0.02)(bbsbmodel)
bbsbmodel = Dense(2048, )(bbsbmodel)
bbsbmodel = LeakyReLU(0.02)(bbsbmodel)

bbsbmodel = Dense(1024, )(bbsbmodel)
bbsbmodel = LeakyReLU(0.02)(bbsbmodel)
bbsbmodel = Dense(1024, )(bbsbmodel)
bbsbmodel = LeakyReLU(0.02)(bbsbmodel)
bbsbmodel = Dropout(0.4)(bbsbmodel)
bbsbmodel = Dense(512, )(bbsbmodel)
bbsbmodel = LeakyReLU(0.02)(bbsbmodel)
bbsbmodel = Dense(512, )(bbsbmodel)
bbsbmodel = LeakyReLU(0.02)(bbsbmodel)

bbsbmodel = Dense(256, )(bbsbmodel)
bbsbmodel = LeakyReLU(0.02)(bbsbmodel)
bbsbmodel = Dense(256, )(bbsbmodel)
bbsbmodel = LeakyReLU(0.02)(bbsbmodel)

bbsbmodel = Dense(128, )(bbsbmodel)
bbsbmodel = LeakyReLU(0.02)(bbsbmodel)

bbsbmodel = Dense(256, )(bbsbmodel)
bbsbmodel = LeakyReLU(0.02)(bbsbmodel)
bbsbmodel = Dense(256, )(bbsbmodel)
bbsbmodel = LeakyReLU(0.02)(bbsbmodel)
bbsbmodel = Dense(512, )(bbsbmodel)
bbsbmodel = LeakyReLU(0.02)(bbsbmodel)
bbsbmodel = Dense(512, )(bbsbmodel)
bbsbmodel = LeakyReLU(0.02)(bbsbmodel)

bbsbmodel = Dense(1024, )(bbsbmodel)
bbsbmodel = LeakyReLU(0.02)(bbsbmodel)
bbsbmodel = Dense(1024, )(bbsbmodel)
bbsbmodel = LeakyReLU(0.02)(bbsbmodel)
bbsbmodel = Dropout(0.4)(bbsbmodel)
bbsbmodel = Dense(2048, )(bbsbmodel)
bbsbmodel = LeakyReLU(0.02)(bbsbmodel)
bbsbmodel = Dense(2048, )(bbsbmodel)
bbsbmodel = LeakyReLU(0.02)(bbsbmodel)
bbsbmodel = Dropout(0.4)(bbsbmodel)

# merge model
model_merge = Concatenate()([sbsmodel, bsbmodel, bbsbmodel])
model = Dense(1024, )(bbsbmodel)
model = LeakyReLU(0.02)(model)
model = Dense(512, )(model)
model = LeakyReLU(0.02)(model)
model = Dense(256,  )(model)
model = LeakyReLU(0.02)(model)

predictions = Dense(57, activation = 'softmax')(model)

model_fin = Model(inputs = [input_1, input_2, input_3], outputs = predictions)
#Setting optimizer as Adam
# =============================================================================
# rmsprop = keras.optimizers.RMSprop(lr=0.1, rho=0.9, epsilon=None, decay=0.0)
# nadam = keras.optimizers.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
# =============================================================================

adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model_fin.compile(loss= 'categorical_crossentropy',
                    optimizer=adam,
                    metrics=['accuracy'])

# In[24]:


#model fitting
history_adam = model_fin.fit([X_train, X_train, X_train], Y_train_transform,
                           batch_size = 1024,
                           epochs = 500,
                           verbose = 1,
                           shuffle=True,
                           validation_split=0.0,
                           callbacks=[loss_history,earlyStopping])

# In[18]:

# =============================================================================
# final_model = load_model('E:/WORKOUT/Statistic/data_sciense_intro/report1/DNN_model/API_merge_sbs_bsb_KNN.h5')
# pred_f =  final_model.predict([X_test ,X_test, X_test], verbose=1)
# pred_f = pred_f.argmax(axis=-1)
# 
# 
# final_pred=[]
# for i in range(len(pred_f)):
#     final_pred.append(Y_name[pred_f[i]])
# output=pd.DataFrame()
# output[0]=list(test_data['shot_id'])
# output[1]=final_pred
# output.to_csv('E:/WORKOUT/Statistic/data_sciense_intro/report1/API_merge_sbs_bsb_KNN.csv', index = False, header = False)
# =============================================================================

#%%
pred = model_fin.predict([X_test,X_test,X_test], verbose=1)
pred = pred.argmax(axis=-1)

#create output
final_pred=[]
for i in range(len(pred)):
    final_pred.append(Y_name[pred[i]])
output=pd.DataFrame()
output[0]=list(test_data['shot_id'])
output[1]=final_pred
output.to_csv('E:/WORKOUT/Statistic/data_sciense_intro/report1/API_str3.csv', index = False, header = False)


#%%
# =============================================================================
# loss_adam = history_adam.history.get('loss')
# acc_adam = history_adam.history.get('acc')
# val_loss_adam = history_adam.history.get('val_loss')
# val_acc_adam = history_adam.history.get('val_acc')
# 
# loss = loss_history.loss
# acc  = loss_history.acc
# val_loss = loss_history.val_loss
# val_acc  = loss_history.val_acc
# 
# #''' Visualize the loss and accuracy of both models'''
# import matplotlib.pyplot as plt
# plt.figure(0)
# plt.subplot(121)
# plt.plot(range(len(loss_adam)), loss_adam,label='Training')
# plt.plot(range(len(val_loss_adam)), val_loss_adam,label='Validation')
# plt.title('Loss history returned by fit function')
# plt.legend(loc='upper left')
# plt.subplot(122)
# plt.plot(range(len(loss)), loss,label='Training')
# plt.plot(range(len(val_loss)), val_loss,label='Validation')
# plt.title('Loss history from Callbacks')
# plt.savefig('E:/WORKOUT/Statistic/data_sciense_intro/report1/DNN_model/API_merge_sbs_bsb_loss_patience1.png',dpi=300,format='png')
# plt.close()
# print('Result saved')
# =============================================================================

#%%
# =============================================================================
# plt.figure(1)
# plt.subplot(121)
# plt.plot(range(len(acc_adam)), acc_adam,label='Training')
# plt.plot(range(len(val_acc_adam)), val_acc_adam,label='Validation')
# plt.title('Acc history returned by fit function')
# plt.legend(loc='upper left')
# plt.subplot(122)
# plt.plot(range(len(acc)), loss,label='Training')
# plt.plot(range(len(val_acc)), val_acc,label='Validation')
# plt.title('Acc history from Callbacks')
# plt.savefig('E:/WORKOUT/Statistic/data_sciense_intro/report1/DNN_model/API_merge_sbs_bsb_acc_patience2.png',dpi=300,format='png')
# plt.close()
# print('Result saved')
# =============================================================================
