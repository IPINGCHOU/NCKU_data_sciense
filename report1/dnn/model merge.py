# -*- coding: utf-8 -*-
"""
Created on Sun Mar 25 14:56:43 2018

@author: kami
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Mar 25 13:51:08 2018

@author: kami
"""


# coding: utf-8

# In[18]:


#import data
import numpy as np
import pandas as pd
import tensorflow as tf
import datetime as dt

data=pd.read_csv("E:/WORKOUT/Statistic/data_sciense_intro/report1/train.csv")
Y_train= np.array(data["action_type"])
test_data=pd.read_csv("E:/WORKOUT/Statistic/data_sciense_intro/report1/test.csv")
X_train = data.drop(columns=['action_type','shot_id'])
X_train["playoffs"]=X_train["playoffs"].astype("str")
X_train["period"]=X_train["period"].astype("str")
X_train["season"]=X_train["season"].astype("str")
X_train['shot_made_flag']=X_train['shot_made_flag'].astype("str")

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

X_test = test_data.drop(columns=['shot_id'])
X_test["playoffs"]=X_test["playoffs"].astype("str")
X_test["period"]=X_test["period"].astype("str")
X_test["season"]=X_test["season"].astype("str")
X_test['shot_made_flag']=X_test['shot_made_flag'].astype("str")

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
from keras.utils.vis_utils import plot_model
from keras.layers import Merge
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import Adam, RMSprop, Adagrad ,adadelta, Nadam
from keras.regularizers import l1,l2
from keras.callbacks import EarlyStopping
from keras.layers.core import Dropout
from keras.callbacks import Callback
from keras.layers.advanced_activations import LeakyReLU

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

#early stopping
earlyStopping=EarlyStopping(monitor= 'val_loss', patience= 20)

#leakyRelu
lrelu= LeakyReLU(alpha = 0.02)


# In[23]:


#DNN model
left = Sequential()
left.add(Dense(4096, input_dim=79))
left.add(lrelu)
left.add(Dense(2048))
left.add(lrelu)
left.add(Dense(1024))
left.add(lrelu)
left.add(Dense(512))
left.add(lrelu)
left.add(Dense(256))
left.add(lrelu)
left.add(Dense(128))
left.add(lrelu)
left.add(Dense(256))
left.add(lrelu)
left.add(Dense(512))
left.add(lrelu)
left.add(Dense(1024))
left.add(lrelu)
left.add(Dense(2048))
left.add(lrelu)
left.add(Dense(4096))
left.add(lrelu)
    

right = Sequential()
right.add(Dense(4096, input_dim=79))
right.add(lrelu)
right.add(Dense(2048))
right.add(lrelu)
right.add(Dense(1024))
right.add(lrelu)
right.add(Dense(512))
right.add(lrelu)
right.add(Dense(256))
right.add(lrelu)
right.add(Dense(128))
right.add(lrelu)
right.add(Dense(256))
right.add(lrelu)
right.add(Dense(512))
right.add(lrelu)
right.add(Dense(1024))
right.add(lrelu)
right.add(Dense(2048))
right.add(lrelu)
right.add(Dense(4096))
right.add(lrelu)

bsbmodel = Sequential()  
bsbmodel.add(Merge([left,right], mode='concat')) 
bsbmodel.add(Dense(4096, kernel_regularizer=l2(0.001)))
bsbmodel.add(lrelu)
bsbmodel.add(Dense(2048, kernel_regularizer=l2(0.001)))
bsbmodel.add(lrelu)
bsbmodel.add(Dense(1024, kernel_regularizer=l2(0.001)))
bsbmodel.add(lrelu)
bsbmodel.add(Dense(512, kernel_regularizer=l2(0.001)))
bsbmodel.add(lrelu)
bsbmodel.add(Dense(256, kernel_regularizer=l2(0.001)))
bsbmodel.add(lrelu)
bsbmodel.add(Dense(128, kernel_regularizer=l2(0.001)))
bsbmodel.add(lrelu)

model.add(Dense(57))
model.add(Activation('softmax'))

#Setting optimizer as Adam
# =============================================================================
# rmsprop = keras.optimizers.RMSprop(lr=0.1, rho=0.9, epsilon=None, decay=0.0)
# nadam = keras.optimizers.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
# =============================================================================

adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=True)
model.compile(loss= 'categorical_crossentropy',
                    optimizer=adam,
                    metrics=['accuracy'])
loss_history = LossHistory()


# In[24]:


#model fitting
history = model.fit([X_train, X_train], Y_train_transform,
                        batch_size=1024,
                        epochs=100,
                        verbose=1,
                        shuffle=True,
                        validation_split=0.1,
                        callbacks=[loss_history])
pred = model.predict_classes([X_test ,X_test], verbose=0)


# In[18]:


#create output
final_pred=[]
for i in range(len(pred)):
    final_pred.append(Y_name[pred[i]])
output=pd.DataFrame()
output[0]=list(test_data['shot_id'])
output[1]=final_pred
output.to_csv('E:/WORKOUT/Statistic/data_sciense_intro/report1/out_model_merge_test2.csv')


# In[22]:
# =============================================================================
# plot_model(model, to_file='E:/WORKOUT/Statistic/data_sciense_intro/report1/model_plot.png', show_shapes=True, show_layer_names=True)
# =============================================================================
