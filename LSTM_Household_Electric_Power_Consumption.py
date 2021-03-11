#!/usr/bin/env python
# coding: utf-8

# In[2]:


import sys 
import numpy as np # linear algebra
from scipy.stats import randint
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv), data manipulation as in SQL
import matplotlib.pyplot as plt # this is used for the plot the graph 
import seaborn as sns # used for plot interactive graph. 
from sklearn.model_selection import train_test_split # to split the data into two parts
from sklearn.preprocessing import StandardScaler # for normalization
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline # pipeline making
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectFromModel
from sklearn import metrics # for the check the error and accuracy of the model
from sklearn.metrics import mean_squared_error,r2_score

## for Deep-learing:
import keras
from keras.layers import Dense
from keras.models import Sequential
from keras.utils import to_categorical
from keras.optimizers import SGD 
from keras.callbacks import EarlyStopping
from keras.utils import np_utils
import itertools
from keras.layers import LSTM
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers import Dropout


# In[3]:


def plot_predictions(test, predicted):
    plt.plot(test[:200], color='red', label='Actual')
    plt.plot(predicted[:200], color='blue', label='Predicted')
    plt.xlabel('Global Active Power', size=15)
    plt.ylabel('Time Step', size=15)
    plt.legend()
    plt.show()
    
def return_rmse(test, predicted):
    rmse = math.sqrt(mean_squared_error(test, predicted))
    print('Root Mean Squared Error is {}.'.format(rmse))


# In[7]:


df = pd.read_csv("C:/Users/HP/Downloads/archive/household_power_consumption.txt", sep=';', parse_dates = {'dt' :['Date','Time']}, 
                 infer_datetime_format = True, na_values=['nan', '?'], index_col = 'dt')


# In[8]:


df.head()


# In[9]:


def _convert(data, n_in=1, n_out=1):
    n_vars = 1 if type(data) is list else data.shape[1]
    names, cols = list(), list()
    dff = pd.DataFrame(data)
    
    for i in range(n_in, 0, -1):
        cols.append(dff.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    for i in range(0, n_out):
        cols.append(dff.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    return agg

#df = df.resample('h').mean()
values = df.values

scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)
df = _convert(scaled, 1, 1)


# In[10]:


df.head()


# In[11]:


df = df.dropna()


# In[12]:


df.columns


# In[13]:


df.drop(['var2(t)', 'var3(t)', 'var4(t)',
       'var5(t)', 'var6(t)', 'var7(t)'], axis=1, inplace=True)


# In[14]:


df.head()


# In[339]:


values = df.values
num = 365*24
train = values[:num, :]
test = values[num:, :]
X_train, y_train = train[:, :-1], train[:, -1]
X_test, y_test = test[:, :-1], test[:, -1]
X_train = np.reshape(X_train, (X_train.shape[0],X_train.shape[1],1))
X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))


# In[340]:


print(X_train.shape, y_train.shape, X_test.shape, y_test.shape) 


# In[341]:


model = Sequential()

model.add(LSTM(100, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences = True))
model.add(Dropout(0.2))

model.add(LSTM(70))
model.add(Dropout(0.3))

model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])


# In[342]:


history = model.fit(X_train, y_train, epochs=50, batch_size=70, validation_data=(X_test, y_test), verbose=1, shuffle=False)
plt.plot(history.history['loss'], color='red')
plt.plot(history.history['val_loss'], color='blue')
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()


# In[343]:


predicted


# In[344]:


y_test


# In[345]:


import math
predicted = model.predict(X_test)
return_rmse(y_test, predicted)


# In[346]:


plot_predictions(y_test, predicted)

