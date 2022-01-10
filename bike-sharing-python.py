#!/usr/bin/env python
# coding: utf-8

# In[1]:


# importing libraries
import numpy as np
import pandas as pd
from datetime import datetime
from datetime import date
import calendar
import matplotlib.pyplot as plt
import seaborn as sn
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')


# In[4]:


train.shape, test.shape


# In[5]:


train.head()


# In[6]:


test.head()


# In[7]:


train.columns


# In[8]:


test.columns


# In[9]:


train.dtypes


# In[10]:


sn.distplot(train["count"])


# In[11]:


sn.distplot(np.log(train["count"]))


# In[12]:


sn.distplot(train["registered"])


# In[14]:


corr = train[["temp","atemp","casual","registered","humidity","windspeed","count"]].corr()
mask = np.array(corr)
mask[np.tril_indices_from(mask)] = False
fig,ax= plt.subplots()
fig.set_size_inches(20,10)
sn.heatmap(corr, mask=mask,vmax=.9, square=True,annot=True, cmap="YlGnBu")


# In[15]:


train.isnull().sum()


# In[16]:


test.isnull().sum()


# In[17]:


train["date"] = train.datetime.apply(lambda x : x.split()[0])
train["hour"] = train.datetime.apply(lambda x : x.split()[1].split(":")[0])
train["month"] = train.date.apply(lambda dateString : datetime.strptime(dateString,"%Y-%m-%d").month)


# In[18]:


test["date"] = test.datetime.apply(lambda x : x.split()[0])
test["hour"] = test.datetime.apply(lambda x : x.split()[1].split(":")[0])
test["month"] = test.date.apply(lambda dateString : datetime.strptime(dateString,"%Y-%m-%d").month)


# In[19]:


training = train[train['datetime']<='2012-03-30 0:00:00']
validation = train[train['datetime']>'2012-03-30 0:00:00']


# In[20]:


train = train.drop(['datetime','date', 'atemp'],axis=1)
test = test.drop(['datetime','date', 'atemp'], axis=1)
training = training.drop(['datetime','date', 'atemp'],axis=1)
validation = validation.drop(['datetime','date', 'atemp'],axis=1)


# In[21]:


from sklearn.linear_model import LinearRegression


# In[22]:


lModel = LinearRegression()


# In[23]:


X_train = training.drop('count', 1)
y_train = np.log(training['count'])
X_val = validation.drop('count', 1)
y_val = np.log(validation['count'])


# In[24]:


X_train.shape, y_train.shape, X_val.shape, y_val.shape


# In[25]:


lModel.fit(X_train,y_train)


# In[26]:


prediction = lModel.predict(X_val)


# In[27]:


# defining a function which will return the rmsle score
def rmsle(y, y_):
    y = np.exp(y),   # taking the exponential as we took the log of target variable
    y_ = np.exp(y_)
    log1 = np.nan_to_num(np.array([np.log(v + 1) for v in y]))
    log2 = np.nan_to_num(np.array([np.log(v + 1) for v in y_]))
    calc = (log1 - log2) ** 2
    return np.sqrt(np.mean(calc))


# In[28]:


rmsle(y_val,prediction)


# In[29]:


from sklearn.tree import DecisionTreeRegressor


# In[30]:


dt_reg = DecisionTreeRegressor(max_depth=5)


# In[31]:


dt_reg.fit(X_train, y_train)


# In[32]:


predict = dt_reg.predict(X_val)


# In[33]:


rmsle(y_val, predict)


# In[34]:


test_prediction = dt_reg.predict(test)


# In[35]:


final_prediction = np.exp(test_prediction)


# In[38]:


outputpro = pd.DataFrame()


# In[39]:


outputpro['count'] = final_prediction


# In[40]:


outputpro.to_csv('outputpro.csv', header=True, index=False)

