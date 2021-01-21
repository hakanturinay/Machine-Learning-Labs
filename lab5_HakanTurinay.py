#!/usr/bin/env python
# coding: utf-8

# In[167]:


import pandas as pd
tb= pd.read_csv("team_big.csv", encoding = "ISO-8859-1", usecols = [4,6,7,8])
df = tb.copy()
df.head()


# In[195]:


from sklearn import metrics
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
import numpy as np


# In[177]:


X = df.drop("Salary", axis = 1)
y = df["Salary"]
X_train, X_test, y_train, y_test = train_test_split(X, y)


# In[178]:


def k_fold_cv(k): 
    
    lm = LinearRegression()
    model = lm.fit(X_train,y_train)
    cross_val =-cross_val_score(model, X, y, cv = k, scoring = "neg_mean_squared_error").mean()
    return cross_val


# In[191]:


def validation():
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
    lm = LinearRegression()
    model = lm.fit(X_train,y_train)
    y_pred = lm.predict(X_test)
    mse = metrics.mean_squared_error(y_test,y_pred)
    return mse


# In[258]:


df = df.reindex(np.random.permutation(df.index))
X = df.drop("Salary", axis = 1)
y = df["Salary"]


# In[259]:


cv_errors1 = k_fold_cv(5)
cv_errors1


# In[260]:


validation1 = validation()
validation1


# In[263]:


df = df.reindex(np.random.permutation(df.index))
X = df.drop("Salary", axis = 1)
y = df["Salary"]


# In[264]:


cv_errors2 = k_fold_cv(5)
cv_errors2


# In[265]:


validation2 = validation()
validation2


# In[266]:


df = df.reindex(np.random.permutation(df.index))
X = df.drop("Salary", axis = 1)
y = df["Salary"]


# In[267]:


cv_errors3 = k_fold_cv(5)
cv_errors3


# In[268]:


validation3 = validation()
validation3


# In[269]:


df = df.reindex(np.random.permutation(df.index))
X = df.drop("Salary", axis = 1)
y = df["Salary"]


# In[270]:


cv_errors4 = k_fold_cv(5)
cv_errors4


# In[271]:


validation4 = validation()
validation4


# In[272]:


df = df.reindex(np.random.permutation(df.index))
X = df.drop("Salary", axis = 1)
y = df["Salary"]


# In[273]:


cv_errors5 = k_fold_cv(5)
cv_errors5


# In[274]:


validation5 = validation()
validation5


# In[430]:


cv_error_array = np.array([cv_errors1,cv_errors2,cv_errors3,cv_errors4,cv_errors5] )
validation_array = np.array([validation1,validation2,validation3,validation4,validation5])
shuffle_number = pd.Series([1,2,3,4,5])


# In[431]:



import matplotlib.pyplot as plt


# In[432]:


plt.plot(shuffle_number,cv_error_array, label = "Cross-validation")
plt.plot( shuffle_number, validation_array, label = "Validation")
plt.xlabel('Shuffle number')
plt.ylabel('Mean squared error')
plt.title('Cross-Validation Vs Validation')
plt.legend()
plt.show()


# In[ ]:




