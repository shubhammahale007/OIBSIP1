#!/usr/bin/env python
# coding: utf-8

# In[98]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
import warnings


# In[99]:


import os
for dirname, _, filenames in os.walk('F:\shubzz\Iris (2).csv'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[100]:


df= pd.read_csv('F:\shubzz\Iris (2).csv')
df.head(10)


# In[101]:


df=df.drop(columns=['Id'])
df


# In[89]:


df.Species.value_counts()


# In[102]:


df.shape


# In[103]:


df["Species"].replace({"Iris-setosa": 2, "Iris-versicolor": 3, "Iris-virginica": 4}, inplace = True)
df.head(10)


# In[92]:



x=pd.DataFrame(df,columns=["SepalLengthCm","SepalWidthCm","PetalLengthCm","PetalWidthCm"]).values
y=df.Species.values.reshape(-1,1)


# In[104]:


x


# In[105]:


y


# In[106]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.30,random_state=42) 


# In[107]:


metrics.accuracy_score(y_test,y_pred)


# In[97]:


from sklearn.model_selection import train_test_split


# In[108]:


X= df.drop('Species', axis= 1)
y= df['Species']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.28, random_state=1)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[ ]:




