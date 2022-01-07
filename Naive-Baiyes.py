#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
data=pd.read_csv(r"covid.csv")
data.head()


# In[2]:


from sklearn import preprocessing


# In[3]:


le=preprocessing.LabelEncoder()


# In[4]:


pc_encoded=le.fit_transform(data["pc"].values)
wbc_encoded=le.fit_transform(data["wbc"].values)
mc_encoded=le.fit_transform(data['mc'].values)
ast_encoded=le.fit_transform(data["ast"].values)
bc_encoded=le.fit_transform(data["bc"].values)
ldh_encoded=le.fit_transform(data["ldh"].values)
Y=le.fit_transform(data["diagnosis"].values)


# In[5]:


X=np.array(list(zip(pc_encoded,wbc_encoded,mc_encoded,ast_encoded,bc_encoded,ldh_encoded)))


# In[7]:


X


# In[9]:


from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score


# In[10]:


model=MultinomialNB()


# In[11]:


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y)
model.fit(X_train,Y_train)
y_pred=model.predict(X_test)


# In[12]:


print(accuracy_score(Y_test,y_pred))


# In[13]:


print(classification_report(Y_test,y_pred))


# In[ ]:




