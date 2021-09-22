#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')


# ## IRIS DATASET

# In[3]:


import seaborn as sns
df = sns.load_dataset('iris')


# In[4]:


df.head()


# ## STANDARDIZE THE VARIABLES

# In[5]:


from sklearn.preprocessing import StandardScaler


# In[6]:


scaler = StandardScaler()


# In[7]:


scaler.fit(df.drop('species',axis=1))


# In[8]:


scaled_features = scaler.transform(df.drop('species',axis=1))


# In[9]:


df_feat = pd.DataFrame(scaled_features,columns=df.columns[:-1])
df_feat.head()


# ## Train Test Split

# In[10]:


from sklearn.model_selection import train_test_split


# In[11]:


X_train, X_test, y_train, y_test = train_test_split(scaled_features,df['species'],test_size=0.30)


# ## Using KNN

# In[12]:


from sklearn.neighbors import KNeighborsClassifier


# In[13]:


knn = KNeighborsClassifier(n_neighbors=1)


# In[14]:


knn.fit(X_train,y_train)


# In[15]:


pred = knn.predict(X_test)


# ## Prediction and Evaluation

# In[16]:


from sklearn.metrics import classification_report,confusion_matrix


# In[17]:


print(confusion_matrix(y_test,pred))


# In[18]:


print(classification_report(y_test,pred))


# ## Choose a K-value

# In[19]:


error_rate = []

# Will take some time
for i in range(1,40):
    
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))


# In[20]:


plt.figure(figsize=(10,6))
plt.plot(range(1,40),error_rate,color='blue', linestyle='dashed', marker='o',
         markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')

plt.ylabel('Error Rate')


# #### Here we see that the error is lowest at K=5 so we retrain the model with K=5

# In[21]:


# FIRST A QUICK COMPARISON TO OUR ORIGINAL K=1
knn = KNeighborsClassifier(n_neighbors=1)

knn.fit(X_train,y_train)
pred = knn.predict(X_test)

print('WITH K=1')
print('\n')
print(confusion_matrix(y_test,pred))
print('\n')
print(classification_report(y_test,pred))


# In[22]:


# NOW WITH K=5
knn = KNeighborsClassifier(n_neighbors=5)

knn.fit(X_train,y_train)
pred = knn.predict(X_test)

print('WITH K=5')
print('\n')
print(confusion_matrix(y_test,pred))
print('\n')
print(classification_report(y_test,pred))


# In[ ]:





# In[ ]:




