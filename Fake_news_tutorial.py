#!/usr/bin/env python
# coding: utf-8

# In[14]:


import numpy as np


# In[15]:


import pandas as pd


# In[16]:


import itertools


# In[17]:


from sklearn.model_selection import train_test_split


# In[18]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[19]:


from sklearn.linear_model import PassiveAggressiveClassifier


# In[20]:


from sklearn.metrics import accuracy_score, confusion_matrix


# In[21]:


#Read the data
df=pd.read_csv('/Users/melissarobinson/Desktop/ADTA_5340/news.csv')


# In[22]:


#Get shape and head
df.shape
df.head()


# In[23]:


#DataFlair - Get the labels
labels=df.label
labels.head()


# In[24]:


#DataFlair - Split the dataset
x_train,x_test,y_train,y_test=train_test_split(df['text'], labels, test_size=0.2, random_state=7)


# In[26]:


#DataFlair - Initialize a TfidVectorizer
tfidf_vectorizer=TfidfVectorizer(stop_words='english', max_df=0.7)


# In[27]:


#DataFlair - Fit and transform train set, transform test set
tfidf_train=tfidf_vectorizer.fit_transform(x_train)
tfidf_test=tfidf_vectorizer.transform(x_test)


# In[29]:


#DataFlair - Initialize a PassiveAggressiveClassifier
pac=PassiveAggressiveClassifier(max_iter=50)
pac.fit(tfidf_train,y_train)

#DataFlair - Predict on the test and calculate accuracy
y_pred=pac.predict(tfidf_test)
score=accuracy_score(y_test,y_pred)
print(f'Accuracy: {round(score*100,2)}%')


# In[31]:


#DataFlair - Build confusion matrix
confusion_matrix(y_test,y_pred, labels=['FAKE','REAL'])


# In[ ]:




