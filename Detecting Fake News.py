#!/usr/bin/env python
# coding: utf-8

# ## Detecting Fake News
# 
# This advanced python project of detecting fake news deals with fake and real news. Using sklearn, we build a TfidfVectorizer on our dataset. Then, we initialize a PassiveAggressive Classifier and fit the model. In the end, the accuracy score and the confusion matrix tell us how well our model fares.
# 
# to download the dataset from [here](https://drive.google.com/file/d/1er9NJTLUA3qnRuyhfzuN0XUsoIC4a-_q/view)

# ### import Libraries

# In[6]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix


# ### Reading DataSet
# 

# In[3]:


df = pd.read_csv("news.csv")


# ### Data Exploration

# In[4]:


df.shape


# In[5]:


df.head()


# ### spliting data into training set and test set

# In[8]:


X = df["text"]
y = df["label"]
X_train , X_test , y_train , y_test = train_test_split(X , y , test_size = 0.2 , random_state = 0)


# ### Apply TfidfVectorizer to training and testing data 

# In[12]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[14]:


tfidf = TfidfVectorizer(stop_words = "english" , max_df = 0.7)


# In[16]:


tfidf_train = tfidf.fit_transform(X_train)
tfidf_test = tfidf.transform(X_test)


# ### Build logistic regression model

# In[17]:


from sklearn.linear_model import LogisticRegression


# In[18]:


logmodel = LogisticRegression()


# In[19]:


logmodel.fit(tfidf_train , y_train)


# In[20]:


predictions = logmodel.predict(tfidf_test)


# In[30]:


score = accuracy_score(y_test , predictions)
print("{} %".format(round(score,2)*100))


# In[31]:


confusion_matrix(y_test , predictions)

