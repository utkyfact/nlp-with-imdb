#!/usr/bin/env python
# coding: utf-8

# ## &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;NLP (Natural Language Processing) KULLANARAK IMDB FİLM 
# 
# ## YORUMLARI KAGGLE DATA SETİ ÜZERİNDE SENTIMENT ANALİZİ
# 

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from bs4 import BeautifulSoup
import re
import nltk
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords


# In[2]:


df = pd.read_csv('NLPlabeledData.tsv',  delimiter="\t", quoting=3)


# In[3]:



df.head()


# In[4]:


len(df)


# In[5]:


len(df["review"])


# In[6]:



nltk.download('stopwords')



# In[ ]:


sample_review= df.review[0]
sample_review


# In[ ]:


sample_review = BeautifulSoup(sample_review).get_text()
sample_review


# In[ ]:


sample_review = re.sub("[^a-zA-Z]",' ',sample_review)
sample_review


# In[ ]:


sample_review = sample_review.lower()
sample_review


# In[ ]:


sample_review = sample_review.split()


  


# In[ ]:


sample_review


# In[ ]:


len(sample_review)


# In[ ]:


# sample_review without stopwords
swords = set(stopwords.words("english"))  # conversion into set for fast searching
sample_review = [w for w in sample_review if w not in swords]               
sample_review


# In[ ]:


len(sample_review)


# In[ ]:




# In[7]:


def process(review):
    # review without HTML tags
    review = BeautifulSoup(review).get_text()
    # review without punctuation and numbers
    review = re.sub("[^a-zA-Z]",' ',review)
    # converting into lowercase and splitting to eliminate stopwords
    review = review.lower()
    review = review.split()
    # review without stopwords
    swords = set(stopwords.words("english")) # conversion into set for fast searching
    review = [w for w in review if w not in swords]               
    # splitted paragraph'ları space ile birleştiriyoruz return
    return(" ".join(review))


# In[8]:



train_x_tum = []
for r in range(len(df["review"])):        
    if (r+1)%1000 == 0:        
        print("No of reviews processed =", r+1)
    train_x_tum.append(process(df["review"][r]))


# ### Train, test split...

# In[9]:


x = train_x_tum
y = np.array(df["sentiment"])

# train test split
train_x, test_x, y_train, y_test = train_test_split(x,y, test_size = 0.1)



# In[10]:


vectorizer = CountVectorizer( max_features = 5000 )

train_x = vectorizer.fit_transform(train_x)


# In[11]:


train_x


# In[12]:


train_x = train_x.toarray()
train_y = y_train


# In[13]:


train_x.shape, train_y.shape


# In[14]:


train_y



# In[15]:


model = RandomForestClassifier(n_estimators = 100, random_state=42)
model.fit(train_x, train_y)


# In[ ]:






# In[16]:



test_xx = vectorizer.transform(test_x)


# In[17]:


test_xx


# In[18]:


test_xx = test_xx.toarray()


# In[19]:


test_xx.shape



# In[20]:


test_predict = model.predict(test_xx)
dogruluk = roc_auc_score(y_test, test_predict)


# In[21]:


print("Doğruluk oranı : % ", dogruluk * 100)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




