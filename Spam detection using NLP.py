#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


messages=pd.read_csv('SMSSpamCollection',sep='\t',names=["label","message"])
print(messages)


# In[3]:


import nltk
import re
nltk.download('stopwords')


# In[4]:


from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()
corpus=[]
for i in range(0,len(messages)):
    review = re.sub('[^a-zA-Z]', ' ',messages['message'][i])
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)
print(corpus)


# In[6]:


from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
X=cv.fit_transform(corpus).toarray()
print(X)


# In[7]:


y = pd.get_dummies(messages['label'])
y = y.iloc[:,1].values


# In[8]:


from sklearn.model_selection import train_test_split
X_train, X_test,y_train, y_test = train_test_split(X,y,test_size=0.20,random_state=0)


# In[9]:


from sklearn.naive_bayes import MultinomialNB
spam_detect_model = MultinomialNB().fit(X_train,y_train)
y_pred=spam_detect_model.predict(X_test)


# In[10]:


from sklearn.metrics import confusion_matrix
confusion_m=confusion_matrix(y_test,y_pred)
print(confusion_m)


# In[12]:


from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
print(accuracy)


# In[ ]:




