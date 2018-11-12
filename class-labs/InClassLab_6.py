
# coding: utf-8

# In[4]:


# Load dataset
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state = 42)


# In[5]:


print(X_train)


# In[7]:


#1. Use PCA to reduce the feature size/dimensionality

from sklearn.decomposition import PCA

pca = PCA(n_components = 2).fit(X_train)
X_train_reduced = pca.transform(X_train)
X_test_reduced = pca.transform(X_test)
print(X_test_reduced)


# In[8]:


# Majority Vote Ensemble algorithm
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import cross_val_score

clf1 = LogisticRegression()
clf2 = GaussianNB()
eclf = VotingClassifier(estimators=[('lr', clf1), ('gnb', clf2)], voting = 'hard')
scores = cross_val_score(eclf, cancer.data, cancer.target, cv = 5)
scores.mean()


# In[10]:


# Bagging ensemble
from sklearn.ensemble import BaggingClassifier

bagging = BaggingClassifier(clf2)
scores = cross_val_score(bagging, cancer.data, cancer.target, cv = 5)
scores.mean()


# In[11]:


# Adaboost algorithm
from sklearn.ensemble import AdaBoostClassifier

adabooster = AdaBoostClassifier(n_estimators = 50)
scores = cross_val_score(adabooster, cancer.data, cancer.target, cv = 5)
scores.mean()


# In[6]:


# Bag-of-words model
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

count = CountVectorizer()
docs = np.array([
    'Today is Wednesday',
    'Today is a day I am learning machine learning',
    'Tonight is the night I will have fun'
])
bag = count.fit_transform(docs) # create vocabulary
print(count.vocabulary_)


# In[7]:


print(bag.toarray())


# In[8]:


# TF-IDF model
from sklearn.feature_extraction.text import TfidfTransformer

tfidf = TfidfTransformer(use_idf = True,
                        norm = 'l2',
                        smooth_idf = True)
np.set_printoptions(precision = 2)
result = tfidf.fit_transform(count.fit_transform(docs))
print(result.toarray())


# In[9]:


type(docs)

