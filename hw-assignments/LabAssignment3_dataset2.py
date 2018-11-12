
# coding: utf-8

# In[1]:


# 2(a) Load dataset 2. 

import pandas
dataset = pandas.read_csv("/home/nbuser/library/Question_Classification_Dataset.csv", sep = ',', encoding = 'ISO-8859-1')


# In[2]:


dataset.shape


# In[8]:


X = dataset.iloc[:1000,1]
y = dataset.iloc[:1000,2]


# In[10]:


import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

count = CountVectorizer()
bag = count.fit_transform(X)
print(count.vocabulary_)


# In[11]:


# TF-IDF model
from sklearn.feature_extraction.text import TfidfTransformer

tfidf = TfidfTransformer(use_idf = True,
                        norm = 'l2',
                        smooth_idf = True)
np.set_printoptions(precision = 2)
result = tfidf.fit_transform(count.fit_transform(X))
X = result.toarray()


# In[14]:


# Dataset 2: SVM classifer

from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

svm_clf = SVC()
kFold = KFold(n_splits = 10)
fold_accuracies = cross_val_score(svm_clf, X, y, cv = kFold)
print("Cross-validation score:\n{}".format(fold_accuracies))
print("Average cross-validation score: {:2f}".format(fold_accuracies.mean()))


# In[15]:


# Dataset 2: Naive Bayes classifer

naive_bayes_clf = GaussianNB()
fold_accuracies = cross_val_score(naive_bayes_clf, X, y, cv = kFold)
print("Cross-validation score:\n{}".format(fold_accuracies))
print("Average cross-validation score: {:2f}".format(fold_accuracies.mean()))


# In[16]:


# Dataset 2: KNN classifer

knn_clf = KNeighborsClassifier(n_neighbors = 10)
fold_accuracies = cross_val_score(knn_clf, X, y, cv = kFold)
print("Cross-validation score:\n{}".format(fold_accuracies))
print("Average cross-validation score: {:2f}".format(fold_accuracies.mean()))


# In[17]:


# Dataset 2: Majority vote classifer

from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

kFold= KFold(n_splits = 10)
clf1 = SVC()
clf2 = GaussianNB()
clf3 = KNeighborsClassifier(n_neighbors = 10)

eclf = VotingClassifier(estimators=[('lr', clf1), ('gnb', clf2), ('rf', clf3)], voting = 'hard')
scores = cross_val_score(eclf, X, y, cv = kFold)
scores.mean()


# In[18]:


# Dataset 2: Bagging method

from sklearn.ensemble import BaggingClassifier

bagging = BaggingClassifier(clf1)
scores = cross_val_score(bagging, X, y, cv = kFold)
scores.mean()


# In[19]:


# Dataset 2: Boosting method

from sklearn.ensemble import AdaBoostClassifier

adabooster = AdaBoostClassifier(n_estimators = 50)
scores = cross_val_score(adabooster, X, y, cv = kFold)
scores.mean()

