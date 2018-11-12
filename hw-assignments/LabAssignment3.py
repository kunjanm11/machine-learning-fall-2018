
# coding: utf-8

# In[1]:


# 1(a) Loading LFW dataset

from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split

lfw = fetch_lfw_people(min_faces_per_person = 60)
X_train, X_test, y_train, y_test = train_test_split(lfw.data, lfw.target, train_size = 0.75, random_state = 42)


# In[14]:


# 1(b) SVM on LFW

from sklearn.decomposition import PCA
from sklearn.svm import SVC
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')

components, comp_score = [], []
for i in range (100, 726, 25):
    pca = PCA(n_components = i, svd_solver = 'randomized').fit(X_train)
    X_train_new = pca.transform(X_train)
    X_test_new = pca.transform(X_test)
    svc_clf = SVC()
    svc_clf.fit(X_train_new, y_train)
    components.append(i)
    comp_score.append(svc_clf.score(X_test_new, y_test))
    
plt.plot(components, comp_score, "b.")


# In[15]:


# Naive Bayes on LFW 

from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')

components, comp_score = [], []
for i in range (100, 726, 25):
    pca = PCA(n_components = i, svd_solver = 'randomized').fit(X_train)
    X_train_new = pca.transform(X_train)
    X_test_new = pca.transform(X_test)
    naive_bayes_clf = GaussianNB()
    naive_bayes_clf.fit(X_train_new, y_train)
    components.append(i)
    comp_score.append(naive_bayes_clf.score(X_test_new, y_test))
    
plt.plot(components, comp_score, "b.")


# In[17]:


#Load MNIST dataset

from sklearn.datasets import fetch_mldata
mnist = fetch_mldata('MNIST original')
mnist


# In[21]:


X, y = mnist["data"], mnist["target"]
X_small = X[:10000]
y_small = y[:10000]
X_train, X_test, y_train, y_test = train_test_split(X_small, y_small, train_size = 0.75, random_state = 42)


# In[22]:


# SVM on MNIST

components, comp_score = [], []
for i in range (100, 726, 25):
    pca = PCA(n_components = i, svd_solver = 'randomized').fit(X_train)
    X_train_new = pca.transform(X_train)
    X_test_new = pca.transform(X_test)
    svc_clf = SVC()
    svc_clf.fit(X_train_new, y_train)
    components.append(i)
    comp_score.append(svc_clf.score(X_test_new, y_test))
    
plt.plot(components, comp_score, "b.")


# In[23]:


# Naive Bayes on MNIST

from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')

components, comp_score = [], []
for i in range (100, 726, 25):
    pca = PCA(n_components = i, svd_solver = 'randomized').fit(X_train)
    X_train_new = pca.transform(X_train)
    X_test_new = pca.transform(X_test)
    naive_bayes_clf = GaussianNB()
    naive_bayes_clf.fit(X_train_new, y_train)
    components.append(i)
    comp_score.append(naive_bayes_clf.score(X_test_new, y_test))
    
plt.plot(components, comp_score, "b.")


# In[211]:


# 2(a) Load dataset 1. 

import pandas
dataset = pandas.read_csv("/home/nbuser/library/spam.csv", sep = ',', encoding = 'ISO-8859-1')


# In[212]:


dataset.shape


# In[213]:


X = dataset.iloc[:1000,1]
y = dataset.iloc[:1000,0]


# In[214]:


import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

count = CountVectorizer()
bag = count.fit_transform(X)
print(count.vocabulary_)


# In[216]:


# TF-IDF model
from sklearn.feature_extraction.text import TfidfTransformer

tfidf = TfidfTransformer(use_idf = True,
                        norm = 'l2',
                        smooth_idf = True)
np.set_printoptions(precision = 2)
result = tfidf.fit_transform(count.fit_transform(X))
X = result.toarray()


# In[217]:


# Dataset 1: SVM classifer

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

svm_clf = SVC()
kFold = KFold(n_splits = 10)
fold_accuracies = cross_val_score(svm_clf, X, y, cv = kFold)
print("Cross-validation score:\n{}".format(fold_accuracies))
print("Average cross-validation score: {:2f}".format(fold_accuracies.mean()))


# In[218]:


# Dataset 1: Naive Bayes classifer

naive_bayes_clf = GaussianNB()
fold_accuracies = cross_val_score(naive_bayes_clf, X, y, cv = kFold)
print("Cross-validation score:\n{}".format(fold_accuracies))
print("Average cross-validation score: {:2f}".format(fold_accuracies.mean()))


# In[219]:


# Dataset 1: KNN classifer

knn_clf = KNeighborsClassifier(n_neighbors = 10)
fold_accuracies = cross_val_score(knn_clf, X, y, cv = kFold)
print("Cross-validation score:\n{}".format(fold_accuracies))
print("Average cross-validation score: {:2f}".format(fold_accuracies.mean()))


# In[220]:


# Dataset 1: Majority vote classifer

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


# In[221]:


# Dataset 1: Bagging method

from sklearn.ensemble import BaggingClassifier

bagging = BaggingClassifier(clf1)
scores = cross_val_score(bagging, X, y, cv = kFold)
scores.mean()


# In[222]:


# Dataset 1: Boosting method

from sklearn.ensemble import AdaBoostClassifier

adabooster = AdaBoostClassifier(n_estimators = 50)
scores = cross_val_score(adabooster, X, y, cv = kFold)
scores.mean()

