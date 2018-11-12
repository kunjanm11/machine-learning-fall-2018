
# coding: utf-8

# In[2]:


# 1. Perceptron
# Load dataset

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state = 42)


# In[5]:


# Instantiate the Perceptron
from sklearn. linear_model import Perceptron

clf = Perceptron(random_state = 0, max_iter = 2)
clf.fit(X_train, y_train)
score = clf.score(X_test, y_test)
print("Accuracy on tets set: {:.2f}".format(score))

