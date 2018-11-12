
# coding: utf-8

# In[1]:


# 1(a)

import numpy
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

wine = load_wine()
X = wine.data
y = wine.target


# In[2]:


print(wine.DESCR)


# In[3]:


# 1(b)

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.8, random_state = 42)


# In[4]:


from sklearn.preprocessing import MinMaxScaler

mms = MinMaxScaler()
X_train_norm = mms.fit_transform(X_train)
X_test_norm = mms.transform(X_test)


# In[17]:


# 2(a)
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier

best_score = 0
for curCriterion in ["gini", "entropy"]:
    for curDepth in [1, 2, 3, 4, 5, 6, 7, 8]:
        decision_tree_clf = DecisionTreeClassifier(criterion = curCriterion, max_depth = curDepth, random_state = 2)
        stratifiedKFold = StratifiedKFold(n_splits = 10, shuffle = True, random_state = 2)
        fold_accuracies = cross_val_score(decision_tree_clf, X_train, y_train, cv = stratifiedKFold, scoring = 'accuracy')
        score = fold_accuracies.mean()
        if score > best_score:
            best_param = {'criterion': curCriterion, 'max_depth' : curDepth}
            best_score = score
        
decision_tree_clf = DecisionTreeClassifier(**best_param)
decision_tree_clf.fit(X_train, y_train)
test_score = decision_tree_clf.score(X_test, y_test)
print("Best score on cross-validation: {:0.2f}".format(best_score))
print("Best parameters: {}".format(best_param))
print("Test set score: {:.2f}".format(test_score))


# In[18]:


# 2(b)

from sklearn.neighbors import KNeighborsClassifier

best_score = 0
for curP in [1, 2]:
    for curK in [1, 2, 3, 4, 5, 6, 7, 8]:
        knn_clf = KNeighborsClassifier(n_neighbors = curK, p = curP, metric = 'minkowski')
        stratifiedKFold = StratifiedKFold(n_splits = 10, shuffle = True, random_state = 2)
        fold_accuracies = cross_val_score(knn_clf, X_train_norm, y_train, cv = stratifiedKFold, scoring = 'accuracy') 
        score = fold_accuracies.mean()
        if score > best_score:
            best_param = {'n_neighbors': curK, 'p' : curP}
            best_score = score
        
knn_clf = KNeighborsClassifier(**best_param)
knn_clf.fit(X_train_norm, y_train)
test_score = knn_clf.score(X_test_norm, y_test)
print("Best score on cross-validation: {:0.2f}".format(best_score))
print("Best parameters: {}".format(best_param))
print("Test set score: {:.2f}".format(test_score))


# In[19]:


# 2(c)
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

ss = StandardScaler()
X_train_scaled = ss.fit_transform(X_train)
X_test_scaled = ss.transform(X_test)

best_score = 0

for curDegree in [2, 3, 4]:
    for curGamma in [0.01, 0.1, 1, 10, 25]:
        for curC in [0.001, 0.01, 0.1, 1, 10, 100]:
            poly_kernel_svm_clf = SVC(C = curC, kernel = "poly", degree = curDegree, gamma = curGamma, random_state = 2)
            stratifiedKFold = StratifiedKFold(n_splits = 10, shuffle = True, random_state = 2)
            fold_accuracies = cross_val_score(poly_kernel_svm_clf, X_train_scaled, y_train, cv = stratifiedKFold, scoring = 'accuracy') 
            score = fold_accuracies.mean()
            if score > best_score:
                best_param = {'C': curC, 'degree': curDegree, 'gamma' : curGamma}
                best_score = score
        
svm = SVC(**best_param)
svm.fit(X_train_scaled, y_train)
test_score = svm.score(X_test_scaled, y_test)
print("Best score on cross-validation: {:0.2f}".format(best_score))
print("Best parameters: {}".format(best_param))
print("Test set score: {:.2f}".format(test_score))


# In[20]:


# 3(a)

decision_tree_clf = DecisionTreeClassifier(criterion = 'entropy', max_depth = 4)
decision_tree_clf.fit(X_train, y_train)

knn_clf = KNeighborsClassifier(n_neighbors = 2, p = 1, metric = 'minkowski')
knn_clf.fit(X_train_norm, y_train)

poly_kernel_svm_clf = SVC(C = 1, kernel = "poly", degree = 3, gamma = 0.1)
poly_kernel_svm_clf.fit(X_train_scaled, y_train)



# In[21]:


# 3(b)
from sklearn.naive_bayes import GaussianNB

naive_bayes_clf = GaussianNB()
naive_bayes_clf.fit(X_train, y_train)


# In[22]:


# 3(c)

from sklearn.metrics import accuracy_score

print("Accuracy scores:")
decision_tree_predicted = decision_tree_clf.predict(X_test)
print("Decision tree: {}".format(accuracy_score(y_test, decision_tree_predicted)))

knn_predicted = knn_clf.predict(X_test_norm)
print("KNN: {}".format(accuracy_score(y_test, knn_predicted)))

svm_predicted = svm.predict(X_test_scaled)
print("SVM: {}".format(accuracy_score(y_test, svm_predicted)))

naive_bayes_predicted = naive_bayes_clf.predict(X_test)
print("Naive Bayes: {}".format(accuracy_score(y_test, naive_bayes_predicted)))


# In[23]:


from sklearn.metrics import precision_score

print("Precision scores:")
print("Decision tree: {}".format(precision_score(y_test, decision_tree_predicted, average = None)))
print("KNN: {}".format(precision_score(y_test, knn_predicted, average = None)))
print("SVM: {}".format(precision_score(y_test, svm_predicted, average = None)))
print("Naive Bayes: {}".format(precision_score(y_test, naive_bayes_predicted, average = None)))



# In[24]:


from sklearn.metrics import recall_score

print("Recall scores:")
print("Decision tree: {}".format(recall_score(y_test, decision_tree_predicted, average = None)))
print("KNN: {}".format(recall_score(y_test, knn_predicted, average = None)))
print("SVM: {}".format(recall_score(y_test, svm_predicted, average = None)))
print("Naive Bayes: {}".format(recall_score(y_test, naive_bayes_predicted, average = None)))


# In[25]:


# 3(d)

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

get_ipython().magic(u'matplotlib inline')
cm_decision_tree = confusion_matrix(y_test, decision_tree_predicted)
sns.heatmap(cm_decision_tree.T, square = True, annot = True, fmt = 'd', cbar = False)
plt.xlabel('true label')
plt.ylabel('predicted label')


# In[26]:


cm_knn = confusion_matrix(y_test, knn_predicted)
sns.heatmap(cm_knn.T, square = True, annot = True, fmt = 'd', cbar = False)
plt.xlabel('true label')
plt.ylabel('predicted label')


# In[27]:


cm_svm = confusion_matrix(y_test, svm_predicted)
sns.heatmap(cm_svm.T, square = True, annot = True, fmt = 'd', cbar = False)
plt.xlabel('true label')
plt.ylabel('predicted label')


# In[28]:


cm_nb = confusion_matrix(y_test, naive_bayes_predicted)
sns.heatmap(cm_nb.T, square = True, annot = True, fmt = 'd', cbar = False)
plt.xlabel('true label')
plt.ylabel('predicted label')

