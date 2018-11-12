
# coding: utf-8

# In[5]:


# Load dataset and split train/test
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 42)


# In[6]:


# Normalize data
from sklearn.preprocessing import MinMaxScaler

mms = MinMaxScaler()
X_train_norm = mms.fit_transform(X_train)
X_test_norm = mms.transform(X_test)


# In[8]:


# Test number of neighbors from 1 to 10
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')

def plotNumNeighborsVsAccuracy(p_value = 2, metric_value = 'minkowski'):
    training_accuracy = []
    test_accuracy = []
    neighbor_settings = range(1,11)
    for curKvalue in neighbor_settings:
        # Build the model
        clf = KNeighborsClassifier(n_neighbors = curKvalue, p = p_value, metric = metric_value)
        clf.fit(X_train_norm, y_train)
        
        # Record training set accuracy
        curTrainAccuracy = clf.score(X_train_norm, y_train)
        training_accuracy.append(curTrainAccuracy)
        
        # Record test set accuracy
        curTestAccuracy = clf.score(X_test_norm, y_test)
        test_accuracy.append(curTestAccuracy)
        
    plt.plot(neighbor_settings, training_accuracy, label = "Training accuracy")
    plt.plot(neighbor_settings, test_accuracy, label = "Test accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Number of neighbors")
    plt.legend()

        
    


# In[9]:


plotNumNeighborsVsAccuracy()


# In[10]:


plotNumNeighborsVsAccuracy(p_value = 1, metric_value = 'minkowski')


# In[11]:


# SVM
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler

# Transform scale of data 
ss = StandardScaler()
X_train_scaled = ss.fit_transform(X_train)
X_test_scaled = ss.transform(X_test)

# Train and evaluate the linear SVM
svm_clf = LinearSVC()
svm_clf.fit(X_train_scaled, y_train)
curTestAccuracy = svm_clf.score(X_test_scaled, y_test)
print(curTestAccuracy)


# In[13]:


# Now train an SVM with polynomial features
from sklearn.svm import SVC

poly_kernel_svm_clf = SVC(kernel = "poly", degree = 3)
poly_kernel_svm_clf.fit(X_train_scaled, y_train)
curTestAccuracy = poly_kernel_svm_clf.score(X_test_scaled, y_test)
print(curTestAccuracy)


# In[17]:


# Perform 3-fold cross-validation
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

kfold = KFold(n_splits = 3)
fold_accuracies = cross_val_score(svm_clf, X, y, cv = kFold)
print("Cross-validation score:\n{}".format(fold_accuracies))
print("Average cross-validation score: {:2f}".format(fold_accuracies.mean()))


# In[18]:


# Reason for poor performance
print(y)


# In[20]:


# Perform 3-fold cross-validation after shuffling data

kfold = StratifiedKFold(n_splits = 3, shuffle = True, random_state = 2)
fold_accuracies = cross_val_score(svm_clf, X, y, cv = kfold)
print("Cross-validation score:\n{}".format(fold_accuracies))
print("Average cross-validation score: {:2f}".format(fold_accuracies.mean()))


# In[21]:


# Tune the hyperparameter using cross-validation 

best_score = 0
for curC in [0.001, 0.01, 0.1, 1, 10, 100]:
    svm = SVC(C = curC)
    fold_accuracies = cross_val_score(svm, X_train, y_train) #IMPORTANT: Tune hyperparamter on training partition only!!
    score = fold_accuracies.mean()
    if score > best_score:
        best_param = {'C': curC}
        best_score = score
        
svm = SVC(**best_param)
svm.fit(X_train, y_train)
test_score = svm.score(X_test, y_test)
print("Best score on cross-validation: {:0.2f}".format(best_score))
print("Best parameters: {}".format(best_param))
print("Test set score: {:.2f}".format(test_score))

