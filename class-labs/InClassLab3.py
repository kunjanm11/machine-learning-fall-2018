
# coding: utf-8

# In[3]:


from sklearn.datasets import load_iris
iris_data = load_iris()


# In[4]:


iris_data.keys()


# In[5]:


dir(iris_data)


# In[6]:


print(iris_data.DESCR)


# In[7]:


X = iris_data.data
y = iris_data.target


# In[8]:


print(X)


# In[9]:


print(y)


# In[10]:


print(X.shape)


# In[11]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 42)


# In[12]:


X_train


# In[13]:


X_test.shape


# In[16]:


from sklearn.tree import DecisionTreeClassifier
tree_giniIndex = DecisionTreeClassifier().fit(X_train, y_train)


# In[17]:


tree_giniIndex


# In[18]:


from sklearn import tree
import graphviz

get_ipython().magic(u'matplotlib inline')

dot_data = tree.export_graphviz(tree_giniIndex, out_file = None, feature_names = iris_data.feature_names, class_names = iris_data.target_names, filled = True, rounded = True)
graph = graphviz.Source(dot_data)
graph


# In[19]:


tree_giniIndexPruned = DecisionTreeClassifier(max_depth = 2).fit(X_train, y_train)

dot_data = tree.export_graphviz(tree_giniIndexPruned, out_file = None, feature_names = iris_data.feature_names, class_names = iris_data.target_names, filled = True, rounded = True)
graph = graphviz.Source(dot_data)
graph


# In[22]:


tree_entropy = DecisionTreeClassifier(criterion = 'entropy').fit(X_train, y_train)


# In[23]:


dot_data = tree.export_graphviz(tree_entropy, out_file = None, feature_names = iris_data.feature_names, class_names = iris_data.target_names, filled = True, rounded = True)
graph = graphviz.Source(dot_data)
graph


# In[24]:


from sklearn import metrics
y_predicted = tree_giniIndexPruned.predict(X_test)


# In[25]:


y_predicted


# In[26]:


print(metrics.classification_report(y_predicted, y_test))


# In[28]:


from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

mat = confusion_matrix(y_predicted, y_test)
sns.heatmap(mat.T, square = True, annot = True, fmt = 'd', cbar = False)
plt.xlabel('true label')
plt.ylabel('predicted label')


# In[29]:


from sklearn.naive_bayes import GaussianNB

gaussian_model = GaussianNB()
gaussian_model.fit(X_train, y_train)


# In[30]:


y_predictedGaussianResults = gaussian_model.predict(X_test)


# In[31]:


print(metrics.classification_report(y_predictedGaussianResults, y_test))


# In[32]:


mat = confusion_matrix(y_predictedGaussianResults, y_test)
sns.heatmap(mat.T, square = True, annot = True, fmt = 'd', cbar = False)
plt.xlabel('true label')
plt.ylabel('predicted label')

