
# coding: utf-8

# In[16]:


# 1. IMPUTE MISSING VALUES
from numpy import nan
import numpy as np

X_train = ([[nan, 0, 3],
           [3, 7, 9],
           [3, 5, 2],
           [4, nan, 6],
           [8, 8, 1]])
X_test = ([[14, 16, -1],
          [nan, 8, -5]])


# In[17]:


# "Train" imputer
from sklearn.preprocessing import Imputer

imputer = Imputer(strategy='median')
imputer.fit(X_train)
print(imputer.statistics_)


# In[18]:


X_train_fixed = imputer.transform(X_train)
print(X_train_fixed)


# In[19]:


X_test_fixed = imputer.transform(X_test)
print(X_test_fixed)


# In[20]:


# 2. Scaling numerical features using min-max scaler
from sklearn.preprocessing import MinMaxScaler

mms = MinMaxScaler()
mms.fit(X_train_fixed)
print(mms.scale_)
print(mms.min_)


# In[21]:


X_train_scaled = mms.transform(X_train_fixed)
print(X_train_scaled)


# In[22]:


X_test_scaled = mms.transform(X_test_fixed)
print(X_test_scaled)


# In[32]:


# Standardization
from sklearn.preprocessing import StandardScaler

stdsc = StandardScaler()
stdsc = stdsc.fit(X_train_fixed)
X_train_std = stdsc.transform(X_train_fixed)
X_test_std = stdsc.transform(X_test_fixed)
print(X_train_std)
print(X_test_std)


# In[33]:


# 3. Convert categorical variables to one-hot encodings

from sklearn.preprocessing import OneHotEncoder

# Train data
# 0 0 3
# 1 1 0
# 0 2 1
# 1 0 2

enc = OneHotEncoder()
data = [[0,0,3], [1, 1, 0], [0, 2, 1], [1, 0, 2]]
enc.fit(data)
encoding = enc.transform([[0, 1, 1]]).toarray()
print("Test data: \n{}".format(encoding))


# In[25]:


# Dimensionality reduction using PCA
# https://jakevdp.github.io/PythonDataScienceHandbook/05.09-principal-component-analysis.html

from sklearn.datasets import fetch_lfw_people

faces = fetch_lfw_people(min_faces_per_person = 60)


# In[26]:


print(faces.target_names)
print(faces.images.shape)


# In[27]:


from sklearn.decomposition import PCA

pca = PCA(150, svd_solver = 'randomized').fit(faces.data)
components = pca.transform(faces.data)


# In[28]:


projected = pca.inverse_transform(components)


# In[31]:


# Visualize the results when have all pixels ~3000 pixels (top row)
# and when using only 150 components (bottom row)
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')

fix, ax = plt.subplots(2, 10, figsize = (10, 2.5),
                     subplot_kw = {'xticks':[], 'yticks':[]},
                     gridspec_kw = dict(hspace = 0.1, wspace = 0.1))

for i in range(10):
    ax[0, i].imshow(faces.data[i].reshape(62, 47), cmap = 'binary_r')
    ax[1, i].imshow(projected[i].reshape(62, 47), cmap = 'binary_r')
    
ax[0, 0].set_ylabel('full-dim\ninput')
ax[1, 0].set_ylabel('150-dim\ncreconstruction')

