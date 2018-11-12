
# coding: utf-8

# In[1]:


# 1. Generate noisy data sample from the linear function y = 3x + 4


# In[2]:


import numpy

numSamples = 100
X = numpy.random.rand(numSamples, 1)
y = 4 + 3X + numpy.random.randn(numSamples, 1)


# In[3]:


import numpy

numSamples = 100
X = numpy.random.rand(numSamples, 1)
y = 4 + 3*X + numpy.random.randn(numSamples, 1)


# In[4]:


# 2. Visualize our data

import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')
plt.plot(X, y, "b.")
plt.show()


# In[5]:


# 3. Create train/test split

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 42)
print("Number samples in training: ", len(X_train))
print("Number samples in testing: ", len(X_test))


# In[6]:


# 4-5. Train linear regression model
from sklearn import linear_model

lr_model = linear_model.LinearRegression().fit(X_train, y_train)
print("Slope/Coefficient: ", lr_model.coef_)
print("Intercept: ", lr_model.intercept_)


# In[7]:


# 6. Visualize predicted versus actual value for the test data
y_predicted = lr_model.intercept_ + lr_model.coef_*X_test

plt.plot(X_test, y_test, "b.")
plt.plot(X_test, y_predicted, "r-")


# In[8]:


# 7-8. Evaluate model performance on test dataset
from sklearn.metrics import mean_absolute_error

mean_absolute_error(y_test, y_predicted)


# In[9]:


from sklearn.metrics import r2_score

r2_score(y_test, y_predicted)


# In[10]:


from sklearn.metrics import mean_squared_error

mean_squared_error(y_test, y_predicted)


# In[12]:


# Train and visualize  Ridge Regression results
ridge_model = linear_model.Ridge(alpha = 0.3)
ridge_model.fit(X_train, y_train)
ridge_predicted = ridge_model.predict(X_test)
plt.plot(X_test, y_test, "b.")
plt.plot(X_test, y_predicted, "r-")


# In[13]:


# Train and visualize Lasso Regression results
lasso_model = linear_model.Lasso(alpha = 0.4)
lasso_model.fit(X_train, y_train)
lasso_predicted = lasso_model.predict(X_test)
plt.plot(X_test, y_test, "b.")
plt.plot(X_test, y_predicted, "r-")


# In[15]:


# Visualize results from linear, ridge, and lasso regression results

plt.plot(X_test, y_test, "b.")
plt.plot(X_test, y_predicted, "r-")
plt.plot(X_test, ridge_predicted, "g-")
plt.plot(X_test, lasso_predicted, "y-")


# In[17]:


# Train polynomial features for model 
from sklearn.preprocessing import PolynomialFeatures 

poly_features = PolynomialFeatures(degree = 2, include_bias = False)
X_poly = poly_features.fit_transform(X)
print(X.shape)
print(X_poly.shape)


# In[18]:


X_polyTrain, X_polyTest, y_polyTrain, y_polyTest = train_test_split(X_poly, y, random_state = 42)
poly_model = linear_model.LinearRegression().fit(X_polyTrain, y_polyTrain)
y_polyPredicted = poly_model.predict(X_polyTest)


# In[19]:


# Load real data
from sklearn.datasets import load_boston

boston_data = load_boston()
boston_data.keys()


# In[20]:


print(boston_data.DESCR)


# In[21]:


print(boston_data.feature_names)


# In[22]:


X_real = boston_data.data
y_real = boston_data.target
X_real_train, X_real_test, y_real_train, y_real_test = train_test_split(X_real, y_real, random_state = 42)


# In[28]:


# Plot impact of training data

def plot_learning_curves(model, X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 23)
    train_errors, test_errors = [], []
    for m in range(1, len(X_train)):
        model.fit(X_train[:m], y_train[:m])
        y_train_predict = model.predict(X_train[:m])
        y_test_predict = model.predict(X_test)
        train_errors.append(mean_squared_error(y_train_predict,y_train[:m])
        test_errors.append(mean_squared_error(y_test_predict,y_test))
                            
    plt.plot(numpy.sqrt(train_errors), "r-+", linewidth = 2, label = "train")
    plt.plot(numpy.sqrt(test_errors), "b-", linewidth = 3, label = "test")


# In[31]:


def plot_learning_curves(model, X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 23)
    train_errors, test_errors = [], []
    for m in range(1, len(X_train)):
        model.fit(X_train[:m], y_train[:m])
        y_train_predict = model.predict(X_train[:m])
        y_test_predict = model.predict(X_test)
        train_errors.append(mean_squared_error(y_train_predict,y_train[:m]))
        test_errors.append(mean_squared_error(y_test_predict,y_test))
                            
    plt.plot(numpy.sqrt(train_errors), "r-+", linewidth = 2, label = "train")
    plt.plot(numpy.sqrt(test_errors), "b-", linewidth = 3, label = "test")


# In[32]:


linear_reg_model = linear_model.LinearRegression()
plot_learning_curves(linear_reg_model, X_real, y_real)

