
# coding: utf-8

# In[1]:


# 1(a)

import numpy
numSamples = 1200
X = numpy.random.rand(numSamples,1)
y = 4 + 3*X*X*X + numpy.random.rand(numSamples,1)


# In[2]:


import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')
plt.plot(X, y, "b.")
plt.show()


# In[6]:


# 1(b)

from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_squared_error


def plot_learning_curves(model, X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.7, random_state = 23)
    train_errors, test_errors = [], []
    for m in range(1, len(X_train), 84):
        model.fit(X_train[:m], y_train[:m])
        y_train_predict = model.predict(X_train[:m])
        y_test_predict = model.predict(X_test)
        train_errors.append(mean_squared_error(y_train_predict,y_train[:m]))
        test_errors.append(mean_squared_error(y_test_predict,y_test))
        
    plt.plot(numpy.sqrt(train_errors), "r-+", linewidth = 2, label = "train")
    plt.plot(numpy.sqrt(test_errors), "b-", linewidth = 3, label = "test")
    


# In[7]:


linear_reg_model = linear_model.LinearRegression()
plot_learning_curves(linear_reg_model, X, y)


# In[8]:


# Train polynomial features for model 
from sklearn.preprocessing import PolynomialFeatures 

poly_features = PolynomialFeatures(degree = 4, include_bias = False)
X_poly = poly_features.fit_transform(X)
print(X.shape)
print(X_poly.shape)

plot_learning_curves(linear_reg_model, X_poly, y)


# In[9]:


# 2(a)

from sklearn.datasets import load_diabetes

diabetes_data = load_diabetes()
diabetes_data.keys()

X_real = diabetes_data.data
y_real = diabetes_data.target
X_real_train, X_real_test, y_real_train, y_real_test = train_test_split(X_real, y_real, train_size = 0.7, random_state = 42)


# In[18]:


diabetes_data.DESCR


# In[10]:


from sklearn.metrics import mean_squared_error

lr_model_real = linear_model.LinearRegression().fit(X_real_train, y_real_train)


# In[11]:


#Linear model

y_real_predicted = lr_model_real.predict(X_real_test)

mean_squared_error(y_real_test, y_real_predicted)


# In[20]:


print("Slope/Coefficient: ", lr_model_real.coef_)
print("Intercept: ", lr_model_real.intercept_)


# In[12]:


#Ridge model
ridge_model = linear_model.Ridge(alpha = 0.3)
ridge_model.fit(X_real_train, y_real_train)
ridge_predicted = ridge_model.predict(X_real_test)
mean_squared_error(y_real_test, ridge_predicted)


# In[13]:


#Lasso model

lasso_model = linear_model.Lasso(alpha = 0.4)
lasso_model.fit(X_real_train, y_real_train)
lasso_predicted = lasso_model.predict(X_real_test)
mean_squared_error(y_real_test, lasso_predicted)


# In[14]:


#Polynomial model 
from sklearn.preprocessing import PolynomialFeatures 

poly_features_real = PolynomialFeatures(degree = 4, include_bias = False)
X_real_poly = poly_features_real.fit_transform(X_real)
print(X_real.shape)
print(X_real_poly.shape)


# In[15]:


X_real_polyTrain, X_real_polyTest, y_real_polyTrain, y_real_polyTest = train_test_split(X_real_poly, y_real, train_size = 0.7, random_state = 42)
poly_model_real = linear_model.LinearRegression().fit(X_real_polyTrain, y_real_polyTrain)
y_real_polyPredicted = poly_model_real.predict(X_real_polyTest)
mean_squared_error(y_real_polyTest, y_real_polyPredicted)


# In[16]:


# 3(b)

array = numpy.arange(0.0, 1.0, 0.1)
for a in range (0,10):
        the_model = linear_model.Ridge(alpha = array[a])
        the_model.fit(X_real_train, y_real_train)
        the_predicted = the_model.predict(X_real_train)
        plt.plot(array[a], mean_squared_error(y_real_train, the_predicted), "b+")
        
array = numpy.arange(0.0, 1.0, 0.1)
for a in range (0,10):
        the_model = linear_model.Lasso(alpha = array[a])
        the_model.fit(X_real_train, y_real_train)
        the_predicted = the_model.predict(X_real_train)
        plt.plot(array[a], mean_squared_error(y_real_train, the_predicted), "r+")

