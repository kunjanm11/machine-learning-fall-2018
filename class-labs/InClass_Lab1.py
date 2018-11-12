
# coding: utf-8

# In[1]:


print("Hello world!")


# In[2]:


list(range(10))


# In[4]:


for n in range(10):
    print("The square of",n, "is",n*n)
print("done")


# In[7]:


def avg(x,y):
    print("first input is", x)
    print("second input is", y)
    a = (x+y)/2.0
    print("average is",a)
    return a


# In[6]:


avg(2,4)


# In[8]:


import numpy


# In[9]:


a = numpy.zeros([3,2])
print(a)


# In[10]:


a[0,0] = 1
a[0,1] = 2
a[1,0] = 3
a[2,1] = 9
print(a)


# In[11]:


v = a[1,0]
print v


# In[12]:


print(v)


# In[13]:


v = a[1,0]
print(v)


# In[14]:


import matplotlib.pyplot


# In[15]:


get_ipython().magic(u'matplotlib inline')


# In[16]:


matplotlib,pyplot.imshow(a)


# In[17]:


matplotlib.pyplot.imshow(a)


# In[18]:


x=numpy.linspace(-10,10,100)
y=numpy.sin(x)
matplotlib.pyplot.plot(x,y,marker="x")


# In[19]:


from sklearn.datasets import load_iris

iris_dataset = load_iris()


# In[20]:


iris_dataset


# In[21]:


iris_dataset.DESCRIPTION


# In[22]:


iris_dataset.DESCR


# In[23]:


dataset = iris_dataset.data


# In[24]:


print dataset


# In[25]:


print(dataset)


# In[26]:


list(iris_dataset.target_names)


# In[27]:


from sklearn.model_selection import train_test_split

y=iris_dataset.target
X_train, X_test, y_train, y_test = train_test_split(dataset,y,test_size=0.33,random_state=42)


# In[28]:


print(y_test)


# In[29]:


print(len(y_test))


# In[30]:


X_train, X_test, y_train, y_test = train_test_split(dataset,y,test_size=0.25,random_state=42)


# In[31]:


print(len(y_test))

