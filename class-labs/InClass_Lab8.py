
# coding: utf-8

# In[1]:


# Load dataset and split into train/test
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state = 1)


# In[2]:


# Standardization
from sklearn.preprocessing import StandardScaler

stdsc = StandardScaler()
stdsc.fit(X_train)
X_train_std = stdsc.transform(X_train)
X_test_std = stdsc.transform(X_test)


# In[4]:


# Multilayer perceptron
from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier(random_state = 42)
mlp.fit(X_train_std, y_train)
print("Accuracy on training set: {:.2f}".format(mlp.score(X_train_std, y_train)))
print("Accuracy = test set: {:.2f}".format(mlp.score(X_test_std, y_test)))


# In[6]:


# Neural network with 2 hidden layers and 10 hidden unit per layer
n_hidden_nodes = 10
mlp2 = MLPClassifier(activation = 'tanh', hidden_layer_sizes = [n_hidden_nodes, n_hidden_nodes])
mlp2.fit(X_train_std, y_train)
print("Accuracy on training set: {:.2f}".format(mlp2.score(X_train_std, y_train)))
print("Accuracy = test set: {:.2f}".format(mlp2.score(X_test_std, y_test)))


# In[2]:


# Load the MNIST dataset and split into training/testing tests
import pandas
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data")
X_train = mnist.train.images
X_test = mnist.test.images
y_train = mnist.train.labels.astype("int")
y_test = mnist.test.labels.astype("int")


# In[6]:


# Need to set Kernel to Python 3
import pandas
from tensorflow.contrib.learn import DNNClassifier
from tensorflow.contrib.learn import SKCompat
#from tensorflow.contrib.RunConfig import RunConfig
from tensorflow.contrib.learn import infer_real_valued_columns_from_input

#config = RunConfig(tf_random_seed = 42)

# Extracting features from the training data
feature_columns = infer_real_valued_columns_from_input(X_train)

# Create the DNN with two hidden layers (300 neurons and 100 neurons)
dnn_clf = DNNClassifier(hidden_units = [300,100],
                       n_classes = 10,
                       feature_columns = feature_columns)
                       #config = config)

#Wrapper
dnn_clf = SKCompat(dnn_clf)

# Train DNN with mini-batch descent
dnn_clf.fit(X_train, y_train, batch_size = 64, steps = 5000)


# In[1]:


# VizWiz daatset
import os
import json
from pprint import pprint


# In[2]:


import requests

base_url = 'https://ivc.ischool.utexas.edu/VizWiz/data'
split = 'train'
annFile = '%s/Annotations/%s.json'%(base_url, split)
imgDir = '%s/Images' %base_url
print(annFile)
print(imgDir)


# In[3]:


annotations = requests.get(annFile)


# In[4]:


fileToRead = '%s.json' %split
with open(fileToRead) as data:
    labels = json.load(data)
    for vq in labels[0:9]:
        image_name = vq['image']
        question = vq['question']
        label = vq['answerable']
        print(image_name)
        print(question)
        print(label)

