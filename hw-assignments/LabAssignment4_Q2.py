
# coding: utf-8

# In[2]:


# 2(a) Load MNIST dataset and split into train/test
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_mldata

mnist = fetch_mldata('MNIST original')
X, y = mnist["data"], mnist["target"]
X_small = X[:10000]
y_small = y[:10000]
X_train, X_test, y_train, y_test = train_test_split(X_small, y_small, train_size = 0.7, random_state = 42)


# In[3]:


# Standardization
from sklearn.preprocessing import StandardScaler

stdsc = StandardScaler()
stdsc.fit(X_train)
X_train_std = stdsc.transform(X_train)
X_test_std = stdsc.transform(X_test)


# In[5]:


def create_tuple(hidden_layer_count, nodes):
    t = ()
    for i in range(hidden_layer_count):
        t = t + (nodes,)
    return t


# In[6]:


# 2(b) Optimize hyperparameters for MLP
from sklearn.neural_network import MLPClassifier
hidden_layer_opt = 5
hidden_node_opt = 10
training_set_accuracy = []
test_set_accuracy = []
prev = 0
for n_hidden_layers in range (5, 33, 2):
    for n_hidden_nodes in range (10, 36, 2):
        mlp2 = MLPClassifier(activation = 'tanh', hidden_layer_sizes = create_tuple(n_hidden_layers,n_hidden_nodes), max_iter = 50, batch_size = 250, solver = 'adam')
        mlp2.fit(X_train_std, y_train)
        training_set_accuracy.append(mlp2.score(X_train_std, y_train))
        test_set_accuracy.append(mlp2.score(X_test_std, y_test))
        
        if(mlp2.score(X_test_std, y_test)>prev):
            hidden_layer_opt = n_hidden_layers
            hidden_node_opt = n_hidden_nodes
            prev = mlp2.score(X_test_std, y_test)
    
print("Optimum number of hidden layers: {:.2f}".format(hidden_layer_opt))
print("Optimum number of hidden nodes: {:.2f}".format(hidden_node_opt))


# In[7]:


max(test_set_accuracy)


# In[8]:


test_set_accuracy

