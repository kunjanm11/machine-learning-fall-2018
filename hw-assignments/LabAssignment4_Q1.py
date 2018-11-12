
# coding: utf-8

# In[1]:


# 1(a) VizWiz dataset
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


import urllib.request, json 

y = []
with urllib.request.urlopen(annFile) as url:
    data = json.loads(url.read().decode())
    for vq in data[0:3]:
        image_name = vq['image']
        question = vq['question']
        label = vq['answerable']
        y.append(label)
        image_url = '%s/%s'%(imgDir,image_name)
        print(image_name)
        print(question)
        print(label)
        print(image_url)



# In[4]:


# Extract text features using Microsoft Azure

subscription_key_text = 'f8256b1d8bcf46beaee69bccb94eb052'
text_analytics_base_url = 'https://eastus.api.cognitive.microsoft.com/text/analytics/v2.0/'

language_api_url = text_analytics_base_url + 'languages'
sentiment_api_url = text_analytics_base_url + 'sentiment'
key_phrase_api_url = text_analytics_base_url + 'keyPhrases'


# In[9]:


from pprint import pprint

text_features = []
for vq in data[0:3]:
    documents = {"documents":[
        {"id": vq['image'], "text": vq['question']},
    ]}

    headers = {"Ocp-Apim-Subscription-Key": subscription_key_text}
    response = requests.post(key_phrase_api_url, headers = headers, json = documents)
    json_response = response.json()
    text_features.append(json_response)
#pprint(text_features)

keyPhrases_list = text_features[0]['documents'][0]['keyPhrases']
print(keyPhrases_list)


# In[30]:


def text_json_response(vq):
    documents = {"documents":[
        {"id": vq['image'], "text": vq['question']},
    ]}
    headers = {"Ocp-Apim-Subscription-Key": subscription_key_text}
    response = requests.post(key_phrase_api_url, headers = headers, json = documents)
    json_response = response.json()
    return json_response



# In[10]:


text_features


# In[38]:


import matplotlib.pyplot as plt
from skimage import io

get_ipython().magic(u'matplotlib inline')


# In[39]:


subscription_key = '96ca6899fd3042558befb4ce4542760a'

vision_base_url = 'https://eastus.api.cognitive.microsoft.com/vision/v1.0'
vision_analyze_url = vision_base_url + '/analyze?'


# In[40]:


# Evaluate an image using microsoft Vision API

def analyze_image(image_url):

    
    # Microsoft API headers, params, etc.
    headers = {'Ocp-Apim-Subscription-key':subscription_key}
    params = {'visualfeatures': 'Adult, Categories, Description, Color, Faces, ImageType, Tags'}
    data = {'url':image_url}
    
    # send request, get API response
    response = requests.post(vision_analyze_url, headers = headers, params = params, json = data)
    response.raise_for_status()
    analysis = response.json()
    return analysis


# In[41]:


import pprint
import urllib.request, json 

def extract_features(data):
    return{
        "description": data["description"],
        "tags": data["tags"],
        "image_format": data["metadata"]["format"],
        #"image_dimensions": str(data["metadata"]["width"]) + "x" + str(data["metadata"]["height"]) + "y",
        #"clip_art_type": data["imageType"]["clipArtType"],
        #"line_drawing_type": data["imageType"]["lineDrawingType"],
        "black_and_white": data["color"]["isBwImg"],
        "adult_content": data["adult"]["adultScore"],
        "racy": data["adult"]["isRacyContent"],
        #"racy_score": data["adult"]["racyScore"],
        "categories": data["categories"],
        "faces": data["faces"],
        "dominant_color_background": data["color"]["dominantColorBackground"],
        "dominant_color_foreground": data["color"]["dominantColorForeground"],
        "accent_color": data["color"]["accentColor"]
    }

image_features = []



# In[42]:


with urllib.request.urlopen(annFile) as url:
    data = json.loads(url.read().decode())
    for vq in data[0:3]:
        image_name = vq['image']
        image_url = '%s/%s'%(imgDir,image_name)
        image_details = analyze_image(image_url)
        features = extract_features(image_details)
        image_features.append(features)
        
tags_list = image_features[0]['description']['tags']


# In[54]:


all_features_list = []
for vq in data[0:3]:
    text_response = text_json_response(vq)
    keyPhrases = text_response['documents'][0]['keyPhrases']
   # print(keyPhrases)
    
    image_name = vq['image']
    image_url = '%s/%s'%(imgDir,image_name)
    image_details = analyze_image(image_url)
    features = extract_features(image_details)
    image_features = features['description']['tags']
   # print(image_features)
    
    all_features = keyPhrases + image_features
    all_features_list.append(all_features)
    print(all_features_list)


# In[45]:


all_features_array = np.asarray(all_features)
print(all_features_array)


# In[16]:


import numpy as np

#image_array = np.array(list(features.items()))
text_array = np.asarray(keyPhrases_list)
image_array = np.asarray(tags_list)


# In[17]:


image_array


# In[18]:


combined_list = []
for i in range(0,3):
    tags_list = image_features[i]['description']['tags']
    keyPhrases_list = text_features[i]['documents'][0]['keyPhrases']
    #print(tags_list+keyPhrases_list)
    combined_list.append(tags_list+keyPhrases_list)
    
combined_array = np.asarray(combined_list)
print(combined_array)


# In[55]:


# Bag-of-words model
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

for i in range(0,3):
    count = CountVectorizer()
    bag = count.fit_transform(all_features_list[i]) # create vocabulary
    print(count.vocabulary_)


# In[49]:


# TF-IDF model
from sklearn.feature_extraction.text import TfidfTransformer

tfidf = TfidfTransformer(use_idf = True,
                        norm = 'l2',
                        smooth_idf = True)
np.set_printoptions(precision = 2)
result = tfidf.fit_transform(count.fit_transform(all_features_list[0]))
print(result.toarray())


# In[51]:


# TF-IDF model
from sklearn.feature_extraction.text import TfidfTransformer
from array import array

tfidf = TfidfTransformer(use_idf = True,
                        norm = 'l2',
                        smooth_idf = True)
np.set_printoptions(precision = 2)
#result = np.zeros(3)
result_list_tfidf = []

allArrays = ()
for i in range(0,3):
    result = tfidf.fit_transform(count.fit_transform(all_features_list[0]))
    allArrays = allArrays+(result.toarray(),)
    #result.toarray()
    result_list_tfidf.append(result.toarray())
    #allArrays = np.concatenate(allArrays,result.toarray())
#print(result.toarray())

allArrays = np.asarray(allArrays)
#print(allArrays)


# In[53]:


#print(result.toarray())


# In[22]:


allArrays.shape


# In[23]:


#result_list_tfidf


# In[50]:


#X = [result_list_tfidf]
#print(X)


# In[25]:


y = np.asarray(y)
y.shape


# In[26]:


# Dataset 1: KNN classifer
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score

naive_bayes_clf = GaussianNB()
fold_accuracies = cross_val_score(naive_bayes_clf, allArrays[0:3], y[0:3])
print("Cross-validation score:\n{}".format(fold_accuracies))
print("Average cross-validation score: {:2f}".format(fold_accuracies.mean()))

