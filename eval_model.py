#!/usr/bin/env python
# coding: utf-8

# In[14]:


import pandas as pd
from keras.models import load_model
from os import listdir


# In[22]:


# get most recent model (highest epoch time)
file_name = sorted(listdir("./models/"), reverse=True)[0]
model = load_model(f"./models/{file_name}")
df = pd.read_pickle("df.pkl")


# In[23]:


# create copy, remove misc. features from X
X = df.copy(deep=True)
del X["Percent Run (%)"]
del X["time (hours)"]
del X["Name"]

# remove all rows where the rod pump is 'dead'
X = X[X["Casing Pressure (psi)"] > 0]

# move response series into y
y = X["failing"]
del X["failing"]

X = X.iloc[:]
y = y.iloc[:]


# In[24]:


score = model.evaluate(X, y, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))

