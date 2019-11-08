#!/usr/bin/env python
# coding: utf-8

# In[1]:


# get_ipython().system('git clone https://github.com/ethanzh/rod-pump-analysis.git')


# In[17]:


import pandas as pd
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
import logging
import datetime


# In[ ]:

file_name = "logs"

logging.basicConfig(
    filename=file_name,
    filemode='a',
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')


# In[3]:


# get list of all csv file
DIR_NAME = "rod_pump"
file_names = [f for f in listdir(DIR_NAME) if isfile(join(DIR_NAME, f))]

logging.info("Files: ")
logging.info(file_names)

# create a list of dataframes for each csv file in ./rod_pump
# then concat to create master dataframe
plt.figure(figsize=(20,10))
df_list = []
file_number = 1
names_length = len(file_names)
print("Begin parsing all files individually")
for file in file_names:
    logging.info(f"[{file_number}/{names_length}] {file}")
    df = pd.read_csv(f'{DIR_NAME}/{file}')
    df['Name'] = file.replace('.csv', '')

    END = len(df["Tubing Pressure (psi)"]) - 10
    diff = 22
    START = END - diff
    time = [x for x in range(0, diff)]
    plt.plot(time, df["Fluid Load (lbs)"][START:END], 
             label="Fluid Load", c='green', alpha=0.5)

    failing = []

    for i, row in df.iterrows():
      is_zero = not dict(df.loc[i])["Percent Run (%)"]
      fail = 1 if i > START and not is_zero else 0
      failing.append(fail)

    df["failing"] = failing

    df_list.append(df)

    file_number += 1
    
df = pd.concat(df_list)


# In[22]:


logging.info("Formatting df into X and y")

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


# In[23]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit_transform(X)


# In[24]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


# In[ ]:


from keras import Sequential
from keras.layers import Dense

logging.info("Begin training model")

classifier = Sequential()

classifier.add(Dense(4, activation='relu', kernel_initializer='random_normal',
                     input_dim=6))
classifier.add(Dense(4, activation='relu', kernel_initializer='random_normal'))
classifier.add(Dense(1, activation='sigmoid', kernel_initializer='random_normal'))

classifier.compile(optimizer ='adam',loss='binary_crossentropy', metrics =['accuracy'])

classifier.fit(X_train,y_train, batch_size=10, epochs=3)

model = classifier.evaluate(X_train, y_train)


# In[ ]:


y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.8)


# In[ ]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
logging.info(cm)


# In[ ]:


logging.info("Saving model")
curr_time = datetime.datetime()
model_name = f"{curr_time}.h5"
classifier.save(model_name)


# In[ ]:




