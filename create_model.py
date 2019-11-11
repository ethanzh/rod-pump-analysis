#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
from os import listdir, path
from os.path import isfile, join
import logging
import matplotlib.pyplot as plt
import datetime


# In[3]:


file_name = "logs"

logging.basicConfig(
    filename=file_name,
    filemode='a',
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')


# In[3]:


# we only want to do this if there isn't a pickle present
if not path.exists("./df.pkl"):
    # get list of all csv file
    DIR_NAME = "rod_pump"
    file_names = [f for f in listdir(DIR_NAME) if isfile(join(DIR_NAME, f))]
    
    logging.info("Files: ")
    logging.info(file_names)
    
    # create a list of dataframes for each csv file in ./rod_pump
    # then concat to create master dataframe
    plt.figure(figsize=(20,10))
    df_list = []
    names_length = len(file_names)
    file_number = 1
    for file in file_names:
        df = pd.read_csv(f'{DIR_NAME}/{file}')
        df['Name'] = file.replace('.csv', '')
        
        logging.info(f"[{file_number}/{names_length}] {file}")

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
    df.to_pickle("./df.pkl")
else:
    df = pd.read_pickle("./df.pkl")


# In[4]:


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


# In[ ]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit_transform(X)


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


# In[ ]:


X_train


# In[ ]:


from keras import Sequential
from keras.layers import Dense

classifier = Sequential()

classifier.add(Dense(4, activation='relu', kernel_initializer='random_normal',
                     input_dim=6))
classifier.add(Dense(4, activation='relu', kernel_initializer='random_normal'))
classifier.add(Dense(1, activation='sigmoid', kernel_initializer='random_normal'))

classifier.compile(optimizer ='adam',loss='binary_crossentropy', metrics =['accuracy'])

classifier.fit(X_train, y_train, batch_size=32, epochs=100)

eval_model = classifier.evaluate(X_train, y_train)
eval_model


# In[ ]:


y_pred = classifier.predict(X_test)
y_pred = y_pred > 0.5


# In[ ]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
logging.info(cm)


# In[4]:


import time
logging.info("Saving model")
curr_time = int(time.time())
model_name = f"{curr_time}.h5"
classifier.save(f"./models/{model_name}")


# In[ ]:




