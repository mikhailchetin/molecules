
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from scipy import stats
#import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from keras.models import Model
from keras.layers import Input, Dense, Dropout, BatchNormalization, LeakyReLU
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, CSVLogger



# In[2]:


data_train = pd.read_csv("data_train.csv")


# In[3]:


#data_train.head()


# In[4]:


data_test = pd.read_csv("data_test.csv")
#data_test.head()


# In[5]:


#print(data_test.shape)


# In[6]:


#print(data_train.shape)


# In[7]:


from keras import backend as K
from keras.losses import mean_absolute_error
import tensorflow as tf


# In[8]:


print(tf.__version__)


# In[9]:


def logloss(y_true, y_pred):
    return tf.log(mean_absolute_error(y_true,y_pred))


# In[10]:


features = set(data_train.columns).intersection(set(data_test.columns))
targets = set(data_train.columns).difference(set(data_test.columns))
features.remove('id')
features.remove('molecule_name')
features.remove('atom_index_1')
features.remove('atom_index_0')
features = sorted(features)
targets = sorted(targets)
print(features, len(features), len(data_test.columns))
print(targets, len(targets))
#exit()

# In[11]:


target_attr = targets.copy()
target_attr.remove('scalar_coupling_constant')


# In[12]:


data_features_train = data_train[list(features)]
data_targets_train = data_train[list(target_attr)]
data_features_test = data_test[list(features)]


# In[13]:


all_inputs = pd.concat([data_features_train, data_features_test])
input_data = StandardScaler().fit_transform(all_inputs)
input_data = pd.DataFrame(input_data, index=all_inputs.index, columns=all_inputs.columns)


# In[14]:


#data_test.columns


# In[15]:


output_data = StandardScaler().fit_transform(data_targets_train)
output_data = pd.DataFrame(output_data, index=data_targets_train.index, columns=data_targets_train.columns)
output_data['scalar_coupling_constant'] = data_train["scalar_coupling_constant"]


# In[16]:


#input_data.head()


# In[17]:


#output_data.head()


# Callbacks

# In[18]:


#output_data.columns


# In[ ]:


# Callback for Early Stopping... May want to raise the min_delta for small numbers of epochs
es = EarlyStopping(monitor='val_loss', min_delta=0.0005, patience=8,verbose=1, mode='auto')
# Callback for Reducing the Learning Rate... when the monitor levels out for 'patience' epochs, then the LR is reduced
rlr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,patience=7, min_lr=1e-6, mode='auto', verbose=1)
# Save the best value of the model for future use
sv_mod = ModelCheckpoint("best_model.hdf5", monitor='val_loss', save_best_only=True, period=1)


csv_logger = CSVLogger('log.csv', append=True, separator=';')
#model.fit(X_train, Y_train, callbacks=[csv_logger])

# In[ ]:


vars_in = Input(shape=(len(features),))
x = Dense(256)(vars_in)
x = BatchNormalization()(x)
x = LeakyReLU(alpha=0.05)(x)
x = Dropout(0.2)(x)
x = Dense(1024)(x)
x = BatchNormalization()(x)
x = LeakyReLU(alpha=0.05)(x)
x = Dropout(0.2)(x)
x = Dense(1024)(x)
x = BatchNormalization()(x)
x = LeakyReLU(alpha=0.05)(x)
x = Dropout(0.2)(x)
x = Dense(1024)(x)
x = BatchNormalization()(x)
x = LeakyReLU(alpha=0.05)(x)
x = Dropout(0.2)(x)
x = Dense(512)(x)
x = BatchNormalization()(x)
x = LeakyReLU(alpha=0.05)(x)
x = Dropout(0.2)(x)
x = Dense(128)(x)
x = BatchNormalization()(x)
x = LeakyReLU(alpha=0.05)(x)
x = Dropout(0.2)(x)
vars_out = Dense(len(targets), activation="linear")(x)
model = Model(inputs=vars_in, outputs=vars_out)
model.summary()
model.compile(loss="mae", optimizer='adam')
history = model.fit(np.array(input_data.iloc[:len(data_train),:]), np.array(output_data), validation_split=0.2, nb_epoch=200, shuffle=True, batch_size=1024, callbacks=[es, rlr, sv_mod,csv_logger])
model.save('model.h5')


# In[ ]:

for el in history.history['loss']:
    print(el)
print()
for el in history.history['val_loss']:
    print(el)
"""
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig("lossplog.png")
"""

# In[ ]:


#input_data.iloc[:len(data_train),:].head()


# In[ ]:


answers_train = model.predict(np.array(input_data.iloc[:len(data_train),:]), batch_size=1024)


# In[ ]:


answers = model.predict(np.array(input_data.iloc[len(data_train):,:]), batch_size=1024)


# In[ ]:




print(np.log(np.mean(np.abs(answers_train[:,27]-data_train['scalar_coupling_constant'].values))))


# In[ ]:


data_test["answer"] = answers[:,27]
data_test[['id','answer']].to_csv("submission_08062321.csv", index=False, header=['id', 'scalar_coupling_constant'])

