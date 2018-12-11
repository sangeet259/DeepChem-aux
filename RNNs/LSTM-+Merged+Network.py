
# coding: utf-8

# In[ ]:


# https://statcompute.wordpress.com/2017/01/08/an-example-of-merge-layer-in-keras/


# In[1]:


from sklearn.utils import class_weight

from sklearn.model_selection import train_test_split
import numpy as np
from keras.models import Sequential
from keras.layers import Dense,Merge,Dropout
from keras.layers import LSTM

# For the plots 
import matplotlib.pyplot as plt

# For controlling the training parameters in the midst of training itself
from keras.callbacks import ModelCheckpoint,EarlyStopping, ReduceLROnPlateau

# For creating directory to save the models
import os
from imblearn.over_sampling import SMOTE


# In[19]:


# Load the data
X = np.load("X (1).npy")
Y = np.load("Y (1).npy")

XT = np.column_stack( (X,Y) )
YT = []
for y in Y:
    if y==0:
        YT.append(0)
    else:
        YT.append(1)
#     elif y>0 and y<=50:
#         YT.append(1)
#     elif y>50 and y<=100:
#         YT.append(2)
#     elif y>100 and y<=150:
#         YT.append(3)
#     elif y>150 and y<=200:
#         YT.append(4)
        
YT = np.array(YT)


# In[20]:


# Split the test train and pass it into validation

# Split the test train and pass it into validation
from sklearn.utils import class_weight

X_train, X_test, Y_train, Y_test = train_test_split(XT, YT, test_size=0.2, random_state=12)
# class_weight = class_weight.compute_class_weight('balanced', np.unique(Y), Y)

sm = SMOTE(random_state=12, ratio = 1.0, k_neighbors = 5)
x_res, y_res = sm.fit_sample(X_train, Y_train)
# print (Y_train.value_counts(), np.bincount(y_res))
# weights = np.array(class_weight)


# In[21]:


X_train = x_res[:,:100]

X_train, X_test, Y_train, Y_test = train_test_split(XT, YT, test_size=0.2, random_state=12)


# In[22]:


# Reshape the stuff according to the the shape Keras expects it to be !
data_lstm = X_train[:,:100].reshape((X_train[:,:100].shape[0], 100, 1))
target = Y_train.reshape((Y_train.shape[0], 1))

val_data_lstm = X_test[:,:100].reshape(X_test[:,:100].shape[0],100,1)
val_target = Y_test.reshape(Y_test.shape[0],1)

# Reshape the stuff according to the the shape Keras expects it to be !
data_lstm = x_res[:,:100].reshape((x_res[:,:100].shape[0], 100, 1))
Y_train = x_res[:,101]
target = np.round(Y_train.reshape((Y_train.shape[0], 1)))
# target = Y_train

val_data_lstm = X_test[:,:100].reshape(X_test[:,:100].shape[0],100,1)
Y_test = X_test[:,101]
val_target = Y_test.reshape(Y_test.shape[0],1)
# val_target = Y_test



# In[23]:


data_lstm.shape


# In[24]:


data_mw = x_res[:,-1].reshape((x_res[:,-1].shape[0],1))
val_data_mw = X_test[:,-1].reshape((X_test[:,-1].shape[0],1))


# In[25]:


val_data_mw.shape


# In[26]:


max(target)


# In[29]:


# The standard model
# Here 50, the dimensionality of the output from LSTM is just a random no and has no suxh signifance, 
# That ^^ shuould be search via a hyperparameter search

branch1 = Sequential()

# The a<final output> will have 50 numbers 
branch1.add(LSTM(50, input_shape=(100, 1)))

branch2 = Sequential()
branch2.add(Dense(10,input_dim=1))
# Get those 50 numbers and put that into one output !
model =Sequential()
model.add(Merge([branch1, branch2], mode = 'concat'))
model.add(Dense(1,activation='tanh'))
branch1.add(Dropout(0.2))
model.add(Dense(1,activation='relu'))
model.add(Dense(1,activation='relu'))


# In[15]:


# Define the output for the saving the checkpoint (best models)

outputFolder = './output-lstm'
if not os.path.exists(outputFolder):
    os.makedirs(outputFolder)
filepath=outputFolder+"/Weights-{epoch:02d}-{val_acc:.2f}.hdf5"


# In[16]:


# The callback function for model checkpoint saving
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1,save_best_only=True,                             save_weights_only=True,mode='auto')

# The callback function for earlystopping

earlystop = EarlyStopping(monitor='val_loss',patience=10,                           verbose=1, mode='auto')

# The callback function for reducing learning rate on plateus

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                              patience=3, min_lr=0.00001)


# In[17]:


# Put all those calback in a list
#callbacks = [earlystop, checkpoint, reduce_lr]
callbacks = [checkpoint, reduce_lr]


# In[32]:


# I have decided to use mean_absolute_error , beacuse for some reason mean squared e
# model.compile(loss='mean_absolute_error', optimizer='adam',metrics=['accuracy'])
model.compile(loss='mean_squared_error', optimizer='adam',metrics=['accuracy'])


model_info = model.fit([data_lstm,data_mw], target, epochs=1000, batch_size=1000, verbose=2,validation_data=([val_data_lstm,val_data_mw],val_target))
                       #,callbacks=callbacks)


# In[24]:


def plot_model_history(model_history):
    fig, axs = plt.subplots(1,2,figsize=(15,5))
    # summarize history for accuracy
    axs[0].plot(range(1,len(model_history.history['acc'])+1),model_history.history['acc'])
    axs[0].plot(range(1,len(model_history.history['val_acc'])+1),model_history.history['val_acc'])
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].set_xticks(np.arange(1,len(model_history.history['acc'])+1),len(model_history.history['acc'])/10)
    axs[0].legend(['train', 'val'], loc='best')
    # summarize history for loss
    axs[1].plot(range(1,len(model_history.history['loss'])+1),model_history.history['loss'])
    axs[1].plot(range(1,len(model_history.history['val_loss'])+1),model_history.history['val_loss'])
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_xticks(np.arange(1,len(model_history.history['loss'])+1),len(model_history.history['loss'])/10)
    axs[1].legend(['train', 'val'], loc='best')
    plt.save('LSTM.jpg')


# In[25]:


plot_model_history(model_info)


# In[31]:


# fig, ax = plt.subplots(figsize=(20, 10))
# ax.scatter(val_target,np.round(model.predict([val_data_lstm,val_data_mw])), s= 10)
# plt.xlabel('Target solubility classes')
# plt.ylabel('Predicted solubility classes')
# plt.title('Performance of CNN Model')
# lims = [
#     np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
#     np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
# ]

# # now plot both limits against eachother
# ax.plot(lims, lims, 'k-', alpha=0.45, zorder=0)
# ax.set_aspect('equal')
# ax.set_xlim(lims)
# ax.set_ylim(lims)
# plt.show()


# # In[27]:


from sklearn.metrics import mean_squared_error
# mse = mean_squared_error(val_target, model.predict([val_data_lstm,val_data_mw]))
mse = mean_squared_error(val_target, np.round(model.predict([val_data_lstm,val_data_mw])))


# In[28]:


mse*19/200


# # In[29]:


# # mean_squared_error(target,model.predict([data_lstm,data_mw]))*19/200
# mean_squared_error(target,np.round(model.predict([data_lstm,data_mw])))*19/200


# # In[30]:


# import sklearn
# sklearn.metrics.precision_recall_fscore_support(val_target, np.round(model.predict([val_data_lstm,val_data_mw]))
#                                                ,average='weighted')

