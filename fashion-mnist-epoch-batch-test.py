
# coding: utf-8

# **To run locally:**
# 1. `pip install keras`
# 2. `pip install tensorflow`
# 3. cd into folder and run `jupyter notebook`

# In[23]:


from keras.datasets import fashion_mnist

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()


# In[24]:


x_train.shape


# In[25]:


import keras; print(keras.__version__);


# In[26]:


import numpy as np
np.random.seed(123) 


# In[27]:


from matplotlib import pyplot as plt
plt.imshow(x_train[0])


# In[28]:


x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32')
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32')


# In[29]:


x_train.shape


# In[30]:


x_train /= 255
x_test /= 255


# In[31]:


y_train[:10]


# In[32]:


from keras.utils import np_utils
y_train = np_utils.to_categorical(y_train, 10)
y_test = np_utils.to_categorical(y_test, 10)


# In[33]:


# First 10 results
y_train[:10]


# In[34]:


from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D


# In[35]:


model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), strides=(1, 1), activation='relu', input_shape=(28,28,1)))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(1000, activation='relu'))
model.add(Dense(10, activation='softmax'))


# In[36]:


model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


# In[37]:


model.fit(x_train, y_train, 
          batch_size=32, nb_epoch=10, verbose=1)


# In[38]:


score = model.evaluate(x_test, y_test, verbose=1)


# In[39]:


score


# In[40]:


model.save('my_model.h5')


# # **Testing hyper parameters**

# In[41]:


from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier


# We will be testing batch size and epochs

# In[42]:


batch_size = [10, 32, 100]
epochs = [10, 15, 20]
param_grid = dict(batch_size=batch_size, epochs=epochs)


# In the interest of time, we will test hyper parameters on a small sample (10%)

# In[43]:


x_train_sample = x_train[:6000]
y_train_sample = y_train[:6000]


# create_model method to generate model

# In[44]:


def create_model():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), strides=(1, 1), activation='relu', input_shape=(28,28,1)))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(1000, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
    return model


# We create a grid of our hyper parameters and create models against all of them.

# In[107]:


model = KerasClassifier(build_fn=create_model, verbose=1)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1)
grid_result = grid.fit(x_train_sample, y_train_sample)


# In[111]:


print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))


# # Build model with batch size 32, epochs 20

# In[45]:


final_model = create_model()
final_model.compile(loss='categorical_crossentropy',
                    optimizer='adam',
                    metrics=['accuracy'])
final_model.fit(x_train, y_train,
                batch_size=32, nb_epoch=20, verbose=1)
final_score = final_model.evaluate(x_test, y_test, verbose=1)


# In[46]:


final_score

