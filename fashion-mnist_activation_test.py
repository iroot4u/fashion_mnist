# coding: utf-8
# In[92]:

from keras.datasets import fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train.shape

# In[94]:
import keras; print(keras.__version__);

# In[95]:
import numpy as np
np.random.seed(123) 

# In[96]:
from matplotlib import pyplot as plt
plt.imshow(x_train[0])

# In[97]:
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32')
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32')

# In[98]:
#  x_train = x_train.astype('float32')
# x_test = x_test.astype('float32')

# In[99]:
x_train.shape

# In[100]:
x_train /= 255
x_test /= 255

# In[101]:
# y_train = y_train.astype('str')
# y_test = y_test.astype('str')

# In[102]:
y_train[:10]

# In[103]:
from keras.utils import np_utils
y_train = np_utils.to_categorical(y_train, 10)
y_test = np_utils.to_categorical(y_test, 10)

# In[104]:
y_train[:10]

# In[105]:
y_train[:10]

# In[106]:
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D

# In[107]:
activation_func = ["relu", "sigmoid"]
figure = plt.figure()

for i, f in enumerate(activation_func):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), strides=(1, 1), activation='relu', input_shape=(28,28,1)))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(64, (3, 3), activation=activation_func))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(1000, activation=activation_func))
    model.add(Dense(10, activation='softmax'))

    # In[108]:
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    # In[109]:
    history = model.fit(x_train, y_train,
              batch_size=32, nb_epoch=2, verbose=1)
    # In[110]:
    score = model.evaluate(x_test, y_test, verbose=0)

    # In[111]:
    print "%s %.5f" % (activation_func, score)

    # plot metrics
    figure.add_subplot(2, 1, i+1)
    plt.plot(history.history['acc'])
    plt.show()
    plt.title(activation_func)

    # In[112]:
    save_as = "my_model_%s" % activation_func
    model.save(save_as)

# figure = plt.figure()
# for i, f in enumerate(actFuncs):
#     figure.add_subplot(3, 2, i+1)
#     out=VisualActivation(activationFunc=f, plot=False)
#     plt.plot(out.index, out.Activated)
#     plt.title(f)