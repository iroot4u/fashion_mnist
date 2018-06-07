# coding: utf-8
from keras.datasets import fashion_mnist

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

print x_train.shape

import keras;
print(keras.__version__);

sample_size = 60000
# sampleN_train = sample_size
# sampleN_test = int(sample_size * .1)

# x_train = x_train[0:sampleN_train]
# x_test = x_test[0:sampleN_test]
# y_train = y_train[0:sampleN_train]
# y_test = y_test[0:sampleN_test]

import numpy as np

np.random.seed(123)

from matplotlib import pyplot as plt
# Uncomment in python editors so plot will display
plt.interactive(False)

plt.imshow(x_train[0])

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32')
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32')

x_train.shape

x_train /= 255
x_test /= 255

y_train[:10]

from keras.utils import np_utils

y_train = np_utils.to_categorical(y_train, 10)
y_test = np_utils.to_categorical(y_test, 10)

print y_train[:10]
print y_test[:10]

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D

# In[17]:
# Define contant parameters for this test
batch_size = 32
nb_epoch = 20
optimizer = 'adam'
metrics = ['accuracy']
loss = 'categorical_crossentropy'

# Define Variable parameters for test
activation_func = ['softmax', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear']

# filepath = ['fashion_mnist/my_model_60000_hard_sigmoid_32_20',
#             'fashion_mnist/my_model_60000_linear_32_20',
#             'fashion_mnist/my_model_60000_relu_32_20',
#             'fashion_mnist/my_model_60000_sigmoid_32_20',
#             '/fashion_mnist/my_model_60000_softmax_32_20',
#             'fashion_mnist/my_model_60000_softplus_32_20',
#             'fashion_mnist/my_model_60000_softsign_32_20',
#             'fashion_mnist/my_model_60000_tanh_32_20']

for i, f in enumerate(activation_func):
    name = "my_model_%d_%s_%d_%d" % (sample_size, f, batch_size, nb_epoch)

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), strides=(1, 1), activation=f, input_shape=(28, 28, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(64, (3, 3), activation=f))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(1000, activation=f))
    model.add(Dense(10, activation='softmax'))

    # In[108]:
    model.compile(loss=loss,
                  optimizer=optimizer,
                  metrics=metrics)

    # LOAD SAVED MODEL(S) WEIGHTS AND APPLY TO COMPILED .model
    filepath = 'fashion_mnist/%s' % name
    model.load_weights(filepath)


    # EVALUATE MODELS
    # In[110]:
    score = model.evaluate(x_test, y_test, verbose=0)

    # Print out accuracy. Change to score[0] for loss.
    print "%s %.5f" % (f, score[1])

    # Could add some code to predict on a new sample
    # model.precict(new_sample)


