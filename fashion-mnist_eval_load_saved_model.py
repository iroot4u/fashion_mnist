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

#plt.imshow(x_train[0])

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
classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Define Variable parameters for test
activation_func = [
                   # 'softmax',
                   # 'softplus',
                   # 'softsign',
                    'relu',
                   # 'tanh',
                   # 'sigmoid',
                   # 'hard_sigmoid',
                   # 'linear'
                  ]

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
    filepath = '/Users/cookie/Documents/Class/Spark/Final/fashion_mnist/models/%s' % name
    model.load_weights(filepath)

    # EVALUATE MODELS
    # In[110]:
    score = model.evaluate(x_test, y_test, verbose=0)

    # Print out accuracy. Change to score[0] for loss.
    print "%s %.5f" % (f, score[1])

    # Could add some code to predict on a new sample
    # model.precict(new_sample)

    # Print visual of model layers
    #import keras_sequential_ascii as ksq
    #ksq.sequential_model_to_ascii_printout(cnn_n)
    from keras_sequential_ascii import keras2ascii
    keras2ascii(model)

    # Confusion Matrix
    from sklearn.metrics import classification_report, confusion_matrix

    Y_pred = model.predict(x_test, verbose=2)
    y_pred = np.argmax(Y_pred, axis=1)

    for ix in range(10):
        print(ix, confusion_matrix(np.argmax(y_test, axis=1), y_pred)[ix].sum())
    cm = confusion_matrix(np.argmax(y_test, axis=1), y_pred)
    print(cm)

    # Visualizing of confusion matrix
    import seaborn as sn
    import pandas as pd

    df_cm = pd.DataFrame(cm, range(10),
                         range(10))
                         #columns=['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'])
    plt.figure(figsize=(10, 7))
    sn.set(font_scale=1.4)  # for label size
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 12}, xticklabels=True, yticklabels=True)  # font size
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title(f)
    plt.show()