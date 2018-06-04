# coding: utf-8
# In[1]:
from keras.datasets import fashion_mnist

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# In[2]:
print x_train.shape

# In[3]:
import keras;
print(keras.__version__);

# In[4]:
sample_size = 60000
sampleN_train = sample_size
sampleN_test = int(sample_size * .1)

x_train = x_train[0:sampleN_train]
x_test = x_test[0:sampleN_test]
y_train = y_train[0:sampleN_train]
y_test = y_test[0:sampleN_test]

# In[5]:
import numpy as np

np.random.seed(123)

# In[6]:
from matplotlib import pyplot as plt
# Uncomment in python editors so plot will display
plt.interactive(False)

plt.imshow(x_train[0])

# In[7]:
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32')
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32')

# In[8]:
x_train.shape

# In[9]:
x_train /= 255
x_test /= 255

# In[10]:
y_train[:10]

# In[11]:
from keras.utils import np_utils

y_train = np_utils.to_categorical(y_train, 10)
y_test = np_utils.to_categorical(y_test, 10)

# In[12]:
print y_train[:10]

# In[13]:
print y_test[:10]

# In[14]:
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

# Plot placeholder
figure = plt.figure()

for i, f in enumerate(activation_func):
    name = "my_model_%d_%s_%d_%d" % (sample_size, f, batch_size, nb_epoch)

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), strides=(1, 1), activation='f', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(64, (3, 3), activation=f))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(1000, activation=f))
    model.add(Dense(10, activation='softmax'))

    # Install pydot and graphiz to plot a flow chart of the layers
    # from keras.utils import plot_model
    # plot_model(model, to_file=concat(name,'.png'), show_shapes=True, show_layer_names=True)

    # In[108]:
    model.compile(loss=loss,
                  optimizer=optimizer,
                  metrics=metrics)
    # In[109]:
    history = model.fit(x_train, y_train,
                        batch_size=batch_size, nb_epoch=nb_epoch, verbose=1)
    # In[110]:
    score = model.evaluate(x_test, y_test, verbose=0)

    # In[111]:
    print "%s %.5f" % (f, score[1])

    # plot metrics
    figure.add_subplot(2, 4, i + 1)
    plt.plot(history.history['acc'])
    plt.title(f)

    # In[112]:
    save_as = name
    model.save(save_as)

# In[ ]:
plt.savefig('activation_all_32_20.png')
plt.show()


