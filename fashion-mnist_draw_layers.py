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
import matplotlib.pyplot as plt
plt.interactive(False)

# plt.imshow(x_train[0])

# In[97]:
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32')
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32')

# In[98]:
# x_train = x_train.astype('float32')
# x_test = x_test.astype('float32')

# In[99]:
print x_train.shape

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
print y_train[:10]

# In[105]:
print y_train[:10]

# In[106]:
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D

# # In[107]:
# model = Sequential()
# model.add(Conv2D(32, kernel_size=(3, 3), strides=(1, 1), activation='relu', input_shape=(28,28,1)))
# model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
# model.add(Dropout(0.25))
# model.add(Conv2D(64, (3, 3), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.5))
# model.add(Flatten())
# model.add(Dense(1000, activation='relu'))
# model.add(Dense(10, activation='softmax'))
#
# # In[108]:
# model.compile(loss='categorical_crossentropy',
#               optimizer='adam',
#               metrics=['accuracy'])
# # In[109]:
# history = model.fit(x_train, y_train,
#           batch_size=32, nb_epoch=1, verbose=1)
#
# # In[110]:
# score = model.evaluate(x_test, y_test, verbose=0)
#
# # In[111]:
# print "%s %.5f" % ("relu", score)
#
# # plot metrics
# figure.add_subplot(2, 1, i+1)
# plt.plot(history.history['acc'])
# plt.title(f)
# plt.show()
#
# # In[112]:
# save_as = "my_model_%s" % "relu"
# model.save(save_as)

# figure = plt.figure()
# for i, f in enumerate(actFuncs):
#     figure.add_subplot(3, 2, i+1)
#     out=VisualActivation(activationFunc=f, plot=False)
#     plt.plot(out.index, out.Activated)
#     plt.title(f)

# ------------------
from keras.models import load_model
model = load_model('/Users/cookie/Documents/Class/Spark/Final/my_model_relu.h5')

print x_train[0].shape
input = x_train[0]
img_batch = np.expand_dims(input, axis=0).astype('float32')
print img_batch.shape

conv_img = model.predict(img_batch)
print conv_img.shape

def visualize_img(img_batch):
    img = np.squeeze(img_batch, axis=0)
    print img.shape
    plt.imshow(img)

visualize_img(conv_img)