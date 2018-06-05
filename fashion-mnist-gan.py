from __future__ import print_function, division

from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.datasets import fashion_mnist

import matplotlib.pyplot as plt

import sys

import numpy as np




# coding: utf-8

# In[92]:

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# In[93]:


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


# x_train = x_train.astype('float32')
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



## First construct the discriminator 
# The discriminator is the network that classifies
# The input images provided by the generator
# It is a standard convolutional neural network.
def build_discriminator(img_shape):
    model = Sequential()
    model.add(Flatten(input_shape=img_shape))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(256))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.summary()

    img = Input(shape=img_shape)
    validity = model(img)

    return Model(img, validity)

# Builds the generator for the network
# This generator will alter the inputs to the 
# descriminator each training epoch
def build_generator(latent_dim, img_shape):
    model = Sequential()

    model.add(Dense(256, input_dim=latent_dim))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(1024))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(np.prod(img_shape), activation='tanh'))
    model.add(Reshape(img_shape))

    model.summary()

    noise = Input(shape=(latent_dim,))
    img = model(noise)

    return Model(noise, img)


def train(
    generator,
    descriminator,
    combined,
    epochs,
    batch_size=128, 
    sample_interval=50
):
    # Rescales all inputs to bbe on a 0 to 1 scale
    X_train = x_train / 127.5 - 1.
    X_train = np.expand_dims(x_train, axis=3)
    print(X_train)

    # Adversarial ground truths
    valid = np.ones((batch_size, 1))
    fake = np.zeros((batch_size, 1))

    for epoch in range(epochs):

        # ---------------------
        #  Train Discriminator
        # ---------------------

        # Select a random batch of images
        idx = np.random.randint(0, X_train.shape[0], batch_size)
        imgs = X_train[idx]

        noise = np.random.normal(0, 1, (batch_size, 100))

        # Generate a batch of new images
        gen_imgs = generator.predict(noise)

        # Train the discriminator
        d_loss_real = discriminator.train_on_batch(imgs, valid)
        d_loss_fake = discriminator.train_on_batch(gen_imgs, fake)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # ---------------------
        #  Train Generator
        # ---------------------

        noise = np.random.normal(0, 1, (batch_size, 100))

        # Train the generator (to have the discriminator label samples as valid)
        g_loss = combined.train_on_batch(noise, valid)

        # Plot the progress
        print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

        # If at save interval => save generated image samples
        if epoch % sample_interval == 0:
            sample_images(generator, epoch)


def sample_images(generator, epoch):
    r, c = 5, 5
    noise = np.random.normal(0, 1, (r * c, 100))
    gen_imgs = generator.predict(noise)

    # Rescale images 0 - 1
    gen_imgs = 0.5 * gen_imgs + 0.5

    fig, axs = plt.subplots(r, c)
    cnt = 0
    for i in range(r):
        for j in range(c):
            axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
            axs[i,j].axis('off')
            cnt += 1
    fig.savefig("images/%d.png" % epoch)
    plt.close()

### Generative Adversarial Network
## In a GAN, the main goal is for the 
# generative network to be ablbe to produce
# images that can be mistaken as real images.
img_rows = 28
img_cols = 28
channels = 1
img_shape = (img_rows, img_cols, channels)
latent_dim = 100

# Adam is an optimization method for 
# achieving best fit that works similarly to 
# stochastic gradient descent
optimizer = Adam(0.0002, 0.5)

# Build and compile the discriminator
discriminator = build_discriminator(img_shape)
discriminator.compile(loss='binary_crossentropy',optimizer=optimizer,metrics=['accuracy'])

# Build the generator
generator = build_generator(latent_dim, img_shape)

# The generator takes noise as input and generates imgs
z = Input(shape=(100,))
img = generator(z)

# For the combined model we will only train the generator
discriminator.trainable = False

# The discriminator takes generated images as input and determines validity
validity = discriminator(img)

# The combined model  (stacked generator and discriminator)
# Trains the generator to fool the discriminator
combined = Model(z, validity)
combined.compile(loss='binary_crossentropy', optimizer=optimizer)


train(
    generator,
    discriminator,
    combined,
    epochs=2,
    batch_size=128, 
    sample_interval=50
)
