#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'notebook')


# In[2]:


import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import keras.backend as K
from tensorflow.keras.callbacks import EarlyStopping
from keras.regularizers import l2
early_stopping = EarlyStopping(patience=3, monitor='loss')
class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
       
        batch = tf.shape(z_mean)[0]
       
        dim = tf.shape(z_mean)[1]
        
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


# In[5]:


'TensorFlow version: ' + K.tf.__version__


# #### Dataset 

# In[7]:


omic_rows, omic_cols, omic_chns = x_train.shape[1:]


# ##### Constant definitions

# In[8]:


original_dim = omic_rows * omic_cols
latent_dim = 100
batch_size = 8
epochs = 100


# ## Model specification

# ### Encoder

# #### Inference network

# In[9]:


x = Input(shape=(original_dim,), name='x')
h = Dense(intermediate_dim, activation='relu', 
          name='hidden_enc')(x)
z_mu = Dense(latent_dim, name='mu')(h)
z_log_var = Dense(latent_dim, name='log_var')(h)
z_sigma = Lambda(lambda t: K.exp(.5*t), name='sigma')(z_log_var)


# In[ ]:



encoder_inputs = keras.Input(shape=((original_dim,)))
#x=layers.Dense(64,activation="tanh")(encoder_inputs)
x = layers.Conv2D(32, 3, activation="tanh", strides=(1,1), padding="same" )(encoder_inputs)
x = layers.Conv2D(64, 3, activation="tanh", strides=(1,1), padding="same", kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01))(x)
x=layers.BatchNormalization()(x)
x = layers.Flatten()(x)
x=layers.Dropout(0.5)(x)
x = layers.Dense(250, activation="tanh", kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01))(x)
z_mean = layers.Dense(latent_dim, name="z_mean")(x)
z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
z = Sampling()([z_mean, z_log_var])
encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
encoder.summary()


# In[ ]:


latent_inputs = keras.Input(shape=(latent_dim,))
x = layers.Dense(omic_rows*omic_cols*64, activation="tanh",  kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01))(latent_inputs)
x = layers.Reshape((omic_rows, omic_cols, 64))(x)
x = layers.Conv2DTranspose(32, 3 ,activation="tanh", strides=(1,1), padding="same")(x)
x = layers.Conv2DTranspose(250, 3, activation="tanh", strides=(1,1), padding="same")(x)
#x=layers.Dropout(0.5)(x)

decoder_outputs=layers.Conv2DTranspose(1, 3, activation="sigmoid", padding="same",kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01))(x)
decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
decoder.summary()


# In[ ]:


class CVAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(CVAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def train_step(self, data):
        if isinstance(data, tuple):
            data = data[0]
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = encoder(data)
            reconstruction = decoder(z)
            reconstruction_loss = tf.reduce_mean(
                keras.losses.binary_crossentropy(data, reconstruction)
            )
            reconstruction_loss *=  h*w      ##### change
            kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
            kl_loss = tf.reduce_mean(kl_loss)
            kl_loss *= -0.5
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return {
            "loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "kl_loss": kl_loss,
        }


# In[ ]:


cvae = CVAE(encoder, decoder)
cvae.compile(optimizer=keras.optimizers.SGD())
cvae.fit(x, epochs=epochs, batch_size=batch_size, callbacks=[early_stopping] )

