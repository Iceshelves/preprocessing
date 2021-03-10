#!/usr/bin/env python
# coding: utf-8

# # Read tiles to input format VAE network

# ### Imports
# Install tensorflow:
# ``%pip install tensorflow``

# In[ ]:


import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
tf.random.set_seed(2) 


# In[ ]:


latent_dim = 4
filter_1 = 3 #32
filter_2 = 2 #64
kernel_size = 5


# ### Create sampling layer

# In[3]:


class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


# ### Build encoder
# 

# In[4]:


def make_encoder():    
    #filter_1 = 3 #32
    #filter_2 = 2 #64
    #kernel_size = 5 #3
    dense_size = 16; 
    encoder_inputs = keras.Input(shape=(20, 20,3)) # enter cut-out shape (20,20,3)
    x = layers.Conv2D(filter_1, kernel_size, activation="relu", strides=2, padding="same")(encoder_inputs)
    x = layers.Conv2D(filter_2, kernel_size, activation="relu", strides=2, padding="same")(x)
    x = layers.Flatten()(x) # to vector
    x = layers.Dense(dense_size, activation="relu")(x) # linked layer
    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])
    encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
    encoder.summary()
    return encoder_inputs, encoder, z , z_mean, z_log_var


# ### Build decoder

# In[5]:


def make_decoder(): 
    latent_inputs = keras.Input(shape=(latent_dim,))
    x = layers.Dense(5 * 5 * filter_2, activation="relu")(latent_inputs) # -- shape corresponding to encoder
    x = layers.Reshape((5, 5, filter_2))(x)
    x = layers.Conv2DTranspose(filter_2, kernel_size, activation="relu", strides=2, padding="same")(x)
    x = layers.Conv2DTranspose(filter_1, kernel_size, activation="relu", strides=2, padding="same")(x)
    decoder_outputs = layers.Conv2DTranspose(3, 3, activation="sigmoid", padding="same")(x) # (1,3) or (3,3)
    decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
    decoder.summary()
    return decoder


# ## Define VAE as model
# With custom train_step

# Update: instead of defining VAE as class, use function-wise definition

# In[6]:


# Define VAE model.
def make_vae(encoder_inputs, z, z_mean, z_log_var, decoder):
    outputs = decoder(z)
    vae = tf.keras.Model(inputs=encoder_inputs, outputs=outputs, name="vae")

    # Add KL divergence regularization loss.
    reconstruction = decoder(z)
    reconstruction_loss = tf.reduce_mean(
        tf.reduce_sum(
            keras.losses.binary_crossentropy(encoder_inputs, reconstruction), axis=(1, 2)
                )
            )
    kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
    kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))

    alpha = 5

    # Play witht different alpha: -2, 0 , 1 ,2 ; 0.2 ; -0.5 ; 50
    # alpha = 10.; 
    total_loss = reconstruction_loss +  alpha * kl_loss # alpha is custom
    vae.add_loss(total_loss)
    return vae
    

