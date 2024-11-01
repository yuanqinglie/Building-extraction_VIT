#!/usr/bin/env python
# coding: utf-8

# In[ ]:



from tensorflow.keras import Input
from tensorflow.keras.layers import concatenate, MaxPool2D
import tensorflow as tf
from tensorflow.keras import Model, layers, initializers
import numpy as np
class DSC(layers.Layer):
    """
    2D Features to multiscale Embedding
    """
    def __init__(self, embed_dim=768):
        super(DSC, self).__init__()
        self.embed_dim = embed_dim

        self.proj = layers.Conv2D(filters=embed_dim,
                      kernel_size=1,
                      padding='SAME',
                      kernel_initializer=initializers.LecunNormal(),
                      bias_initializer=initializers.Zeros())


    def call(self, inputs,scale):

        x1 = self.proj(inputs)

        X=[]
        for i in range(scale):
          k=2*i+3

          x = layers.DepthwiseConv2D(kernel_size=k,
               strides=k, padding='valid')(x1)
          # [B, H, W, C] -> [B, H*W, C]
          x = tf.reshape(x, [tf.shape(x)[0], tf.shape(x)[1]*tf.shape(x)[2], self.embed_dim])
          X.append(x)
        x=concatenate(X, axis=1)

        return x

