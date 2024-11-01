#!/usr/bin/env python
# coding: utf-8

# In[ ]:


class SCRD(layers.Layer):
    def __init__(self,embed_dim=32):


        self.embed_dim = embed_dim
        self.proj = layers.Conv2D(filters=self.embed_dim,
                      kernel_size=1,
                      padding='SAME',
                      kernel_initializer=initializers.LecunNormal(),
                      bias_initializer=initializers.Zeros())
        super(SCRD, self).__init__()


    def call(self,inputs,embed_dim):
      xh=inputs[0]
      xl=inputs[1]
      B,H1,W1,C1=xl.shape
      B,H2,W2,C1=xh.shape
      k=H1//H2
      xh=self.proj(xh)
      xl=self.proj(xl)

      
      

      # B, HW, embed_dim ---->> B, embed_dim, HW 
      T = tf.reshape(xh, [-1,H2*W2,self.embed_dim])
      T = tf.transpose(T,[0,2,1])
      

      xl_1=MaxPool2D(pool_size=(k,k))(xl)
      x1_1=tf.reshape(xl_1,[-1,H2*W2,self.embed_dim])
      


      # (B, embed_dim, HW)*(B, HW, embed_dim)
      aff = tf.matmul(T, x1_1)
      aff = tf.nn.softmax(aff, axis=-1)



      # HW*T,T*embed_dim
      x = tf.reshape(xl,[-1,H1*W1,self.embed_dim])
      x = tf.matmul(x, aff)
      x = tf.reshape(x,[-1,H1,W1,self.embed_dim])
      return x

