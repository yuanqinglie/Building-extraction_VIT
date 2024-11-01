#!/usr/bin/env python
# coding: utf-8

# In[ ]:


class Attention_dsc(layers.Layer):
    k_ini = initializers.GlorotUniform()
    b_ini = initializers.Zeros()

    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=1,
                 attn_drop_ratio=0.,
                 proj_drop_ratio=0.,
                 name=None):
        super(Attention_dsc, self).__init__(name=name)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.kv = layers.Dense(dim * 2, use_bias=qkv_bias, name="qkv",
                                kernel_initializer=self.k_ini, bias_initializer=self.b_ini)
        self.attn_drop = layers.Dropout(attn_drop_ratio)
        self.proj = layers.Dense(dim, name="out",
                                 kernel_initializer=self.k_ini, bias_initializer=self.b_ini)
        self.proj_drop = layers.Dropout(proj_drop_ratio)

    def call(self, inputs, training=None):
        # [batch_size, num_patches + 1, total_embed_dim]
        x_q=inputs[0]

        B, N, C = inputs[1].shape
        B1,H,W,C1 = inputs[0].shape

        # qkv(): -> [batch_size, num_patches + 1, 3 * total_embed_dim]
        q=self.proj(x_q)
        q=tf.reshape(q, [-1, H*W, self.num_heads, C // self.num_heads])
        q=tf.transpose(q, [0, 2, 1, 3])


        kv = self.kv(inputs[1])
        # reshape: -> [batch_size, num_patches + 1, 2, num_heads, embed_dim_per_head]
        kv = tf.reshape(kv, [-1, N, 2, self.num_heads, C // self.num_heads])
        # transpose: -> [2, batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        kv = tf.transpose(kv, [2, 0, 3, 1, 4])
        # [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        k, v = kv[0], kv[1]


        # transpose: -> [batch_size, num_heads, embed_dim_per_head, num_patches + 1]
        # multiply -> [batch_size, num_heads, num_patches + 1, num_patches + 1]
        attn = tf.matmul(a=q, b=k, transpose_b=True) * self.scale
        attn = tf.nn.softmax(attn, axis=-1)
        attn = self.attn_drop(attn, training=training)

        # multiply -> [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        x = tf.matmul(attn, v)

        # transpose: -> [batch_size, num_patches + 1, num_heads, embed_dim_per_head]
        x = tf.transpose(x, [0, 2, 1, 3])

        # reshape: -> [batch_size, num_patches + 1, total_embed_dim]
        x = tf.reshape(x, [-1, H,W, C])
        x = x+x_q
        return x

