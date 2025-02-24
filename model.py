

from tensorflow.keras.layers import Input, ZeroPadding2D, Conv2D,Lambda,BatchNormalization, MaxPooling2D, UpSampling2D, Add, DepthwiseConv2D
    from tensorflow.keras.models import Model

    # input
    classes =2
    inputs = Input(shape=(512, 512, 3))
    x = ZeroPadding2D((3, 3))(inputs)

    # stage 1
    x = Conv2d_BN(x, 64, 7, strides=(2, 2), padding='valid')
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    # stage 2
    x = bottleneck_Block(x, 128, strides=(1, 1), with_conv_shortcut=True)
    x = bottleneck_Block(x, 128)
    x = bottleneck_Block(x, 128)
    fuse2=x
    x = Conv2d_BN(x, 192, 1, strides=(1, 1), padding='same')



    # stage 3
    x = Attention_Block(x,embed_dim=192,num_heads=6)
    x = Attention_Block(x,embed_dim=192,num_heads=6)
    fuse3=x
    x = Conv2d_BN(x, 384, 1, strides=(1, 1), padding='valid')
    x = DepthwiseConv2D(kernel_size=3,strides=2, padding='same')(x)


    # stage 4
    for i in range(10):
      x = Attention_Block(x,embed_dim=384,num_heads=12)
    fuse4=x
    x = Conv2d_BN(x, 768, 1, strides=(1, 1), padding='valid')
    x = DepthwiseConv2D(kernel_size=3,strides=2, padding='same')(x)


    # stage 5
    x = Attention_Block(x,embed_dim=768,num_heads=24)
    x = Attention_Block(x,embed_dim=768,num_heads=24)
    fuse5=x

    # Decoder

    # fuse5+ fuse4
    upsample_fuse5 = UpSampling2D(size=(2, 2), interpolation='bilinear')(fuse5)
    conv_upsample_fuse5 = Conv2d_BN(upsample_fuse5, 384, 1, strides=(1, 1), padding='same')
    fused4_5 = Add()([conv_upsample_fuse5, fuse4])
    fused4_5 =Conv2d_BN(fused4_5, 192, 1, strides=(1, 1), padding='same')

    # fuse2+ fuse3
    _fuse2 = Conv2d_BN(fuse2, 192, 1, strides=(1, 1), padding='same')
    fused3_2 = Add()([_fuse2, fuse3])
    scrd= SCRD()
    output=scrd([fused4_5,fused3_2],embed_dim=32)

    output = UpSampling2D(size=(4, 4), interpolation='bilinear')(output)
    output = Conv2D(classes, 3, use_bias=False, padding='same',kernel_initializer='he_normal')(output)



    model=Model(inputs=inputs, outputs=output)
