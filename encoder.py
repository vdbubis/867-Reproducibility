import tensorflow.keras as kl
from tensorflow.keras import layers

class ConvNormActBlock(layers.Layer):

    def __init__(self, 
                 filters, 
                 kernel_size, 
                 strides=1, 
                 groups=1,
                 data_format="channels_last",
                 dilation_rate=1, 
                 kernel_initializer="he_normal",
                 trainable=True,
                 use_bias=False,
                 normalization=dict(normalization="batch_norm", axis=-1, trainable=True),
                 name=None):
        super(ConvNormActBlock, self).__init__(name=name)
        # if strides != (1, 1):
        self.strides = strides
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)

        if isinstance(strides, int):
            strides = (strides, strides)

        if isinstance(dilation_rate, int):
            dilation_rate = (dilation_rate, dilation_rate)

        if strides == (1, 1):
            p =1
            padding = "same"
        else:
            p = ((kernel_size[0] - 1) // 2 * dilation_rate[0], (kernel_size[1] - 1) * dilation_rate[1] // 2)
            self.pad = layers.ZeroPadding2D(p, data_format=data_format)
            padding = "valid"
        
        self.conv = layers.Conv2D(filters=filters,
                                           kernel_size=kernel_size, 
                                           strides=strides, 
                                           padding="same",
                                           data_format=data_format, 
                                           dilation_rate=dilation_rate, 
                                           groups=groups,
                                           use_bias=normalization is None or use_bias, 
                                           trainable=trainable,
                                           kernel_initializer=kernel_initializer,
                                           name="conv2d")




#dilated encoder
def encoder(input_shapes,
                   dilation_rates,
                   filters=512,
                   midfilters=128,
                   normalization=dict(normalization="batch_norm", momentum=0.03, epsilon=1e-3, axis=-1, trainable=True),
                   activation=dict(activation="ReLU"),
                   kernel_initializer="he_normal",
                   data_format="channels_last",
                   name="dilated_encoder"):

    # input channel size | if use darknet53 as the backbone ,the input is 1024 channels | if use resnet50 as the backbone, the input is 2048 channels
    inputs = kl.Input(shape=input_shapes)


    # projection layer 1
    x = ConvNormActBlock(filters=filters,
                         kernel_size=1,
                         data_format=data_format,
                         normalization=normalization,
                         activation=None,
                         use_bias=False)(inputs)   #input:1024/ 2048 | output:512
    # projection layer 2
    x = ConvNormActBlock(filters=filters,
                         kernel_size=3,
                         data_format=data_format,
                         normalization=normalization,
                         use_bias=False,
                         activation=None)(x)     #input:512 | output:512


    # residual block 1
    shortcut = x
    x = ConvNormActBlock(filters=midfilters,
                         kernel_size=1,
                         normalization=normalization,
                         activation=activation,
                         data_format=data_format,
                         use_bias=False)(x) #input:512 | output:128
    x = ConvNormActBlock(filters=midfilters,
                         kernel_size=3,
                         dilation_rate=2,
                         normalization=normalization,
                         activation=activation,
                         data_format=data_format,
                         use_bias=False)(x) #input:128 | output:128 | dilation rate:2
    x = ConvNormActBlock(filters=filters,
                         kernel_size=1,
                         normalization=normalization,
                         activation=activation,
                         data_format=data_format,
                         use_bias=False)(x) #input:128 | output:512
    x = layers.Add()([x, shortcut])
    # residual block 2
    shortcut = x
    x = ConvNormActBlock(filters=128,
                         kernel_size=1,
                         normalization=normalization,
                         activation=activation,
                         data_format=data_format,
                         use_bias=False)(x) #input:512 | output:128
    x = ConvNormActBlock(filters=midfilters,
                         kernel_size=3,
                         dilation_rate=4,
                         normalization=normalization,
                         activation=activation,
                         data_format=data_format,
                         use_bias=False)(x) #input:128 | output:512 | dilation rate:4
    x = ConvNormActBlock(filters=filters,
                         kernel_size=1,
                         normalization=normalization,
                         activation=activation,
                         data_format=data_format,
                         use_bias=False)(x) #input:512 | output:512
    x = layers.Add()([x, shortcut])

    # residual block 3

    shortcut = x
    x = ConvNormActBlock(filters=128,
                         kernel_size=1,
                         normalization=normalization,
                         activation=activation,
                         data_format=data_format,
                         use_bias=False)(x) #input:512 | output:512
    x = ConvNormActBlock(filters=128,
                         kernel_size=3,
                         dilation_rate=6,
                         normalization=normalization,
                         activation=activation,
                         data_format=data_format,
                         use_bias=False)(x) #input:512 | output:512 | dilation rate:6
    x = ConvNormActBlock(filters=filters,
                         kernel_size=1,
                         normalization=normalization,
                         activation=activation,
                         data_format=data_format,
                         use_bias=False)(x) #input:512 | output:512
    x = layers.Add()([x, shortcut])

    # residual block 4
    shortcut = x
    x = ConvNormActBlock(filters=128,
                         kernel_size=1,
                         normalization=normalization,
                         activation=activation,
                         data_format=data_format,
                         use_bias=False)(x) #input:512 | output:512

    x = ConvNormActBlock(filters=128,
                         kernel_size=3,
                         dilation_rate=8,
                         normalization=normalization,
                         activation=activation,
                         data_format=data_format,
                         use_bias=False)(x) #input:512 | output:512 | dilation rate:8

    x = ConvNormActBlock(filters=filters,
                         kernel_size=1,
                         normalization=normalization,
                         activation=activation,
                         data_format=data_format,
                         use_bias=False)(x) #input:512 | output:512
    x = layers.Add()([x, shortcut])

    
    return kl.Model(inputs=inputs, outputs=x)  #input:1024/2048 | output:512


