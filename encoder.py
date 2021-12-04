import tensorflow as tf
from tensorflow.keras import layers

class DilatedEncoder(tf.keras.Model):
    def __init__(
        self,
        in_channels = 2048,
        encoder_channels = 512,
        block_mid_channels = 128,
        #num_residual_blocks = 4,
        block_dilations = [2,4,6,8], #This is a list of the dilations in each respective residual block
        activation = 'relu',
        kernel_init=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None)
    ):
                
        super().__init__()
                
        self.in_channels = in_channels
        self.encoder_channels = encoder_channels
        self.block_mid_channels = block_mid_channels
        
        #self.num_residual_blocks = num_residual_blocks
        self.block_dilations = block_dilations
        self.num_residual_blocks = len(block_dilations)
        
        self.activation = activation
        self.kernel_init = kernel_init

        self._init_layers()
        #self._init_weights() #TF includes the ini
        
    def _init_layers(self):
        self.proj_subnet = tf.keras.Sequential()
        
        self.proj_subnet.add(layers.Conv2D(self.encoder_channels, kernel_size=1, strides=1, padding="valid", 
                                              kernel_initializer=self.kernel_init)) #Equivalent to padding=0 (default)
        self.proj_subnet.add(layers.BatchNormalization())
        self.proj_subnet.add(layers.Conv2D(self.encoder_channels, kernel_size=3, strides=1, padding="same", 
                                              kernel_initializer=self.kernel_init)) #Equivalent to padding=1
        self.proj_subnet.add(layers.BatchNormalization())
        
        self.encoder_blocks = tf.keras.Sequential()
        for dilation in self.block_dilations:
            self.encoder_blocks.add(Bottleneck(self.encoder_channels, self.block_mid_channels,
                                               dilation, self.activation, self.kernel_init))
            
    def call(self, inputs, training=False):
        x = self.proj_subnet(inputs)
        x = self.encoder_blocks(x)
        
        return x
    
#This is the repeated residual block, which we make a separate class in order to easier include repeated residual skips.
class Bottleneck(tf.keras.Model):
    def __init__(self,
                 encoder_channels,
                 mid_channels,
                 dilation,
                 activation,
                 kernel_init
                ):
        super().__init__()
        
        #First part; note that this outputs for bottleneck size, not normal size
        self.startconv = tf.keras.Sequential()
        self.startconv.add(layers.Conv2D(mid_channels, kernel_size=1, strides=1, padding="valid",
                                              kernel_initializer=kernel_init))
        self.startconv.add(layers.BatchNormalization()) #Default initializations for this already match the paper's code
        self.startconv.add(layers.Activation(activation))
        
        #Dilated middle
        self.midconv = tf.keras.Sequential()
        self.midconv.add(layers.ZeroPadding2D(dilation)) #padding=dilation
        self.midconv.add(layers.Conv2D(encoder_channels, kernel_size=3, strides=1, padding="valid",
                                              dilation_rate=dilation, kernel_initializer=kernel_init))
        self.midconv.add(layers.BatchNormalization())
        self.midconv.add(layers.Activation(activation))
        
        #Ending part
        self.endconv = tf.keras.Sequential()
        self.endconv.add(layers.Conv2D(encoder_channels, kernel_size=1, strides=1, padding="valid",
                                              kernel_initializer=kernel_init))
        self.endconv.add(layers.BatchNormalization())
        self.endconv.add(layers.Activation(activation))
        
    def call(self, inputs, training=False):
        identity = inputs #Save input for resiidual block skip connection
        x = self.startconv(inputs)
        x = self.midconv(x)
        x = self.endconv(x)
        x = layers.Add()([x, identity]) #Close residual block skip connection
        
        return x