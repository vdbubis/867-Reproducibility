import math
import tensorflow as tf
from tensorflow.keras import layers

class Decoder(tf.keras.Model):
    def __init__(self,
                 num_classes,
                 num_anchors,
                 decoder_channels=512,
                 cls_layers=2,
                 reg_layers=4,
                 activation='relu',
                 kernel_init=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None),
                 prior_prob=0.5): #0.5 is effectively a zeros initialization.
        
        super().__init__()
        
        self.cls_layers = cls_layers #Value used in paper is 2
        self.reg_layers = reg_layers #Value used in paper is 4
        self.activation = activation #For now, this is implicitly fixed as 'relu'
        self.kernel_init = kernel_init #In the paper, this is a random normal of mean=0 and std=0.01
        
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.decoder_channels = decoder_channels
        
        self.prior_prob = prior_prob
        
        self.INF = 1e8 #Maximum value
        
        self._init_layers()
        
    def _init_layers(self):
        self.cls_subnet = tf.keras.Sequential() #These two subnets are parallel; objectness multiplication follows their outputs
        self.reg_subnet = tf.keras.Sequential()
        
        for i in range(self.cls_layers):
            self.cls_subnet.add(layers.Conv2D(self.decoder_channels,(3, 3), strides=1, padding="same",
                                              kernel_initializer=self.kernel_init))
            self.cls_subnet.add(layers.BatchNormalization()) #Default initializations for this already match the paper's code
            self.cls_subnet.add(layers.Activation(self.activation))
            
        for i in range(self.reg_layers):
            self.reg_subnet.add(layers.Conv2D(self.decoder_channels,(3, 3), strides=1, padding="same", kernel_initializer=self.kernel_init))
            self.reg_subnet.add(layers.BatchNormalization())
            self.reg_subnet.add(layers.Activation(self.activation))
            
        bias_value = -math.log((1 - self.prior_prob) / self.prior_prob) #This bias value is only used for classification
            
        self.cls_score = layers.Conv2D(self.num_anchors * self.num_classes,(3, 3), strides=1, padding="same",
                                       kernel_initializer=self.kernel_init,
                                       bias_initializer=tf.keras.initializers.Constant(bias_value))
        self.bbox_pred = layers.Conv2D(self.num_anchors * 4,(3, 3), strides=1, padding="same",
                                       kernel_initializer=self.kernel_init)
        self.object_pred = layers.Conv2D(self.num_anchors,(3, 3), strides=1, padding="same",
                                         kernel_initializer=self.kernel_init)
        
    def call(self, inputs, training=False): #This is the forward pass
        cls_score = self.cls_score(self.cls_subnet(inputs))
        N, W, H, _ = cls_score.shape
        cls_score = tf.reshape(cls_score, [N, W, H, -1, self.num_classes])
        
        reg_feat = self.reg_subnet(inputs) #We are effectively saving this activation to input twice
        bbox_reg = self.bbox_pred(reg_feat)
        objectness = self.object_pred(reg_feat)
        
        #Objectness for multiplication with pred logits
        objectness = tf.reshape(objectness, [N, W, H, -1, 1])
        
        #This is a softmax with guards against exp overflows (deltas are logarithmic)
        normalized_cls_score = cls_score + objectness - tf.math.log(
            1. + tf.clip_by_value(tf.math.exp(cls_score), clip_value_min=0, clip_value_max=self.INF) + tf.clip_by_value(
                tf.math.exp(objectness), clip_value_min=0, clip_value_max=self.INF))
        
        normalized_cls_score = tf.reshape(normalized_cls_score, [N, W, H, -1])
        
        return normalized_cls_score, bbox_reg