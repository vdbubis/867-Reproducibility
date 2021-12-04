import os
import numpy as np
from typing import Dict, List, Tuple

#We are skipping on visualized training, so no event_storage is needed
#Post-processing was in the implementation, but not mentioned in the paper, so we are excluding it

import tensorflow as tf
from tensorflow import Tensor
import tensorflow_datasets as tfds

from retinanet_utility import compute_iou, AnchorBox, get_backbone, swap_xy, convert_to_xywh

#Addon loss functions, equivalent to sigmoid_focal_loss_jit and giou_loss

from tensorflow_addons.losses import SigmoidFocalCrossEntropy, giou_loss

from encoder import DilatedEncoder
from decoder import Decoder
from box_regression import YOLOFBox2BoxTransform
from uniform_matcher import UniformMatcher

from anchor_gen import AnchorGenerator
from matching_loss import YolofLoss

from YOLOF import YOLOF

import zipfile

"""
## Downloading the COCO2017 dataset
Training on the entire COCO2017 dataset which has around 118k images takes a
lot of time, hence we will be using a smaller subset of ~500 images for
training in this example.
"""

url = "https://github.com/srihari-humbarwadi/datasets/releases/download/v0.1.0/data.zip"
filename = os.path.join(os.getcwd(), "data.zip")

with zipfile.ZipFile("data.zip", "r") as z_fp:
    z_fp.extractall("./")

(train_dataset, val_dataset), dataset_info = tfds.load(
    "coco/2017", split=["train", "validation"], with_info=True, data_dir="data"
)

#Format: 'image', 'image/filename', 'image/id', 'objects'

#Refer to: https://github.com/chensnathan/YOLOF/blob/master/configs/yolof_R_50_DC5_1x.yaml

coco_num_classes=80 #Number of classes used in the COCO2017 dataset; 91 are defined however, though not all used
anchor_sizes=[32, 64, 128, 256, 512]
anchor_gen = AnchorGenerator(sizes=anchor_sizes)
num_anchors=len(anchor_sizes)

model = YOLOF(
        backbone = get_backbone(),
        encoder = DilatedEncoder(),
        decoder = Decoder(num_classes=coco_num_classes, num_anchors=num_anchors),
        #anchor_generator = AnchorBox(areas=anchor_sizes,
        #                             aspect_ratios = [1.0],
        #                             scales = [1],
        #                             single_anchor = True
        #                             #strides = [2 ** i for i in range(3, 8)],
        #                            ), #This is an overload
        anchor_generator = anchor_gen,
        box2box_transform = YOLOFBox2BoxTransform(weights=(1, 1, 1, 1)), #Unit variance is default
        loss_function = YolofLoss(anchor_gen.get_anchors(19, 19),
                                  anchor_matcher = UniformMatcher(match_times=4), #Default is 4 matches
                                  num_classes=coco_num_classes),
        num_classes = coco_num_classes
    )

from tensorflow.keras import optimizers

lr_schedule = optimizers.schedules.ExponentialDecay(
    initial_learning_rate=5e-4,
    decay_steps=100,
    decay_rate=0.9)

model.compile( #Specify auxilarry details
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule), #Adaptive moment estimation. Usually used for larger datasets.
        loss=model.loss_function #Information inefficiency for recoding through another distribution. Standard loss.
        )

autotune = tf.data.AUTOTUNE

#batch_size = 2 #Number of images used to calculate gradients per step; more consistent but less efficient when increased
#Adam uses similar principles as SGD in this regard

train_dataset = train_dataset.map(model.preprocess_image, num_parallel_calls=autotune)
train_dataset = train_dataset.apply(tf.data.experimental.ignore_errors()) #This drops error-inducing data
#train_dataset = train_dataset.batch(batch_size)
train_dataset = train_dataset.prefetch(autotune)

val_dataset = val_dataset.map(model.preprocess_image, num_parallel_calls=autotune)
val_dataset = val_dataset.apply(tf.data.experimental.ignore_errors()) #This drops error-inducing data
#val_dataset = val_dataset.batch(batch_size)
val_dataset = val_dataset.prefetch(autotune)

#epochs = 100
#
#history = model.fit(
#    train_dataset.take(100),
#    validation_data=val_dataset.take(50),
#    epochs=epochs,
#    verbose=1,
#)

small_x = []
small_y = []

batch_size = 12

for (x, y) in train_dataset.take(batch_size):
    small_x.append(x)
    small_y.append(y)
    
small_x = tf.stack(small_x)

sample_out = model(small_x)

loss = model.loss_function(small_y, sample_out)

print("On the initialized network, inferring from a batch size of", batch_size, " we have a loss of", loss.numpy())