

"""
Title: Object Detection with RetinaNet
Author: [Srihari Humbarwadi](https://twitter.com/srihari_rh)
Date created: 2020/05/17
Last modified: 2020/07/14
Description: Implementing RetinaNet: Focal Loss for Dense Object Detection.
"""

"""
## Introduction
Object detection a very important problem in computer
vision. Here the model is tasked with localizing the objects present in an
image, and at the same time, classifying them into different categories.
Object detection models can be broadly classified into "single-stage" and
"two-stage" detectors. Two-stage detectors are often more accurate but at the
cost of being slower. Here in this example, we will implement RetinaNet,
a popular single-stage detector, which is accurate and runs fast.
RetinaNet uses a feature pyramid network to efficiently detect objects at
multiple scales and introduces a new loss, the Focal loss function, to alleviate
the problem of the extreme foreground-background class imbalance.
**References:**
- [RetinaNet Paper](https://arxiv.org/abs/1708.02002)
- [Feature Pyramid Network Paper](https://arxiv.org/abs/1612.03144)
"""


import os
import re
import zipfile

import numpy as np
import tensorflow as tf
from tensorflow import keras
#from tensorflow.keras import layers
from tensorflow.keras import datasets, layers, models

import matplotlib.pyplot as plt
import tensorflow_datasets as tfds


"""
## Downloading the COCO2017 dataset
Training on the entire COCO2017 dataset which has around 118k images takes a
lot of time, hence we will be using a smaller subset of ~500 images for
training in this example.
"""

url = "https://github.com/srihari-humbarwadi/datasets/releases/download/v0.1.0/data.zip"
filename = os.path.join(os.getcwd(), "data.zip")
keras.utils.get_file(filename, url)


with zipfile.ZipFile("data.zip", "r") as z_fp:
    z_fp.extractall("./")


"""
## Implementing utility functions
Bounding boxes can be represented in multiple ways, the most common formats are:
- Storing the coordinates of the corners `[xmin, ymin, xmax, ymax]`
- Storing the coordinates of the center and the box dimensions
`[x, y, width, height]`
Since we require both formats, we will be implementing functions for converting
between the formats.
"""


def swap_xy(boxes):
    """Swaps order the of x and y coordinates of the boxes.
    Arguments:
      boxes: A tensor with shape `(num_boxes, 4)` representing bounding boxes.
    Returns:
      swapped boxes with shape same as that of boxes.
    """
    return tf.stack([boxes[:, 1], boxes[:, 0], boxes[:, 3], boxes[:, 2]], axis=-1)


def convert_to_xywh(boxes):
    """Changes the box format to center, width and height.
    Arguments:
      boxes: A tensor of rank 2 or higher with a shape of `(..., num_boxes, 4)`
        representing bounding boxes where each box is of the format
        `[xmin, ymin, xmax, ymax]`.
    Returns:
      converted boxes with shape same as that of boxes.
    """
    return tf.concat(
        [(boxes[..., :2] + boxes[..., 2:]) / 2.0, boxes[..., 2:] - boxes[..., :2]],
        axis=-1,
    )


def convert_to_corners(boxes):
    """Changes the box format to corner coordinates
    Arguments:
      boxes: A tensor of rank 2 or higher with a shape of `(..., num_boxes, 4)`
        representing bounding boxes where each box is of the format
        `[x, y, width, height]`.
    Returns:
      converted boxes with shape same as that of boxes.
    """
    return tf.concat(
        [boxes[..., :2] - boxes[..., 2:] / 2.0, boxes[..., :2] + boxes[..., 2:] / 2.0],
        axis=-1,
    )


"""
## Computing pairwise Intersection Over Union (IOU)
As we will see later in the example, we would be assigning ground truth boxes
to anchor boxes based on the extent of overlapping. This will require us to
calculate the Intersection Over Union (IOU) between all the anchor
boxes and ground truth boxes pairs.
"""


def compute_iou(boxes1, boxes2):
    """Computes pairwise IOU matrix for given two sets of boxes
    Arguments:
      boxes1: A tensor with shape `(N, 4)` representing bounding boxes
        where each box is of the format `[x, y, width, height]`.
        boxes2: A tensor with shape `(M, 4)` representing bounding boxes
        where each box is of the format `[x, y, width, height]`.
    Returns:
      pairwise IOU matrix with shape `(N, M)`, where the value at ith row
        jth column holds the IOU between ith box and jth box from
        boxes1 and boxes2 respectively.
    """
    boxes1_corners = convert_to_corners(boxes1)
    boxes2_corners = convert_to_corners(boxes2)
    lu = tf.maximum(boxes1_corners[:, None, :2], boxes2_corners[:, :2])
    rd = tf.minimum(boxes1_corners[:, None, 2:], boxes2_corners[:, 2:])
    intersection = tf.maximum(0.0, rd - lu)
    intersection_area = intersection[:, :, 0] * intersection[:, :, 1]
    boxes1_area = boxes1[:, 2] * boxes1[:, 3]
    boxes2_area = boxes2[:, 2] * boxes2[:, 3]
    union_area = tf.maximum(
        boxes1_area[:, None] + boxes2_area - intersection_area, 1e-8
    )
    return tf.clip_by_value(intersection_area / union_area, 0.0, 1.0)


def visualize_detections(
    image, boxes, classes, scores, figsize=(7, 7), linewidth=1, color=[0, 0, 1]
):
    """Visualize Detections"""
    image = np.array(image, dtype=np.uint8)
    plt.figure(figsize=figsize)
    plt.axis("off")
    plt.imshow(image)
    ax = plt.gca()
    for box, _cls, score in zip(boxes, classes, scores):
        text = "{}: {:.2f}".format(_cls, score)
        x1, y1, x2, y2 = box
        w, h = x2 - x1, y2 - y1
        patch = plt.Rectangle(
            [x1, y1], w, h, fill=False, edgecolor=color, linewidth=linewidth
        )
        ax.add_patch(patch)
        ax.text(
            x1,
            y1,
            text,
            bbox={"facecolor": color, "alpha": 0.4},
            clip_box=ax.clipbox,
            clip_on=True,
        )
    plt.show()
    return ax


"""
## Implementing Anchor generator
Anchor boxes are fixed sized boxes that the model uses to predict the bounding
box for an object. It does this by regressing the offset between the location
of the object's center and the center of an anchor box, and then uses the width
and height of the anchor box to predict a relative scale of the object. In the
case of RetinaNet, each location on a given feature map has nine anchor boxes
(at three scales and three ratios).
"""


class AnchorBox:
    """Generates anchor boxes.
    This class has operations to generate anchor boxes for feature maps at
    strides `[8, 16, 32, 64, 128]`. Where each anchor each box is of the
    format `[x, y, width, height]`.
    Attributes:
      aspect_ratios: A list of float values representing the aspect ratios of
        the anchor boxes at each location on the feature map
      scales: A list of float values representing the scale of the anchor boxes
        at each location on the feature map.
      num_anchors: The number of anchor boxes at each location on feature map
      areas: A list of float values representing the areas of the anchor
        boxes for each feature map in the feature pyramid.
      strides: A list of float value representing the strides for each feature
        map in the feature pyramid.
    """

    def __init__(self):
        self.aspect_ratios = [0.5, 1.0, 2.0]
        self.scales = [2 ** x for x in [0, 1 / 3, 2 / 3]]

        self._num_anchors = len(self.aspect_ratios) * len(self.scales)
        self._strides = [2 ** i for i in range(3, 8)]
        self._areas = [x ** 2 for x in [32.0, 64.0, 128.0, 256.0, 512.0]]
        self._anchor_dims = self._compute_dims()

    def _compute_dims(self):
        """Computes anchor box dimensions for all ratios and scales at all levels
        of the feature pyramid.
        """
        anchor_dims_all = []
        for area in self._areas:
            anchor_dims = []
            for ratio in self.aspect_ratios:
                anchor_height = tf.math.sqrt(area / ratio)
                anchor_width = area / anchor_height
                dims = tf.reshape(
                    tf.stack([anchor_width, anchor_height], axis=-1), [1, 1, 2]
                )
                for scale in self.scales:
                    anchor_dims.append(scale * dims)
            anchor_dims_all.append(tf.stack(anchor_dims, axis=-2))
        return anchor_dims_all

    def _get_anchors(self, feature_height, feature_width, level):
        """Generates anchor boxes for a given feature map size and level
        Arguments:
          feature_height: An integer representing the height of the feature map.
          feature_width: An integer representing the width of the feature map.
          level: An integer representing the level of the feature map in the
            feature pyramid.
        Returns:
          anchor boxes with the shape
          `(feature_height * feature_width * num_anchors, 4)`
        """
        rx = tf.range(feature_width, dtype=tf.float32) + 0.5
        ry = tf.range(feature_height, dtype=tf.float32) + 0.5
        centers = tf.stack(tf.meshgrid(rx, ry), axis=-1) * self._strides[level - 3]
        centers = tf.expand_dims(centers, axis=-2)
        centers = tf.tile(centers, [1, 1, self._num_anchors, 1])
        dims = tf.tile(
            self._anchor_dims[level - 3], [feature_height, feature_width, 1, 1]
        )
        anchors = tf.concat([centers, dims], axis=-1)
        return tf.reshape(
            anchors, [feature_height * feature_width * self._num_anchors, 4]
        )

    def get_anchors(self, image_height, image_width):
        return self._get_anchors(tf.math.ceil(image_height / 32), tf.math.ceil(image_width / 32), 3)
        """Generates anchor boxes for all the feature maps of the feature pyramid.
        Arguments:
          image_height: Height of the input image.
          image_width: Width of the input image.
        Returns:
          anchor boxes for all the feature maps, stacked as a single tensor
            with shape `(total_anchors, 4)`
        """
        # anchors = [
        #     self._get_anchors(
        #         tf.math.ceil(image_height / 2 ** i),
        #         tf.math.ceil(image_width / 2 ** i),
        #         i,
        #     )
        #     for i in range(3, 8)
        # ]
        # return self._get_anchors(tf.math.ceil(image_height / 32),tf.math.ceil(image_width / 32),3)


"""
## Preprocessing data
Preprocessing the images involves two steps:
- Resizing the image: Images are resized such that the shortest size is equal
to 800 px, after resizing if the longest side of the image exceeds 1333 px,
the image is resized such that the longest size is now capped at 1333 px.
- Applying augmentation: Random scale jittering  and random horizontal flipping
are the only augmentations applied to the images.
Along with the images, bounding boxes are rescaled and flipped if required.
"""


def random_flip_horizontal(image, boxes):
    """Flips image and boxes horizontally with 50% chance
    Arguments:
      image: A 3-D tensor of shape `(height, width, channels)` representing an
        image.
      boxes: A tensor with shape `(num_boxes, 4)` representing bounding boxes,
        having normalized coordinates.
    Returns:
      Randomly flipped image and boxes
    """
    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_left_right(image)
        boxes = tf.stack(
            [1 - boxes[:, 2], boxes[:, 1], 1 - boxes[:, 0], boxes[:, 3]], axis=-1
        )
    return image, boxes


def resize_and_pad_image(
    image, min_side=800.0, max_side=1333.0, jitter=[640, 1024], stride=128.0
):
    """Resizes and pads image while preserving aspect ratio.
    1. Resizes images so that the shorter side is equal to `min_side`
    2. If the longer side is greater than `max_side`, then resize the image
      with longer side equal to `max_side`
    3. Pad with zeros on right and bottom to make the image shape divisible by
    `stride`
    Arguments:
      image: A 3-D tensor of shape `(height, width, channels)` representing an
        image.
      min_side: The shorter side of the image is resized to this value, if
        `jitter` is set to None.
      max_side: If the longer side of the image exceeds this value after
        resizing, the image is resized such that the longer side now equals to
        this value.
      jitter: A list of floats containing minimum and maximum size for scale
        jittering. If available, the shorter side of the image will be
        resized to a random value in this range.
      stride: The stride of the smallest feature map in the feature pyramid.
        Can be calculated using `image_size / feature_map_size`.
    Returns:
      image: Resized and padded image.
      image_shape: Shape of the image before padding.
      ratio: The scaling factor used to resize the image
    """
    image_shape = tf.cast(tf.shape(image)[:2], dtype=tf.float32)
    if jitter is not None:
        min_side = tf.random.uniform((), jitter[0], jitter[1], dtype=tf.float32)
    ratio = min_side / tf.reduce_min(image_shape)
    if ratio * tf.reduce_max(image_shape) > max_side:
        ratio = max_side / tf.reduce_max(image_shape)
    image_shape = ratio * image_shape
    image = tf.image.resize(image, tf.cast(image_shape, dtype=tf.int32))
    padded_image_shape = tf.cast(
        tf.math.ceil(image_shape / stride) * stride, dtype=tf.int32
    )
    image = tf.image.pad_to_bounding_box(
        image, 0, 0, padded_image_shape[0], padded_image_shape[1]
    )
    return image, image_shape, ratio


def preprocess_data(sample):
    """Applies preprocessing step to a single sample
    Arguments:
      sample: A dict representing a single training sample.
    Returns:
      image: Resized and padded image with random horizontal flipping applied.
      bbox: Bounding boxes with the shape `(num_objects, 4)` where each box is
        of the format `[x, y, width, height]`.
      class_id: An tensor representing the class id of the objects, having
        shape `(num_objects,)`.
    """
    image = sample["image"]
    bbox = swap_xy(sample["objects"]["bbox"])
    class_id = tf.cast(sample["objects"]["label"], dtype=tf.int32)

    image, bbox = random_flip_horizontal(image, bbox)
    image, image_shape, _ = resize_and_pad_image(image)

    bbox = tf.stack(
        [
            bbox[:, 0] * image_shape[1],
            bbox[:, 1] * image_shape[0],
            bbox[:, 2] * image_shape[1],
            bbox[:, 3] * image_shape[0],
        ],
        axis=-1,
    )
    bbox = convert_to_xywh(bbox)
    return image, bbox, class_id


"""
## Encoding labels
The raw labels, consisting of bounding boxes and class ids need to be
transformed into targets for training. This transformation consists of
the following steps:
- Generating anchor boxes for the given image dimensions
- Assigning ground truth boxes to the anchor boxes
- The anchor boxes that are not assigned any objects, are either assigned the
background class or ignored depending on the IOU
- Generating the classification and regression targets using anchor boxes
"""


class LabelEncoder:
    """Transforms the raw labels into targets for training.
    This class has operations to generate targets for a batch of samples which
    is made up of the input images, bounding boxes for the objects present and
    their class ids.
    Attributes:
      anchor_box: Anchor box generator to encode the bounding boxes.
      box_variance: The scaling factors used to scale the bounding box targets.
    """

    def __init__(self):
        self._anchor_box = AnchorBox()
        self._box_variance = tf.convert_to_tensor(
            [0.1, 0.1, 0.2, 0.2], dtype=tf.float32
        )

    def _match_anchor_boxes(
        self, anchor_boxes, gt_boxes, match_iou=0.5, ignore_iou=0.4,max_top_k=100
    ):
        """Matches ground truth boxes to anchor boxes based on IOU.
        1. Calculates the pairwise IOU for the M `anchor_boxes` and N `gt_boxes`
          to get a `(M, N)` shaped matrix.
        2. The ground truth box with the maximum IOU in each row is assigned to
          the anchor box provided the IOU is greater than `match_iou`.
        3. If the maximum IOU in a row is less than `ignore_iou`, the anchor
          box is assigned with the background class.
        4. The remaining anchor boxes that do not have any class assigned are
          ignored during training.
        Arguments:
          anchor_boxes: A float tensor with the shape `(total_anchors, 4)`
            representing all the anchor boxes for a given input image shape,
            where each anchor box is of the format `[x, y, width, height]`.
          gt_boxes: A float tensor with shape `(num_objects, 4)` representing
            the ground truth boxes, where each box is of the format
            `[x, y, width, height]`.
          match_iou: A float value representing the minimum IOU threshold for
            determining if a ground truth box can be assigned to an anchor box.
          ignore_iou: A float value representing the IOU threshold under which
            an anchor box is assigned to the background class.
        Returns:
          matched_gt_idx: Index of the matched object
          positive_mask: A mask for anchor boxes that have been assigned ground
            truth boxes.
          ignore_mask: A mask for anchor boxes that need to by ignored during
            training
        """
        iou_matrix = compute_iou(anchor_boxes, gt_boxes)
        max_iou = tf.reduce_max(iou_matrix, axis=1)
        matched_gt_idx = tf.argmax(iou_matrix, axis=1)
        positive_mask = tf.greater_equal(max_iou, match_iou)
        negative_mask = tf.less(max_iou, ignore_iou)
        ignore_mask = tf.logical_not(tf.logical_or(positive_mask, negative_mask))
        return (
            matched_gt_idx,
            tf.cast(positive_mask, dtype=tf.float32),
            tf.cast(ignore_mask, dtype=tf.float32),
        )

    def _compute_box_target(self, anchor_boxes, matched_gt_boxes):
        """Transforms the ground truth boxes into targets for training"""
        box_target = tf.concat(
            [
                (matched_gt_boxes[:, :2] - anchor_boxes[:, :2]) / anchor_boxes[:, 2:],
                tf.math.log(matched_gt_boxes[:, 2:] / anchor_boxes[:, 2:]),
            ],
            axis=-1,
        )
        box_target = box_target / self._box_variance
        return box_target

    def _encode_sample(self, image_shape, gt_boxes, cls_ids):
        """Creates box and classification targets for a single sample"""
        anchor_boxes = self._anchor_box.get_anchors(image_shape[1], image_shape[2])
        cls_ids = tf.cast(cls_ids, dtype=tf.float32)
        matched_gt_idx, positive_mask, ignore_mask = self._match_anchor_boxes(
            anchor_boxes, gt_boxes
        )
        matched_gt_boxes = tf.gather(gt_boxes, matched_gt_idx)
        box_target = self._compute_box_target(anchor_boxes, matched_gt_boxes)
        matched_gt_cls_ids = tf.gather(cls_ids, matched_gt_idx)
        cls_target = tf.where(
            tf.not_equal(positive_mask, 1.0), -1.0, matched_gt_cls_ids
        )
        cls_target = tf.where(tf.equal(ignore_mask, 1.0), -2.0, cls_target)
        cls_target = tf.expand_dims(cls_target, axis=-1)
        label = tf.concat([box_target, cls_target], axis=-1)
        return label

    def encode_batch(self, batch_images, gt_boxes, cls_ids):
        """Creates box and classification targets for a batch"""
        images_shape = tf.shape(batch_images)
        batch_size = images_shape[0]

        labels = tf.TensorArray(dtype=tf.float32, size=batch_size, dynamic_size=True)
        for i in range(batch_size):
            label = self._encode_sample(images_shape, gt_boxes[i], cls_ids[i])
            labels = labels.write(i, label)
        batch_images = tf.keras.applications.resnet.preprocess_input(batch_images)
        return batch_images, labels.stack()


"""
## Building the ResNet50 backbone
RetinaNet uses a ResNet based backbone, using which a feature pyramid network
is constructed. In the example we use ResNet50 as the backbone, and return the
feature maps at strides 8, 16 and 32.
"""


def get_backbone():
    """Builds ResNet50 with pre-trained imagenet weights"""
    backbone = keras.applications.ResNet50(
        include_top=False, input_shape=[None, None, 3]
    )
    c3_output, c4_output, c5_output = [
        backbone.get_layer(layer_name).output
        for layer_name in ["conv3_block4_out", "conv4_block6_out", "conv5_block3_out"]
    ]
    return keras.Model(
        inputs=[backbone.inputs], outputs=[c5_output]
    )


"""
## Building Feature Pyramid Network as a custom layer
"""


class ConvNormActBlock(layers.Layer):
    def __init__(self,
                 filters,
                 kernel_size,
                 strides=1,
                 groups=1,

                 data_format="channels_last",
                 dilation_rate=1,
                 kernel_initializer="he_normal",
                 activation=None,
                 trainable=True,
                 use_bias=False,
                 normalization=dict(normalization="batch_norm", axis=-1, trainable=True),
                 name=None):
        super(ConvNormActBlock, self).__init__(name=name)

        self.strides = strides
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)

        if isinstance(strides, int):
            strides = (strides, strides)

        if isinstance(dilation_rate, int):
            dilation_rate = (dilation_rate, dilation_rate)
        if activation is not None:
            self.act = layers.Activation(activation['activation'])
        else:
            self.act = layers.Lambda(lambda x: x)
        if strides == (1, 1):
            p = 1
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
        self.bn = layers.BatchNormalization(trainable=normalization["trainable"])
    def call(self, inputs):
        x = self.conv(inputs)
        x = self.bn(x)
        o = self.act(x)
        return o


def Encoder(input_shapes,
                   dilation_rates,
                   filters=512,
                   midfilters=128,
                   normalization=dict(normalization="batch_norm", momentum=0.03, epsilon=1e-3, axis=-1, trainable=True),
                   activation=dict(activation="relu"),
                   kernel_initializer="he_normal",
                   data_format="channels_last",
                   name="dilated_encoder"):
    inputs = layers.Input(shape=(None,None,2048))

    # projection layer 1
    x = ConvNormActBlock(filters=filters,
                         kernel_size=1,
                         data_format=data_format,
                         normalization=normalization,
                         activation=None,
                         use_bias=False)(inputs)
    # projection layer 2
    x = ConvNormActBlock(filters=filters,
                         kernel_size=3,
                         data_format=data_format,
                         normalization=normalization,
                         use_bias=False,
                         activation=None)(x)

    # residual block 1
    shortcut = x
    x = ConvNormActBlock(filters=midfilters,
                         kernel_size=1,
                         normalization=normalization,
                         activation=activation,
                         data_format=data_format,
                         use_bias=False)(x)
    x = ConvNormActBlock(filters=midfilters,
                         kernel_size=3,
                         dilation_rate=2,
                         normalization=normalization,
                         activation=activation,
                         data_format=data_format,
                         use_bias=False)(x)
    x = ConvNormActBlock(filters=filters,
                         kernel_size=1,
                         normalization=normalization,
                         activation=activation,
                         data_format=data_format,
                         use_bias=False)(x)
    x = layers.Add()([x, shortcut])
    # residual block 2
    shortcut = x
    x = ConvNormActBlock(filters=midfilters,
                         kernel_size=1,
                         normalization=normalization,
                         activation=activation,
                         data_format=data_format,
                         use_bias=False)(x)
    x = ConvNormActBlock(filters=midfilters,
                         kernel_size=3,
                         dilation_rate=4,
                         normalization=normalization,
                         activation=activation,
                         data_format=data_format,
                         use_bias=False)(x)
    x = ConvNormActBlock(filters=filters,
                         kernel_size=1,
                         normalization=normalization,
                         activation=activation,
                         data_format=data_format,
                         use_bias=False)(x)
    x = layers.Add()([x, shortcut])

    # residual block 3

    shortcut = x
    x = ConvNormActBlock(filters=midfilters,
                         kernel_size=1,
                         normalization=normalization,
                         activation=activation,
                         data_format=data_format,
                         use_bias=False)(x)
    x = ConvNormActBlock(filters=midfilters,
                         kernel_size=3,
                         dilation_rate=6,
                         normalization=normalization,
                         activation=activation,
                         data_format=data_format,
                         use_bias=False)(x)
    x = ConvNormActBlock(filters=filters,
                         kernel_size=1,
                         normalization=normalization,
                         activation=activation,
                         data_format=data_format,
                         use_bias=False)(x)
    x = layers.Add()([x, shortcut])

    # residual block 4
    shortcut = x
    x = ConvNormActBlock(filters=midfilters,
                         kernel_size=1,
                         normalization=normalization,
                         activation=activation,
                         data_format=data_format,
                         use_bias=False)(x)

    x = ConvNormActBlock(filters=midfilters,
                         kernel_size=3,
                         dilation_rate=8,
                         normalization=normalization,
                         activation=activation,
                         data_format=data_format,
                         use_bias=False)(x)

    x = ConvNormActBlock(filters=filters,
                         kernel_size=1,
                         normalization=normalization,
                         activation=activation,
                         data_format=data_format,
                         use_bias=False)(x)
    x = layers.Add()([x, shortcut])
    return keras.Model(inputs=inputs, outputs=x)  # input:1024/2048 | output:512


class Decoder(tf.keras.Model):
        def __init__(self,
                     num_classes,
                     num_anchors,
                     cls_layers=2,
                     reg_layers=4,
                     activation='relu',
                     kernel_init=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None),
                     prior_prob=0.5):  # 0.5 is effectively a zeros initialization.
            super(Decoder,self).__init__()
            self.cls_layers = cls_layers  # Value used in paper is 2
            self.reg_layers = reg_layers  # Value used in paper is 4
            self.activation = activation  # For now, this is implicitly fixed as 'relu'
            self.kernel_init = kernel_init  # In the paper, this is a random normal of mean=0 and std=0.01

            self.num_classes = num_classes
            self.num_anchors = num_anchors

            self.prior_prob = prior_prob

            self.INF = 1e8  # Maximum value

            self._init_layers()
            # self._init_weights() #TF includes the ini

        def _init_layers(self):
            self.cls_subnet = tf.keras.Sequential()
            self.reg_subnet = tf.keras.Sequential()

            for i in range(self.cls_layers):
                self.cls_subnet.add(layers.Conv2D(512, (3, 3), strides=1, padding="same",
                                                  kernel_initializer=self.kernel_init))
                self.cls_subnet.add(layers.BatchNormalization())  # Default initializations for this already match the paper's code
                self.cls_subnet.add(layers.Activation(self.activation))

            for i in range(self.reg_layers):
                self.reg_subnet.add(layers.Conv2D(512, (3, 3), strides=1, padding="same", kernel_initializer=self.kernel_init))
                self.reg_subnet.add(layers.BatchNormalization())
                self.reg_subnet.add(layers.Activation(self.activation))

            bias_value = -tf.math.log((1 - self.prior_prob) / self.prior_prob)

            self.cls_score = layers.Conv2D(self.num_anchors * self.num_classes, (3, 3), strides=1, padding="same",
                                           kernel_initializer=self.kernel_init,
                                           bias_initializer=tf.keras.initializers.Constant(bias_value))
            self.bbox_pred = layers.Conv2D(self.num_anchors * 4, (3, 3), strides=1, padding="same",
                                           kernel_initializer=self.kernel_init)
            self.object_pred = layers.Conv2D(self.num_anchors, (3, 3), strides=1, padding="same",
                                             kernel_initializer=self.kernel_init)

        def call(self, inputs, training=False):  # This is the forward pass
            cls_score = self.cls_score(self.cls_subnet(inputs))
            N, H, W = tf.reshape(cls_score)[0], tf.reshape(cls_score)[1],tf.reshape(cls_score)[2]
            cls_score = tf.reshape(cls_score,[N,H,W,-1,self.num_classes])

            reg_feat = self.bbox_subnet(inputs)  # We are effectively saving this activation to input twice
            bbox_reg = self.bbox_pred(reg_feat)
            bbox_reg = tf.reshape(bbox_reg,(N,-1,4))
            objectness = self.object_pred(reg_feat)

            # Objectness multiplication
            objectness = tf.reshape(objectness, [N, H, W,-1,1])

            # This is a softmax with guards against exp overflows
            normalized_cls_score = cls_score + objectness - tf.math.log(
                1. + tf.minimum(tf.math.exp(cls_score), self.INF) + tf.minimum(tf.math.exp(objectness), self.INF))
            normalized_cls_score = tf.reshape(normalized_cls_score,[N,-1,self.num_classes])

            #normalized_cls_score = tf.reshape(normalized_cls_score, [N, -1, H, W])

            return normalized_cls_score, bbox_reg

###Inserted Class Codes

import math
from typing import Tuple

_DEFAULT_SCALE_CLAMP = math.log(1000.0 / 16)

class YOLOFBox2BoxTransform(object):
    """
    The box-to-box transform defined in R-CNN. The transformation is
    parameterized by 4 deltas: (dx, dy, dw, dh). The transformation scales
    the box's width and height by exp(dw), exp(dh) and shifts a box's center
    by the offset (dx * width, dy * height).
    We add center clamp for the predict boxes.
    """

    def __init__(
            self,
            weights: Tuple[float, float, float, float],
            scale_clamp: float = _DEFAULT_SCALE_CLAMP,
            add_ctr_clamp: bool = False,
            ctr_clamp: int = 32
    ):
        """
        Args:
            weights (4-element tuple): Scaling factors that are applied to the
                (dx, dy, dw, dh) deltas. In Fast R-CNN, these were originally
                set such that the deltas have unit variance; now they are
                treated as hyperparameters of the system.
            scale_clamp (float): When predicting deltas, the predicted box
                scaling factors (dw and dh) are clamped such that they are
                <= scale_clamp.
            add_ctr_clamp (bool): Whether to add center clamp, when added, the
                predicted box is clamped is its center is too far away from
                the original anchor's center.
            ctr_clamp (int): the maximum pixel shift to clamp.
        """
        self.weights = weights
        self.scale_clamp = scale_clamp
        self.add_ctr_clamp = add_ctr_clamp
        self.ctr_clamp = ctr_clamp
        
    def get_deltas(self, src_boxes, target_boxes):
        """
        Get box regression transformation deltas (dx, dy, dw, dh) that can be
        used to transform the `src_boxes` into the `target_boxes`. That is,
        the relation ``target_boxes == self.apply_deltas(deltas,
        src_boxes)`` is true (unless any delta is too large and is clamped).
        Args:
            src_boxes (Tensor): source boxes, e.g., object proposals
            target_boxes (Tensor): target of the transformation, e.g.,
                ground-truth boxes.
        """
        assert isinstance(src_boxes, tf.Tensor), type(src_boxes)
        assert isinstance(target_boxes, tf.Tensor), type(target_boxes)

        src_widths = src_boxes[..., 2] - src_boxes[..., 0] #Subtracting corner x coordinates
        src_heights = src_boxes[..., 3] - src_boxes[..., 1] #Subtracting corner y coordinates
        src_ctr_x = src_boxes[..., 0] + 0.5 * src_widths #Getting center point x
        src_ctr_y = src_boxes[..., 1] + 0.5 * src_heights #Getting center point y

        target_widths = target_boxes[..., 2] - target_boxes[..., 0] #We repeat all the above for targets
        target_heights = target_boxes[..., 3] - target_boxes[..., 1]
        target_ctr_x = target_boxes[..., 0] + 0.5 * target_widths
        target_ctr_y = target_boxes[..., 1] + 0.5 * target_heights

        wx, wy, ww, wh = self.weights
        dx = wx * (target_ctr_x - src_ctr_x) / src_widths
        dy = wy * (target_ctr_y - src_ctr_y) / src_heights
        dw = ww * tf.math.log(target_widths / src_widths)
        dh = wh * tf.math.log(target_heights / src_heights)

        deltas = tf.stack((dx, dy, dw, dh))
        assert (src_widths > 0).all().item(), \
            "Input boxes to Box2BoxTransform are not valid!"
        return deltas
    
    def apply_deltas(self, deltas, boxes):
        """
        Apply transformation `deltas` (dx, dy, dw, dh) to `boxes`.
        Args:
            deltas (Tensor): transformation deltas of shape (N, k*4),
                where k >= 1. deltas[i] represents k potentially different
                class-specific box transformations for the single box boxes[i].
            boxes (Tensor): boxes to transform, of shape (N, 4)
        """
        deltas = deltas.float()  # ensure fp32 for decoding precision
        boxes = tf.dtypes.cast(boxes, deltas.dtype)

        widths = boxes[..., 2] - boxes[..., 0]
        heights = boxes[..., 3] - boxes[..., 1]
        ctr_x = boxes[..., 0] + 0.5 * widths
        ctr_y = boxes[..., 1] + 0.5 * heights

        wx, wy, ww, wh = self.weights
        dx = deltas[..., 0::4] / wx
        dy = deltas[..., 1::4] / wy
        dw = deltas[..., 2::4] / ww
        dh = deltas[..., 3::4] / wh

        # Prevent sending too large values into exp by using clip_by_value (substitute for clamp)
        dx_width = dx * widths[..., None]
        dy_height = dy * heights[..., None]
        if self.add_ctr_clamp:
            dx_width = tf.clip_by_value(dx_width,
                                   clip_value_max=self.ctr_clamp,
                                   clip_value_min=-self.ctr_clamp)
            dy_height = tf.clip_by_value(dy_height,
                                    clip_value_max=self.ctr_clamp,
                                    clip_value_min=-self.ctr_clamp)
        dw = tf.clip_by_value(dw, clip_value_max=self.scale_clamp)
        dh = tf.clip_by_value(dh, clip_value_max=self.scale_clamp)

        pred_ctr_x = dx_width + ctr_x[..., None]
        pred_ctr_y = dy_height + ctr_y[..., None]
        pred_w = tf.math.exp(dw) * widths[..., None]
        pred_h = tf.math.exp(dh) * heights[..., None]

        x1 = pred_ctr_x - 0.5 * pred_w
        y1 = pred_ctr_y - 0.5 * pred_h
        x2 = pred_ctr_x + 0.5 * pred_w
        y2 = pred_ctr_y + 0.5 * pred_h
        pred_boxes = tf.stack((x1, y1, x2, y2))
        return pred_boxes.reshape(deltas.shape)

def box_xyxy_to_cxcywh(x): #Convert coordinates from corners to center and width+height
    x0, y0, x1, y1 = tf.unstack(x) #Iterable unpacking that's compatible with tf execution
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return tf.stack(b)

def abs_norm(x, y): #equivalent to pytorch's cdist for p=1
    return tf.reduce_sum(tf.math.abs(x - y))

class UniformMatcher(tf.keras.Model):
    
    def __init__(self, match_times=4):
        super().__init__()
        self.match_times = match_times
        
    def call(self, pred_boxes, anchors, targets):
        bs, num_queries = pred_boxes.shape[:2]
        
        out_bbox = tf.reshape(pred_boxes, [-1])
        anchors = tf.reshape(anchors, [-1])
        
        tgt_bbox = tf.concat([v.gt_boxes.tensor for v in targets])
        
        # Compute the L1 cost between boxes (which is just the sum of elements for the difference)
        # Note that we use anchors and predict boxes both
        tf.reduce_sum(tf.math.abs(x - y), 1)
        
        #These are L1-norm costs
        cost_bbox = abs_norm(box_xyxy_to_cxcywh(out_bbox), box_xyxy_to_cxcywh(tgt_bbox))
        cost_bbox_anchors = abs_norm(box_xyxy_to_cxcywh(anchors), box_xyxy_to_cxcywh(tgt_bbox))
        
        # Final cost matrix
        C = cost_bbox
        C = tf.reshape(C, [bs, num_queries, -1])#.cpu() #Note: This means that C has 3 dimensions
        C1 = cost_bbox_anchors
        C1 = tf.reshape(C1, [bs, num_queries, -1])#.cpu()
        
        sizes = [len(v.gt_boxes.tensor) for v in targets]
        all_indices_list = [[] for _ in range(bs)]
        
        # positive indices when matching predict boxes and gt boxes [This is for C, the bounding box predictions]
        indices = [
            tuple(
                torch.topk( #Note this has Sorting by default, as in original implementation
                    tf.transpose(c[i]), #We have to transpose so that we're searching the first dim rather than last
                    k=self.match_times,
                    largest=False)[1].numpy().tolist() #topk returns both the values and the indices, [1] specifies indices
            )
            for i, c in enumerate(C.split(sizes, -1)) #Final output is a list of tuples
        ]
        
        indices1 = [
            tuple(
                torch.topk( #Note this has Sorting by default, as in original implementation
                    tf.transpose(c[i]), #We have to transpose so that we're searching the first dim rather than last
                    k=self.match_times,
                    largest=False)[1].numpy().tolist() #topk returns both the values and the indices, [1] specifies indices
            )
            for i, c in enumerate(C1.split(sizes, -1)) #Final output is a list of tuples
        ]
        
        # concat the indices according to image ids
        for img_id, (idx, idx1) in enumerate(zip(indices, indices1)):
            img_idx_i = [
                np.array(idx_ + idx1_)
                for (idx_, idx1_) in zip(idx, idx1)
            ]
            img_idx_j = [
                np.array(list(range(len(idx_))) + list(range(len(idx1_))))
                for (idx_, idx1_) in zip(idx, idx1)
            ]
            all_indices_list[img_id] = [*zip(img_idx_i, img_idx_j)]
            
        # re-organize the positive indices
        all_indices = []
        for img_id in range(bs):
            all_idx_i = []
            all_idx_j = []
            for idx_list in all_indices_list[img_id]:
                idx_i, idx_j = idx_list
                all_idx_i.append(idx_i)
                all_idx_j.append(idx_j)
            all_idx_i = np.hstack(all_idx_i)
            all_idx_j = np.hstack(all_idx_j)
            all_indices.append((all_idx_i, all_idx_j))
            
        return [
            (tf.convert_to_tensor(i, dtype=tf.int64),
             tf.convert_to_tensor(j, dtype=tf.int64))
            for i, j in all_indices
        ]

###End of Insertion

class YOLOF_model(keras.Model):
    def __init__(self,num_classes,num_anchors=9,dilation_rates=2):
        super(YOLOF_model, self).__init__()
        self.backbone = get_backbone()
        self.encoder =Encoder(input_shapes=(None,None,2048), dilation_rates=dilation_rates)
        self.decoder = Decoder(num_classes=num_classes,num_anchors=num_anchors)
        
        self.anchor_generator = AnchorBox() 
        
        self.box2box_transform = YOLOFBox2BoxTransform(weights=(1, 1, 1, 1)) #We have to specify a tuple of 4 weights; do not keep these default
        self.anchor_matcher = UniformMatcher() #This has a default number of match times
        
    def call(self, inputs): #Where is our preprocessing?
        x = self.backbone(inputs)
        
        #Here is where we would get the anchors
        anchors_image = self.anchor_generator.get_anchors(x.shape[-2], x.shape[-1]) #Height and Width
        anchors = [copy.deepcopy(anchors_image) for _ in range(num_images)]
        
        x = self.encoder(x)
        cls, bbox = self.decoder(x) #cls is pred_logits, bbox is pred_anchor_deltas
        
        #o = tf.concat([bbox,cls],aris=-1)
        return o
        
    def get_ground_truth(self, anchors, bbox_preds, targets):
        anchors = [tf.concat(anchors_i) for anchors_i in anchors]
        N = len(anchors)
        # list[Tensor(R, 4)], one for each image
        
        all_anchors = tf,reshape(tf.concat(anchors), [N, -1, 4])
        # Boxes(Tensor(N*R, 4))
        box_delta = tf.concat(bbox_preds, axis=1)
        # box_pred: xyxy; targets: xyxy
        box_pred = self.box2box_transform.apply_deltas(box_delta, all_anchors)
        indices = self.anchor_matcher(box_pred, all_anchors, targets)
        return indices
        
    

"""
## Implementing a custom layer to decode predictions
"""


class DecodePredictions(tf.keras.layers.Layer):
    """A Keras layer that decodes predictions of the RetinaNet model.
    Attributes:
      num_classes: Number of classes in the dataset
      confidence_threshold: Minimum class probability, below which detections
        are pruned.
      nms_iou_threshold: IOU threshold for the NMS operation
      max_detections_per_class: Maximum number of detections to retain per
       class.
      max_detections: Maximum number of detections to retain across all
        classes.
      box_variance: The scaling factors used to scale the bounding box
        predictions.
    """

    def __init__(
        self,
        num_classes=80,
        confidence_threshold=0.05,
        nms_iou_threshold=0.5,
        max_detections_per_class=100,
        max_detections=100,
        box_variance=[0.1, 0.1, 0.2, 0.2],
        **kwargs
    ):
        super(DecodePredictions, self).__init__(**kwargs)
        self.num_classes = num_classes
        self.confidence_threshold = confidence_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self.max_detections_per_class = max_detections_per_class
        self.max_detections = max_detections

        self._anchor_box = AnchorBox()
        self._box_variance = tf.convert_to_tensor(
            [0.1, 0.1, 0.2, 0.2], dtype=tf.float32
        )

    def _decode_box_predictions(self, anchor_boxes, box_predictions):
        boxes = box_predictions * self._box_variance
        boxes = tf.concat(
            [
                boxes[:, :, :2] * anchor_boxes[:, :, 2:] + anchor_boxes[:, :, :2],
                tf.math.exp(boxes[:, :, 2:]) * anchor_boxes[:, :, 2:],
            ],
            axis=-1,
        )
        boxes_transformed = convert_to_corners(boxes)
        return boxes_transformed

    def call(self, images, predictions):
        image_shape = tf.cast(tf.shape(images), dtype=tf.float32)
        anchor_boxes = self._anchor_box.get_anchors(image_shape[1], image_shape[2])
        box_predictions = predictions[:, :, :4]
        cls_predictions = tf.nn.sigmoid(predictions[:, :, 4:])
        boxes = self._decode_box_predictions(anchor_boxes[None, ...], box_predictions)

        return tf.image.combined_non_max_suppression(
            tf.expand_dims(boxes, axis=2),
            cls_predictions,
            self.max_detections_per_class,
            self.max_detections,
            self.nms_iou_threshold,
            self.confidence_threshold,
            clip_boxes=False,
        )


"""
## Implementing Smooth L1 loss and Focal Loss as keras custom losses
"""


class RetinaNetBoxLoss(tf.losses.Loss):
    """Implements Smooth L1 loss"""

    def __init__(self, delta):
        super(RetinaNetBoxLoss, self).__init__(
            reduction="none", name="RetinaNetBoxLoss"
        )
        self._delta = delta

    def call(self, y_true, y_pred):
        difference = y_true - y_pred
        absolute_difference = tf.abs(difference)
        squared_difference = difference ** 2
        loss = tf.where(
            tf.less(absolute_difference, self._delta),
            0.5 * squared_difference,
            absolute_difference - 0.5,
        )
        return tf.reduce_sum(loss, axis=-1)


class RetinaNetClassificationLoss(tf.losses.Loss):
    """Implements Focal loss"""

    def __init__(self, alpha, gamma):
        super(RetinaNetClassificationLoss, self).__init__(
            reduction="none", name="RetinaNetClassificationLoss"
        )
        self._alpha = alpha
        self._gamma = gamma

    def call(self, y_true, y_pred):
        cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=y_true, logits=y_pred
        )
        probs = tf.nn.sigmoid(y_pred)
        alpha = tf.where(tf.equal(y_true, 1.0), self._alpha, (1.0 - self._alpha))
        pt = tf.where(tf.equal(y_true, 1.0), probs, 1 - probs)
        loss = alpha * tf.pow(1.0 - pt, self._gamma) * cross_entropy
        return tf.reduce_sum(loss, axis=-1)


class RetinaNetLoss(tf.losses.Loss):
    """Wrapper to combine both the losses"""

    def __init__(self, num_classes=80, alpha=0.25, gamma=2.0, delta=1.0):
        super(RetinaNetLoss, self).__init__(reduction="auto", name="RetinaNetLoss")
        self._clf_loss = RetinaNetClassificationLoss(alpha, gamma)
        self._box_loss = RetinaNetBoxLoss(delta)
        self._num_classes = num_classes

    def call(self, y_true, y_pred):
        y_pred = tf.cast(y_pred, dtype=tf.float32)
        box_labels = y_true[:, :, :4]
        box_predictions = y_pred[:, :, :4]
        cls_labels = tf.one_hot(
            tf.cast(y_true[:, :, 4], dtype=tf.int32),
            depth=self._num_classes,
            dtype=tf.float32,
        )
        cls_predictions = y_pred[:, :, 4:]
        positive_mask = tf.cast(tf.greater(y_true[:, :, 4], -1.0), dtype=tf.float32)
        ignore_mask = tf.cast(tf.equal(y_true[:, :, 4], -2.0), dtype=tf.float32)
        clf_loss = self._clf_loss(cls_labels, cls_predictions)
        box_loss = self._box_loss(box_labels, box_predictions)
        clf_loss = tf.where(tf.equal(ignore_mask, 1.0), 0.0, clf_loss)
        box_loss = tf.where(tf.equal(positive_mask, 1.0), box_loss, 0.0)
        normalizer = tf.reduce_sum(positive_mask, axis=-1)
        clf_loss = tf.math.divide_no_nan(tf.reduce_sum(clf_loss, axis=-1), normalizer)
        box_loss = tf.math.divide_no_nan(tf.reduce_sum(box_loss, axis=-1), normalizer)
        loss = clf_loss + box_loss
        return loss


"""
## Setting up training parameters
"""

model_dir = "retinanet/"
label_encoder = LabelEncoder()

num_classes = 80
batch_size = 1

learning_rates = [2.5e-06, 0.000625, 0.00125, 0.0025, 0.00025, 2.5e-05]
learning_rate_boundaries = [125, 250, 500, 240000, 360000]
learning_rate_fn = tf.optimizers.schedules.PiecewiseConstantDecay(
    boundaries=learning_rate_boundaries, values=learning_rates
)

"""
## Initializing and compiling model
"""
loss_fn = RetinaNetLoss(num_classes)
model = YOLOF_model(num_classes)

optimizer = tf.optimizers.SGD(learning_rate=learning_rate_fn,momentum=0.9)
model.compile(loss=loss_fn,optimizer=optimizer)




"""
## Setting up callbacks
"""

callbacks_list = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(model_dir, "weights" + "_epoch_{epoch}"),
        monitor="loss",
        save_best_only=False,
        save_weights_only=True,
        verbose=1,
    )
]

"""
## Load the COCO2017 dataset using TensorFlow Datasets
"""

#  set `data_dir=None` to load the complete dataset

(train_dataset, val_dataset), dataset_info = tfds.load(
    "coco/2017", split=["train", "validation"], with_info=True, data_dir="data"
)

"""
## Setting up a `tf.data` pipeline
To ensure that the model is fed with data efficiently we will be using
`tf.data` API to create our input pipeline. The input pipeline
consists for the following major processing steps:
- Apply the preprocessing function to the samples
- Create batches with fixed batch size. Since images in the batch can
have different dimensions, and can also have different number of
objects, we use `padded_batch` to the add the necessary padding to create
rectangular tensors
- Create targets for each sample in the batch using `LabelEncoder`
"""

autotune = tf.data.AUTOTUNE
#autotune = tf.data.experimental.AUTOTUNE
train_dataset = train_dataset.map(preprocess_data, num_parallel_calls=autotune)
train_dataset = train_dataset.shuffle(8 * batch_size)
train_dataset = train_dataset.padded_batch(
    batch_size=batch_size, padding_values=(0.0, 1e-8, -1), drop_remainder=True
)
train_dataset = train_dataset.map(
    label_encoder.encode_batch, num_parallel_calls=autotune
)
train_dataset = train_dataset.apply(tf.data.experimental.ignore_errors())
train_dataset = train_dataset.prefetch(autotune)

val_dataset = val_dataset.map(preprocess_data, num_parallel_calls=autotune)
val_dataset = val_dataset.padded_batch(
    batch_size=1, padding_values=(0.0, 1e-8, -1), drop_remainder=True
)
val_dataset = val_dataset.map(label_encoder.encode_batch, num_parallel_calls=autotune)
val_dataset = val_dataset.apply(tf.data.experimental.ignore_errors())
val_dataset = val_dataset.prefetch(autotune)

"""
## Training the model
"""

# Uncomment the following lines, when training on full dataset
# train_steps_per_epoch = dataset_info.splits["train"].num_examples // batch_size
# val_steps_per_epoch = \
#     dataset_info.splits["validation"].num_examples // batch_size

# train_steps = 4 * 100000
# epochs = train_steps // train_steps_per_epoch

epochs = 1

# Running 100 training and 50 validation steps,
# remove `.take` when training on the full dataset
model.fit(
    train_dataset.take(100),
    validation_data=val_dataset.take(50),
    epochs=epochs,
    callbacks=callbacks_list,
    verbose=1,
)
# with tf.device('/cpu:0'):
#     model.fit(
#         train_dataset.skip(455),
#         validation_data=val_dataset.take(50),
#         epochs=epochs,
#         callbacks=callbacks_list,
#         verbose=1,
# )

"""
## Loading weights
"""

# Change this to `model_dir` when not using the downloaded weights
weights_dir = "data"

latest_checkpoint = tf.train.latest_checkpoint(weights_dir)
model.load_weights(latest_checkpoint)

"""
## Building inference model
"""

image = tf.keras.Input(shape=[None, None, 3], name="image")
predictions = model(image, training=False)
detections = DecodePredictions(confidence_threshold=0.5)(image, predictions)
inference_model = tf.keras.Model(inputs=image, outputs=detections)

"""
## Generating detections
"""


def prepare_image(image):
    image, _, ratio = resize_and_pad_image(image, jitter=None)
    image = tf.keras.applications.resnet.preprocess_input(image)
    return tf.expand_dims(image, axis=0), ratio


val_dataset = tfds.load("coco/2017", split="validation", data_dir="data")
int2str = dataset_info.features["objects"]["label"].int2str

for sample in val_dataset.take(2):
    image = tf.cast(sample["image"], dtype=tf.float32)
    input_image, ratio = prepare_image(image)
    detections = inference_model.predict(input_image)
    num_detections = detections.valid_detections[0]
    class_names = [
        int2str(int(x)) for x in detections.nmsed_classes[0][:num_detections]
    ]
    visualize_detections(
        image,
        detections.nmsed_boxes[0][:num_detections] / ratio,
        class_names,
        detections.nmsed_scores[0][:num_detections],
    )


