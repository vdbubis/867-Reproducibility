import numpy as np
from typing import Dict, List, Tuple

#We are skipping on visualized training, so no event_storage is needed
#Post-processing was in the implementation, but not mentioned in the paper, so we are excluding it

import tensorflow as tf
from tensorflow import Tensor
import tensorflow_datasets as tfds

from retinanet_utility import compute_iou, get_backbone, swap_xy, convert_to_xywh

from encoder import DilatedEncoder
from decoder import Decoder
from box_regression import YOLOFBox2BoxTransform

from anchor_gen import AnchorGenerator

class YOLOF(tf.keras.Model):    
    def __init__( #Don't forget the super initialization
            self,
            *,
            image_size = (608,608),
            backbone,
            encoder,
            decoder,
            anchor_generator,
            box2box_transform,
            loss_function,
            num_classes,
            test_score_thresh=0.05,
            test_topk_candidates=1000,
            test_nms_thresh=0.6,
            max_detections_per_image=100,
            pixel_mean=(103.53, 116.28, 123.675), #Obtained from original repo comment on ImageNet pixel_mean
            pixel_std=(57.375, 57.120, 58.395), #ImageNet std as noted in original repo comment
            input_format="BGR"
    ):
        super().__init__()
        
        self.image_size = image_size
        
        self.backbone = backbone
        self.encoder = encoder
        self.decoder = decoder
        
        # Anchors
        self.anchor_generator = anchor_generator
        self.box2box_transform = box2box_transform
        self.loss_function = loss_function

        self.num_classes = num_classes
        
        # Inference parameters:
        self.test_score_thresh = test_score_thresh
        self.test_topk_candidates = test_topk_candidates
        self.test_nms_thresh = test_nms_thresh
        self.max_detections_per_image = max_detections_per_image
        
        #Preprocessing
        self.pixel_mean = pixel_mean
        self.pixel_std = pixel_std

    def call(self, inputs, training=False):
        N = inputs.shape[0] #This can't be regular len since it's not well defined for tensors or datasets
        #Data input format from backbone is [N, W, H, 2048], N is number of images
        
        _, _, features = self.backbone(inputs) #Backbone has 3 outputs. We want the one with 2048 channels
        
        feature_width = features.shape[1]
        feature_height = features.shape[2]
        
        pred_logits, pred_anchor_deltas = self.decoder(self.encoder(features))
        
        #Note: The format of pred_anchor_deltas is [N, W, H, AK], so we do not need to permute like original
        
        #We alter the original structure of this code in order to apply the deltas to the anchors, and return
        #the full predicted boxes
        
        anchors = self.anchor_generator.get_anchors(feature_width, feature_height)
        #Anchors in (W, H, A, 4)
        #Our first step is to reshape so that we're comparing boxes on N, WHA, 4 instead
        anchors = tf.reshape(anchors, [1, -1, 4])
        anchors = tf.tile(anchors, [N, 1, 1]) #tile for each image in batch
        
        pred_anchor_deltas = tf.reshape(pred_anchor_deltas, [N, -1, 4]) #Need this shape to easier assign deltas
        
        #Next we apply deltas to the anchors
        pred_boxes = self.box2box_transform.apply_deltas(pred_anchor_deltas, anchors)
        pred_boxes = tf.reshape(pred_boxes, [N, feature_width, feature_height, -1])
        #We return to [N, W, H, A4] format
        
        return pred_logits, pred_boxes
    
    def inference(self, inputs, pred_logits, pred_boxes):
        results = [] #We want to output a tensor, for each image, a set of class ids, confidence scores, and bounding boxes
        N = inputs.shape[0]
        for i in range(N):
            pred_logits_per_image = [x[img_idx] for x in pred_logits]
            boxes_per_image = [x[img_idx] for x in pred_boxes]
            results_per_image = self.inference_single_image(
                anchors, pred_logits_per_image, boxes_per_image
            ) #We don't need anchors. Do we need the inputs at this point?
            results.append(results_per_image)
        return results
    
    def preprocess_image(self, sample):
        #Applies to a single image
        
        image = sample["image"]
        bbox = swap_xy(sample["objects"]["bbox"]) #It's important that we do this, as by default, coco2017 is yxyx format
        class_id = tf.cast(sample["objects"]["label"], dtype=tf.int32)
        
        #image, bbox = random_flip_horizontal(image, bbox)
        #image, image_shape, _ = resize_and_pad_image(image)
        
        image = tf.image.resize(image, self.image_size)
        image = (image - self.pixel_mean) / self.pixel_std
        
        #Our model architecture is measured in pixels, not percentage, meaning that we scale up bboxes here
        image_factor = tf.convert_to_tensor(self.image_size * 2)
        image_factor = tf.cast(image_factor, bbox.dtype)
        bbox = convert_to_xywh(bbox)
        bbox = bbox * image_factor[None, ...] #Ensure that we multiply the last axis in broadcast by expanding dim
        
        return image, (class_id, bbox) #Swapping order to match decoder output order