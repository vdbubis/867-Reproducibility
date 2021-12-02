import math
from typing import Tuple

import tensorflow as tf

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
        pass #This is a dummy function; it's not needed anywhere in the implementation, so we've left it for now.
    
    def apply_deltas(self, deltas, boxes):
        deltas = tf.cast(deltas, 'float32') #Ensure fp32 for compatability
        boxes = tf.cast(boxes, deltas.dtype) #Match type

        #deltas are in format (N, WHA, 4)
        #boxes are in format (N, WHA, 4)
        #both are already in xywh

        variance = tf.convert_to_tensor(self.weights)
        variance = variance[None, None, ...] #Expand dimensions for broadcastability with (N, WHA, 4)
        variance = tf.cast(variance, deltas.dtype)
        
        #center deltas are proportions of the original anchor dimensions (to apply to the center)
        #dimension deltas are logarithmic units for multiplying the original anchor dimensions
        
        deltas = deltas / variance
        
        #Next we briefly split our tensors so we can apply separate clamps and update rules
        ctr_deltas = deltas[..., :2]
        dim_deltas = deltas[..., 2:]
        
        ctr_boxes = boxes[..., :2]
        dim_boxes = boxes[..., 2:]
        
        ctr_deltas = ctr_deltas * dim_boxes #Adjust according to size of box
        if self.add_ctr_clamp:
            ctr_deltas = tf.clip_by_value(ctr_deltas, clip_value_min= -self.ctr_clamp, clip_value_max= self.ctr_clamp)
        
        #clipping the dimensions deltas is necessary to prevent exp overflow, so it is not optional
        dim_deltas = tf.clip_by_value(dim_deltas, clip_value_min= -self.scale_clamp, clip_value_max= self.scale_clamp)

        ctr_pred = ctr_boxes + ctr_deltas
        dim_pred = dim_boxes * tf.math.exp(dim_deltas)
        
        pred_boxes = tf.concat([ctr_pred, dim_pred], axis=2) #Reunite into (N, WHA, 4)

        return pred_boxes