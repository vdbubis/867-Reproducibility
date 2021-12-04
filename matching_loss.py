import tensorflow as tf

from box_regression import YOLOFBox2BoxTransform
from uniform_matcher import UniformMatcher

from retinanet_utility import compute_iou, convert_to_corners

from tensorflow_addons.losses import SigmoidFocalCrossEntropy, giou_loss

class YolofLoss (tf.keras.losses.Loss):
    def __init__(self,
                 anchors,
                 anchor_matcher,
                 num_classes,
                 pos_ignore_thresh=0.15,
                 neg_ignore_thresh=0.7,
                 focal_loss_alpha=0.25,
                 focal_loss_gamma=2.0,
                 box2box_transform=YOLOFBox2BoxTransform(weights=(1, 1, 1, 1))
                ):
        super().__init__()
        
        self.anchors = anchors
        self.anchor_matcher = anchor_matcher
        
        self.num_classes = num_classes
        
        self.box2box_transform = box2box_transform
        
        # Ignore thresholds:
        self.pos_ignore_thresh = pos_ignore_thresh
        self.neg_ignore_thresh = neg_ignore_thresh
        # Loss parameters:
        self.focal_loss_alpha = focal_loss_alpha
        self.focal_loss_gamma = focal_loss_gamma
        
        self.focal_loss = SigmoidFocalCrossEntropy(alpha=focal_loss_alpha, gamma=focal_loss_gamma) #Currently no alternative to this
        
    def call(self, y_true, y_pred):
        #y_true is a list of 2-tuples of tensors of varying lengths K*, the number of objects on an image
        #cls shape is [K*,], bbox shape is [K*, 4]
        
        cls_true = []
        bbox_true = []
        
        for cls, bbox in y_true:
            cls_true.append(cls)
            bbox_true.append(bbox)
        
        #cls_true and bbox true are now lists of the true tensors, length N
        
        #Next, we unpack the prediction tensors from y_pred, which is a 2-tuple
        
        pred_logits, pred_boxes = y_pred
        
        N = pred_logits.shape[0]
        
        #pred_logits is in (N, W, H, AK)
        #pred_boxes is in (N, W, H, 4A)
        
        pred_logits = tf.reshape(pred_logits, [-1, self.num_classes]) # [NWHA, K] shape
        pred_boxes = tf.reshape(pred_boxes, [N, -1, 4]) #Reshape for uniform matching
        
        anchors = self.anchors
        anchors = tf.reshape(anchors, [1, -1, 4])
        anchors = tf.tile(anchors, [N, 1, 1]) #tile for each image in batch
        
        indices = self.anchor_matcher(pred_boxes, anchors, bbox_true)
        
        #Our code is running up to here
        
        ious = []
        pos_ious = []
        for i in range(N):
            iou = compute_iou(pred_boxes[i, ...], bbox_true[i]) #pred_boxes[i, ...] is (WHA, 4) and bbox_true[i] is (K*, 4)
            if tf.size(iou) == 0: #The compute_iou tensor is (WHA, K*)
                max_iou = tf.fill([pred_boxes.shape[0]], 0) #We are finding the closest iou for each prediction, i.e. WHA queries
            else:
                max_iou = tf.reduce_max(iou, axis=1) #Which of the true boxes is each prediction closest to? (WHA) Tensor
            
            #We can use max_iou to determine if a pred_box falls below the negative threshold for all
            
            anchor_candidates = indices[i]
            
            a_iou = compute_iou(anchors[i, ...], bbox_true[i])
            if tf.size(a_iou) == 0:
                pos_iou = tf.fill([0], 0)
            else:
                pos_iou = tf.gather_nd(a_iou, anchor_candidates)
                
            ious.append(max_iou)
            pos_ious.append(pos_iou)
        
        
        ious = tf.concat(ious, axis=0) #Tensor of (NWHA, )
        ignore_idx = ious > self.neg_ignore_thresh #Booldean tensor denoting insignificant overlaps
        pos_ious = tf.concat(pos_ious, axis=0) #Tensor of (N x K* x match_times, )
        pos_ignore_idx = pos_ious < self.pos_ignore_thresh #Boolean tensor 
        
        src_idx = tf.concat([src[..., 0] + idx * anchors.shape[1] for idx, src in enumerate(indices)], axis=0) #Index out of NWHA, with WHA range for matched index, so we multiply by that
        #Note that anchors are [N, WHA, 4], so shape[1] is WHA. We also need to grab only the source indices from the indices tensor,  hence the [..., 0] slice
        
        #Braodly speaking, each pair of predicted boxes and classes falls within three catagories
        #Those that we ignore completely are described by the union of ignore_idx and pos_ignore_idx
        #Those that this paper calls "negative", or that retinanet considers "background",
        #are those that are neither in valid_idxs (complement of ignore union) or described by src_idx
        #src_idx describes foreground/positive elements, which is what we train for their given classes
        
        #Since we slice to valid_idxs, we consider background ids to be anything leftover not described by src_idx
        
        #Therefore, for target classes, we will take a flat concatenation of the true classes by our matched indices,
        #create a boolean tensor describing classes not under src_idx, and mulitply it by the number of classes.
        #Non-background classes range [0, num_classes), so num_classes itself is the background class.
        #We will then take the minimum of the original labels with this background class tensor in order
        #to get our targets for the focal loss.
        
        #Important note: src_idx elements are not necessarily unique, as the same anchor can be shared by multiple true boxes
        
        num_foreground = src_idx[~pos_ignore_idx].shape[0]
        
        #We will start by taking the complement of the above assigned indices (the ones we've ignored)
        valid_idxs = tf.reduce_sum(tf.one_hot(src_idx[pos_ignore_idx], pred_logits.shape[0]), axis=0)
        valid_idxs = tf.cast(valid_idxs, 'bool')
        valid_idxs = ~tf.logical_or(valid_idxs, ignore_idx) #Complement of all ignored elements

        #We will slice src_idx with ~pos_ignore_idx
        foreground_idxs = tf.reduce_sum(tf.one_hot(src_idx[~pos_ignore_idx], pred_logits.shape[0]), axis=0) #Get matched locations booleans as integers
        foreground_idxs = foreground_idxs > 0 #Convert to bool form, also implicitly drop duplicates
        
        background_labels = tf.cast(foreground_idxs, 'int32')
        background_labels = background_labels * -1 + 1 #Invert for background location booleans as integers
        background_labels = background_labels * self.num_classes #Apply background class
        
        target_classes_o = []
        for i in range(N):
            target_classes_o.append(tf.concat([cls_true[i][J] for J in indices[i][..., 1]], axis=0))
        target_classes_o = tf.concat(target_classes_o, axis=0) #original target classes in context of matched anchors
        
        gt_classes = tf.one_hot(src_idx, pred_logits.shape[0], dtype='int32') #Must be int for compatability with other class label structures
        gt_classes = gt_classes * target_classes_o[..., None] #Apply the above labels to the properly shaped tensor
        gt_classes = tf.reduce_max(gt_classes, axis=0) #Flatten to proper shape
        gt_classes = tf.where(foreground_idxs, gt_classes, tf.fill([pred_logits.shape[0]], self.num_classes))
        
        gt_classes_target = tf.one_hot(gt_classes, self.num_classes) #Confidence values for each label
        
        # cls loss
        loss_cls = self.focal_loss(gt_classes_target[valid_idxs], pred_logits[valid_idxs])
        #This is a layer initialized with alpha and gamma during this class's __init__()

        # reg loss
        target_boxes = []
        for i in range(N):
            target_boxes.append(tf.stack([bbox_true[i][J] for J in indices[i][..., 1]]))
        target_boxes = tf.concat(target_boxes, axis=0)
        target_boxes = target_boxes[~pos_ignore_idx]
        
        matched_predicted_boxes = tf.gather(tf.reshape(pred_boxes, (-1, 4)), src_idx[~pos_ignore_idx])
        
        #It's important to remember that up until this point, our boxes have been in xywh format, and giou_loss assumes corners
        #Therefore, we will convert to corners now
        target_boxes = convert_to_corners(target_boxes)
        matched_predicted_boxes = convert_to_corners(matched_predicted_boxes)
        
        loss_box_reg = giou_loss(target_boxes, matched_predicted_boxes) #This is not a layer, so there's no initialization for it
        
        #Our losses are different sizes, so we have to manually reduce them each
        
        loss_cls = tf.reduce_sum(loss_cls)
        loss_box_reg = tf.reduce_sum(loss_box_reg)

        return (loss_cls + loss_box_reg) / max(1, num_foreground) #TODO: Add weighting