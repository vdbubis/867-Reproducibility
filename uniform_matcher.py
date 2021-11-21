import numpy as np
import tensorflow as tf

from tensorflow.keras import datasets, layers, models

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