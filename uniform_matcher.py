import numpy as np
import tensorflow as tf

from tensorflow.keras import datasets, layers, models

#def box_xyxy_to_cxcywh(x): #Convert coordinates from corners to center and width+height
#    x0, y0, x1, y1 = tf.unstack(x) #Iterable unpacking that's compatible with tf execution
#    b = [(x0 + x1) / 2, (y0 + y1) / 2,
#         (x1 - x0), (y1 - y0)]
#    return tf.stack(b)

#We don't need this because our boxes are already in xywh

def abs_norm(x, y): #equivalent to pytorch's cdist for p=1
    return tf.reduce_sum(tf.math.abs(x - y), axis=1)

def cost_matrix(x, y): #This only works on one image at a time
    return tf.stack([abs_norm(x, y_sub) for y_sub in y])

class UniformMatcher(tf.keras.Model):
    
    def __init__(self, match_times=4):
        super().__init__()
        self.match_times = match_times
        
    def call(self, pred_boxes, anchors, targets):
        #pred_boxes and anchors are both shape (N, WHA, 4)
        #targets is a N-len list of tensors each shape (K*, 4), with variable K*
    
        bs, num_queries = pred_boxes.shape[:2] #N and WHA respectively
        
        out_bbox = tf.reshape(pred_boxes, [-1, 4]) #Flatten for abs_norm
        anchors = tf.reshape(anchors, [-1, 4]) #Shape is (NWHA, 4)
        
        #Let K** be the sum of K* for each image
        tgt_bbox = tf.concat(targets, axis=0) #This is a (K**, 4) tensor
        
        #These are L1-norm costs
        C_pred = cost_matrix(out_bbox, tgt_bbox)
        C_anchors = abs_norm(anchors, tgt_bbox)
        #Cost matrix returns a (K**, WHA) tensor
        
        #Reshape to sweep costs
        C_pred = tf.reshape(C_pred, [-1, bs, num_queries]) #Separate NWHA into N and WHA
        C_pred = tf.transpose(C_pred, [1, 0, 2]) #Move N to front to iterate over batch
        C_pred = -C_pred #We invert this so top k gives us the least elements instead of greatest
        
        C_anchors = tf.reshape(C_anchors, [-1, bs, num_queries]) #Separate NWHA into N and WHA
        C_anchors = tf.transpose(C_anchors, [1, 0, 2]) #Move N to front to iterate over batch
        C_anchors = -C_pred #We invert this so top k gives us the least elements instead of greatest
        
        num_boxes = [image_boxes.shape[0] for image_boxes in targets] #This gets us K* for each image
        
        all_indices_list = [[] for _ in range(bs)] #Initialize list for each image in batch
        
        #At this point, both cost tensors are shape (N, K**, WHA)
        #We will split on K** to get a series of (N, K* WHA) tensors, from which we can access (K*, WHA)
        #With that, we can top_k the (K*, WHA) for a tensor of (K*, match_times)
        
        # positive indices when matching predict boxes and gt boxes
        indices_pred = [
            tuple( #Note that C[i] is (K*, WHA)
                tf.math.top_k(C_pred[i], k=self.match_times).indices.numpy().tolist() #Convert the top k to list representation
            )
            for i, c in enumerate(tf.split(C_pred, num_boxes, 1)) #We split along K** for K* in c, and i is over N
        ]
        
        # positive indices when matching anchor boxes and gt boxes
        indices_anchors = [
            tuple( #Note that C[i] is (K*, WHA)
                tf.math.top_k(C_anchors[i], k=self.match_times).indices.numpy().tolist() #Convert the top k to list representation
            )
            for i, c in enumerate(tf.split(C_anchors, num_boxes, 1)) #We split along K** for K* in c, and i is over N
        ]
        
        #Concatenate indices according to image ids
        for img_id, (idx_pred, idx_anchor) in enumerate(zip(indices_pred, indices_anchors)):
            img_idx_i = [
                np.array(idx_pred_sub + idx_anchor_sub) #We concatenate the actual lists of indices
                for (idx_pred_sub, idx_anchor_sub) in zip(idx_pred, idx_anchor)
            ]
            img_idx_j = [
                np.array(list(range(len(idx_pred_sub))) + list(range(len(idx_anchor_sub)))) #Sequential lists according to size of above lists
                for (idx_pred_sub, idx_anchor_sub) in zip(idx_pred, idx_anchor)
            ]
            all_indices_list[img_id] = [*zip(img_idx_i, img_idx_j)] #Value of top index, and index value according to origin
        
        # re-organize the positive indices; that is to say, to organize them by image id
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
            
        #Our final format is: for every image (in order of entry), we have a tuple of all the top indices for predictions and anchors
        #As well as the sequential indices representing each bounding box within that image.
            
        return [
            (tf.convert_to_tensor(i, dtype=tf.int64),
             tf.convert_to_tensor(j, dtype=tf.int64))
            for i, j in all_indices
        ]