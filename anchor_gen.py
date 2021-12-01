import tensorflow as tf

#Generator for a single pyramid level of anchors, P5
#These are returned in (W, H, A, 4) with xywh format
class AnchorGenerator:
    def __init__(self,
                 sizes = [32, 64, 128, 256, 512], #Default for retinanet
                 downsample_rate = 32 #Default for C5
                ):
        self.sizes = sizes #Sidelengths for square boxes
        self.num_anchors = len(sizes) #This would be multiplied by len of aspect_ratios if we include them
        
        self.downsample_rate = downsample_rate #This is also stride
    
    def get_anchors(self, feature_width, feature_height):
        columns = tf.range(feature_width) #Start with sequenial integers for both xy; starts at 0
        rows = tf.range(feature_height)
        centers = tf.meshgrid(columns, rows) #Repeat both to obtain two tensors the shape of (H, W)
        centers = tf.stack(centers, axis=2) #Stack to obtain a (H, W, 2) tensor, a pair of center indexes for each loc

        centers = tf.cast(centers, 'float32') #Convert in case we have unusual strides (and python type compatability)

        centers = centers * self.downsample_rate + (self.downsample_rate/2) #Apply strides, move coord to the box center
        centers = tf.expand_dims(centers, axis=2) #Add dimension for anchor sizes
        centers = tf.tile(centers, [1, 1, self.num_anchors, 1]) #Repeat location for each anchor size. Now (H, W, A, 2)

        dims = tf.convert_to_tensor(self.sizes) #Create list of dimensions
        dims = tf.stack([dims, dims], axis=1) #Create list of square dimensions
        dims = tf.cast(dims, 'float32') #Convert ahead of aspect ratio, and for compatability

        #If we were adding aspect ratios, this is where we would do it.
        #We would do this with a list comprehension for element-wise multiplication for each ratio, then concat the list

        dims = tf.expand_dims(tf.expand_dims(dims, axis=0), axis=0) #Add xy dimensions
        dims = tf.tile(dims, [feature_width, feature_height, 1, 1]) #Tile for (H, W, A, 2)

        anchors = tf.concat([centers, dims], axis=3) #This yields the desired (H, W, A, 4), bboxes in xywh format

        return anchors