## Reproducibility Code for You Only Look One Level Feature

This repository contains code relevant to the attempted re-implementation of You Only Look One Level Feature, which can be found on https://arxiv.org/abs/2103.09460. As we deemed full reproduction beyond what was possible under the scope of this project, we have included the code with which we reached that conclusion. Due to the problems discussed in our report, this code currently can not function beyond forward passes, an example of which is in forward_pass_prototype.py.

### Acknowledgement and Links

The retinanet_utils.py script is a slightly modified version of the retinanet demo found at https://github.com/keras-team/keras-io/blob/master/examples/vision/retinanet.py. We have included it for the data pipeline, the backbone, and some of the functions used for manipulating bounding boxes. We have removed the demo itself, as attempting to import it unmodified will cause it to run the entire model to calculate the global variables used for the demo.

This repository is an attempted port of https://github.com/chensnathan/YOLOF, which is the official repository for the paper in question. In bibtex, the citation for their paper is:

    @inproceedings{chen2021you,
      title={You Only Look One-level Feature},
      author={Chen, Qiang and Wang, Yingming and Yang, Tong and Zhang, Xiangyu and Cheng, Jian and Sun, Jian},
      booktitle={IEEE Conference on Computer Vision and Pattern Recognition},
      year={2021}
    }

### Requirements

This is primarily a Tensorflow 2 implementation. Please refer to the official tutorial (https://www.tensorflow.org/install/gpu) for detailed instructions and further requirements for installing Tensorflow with GPU support.

    pip install numpy
    pip install tensorflow-addons
    pip install tensorflow-datasets
    
Note that Tensorflow Addons and Tensorflow Datasets must be installed separately from Tensorflow itself, as they are not part of its package. The addons are used for GIoU and focal loss functions, and the datasets package is required for the data pipeline in the retinanet demo, which we have adapted for our own code.

### Architectural Details

[Figure 9 from Chen et al](/figures/model.png)

The above is Figure 9 from the original paper. The layout of our network is identical, with modifications to tensor format, bounding box format, and with predicted boxes returned rather than box deltas.

Wherever a tensor in the original implementation is of the form [N, \_, H, W], our implementation instead uses tensors of form [N, W, H, \_]. Feature outputs from network layers in Tensorflow will default to the last dimension, where they remain through all reshapings and broadcasts. In the case of W and H, we have their order match the format of our bounding boxes, which are in xywh format for most of the process.  We convert boxes to xywh format as part of our preprocessing, and in the existing implementation, it is only necessary to convert to the ocrner xyxy format for TFA's implementation of GIoU loss. Additionally, our bounding boxes are expressed in pixel units, and not percentages of their respective images.

We have moved the delta application out of the loss function and into the model itself, so that our model is returning box predictions rather than the deltas. This also means that the only thing dependent on box_reg.py is the YOLOF model class itself. Additionally, we do not flatten our anchors tensor when applying deltas or matching; this is to reduce the need to slice for specific coordinates.

[Figure 4 from our report](/figures/YOLOF_training.png)

This figure from our report illustrates the training process. Note that we have created a simple anchor generator, which is called both for generating predicted boxes, and for matching anchors in uniform matching. Given the structure of the original paper and the design of the uniform matcher, it is impossible to separate it from the training process.

At this time, there is no support for config files; the model can only be initialized using \_\_init\_\_(). For an example of this, refer to the forward_pass_prototype.py script.

### Future Work

At present, we have no plans to continue work on this repository. In order for this implementation to be completed, an inference function must be implemented for the model that is compatible with COCOEvals (which are available in the tensorflow models repository: https://github.com/tensorflow/models/tree/master/official/vision/detection/evaluation). Following this, the uniform matching method would have to be made functional within Tensorflow through a custom training loop, or have its use of prediction tensors abandoned entirely, in which case, it would be moved to preprocessing as the label encoding step (and in such a case, would no longer be the method used by Chen et Al). The step following this would be to add support for configs and checkpoints, as well as modularity for the matching method (which is currently impractical due to the matching being part of training), as well as the box regression and presence of the encoder (for the purpose of ablation study).