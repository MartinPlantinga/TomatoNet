# --------------------------------------------------------
# Select top 100 predicted boxes for mask branch
# Written by Thanh-Toan Do
# --------------------------------------------------------
import caffe
import yaml
import numpy as np
import numpy.random as npr
from fast_rcnn.config import cfg
from fast_rcnn.bbox_transform import bbox_transform
from fast_rcnn.bbox_transform import clip_boxes, bbox_transform_inv
from fast_rcnn.nms_wrapper import nms
from utils.cython_bbox import bbox_overlaps

import cv2
import os.path as osp
import cPickle
import ipdb
DEBUG = False

class PredToProposalLayer(caffe.Layer):
    """
    Assign object detection proposals to ground-truth targets. Produces proposal
    classification labels and bounding-box regression targets.
    """
    def setup(self, bottom, top):
        layer_params = yaml.load(self.param_str_)
        #self._num_classes = layer_params['num_classes']
        self._max_per_image = layer_params['max_per_image']
        self._thresh = layer_params['thresh'] # MP: default thresh set to 0.05 in test.prototxt file
        # rois_for_mask (0, x1, y1, x2, y2)
        top[0].reshape(1, 5)
        # rois_class_score
        top[1].reshape(1, 1)
        # rois_class_ind
        top[2].reshape(1, 1)
        # rois_final
        top[3].reshape(1, 5)

    def forward(self, bottom, top):
        # Proposal ROIs (0, x1, y1, x2, y2) coming from RPN
        # (i.e., rpn.proposal_layer.ProposalLayer), or any other source
        ipdb.set_trace()
        rois = bottom[0].data

        # print ("==================================rois====================")
        # print rois

        ######## bbox_pred
        box_deltas = bottom[1].data
        print("box_deltas with shape {}: {}".format(box_deltas.shape,box_deltas))
        ########class score
        scores = bottom[2].data
        ########image info
        im_info = bottom[3].data
        im_scale = im_info[0][2]

        # unscale back to raw image space
        boxes_0 = rois[:, 1:5] / im_scale # MP: only for x1,y1,x2,y2 and excluding n.
        pred_boxes = bbox_transform_inv(boxes_0, box_deltas) # MP: reshape bboxes based on delta values (Eqs 1-4 original RCNN paper)
        im_shape = [im_info[0][0], im_info[0][1]]/im_scale #original size of input image
        boxes = clip_boxes(pred_boxes, im_shape) #clip predicted box using original input size

        # MP: printing
        print("im_info with shape {}: {}".format(im_info.shape, im_info))
        print("im_scale with shape {}: {}".format(im_scale.shape, im_scale))
        print("im_shape with shape {} : {}".format(im_shape.shape, im_shape))
        print("pred_boxes shape: {}".format(pred_boxes.shape))
        print("boxes shape: {}".format(boxes.shape))
        print ('+++++++++++++++++++++++++++++++++')

        # print("=========================rois from rpn.proposal_layer")
        # print("=========================shape: " + str(rois.shape))
        # print rois

        # print("=========================rois from rpn.proposal_layer")
        # print("=========================shape: " + str(boxes.shape))
        # print boxes

        max_per_image = self._max_per_image
        thresh = self._thresh
        num_classes = scores.shape[1]
        i = 0 #only support single image
        num_images = 1
        #    all_boxes[cls][image] = N x 5 array of detections in
        #    (x1, y1, x2, y2, score)
        all_boxes = [[[] for _ in xrange(num_images)]    # MP: list of 11 [[]] np.arrays
                     for _ in xrange(num_classes)]

        # print ("=========================num_classes: " + str(num_classes))
        # print ("=========================image size: " + str(im_shape))

        ## for each class (ignoring background class)
        for j in xrange(1, num_classes):

            # if j == 23:
            #     print ("=========================scores[:,j]. j = " + str(j))
            #     print scores[:, j]

            inds = np.where(scores[:, j] > thresh)[0]
            cls_scores = scores[inds, j]
            cls_boxes = boxes[inds, j * 4:(j + 1) * 4] #get boxes correspond to class j

            cls_dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])).astype(np.float32, copy=False)
            # MP: can also be written as cls_dets=np.hstack((cls_boxes, cls_scores.reshape(len(cls_scores,1))))
            # print ("===============================size of dets before nms: " + str(cls_dets.shape))
            # cfg.TEST.NMS = 0.3
            if len(inds) > 0:
                print("cls_dets with length {}: {}".format(len(cls_dets), cls_dets))
            keep = nms(cls_dets, cfg.TEST.NMS) # MP: use nms to get the boxes with highest score and eliminate overlap
            # print ("===============keep in rpn/pred_to_proposal_layer.py======: " + str(keep))
            cls_dets = cls_dets[keep, :]
            # print ("===============================size of dets after nms: " + str(cls_dets.shape))
            all_boxes[j][i] = cls_dets # MP: this array contains a list of 11 arrays, one for each class, where the array contains the bounding boxes for the class and the probability, i.e. [[x1,y1,x2,y2,p1],[x1,y1,x2,y2,p2],...]
            ''' 
            if len(inds) > 0:
                print("inds with length {}: {}".format(len(inds),inds))
                print("cls_scores with length {}: {}".format(len(cls_scores), cls_scores)) 
                print("maximum class score: {}".format(max(cls_scores)))
                print("second maximum class score: {}".format(sorted(cls_scores)[-2]))
                print("minimum class score: {}".format(min(cls_scores)))
                print("cls_boxes with length {}: {}".format(len(cls_boxes),cls_boxes))
                print("cls_dets with length {}: {}".format(len(cls_dets),cls_dets))
                print("all_boxes[j][i] of type {} with shape {}: {}".format(type(all_boxes),len(all_boxes), all_boxes))
            # print ("===================image: " + str(i) + " class: " + str(j))
            # print ("===================shape of all_boxes[j][i]: " + str(all_boxes[j][i].shape))
            # print all_boxes[j][i]
            '''
            # Limit to max_per_image detections *over all classes*
        
        if max_per_image > 0: # MP: max_per_image set to 100
            image_scores = np.hstack([all_boxes[j][i][:, -1]
                                      for j in xrange(1, num_classes)]) # MP: extract the images scores from all_boxes
            if len(image_scores) > max_per_image: # MP: take only 100 boxes maximum per image, sorted on a cls_score basis.
                image_thresh = np.sort(image_scores)[-max_per_image]
                for j in xrange(1, num_classes):
                    keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
                    all_boxes[j][i] = all_boxes[j][i][keep, :]
                    # print ("===================image: " + str(i) + "class: " + str(j))
                    # print ("===================shape of all_boxes[j][i]: " + str(all_boxes[j][i].shape))

        num_boxes = 0
        for j in xrange(1, num_classes):
            num_boxes = num_boxes + all_boxes[j][i].shape[0]

        # print ("===========num_boxes========:" + str(num_boxes))
        num_boxes = max(num_boxes,1) #tranh loi 'Floating point exception(core dumped)'

        # MP: preparing the blobs that are passed to the mask layers.
        rois_for_mask = np.zeros((num_boxes, 5), dtype=np.float32)
        rois_class_score = -1*np.ones((num_boxes, 1), dtype=np.float32)
        rois_class_ind = -1*np.ones((num_boxes, 1), dtype=np.float32)
        rois_final = np.zeros((num_boxes, 5), dtype=np.float32)

        count = 0
        for j in xrange(1, num_classes):
            # MP: the values of i is defined in line 81 and is set to constant value 0 (for one image only)
            all_boxes_j = all_boxes[j][i] #boxes correspond to class ji 
            c = all_boxes_j.shape[0] # MP: c is the number of boxes for class j, e.g. 2 bboxes for class racket
            if c > 0:
                # MP: extract coordinates and class scores of bboxes:
                coors = all_boxes_j[:, 0:4]
                cl_scores = all_boxes_j[:, 4:5]

                # MP: rois_for_mask has shape np.array( [ [0, x1, y1, x2, y2], [0, x1, y1, x2, y2], ... , [0, x1, y1, x2, y2] ] )
                # rois_for_mask is scaled up for 600x1000 images
                # rois_final is not scaled up, but uses the image's size
                rois_for_mask[count:count+c, 1:5] = coors*im_scale # w.r.t big size, e.g., 600x1000
                
                # MP: rois_final has shape np.array( [ [0, x1, y1, x2, y2], [0, x1, y1, x2, y2] ,..., [0, x1, y1, x2, y2] ] )
                rois_final[count:count+c, 1:5] = coors # w.r.t. original image size. rois_final same rois_for_mask but with different scale
                # MP: rois_class_score has shape np.array([[score_1], [score_2], ... , [score_n]])
                rois_class_score[count:count+c, 0:1] = cl_scores
                # MP: rois_class_ind has shape np.array([[label#1], [label#2], ... , [label#n]])
                rois_class_ind[count:count+c, 0:1] = np.tile(j, [c, 1])
                count = count + c

                print("rois_for_mask: {}".format(rois_for_mask))
                print("rois_class_score: {}".format(rois_class_score))
                print("rois_class_ind: {}".format(rois_class_ind))
                print("count: {}".format(count))
        # print ("===================================rois_for_mask")
        # print ("===================================shape: " + str(rois_for_mask.shape))
        # print rois_for_mask

        # rois_for_mask
        # print ("===========OK or NOT========")
        top[0].reshape(*rois_for_mask.shape)
        top[0].data[...] = rois_for_mask
        # print ("===========OK or NOT========")
        # classification score
        top[1].reshape(*rois_class_score.shape)
        top[1].data[...] = rois_class_score

        # class index
        top[2].reshape(*rois_class_ind.shape)
        top[2].data[...] = rois_class_ind

        # rois_final
        top[3].reshape(*rois_final.shape)
        top[3].data[...] = rois_final

        # print ("=======================number of rois_final:====================" + str(rois_final.shape))

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass


