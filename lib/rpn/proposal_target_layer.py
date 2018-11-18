# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Sean Bell
# --------------------------------------------------------
import caffe
import yaml
import numpy as np
import numpy.random as npr
from fast_rcnn.config import cfg
from fast_rcnn.bbox_transform import bbox_transform
from utils.cython_bbox import bbox_overlaps
from utils.cython_overlap_ratio import bbox_overlaps as c_bbox_overlaps

import cv2
import os.path as osp
import cPickle
import time
import ipdb
DEBUG = False

# MP: modified version of proposal_target_layer.py.
# Original version renamed as proposal_target_layer_original.py
# VISUALIZE
background = [200, 222, 250]    # Light Sky Blue
c1 = [0, 0, 205]		# ok
c2 = [34, 139, 34]		# ok
c3 = [192, 192, 128]		# 3
c4 = [165, 42, 42]		# ok
c5 = [128, 64, 128]		# 5
c6 = [204, 102, 0]		# 6
c7 = [184, 134, 11]		# ok
c8 = [0, 153, 153]		# ok
c9 = [0, 134, 141]		# ok
c10 = [184, 0, 141]		# ok
label_colors = np.array([background, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10])

# MP: create a new (random) list of label_colors.
from random import randint

mask_colors = np.empty([0])
for i in range(90):
    hex_color = '%06X'%randint(0, 0xFFFFFF)
    rgb_color = np.asarray([int(hex_color[i:i+2], 16) for i in (0, 2 ,4)])
    try:
        mask_colors = np.vstack((mask_colors,rgb_color))
    except ValueError:
        mask_colors = rgb_color

label_colors = mask_colors

out_verbose = 0
verbose_showim = 0


class ProposalTargetLayer(caffe.Layer):
    """
    Assign object detection proposals to ground-truth targets.
    Produces proposal classification labels and bounding-box
    regression targets.
    """

    def setup(self, bottom, top):
        layer_params = yaml.load(self.param_str_)
        self._num_classes = layer_params['num_classes']

        # sampled rois (0, x1, y1, x2, y2)
        top[0].reshape(1, 5)
        # labels
        top[1].reshape(1, 1)
        # bbox_targets
        top[2].reshape(1, self._num_classes * 4) #(x1, y1, x2, y2, cls)
        # bbox_inside_weights
        top[3].reshape(1, self._num_classes * 4)
        # bbox_outside_weights
        top[4].reshape(1, self._num_classes * 4)
	#segmentation mask for positive rois
        top[5].reshape(1, cfg.TRAIN.MASK_SIZE, cfg.TRAIN.MASK_SIZE)
        #positive rois for mask branch
	top[6].reshape(1, 5)

    def forward(self, bottom, top):
    	'''
	MP:
	input:
		blob		name		size 		origin
		bottom[0].data: rpn_rois	(2000 x 4)	proposal (lib/rpn/proposal_layer.py)
		bottom[1].data: gt_boxes	(r x 1)		input-data (lib/roi_data_layer/layer.py)
		bottom[2].data: im_info		(1 x 3)		input-data (lib/roi_data_layer/layer.py)
		bottom[3].data: seg_mask_inds	(n x 2)		input-data (lib/roi_data_layer/layer.py)
		bottom[4].data: flipped		(1 x 1)		input-data (lib/roi_data_layer/layer.py)
	output:
		blob 	name			size			destination
		top[0]:	rois 			(r x 5)			roi_align5
		top[1]:	labels			(r x 1)			loss_cls
		top[2]:	bbox_targets 		(r x 4*no_classes)	loss_bbox
		top[3]:	bbox_inside_weights 	(r x 4*no_classes)	loss_bbox
		top[4]:	bbox_outside_weights 	(r x 4*no_classes)	loss_bbox
		top[5]:	mask_targets 		(p x 244 x 244)		loss_mask
		top[6]:	rois_pos 		(p x 5)			roi_align5_2
	'''
        # MP: Proposal ROIs of shape (0, x1, y1, x2, y2) that originate from the RPN
        rpn_rois = bottom[0].data
        # MP: GT boxes (x1, y1, x2, y2, label)
        gt_boxes = bottom[1].data
        # MP: image info (1x3)
        im_info = bottom[2].data
        # MP: segmenation mask index: array of type [[file_name_1 bbox_index_1], ... , [file_name_n bbox_index_n]]
        seg_mask_inds = bottom[3].data
        # MP: flipped (True or False)
        flipped = bottom[4].data
        '''
	# MP:
	print "bottom[0] (rpn_rois) of size {}: {}".format(all_rois.shape, all_rois)
	print "bottom[1] (gt_boxes) of size {}: {}".format(gt_boxes.shape, gt_boxes)
	print "bottom[2] (im_info of size {}: {}".format(im_info.shape, im_info)
        print "bottom[3] (seg_mask_inds) of size {}: {}".format(seg_mask_inds.shape, seg_mask_inds)
        print "bottom[4] (flipped) of size {}: {}".format(flipped.shape, flipped)
        '''



        num_images = 1
        rois_per_image = cfg.TRAIN.BATCH_SIZE / num_images #cfg.TRAIN.BATCH_SIZE = 32; num_images = 1
        fg_rois_per_image = np.round(cfg.TRAIN.FG_FRACTION * rois_per_image) #cfg.TRAIN.FG_FRACTION = 0.25 => 0.25*32 = 8

	'''
	# MP:
	print "rois_per_image: {}".format(rois_per_image)
	print "fg_rois_per_image: {}".format(fg_rois_per_image)
	'''

        # MP: Stack ground truth and RPN proposed boxes
	# [[0, x1, y1, x2, y2]_rpn ,...,[0, x1, y1, x2, y2]_rpn, [0, x1, y1, x2, y2]_gt, ... , [0, x1, y1, x2, y2]_gt]
        zeros = np.zeros((gt_boxes.shape[0], 1), dtype=gt_boxes.dtype)
        all_rois = np.vstack(
            (rpn_rois, np.hstack((zeros, gt_boxes[:, :-1])))
        )

	# Sanity check: single batch only
        assert np.all(all_rois[:, 0] == 0), \
                'Only single item batches are supported' #first element in each roi is image index (in this mini batch)



        # Sample rois with classification labels and bounding box regression
        # targets

	'''
	MP:
	labels: 		size (32,1)	list w/ label numbers for fg bboxes, otherwise 0
	rois: 			size(32,5)	array with bbox (fg & bg) coordinates [0, x1, y1, x2, y2]
	bbox_targets: 		size(32,44)	array with diff for bboxes
	bbox_inside_weights: 	size(32,44)	array where 1 indicates that there is a diff
	bbox_targets_oris:	size(32,5)	array with coordinates of gt and correspondng label [x1,y1,x2,y2,label]
						the gt corresponds to the proposed roi
	rois_pos:		size(4,5)	same as rois, but only fg bboxes
	gt_assignment_pos:	size(4,1)	list of indices that indicate to which of the gt boxes the fg boxes correspond to
        '''
        labels, \
	rois, \
	bbox_targets, \
	bbox_inside_weights, \
	bbox_targets_oris, \
	rois_pos,\
	gt_assignment_pos = _sample_rois(all_rois,		# (2000+ x 5)
					 gt_boxes,		# (n x 5)
					 fg_rois_per_image,	# 8
					 rois_per_image, 	# 32
					 self._num_classes)	# 11
	# MP: cfg.TRAIN.MASK_REG = True
	if cfg.TRAIN.MASK_REG:
            num_roi = len(rois_pos) #incase output rois_pos (so only need to define mask to rois_pos)

            # out_verbose = False
            if out_verbose: print ("number of rois in the image: " + str(num_roi))
            if out_verbose: print 'rois pos shape: ', rois_pos.shape
            if out_verbose: print 'rois pos: ', rois_pos
            if out_verbose: print "gt_assignment_pos shape: ", gt_assignment_pos.shape
            if out_verbose: print "gt_assignment_pos: ", gt_assignment_pos

            mask_targets = -1 * np.ones((num_roi, cfg.TRAIN.MASK_SIZE, cfg.TRAIN.MASK_SIZE), dtype=np.float32)

            im_ind = seg_mask_inds[0][0]
            im_scale = im_info[0][2]
            im_height = im_info[0][0]
            im_width = im_info[0][1]
            flipped = flipped[0][0]

	    # out_verbose = False
            if out_verbose: print 'im_ind: ', im_ind, '- im height: ', im_height, '-- im_width: ', im_width

            #######################
            # ==== MASK PART ==== #
            #######################

            # read all segmentation masks of this image from hard disk
            mask_ims = []
            mask_flipped_ims = []
            count = 0

            ipdb.set_trace()

            while True:  ## load mask
                '''
                MP: Two lists of masks are created (mask_ims and mask_flipped_ims).
                - Masks are loaded from the data directory.
                - Masks are scaled according to the scale defined in im_info.
                - Nearest values are determined as scaling generates floating point numbers.
                - Masks are appended to the list mask_ims.
                - Masks are flipped and appended to mask_flipped_ims.
                '''
                count += 1
                #seg_mask_path = './data/cache/seg_mask_coco_gt/' + str(int(im_ind)) + '_' + str(int(count)) + '_segmask.sm'
                if cfg.TRAIN.TRAINING_DATA == 'coco_2017_train':
                    seg_mask_path = '../../data/mscoco/mask_rcnn/Segmentation_Masks/trainVal2017/{}_{}_segmask.sm'.format(str(int(im_ind)), str(int(count)))
                elif cfg.TRAIN.TRAINING_DATA == 'coco_2017_train_8000':
                    seg_mask_path = '../../data/mscoco/mask_rcnn/Segmentation_Masks/trainVal2017_81_8000/{}_{}_segmask.sm'.format(str(int(im_ind)), str(int(count)))
                elif cfg.TRAIN.TRAINING_DATA == 'coco_2014_train':
                    seg_mask_path = './data/cache/'+'GTsegmask_'+cfg.TRAIN.TRAINING_DATA+'/' + str(int(im_ind)) + '_' + str(int(count)) + '_segmask.sm'
                elif cfg.TRAIN.TRAINING_DATA == 'TomatoDB':
                    seg_mask_path = '../../data/TomatoDB/train_testval_data/ground_truth/{}_{}_segmask.sm'.format(str(int(im_ind)), str(int(count)))
                elif cfg.TRAIN.TRAINING_DATA == 'VOC_2012_train':
                    im_ind = int(im_ind) ## Not use
                    t = str(int(im_ind)) #i.e. t = '8000008'
                    p = t[-6:] #p = '000008'
                    p2 = t[0:len(t) - len(p)] #p2 = '8'
                    if len(p2) == 1:
                        p2 = '200' + p2 #p2 = '2008' or '2009'
                    if len(p2) == 2:
                        p2 = '20' + p2 #p2 = '2010' or '2011' or '2012'

                    # FOR PASCAL
                    #seg_mask_path = './data/cache/' + 'GTsegmask_' + cfg.TRAIN.TRAINING_DATA + '/' + str(p2) + '_' + str(p) + '_' + str(int(count)) + '_segmask.sm'

                    # FOR IIT-AFF dataset
                    seg_mask_path = './data/cache/' + 'GTsegmask_' + cfg.TRAIN.TRAINING_DATA + '/' + t + '_' + str(int(count)) + '_segmask.sm'

                else:
                    print ("lib/rpn/proposal_target_layer.py: DO NOT KNOW TRAINING DATASET.")

                if osp.exists(seg_mask_path):

                    with open(seg_mask_path, 'rb') as f:
                        mask_im = cPickle.load(f)
                    uni_ids = np.unique(mask_im)
                    org_uni_label = np.unique(mask_im)

                    mask_im = (mask_im).astype('float32')
                    mask_im = cv2.resize(mask_im, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)
                    uni_ids = np.unique(mask_im)

                    mask_im = _convert_mask_to_original_ids_manual_FOR_IMAGE(mask_im, org_uni_label)
                    mask_im = np.asarray(mask_im) #convert mask_im to np array

                    diffmask = mask_im - np.around(mask_im, decimals = 0)

                    # MP: below two lines don't seem to add any value
                    #mask_im = np.around(mask_im, decimals=0)  # round 1.6 -> 2., 1.1 -> 1.
                    #mask_im = mask_im.astype('uint8') # 2. --> 2
                    uni_res = np.unique(mask_im)

                    mask_ims.append(mask_im)

                    # flip images
                    mask_flipped_im = cv2.flip(mask_im, 1) #vertical flip
                    mask_flipped_ims.append(mask_flipped_im)
                else:
                    break

            # create flag array to check if bbox is all zero or not
            flag_arr = np.full((rois_pos.shape[0], ), 1, 'uint8') # (shape = (num_bbox, 1)) # fill with 1: default is good, if not good, set to 0

            for ix, roi in enumerate(rois_pos):
                k = gt_assignment_pos[ix]
                # index of seg mask. minus 1 because the stored mask index in seg_mask_inds is 1-based
                gt_mask_ind = int(seg_mask_inds[k][1]) - 1
                gt_box = gt_boxes[k]
                #roi coordinate
                x1 = round(roi[1])
                y1 = round(roi[2])
                x2 = round(roi[3])
                y2 = round(roi[4])

                ## Anh them vo!
                if x1 == y1 == x2 == y2 == 0:
                    # set flag array [ix] to 0 : found bad bbox
                    flag_arr[ix] = 0


                # MP: make sure bounding box is smaller than image
                x1 = np.min((im_width - 1, np.max((0, x1))))
                y1 = np.min((im_height - 1, np.max((0, y1))))
                x2 = np.min((im_width - 1, np.max((0, x2))))
                y2 = np.min((im_height - 1, np.max((0, y2))))
                w = (x2 - x1) + 1
                h = (y2 - y1) + 1
                # MP: ground truth bbox coordinates
                x1t = round(gt_box[0])
                y1t = round(gt_box[1])
                x2t = round(gt_box[2])
                y2t = round(gt_box[3])
                # sanity check
                x1t = np.min((im_width - 1, np.max((0, x1t))))
                y1t = np.min((im_height - 1, np.max((0, y1t))))
                x2t = np.min((im_width - 1, np.max((0, x2t))))
                y2t = np.min((im_height - 1, np.max((0, y2t))))

                # MP: ground truth bbox label
                cls = gt_box[4]

                # MP:   crop the ground truth mask according to the rois
                #       initiate all mask values with -1
                roi_mask = -1 * np.ones((int(h), int(w)), dtype=np.float32)
                # MP: flipped = False
                if flipped:
                    gt_mask = mask_flipped_ims[gt_mask_ind]
                else:
                    gt_mask = mask_ims[gt_mask_ind]

                uni_ids = np.unique(gt_mask)

                # compute overlap between roi coordinate and gt_roi coordinate
                x1o = int(max(x1, x1t))
                y1o = int(max(y1, y1t))
                x2o = int(min(x2, x2t))
                y2o = int(min(y2, y2t))
                ho = y2o-y1o
                wo = x2o-x1o

                mask_overlap = np.zeros((ho, wo), dtype=np.float32)

                verbose_showim = False
                if verbose_showim:
                    '''
                    MP: display ground truth mask with bounding boxes:
                        red:    ground truth bbox
                        green:  overlap gt and rois bbox
                        blue:   rois bbox
                    '''
                    if cfg.TRAIN.TRAINING_DATA == 'coco_2017_train':
                        imgIdx = int(seg_mask_inds[0][0])
                        imgIdx = "%012d"%imgIdx
                        imgPath = '../../data/mscoco/cocoapi/images/train2017/{}.jpg'.format(imgIdx)
                        import skimage.io as io
                        I = io.imread(imgPath)

                        cv2.imshow("Original Image", I)

                    color_img = label_colors.take(gt_mask, axis=0).astype('uint8')
                    cv2.rectangle(color_img, (int(x1t), int(y1t)), (int(x2t), int(y2)), (0, 0, 255), 8)        # plot roi with Red
                    cv2.rectangle(color_img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 4)    # plot gt with Blue
                    cv2.rectangle(color_img, (int(x1o), int(y1o)), (int(x2o), int(y2o)), (0, 255, 0), 2)    # plot overlap with Green
                    cv2.imshow("color_img", color_img)          # full ground truth mask

                    '''
                    MP: display cropped overlap gt and rois bbox
                    '''
                    mask_overlap_draw = color_img[int(y1o):int(y2o), int(x1o):int(x2o), :]
                    cv2.imshow("overlap", mask_overlap_draw)
                    cv2.waitKey(0)


                mask_overlap[:, :] = gt_mask[int(y1o):int(y2o), int(x1o):int(x2o)]

                if roi_mask.shape[0] > 3 and roi_mask.shape[1] > 3:  #only resize if shape != (0, 0)

                    '''
                    MP: 1. take overlap values and put them im a bbox of size equal to the rois bbox
                        2. resize box to 244 x 244
                    '''
                    roi_mask[int(y1o-y1):int(y2o-y1), int(x1o-x1):int(x2o-x1)] = mask_overlap
                    original_uni_ids = np.unique(roi_mask)

                    roi_mask = cv2.resize(roi_mask.astype('float'), (cfg.TRAIN.MASK_SIZE, cfg.TRAIN.MASK_SIZE), interpolation=cv2.INTER_LINEAR)
                    roi_mask = _convert_mask_to_original_ids_manual(roi_mask, original_uni_ids)

                    if verbose_showim:
                        '''
                        MP: display above defined bbox roi_mask
                       '''

                        color_roi_mask = label_colors.take(roi_mask.astype('int8'), axis=0).astype('int8')
                        cv2.imshow('roi_mask', color_roi_mask)
                        cv2.waitKey(0)

                else:
                    roi_mask = -1 * np.ones((cfg.TRAIN.MASK_SIZE, cfg.TRAIN.MASK_SIZE), dtype=np.float32) # set roi_mask to -1

                '''
                MP: store the roi_masks in mask_arrays
                '''
                mask_targets[ix, :, :] = roi_mask

        if DEBUG:
            print 'num fg: {}'.format((labels > 0).sum())
            print 'num bg: {}'.format((labels == 0).sum())
            self._count += 1
            self._fg_num += (labels > 0).sum()
            self._bg_num += (labels == 0).sum()
            print 'num fg avg: {}'.format(self._fg_num / self._count)
            print 'num bg avg: {}'.format(self._bg_num / self._count)
            print 'ratio: {:.3f}'.format(float(self._fg_num) / float(self._bg_num))


        ## DEBUG

        if out_verbose: print '--- rois shape        : ', rois.shape
        if out_verbose: print '--- labels shape      : ', labels.shape
        if out_verbose: print '--- bbox_targets shape: ', bbox_targets.shape
        if out_verbose: print '--- bbox ins wei shape: ', bbox_inside_weights.shape
        if out_verbose: print '--- mask targets shape: ', mask_targets.shape
        if out_verbose: print '--- rois_pos shape    : ', rois_pos.shape



        top[0].reshape(*rois.shape)
        top[0].data[...] = rois

        # classification labels
        top[1].reshape(*labels.shape)
        top[1].data[...] = labels

        # bbox_targets
        top[2].reshape(*bbox_targets.shape)
        top[2].data[...] = bbox_targets

        # bbox_inside_weights
        top[3].reshape(*bbox_inside_weights.shape)
        top[3].data[...] = bbox_inside_weights

        # bbox_outside_weights
        top[4].reshape(*bbox_inside_weights.shape)
        top[4].data[...] = np.array(bbox_inside_weights > 0).astype(np.float32)

	# mask targets
        top[5].reshape(*mask_targets.shape)
        top[5].data[...] = mask_targets
        ####incase output rois_pos
        top[6].reshape(*rois_pos.shape)
        top[6].data[...] = rois_pos

	'''
	# MP: printing
	print "top[0] (rois) of size {}: {}".format(rois.shape, rois)
	print "top[1] (labels) of size {}: {}".format(labels.shape, labels)
	print "top[2] (bbox_targets) of size {}: {}".format(bbox_targets.shape, bbox_targets)
	print "top[3] (bbox_inside_weights) of size {} and max at {} / {}: {}".format(bbox_inside_weights.shape, np.argmax(bbox_inside_weights[1,:]), [i for i, val in enumerate(bbox_inside_weights[1,:]) if val>0.], bbox_inside_weights)
	print "top[4] is bbox outside weigths, but is set to the same values as bbox inside weights"
        print "top[5] (mask_targets) of size {}: {}".format(mask_targets.shape, mask_targets)
        print "top[6] (rois_pos) of size {}: {}".format(rois_pos.shape, rois_pos)
	iii =2
	mtrange = mask_targets.shape[0]
	lmt = []
	for i in range(mtrange):
		mask = mask_targets[i,:,:]
		for j in range(243):
			mask_row = mask[j,:]
			for k in range(243):
				mask_ind = mask_row[k]
				if mask_ind not in lmt:
					lmt.append(mask_ind)
	print "lmt: {}".format(lmt)
	#print "mask_targets: {}".format(mask_targets[1,10,:])
	#mtlist = [i for i, val in enumerate(mask_targets[j,iii,:]) if val == 1.]
	#print "mask_targets with ids {} where value != -1.: {}".format(mtlist, mask_targets[j,iii, mtlist]

	#list = [i for i, val in enumerate(bbox_targets[iii,: ]) if abs(val) >0]
	#print "List with ids {} where abs(value) >0. and values {}".format(list, bbox_targets[iii,list])
	'''

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass



def _reset_mask_ids_FOR_IMAGE(mask, before_uni_ids):
    # reset ID mask values from [0, 1, 4] to [0, 1, 2] to resize later
#     if -1 in before_uni_ids:
#         counter = -1
#     else:
#         counter = 0

    counter = 0  # INDEX START AT 0

    for id in before_uni_ids:
        mask[mask == id] = counter
        counter += 1


    return mask


def _reset_mask_ids(mask, before_uni_ids):
    # reset ID mask values from [0, 1, 4] to [0, 1, 2] to resize later
#     if -1 in before_uni_ids:
#         counter = -1
#     else:
#         counter = 0

    counter = -1

    for id in before_uni_ids:
        mask[mask == id] = counter
        counter += 1


    return mask



def _convert_mask_to_original_ids_manual_FOR_IMAGE(mask, original_uni_ids):

    const = 0.005
    out_mask = original_uni_ids[(np.abs(mask - original_uni_ids[:,None,None]) < const).argmax(0)]
    return out_mask

def _convert_mask_to_original_ids_manual(mask, original_uni_ids):

    const = 0.005
    out_mask = original_uni_ids[(np.abs(mask - original_uni_ids[:,None,None]) < const).argmax(0)]


    #return mask
    return out_mask

def _set_unwanted_label_to_zero(mask, before_uni_label):
    # 1. round mask
    # 2. find unique mask value
    # 3. remove based on set different

    mask = np.around(mask, decimals=0)  # round 1.6 -> 2., 1.1 -> 1.
    #mask = mask.astype('int') # 2. --> 2

    uni_mask_values = np.unique(mask)

    unwanted_labels = list(set(uni_mask_values).symmetric_difference(set(before_uni_label)))
    if out_verbose: print '--- unwanted labels: ', unwanted_labels

    for ul in unwanted_labels:
        mask[mask == ul] = 0 ## set value of unwanted label to zeo

    return mask


def _get_bbox_regression_labels(bbox_target_data, num_classes):
    """Bounding-box regression targets (bbox_target_data) are stored in a
    compact form N x (class, tx, ty, tw, th)

    This function expands those targets into the 4-of-4*K representation used
    by the network (i.e. only one class has non-zero targets).

    Returns:
        bbox_target (ndarray): N x 4K blob of regression targets
        bbox_inside_weights (ndarray): N x 4K blob of loss weights
    """

    clss = bbox_target_data[:, 0]
    # print ("===============class size: " + str(clss))
    bbox_targets = np.zeros((clss.size, 4 * num_classes), dtype=np.float32) #clss.size = 128 ---> bbox_targets = 128 * 84, moi roi la 1*84 dimesion
    bbox_inside_weights = np.zeros(bbox_targets.shape, dtype=np.float32)
    inds = np.where(clss > 0)[0]
    for ind in inds:
        cls = clss[ind]
        start = 4 * cls
        end = start + 4
        start=int(start)
	end=int(end)
	bbox_targets[ind, start:end] = bbox_target_data[ind, 1:] #gan gia tri tai class tuong ung la bbox_target_data, con lai la so 0
        bbox_inside_weights[ind, start:end] = cfg.TRAIN.BBOX_INSIDE_WEIGHTS
    return bbox_targets, bbox_inside_weights


def _compute_targets(ex_rois, gt_rois, labels):
    """Compute bounding-box regression targets for an image."""

    assert ex_rois.shape[0] == gt_rois.shape[0]
    assert ex_rois.shape[1] == 4
    assert gt_rois.shape[1] == 4

    targets = bbox_transform(ex_rois, gt_rois)
    if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
        # Optionally normalize targets by a precomputed mean and stdev
        targets = ((targets - np.array(cfg.TRAIN.BBOX_NORMALIZE_MEANS))
                / np.array(cfg.TRAIN.BBOX_NORMALIZE_STDS))
    return np.hstack(
            (labels[:, np.newaxis], targets)).astype(np.float32, copy=False)

def _sample_rois(all_rois, gt_boxes, fg_rois_per_image, rois_per_image, num_classes):
    """Generate a random sample of RoIs comprising foreground and background
    examples.
    """
    # MP:
    # overlaps: (no_rois x no_gt_bbox) each row gives the overlap of the proposed region with the gt boxes. Overlap is measured as: (overlapping area)/(union area).
    # gt_assignment: determines which of the gt boxes has more overlap with the regions
    # max_overlaps: takes the maximum overlap of a region
    # labels: defines which which gt box corresponds best with the region and assigns its label to the region
    # fg_rois_per_image = 8
    # overlaps: (rois x gt_boxes)

    # MP: bbox_overlaps rewritten as c_bbox_overlaps
    #overlaps =c_bbox_overlaps(np.ascontiguousarray(all_rois[:, 1:5], dtype=np.float),
    #    		     np.ascontiguousarray(gt_boxes[:, :4], dtype=np.float))
    overlaps = bbox_overlaps(np.ascontiguousarray(all_rois[:, 1:5], dtype=np.float),
        		     np.ascontiguousarray(gt_boxes[:, :4], dtype=np.float))
    # MP: which column index has maximum value
    gt_assignment = overlaps.argmax(axis=1)
    max_overlaps = overlaps.max(axis=1)
    labels = gt_boxes[gt_assignment, 4]


    # MP: Extract RoIs where overlap >= FG_THRESH
    fg_inds = np.where(max_overlaps >= cfg.TRAIN.FG_THRESH)[0]

    # Guard against the case when an image has fewer than fg_rois_per_image (i.e. 8)
    fg_rois_per_this_image = min(fg_rois_per_image, fg_inds.size)

    # Sample foreground regions without replacement
    if fg_inds.size > 0:
        fg_inds = npr.choice(fg_inds, size=int(fg_rois_per_this_image), replace=False)

    # MP: Extract RoIs where overlap in [BG_THRESH_LO, BG_THRESH_HI), i.e. [0.0, 0.5)
    bg_inds = np.where((max_overlaps < cfg.TRAIN.BG_THRESH_HI) &
                       (max_overlaps >= cfg.TRAIN.BG_THRESH_LO))[0]

    # Compute number of background RoIs to take from this image (guarding
    # against there being fewer than desired)
    # MP: Take the no of bg_inds such that fg_inds.shape + bg_inds.shape = 32
    bg_rois_per_this_image = rois_per_image - fg_rois_per_this_image
    bg_rois_per_this_image = min(bg_rois_per_this_image, bg_inds.size)
    if bg_inds.size > 0:
        bg_inds = npr.choice(bg_inds, size=int(bg_rois_per_this_image), replace=False)


    # MP: concatenate the fg_inds and bg_inds, such that keep_inds.shape = 32
    keep_inds = np.append(fg_inds, bg_inds)
    # MP: obtain the labels set the ones corresponding to bg_inds to zero
    labels = labels[keep_inds]
    labels[int(fg_rois_per_this_image):] = 0

    # MP: select the 32 rois (fg & bg) from the 2000+ rois with the keep_inds
    rois = all_rois[keep_inds]
    # MP: fg rois
    rois_pos = np.zeros((fg_inds.size, 5), dtype=np.float32) #because return rois_pos as top ---> allocate memory for it
    rois_pos[:, :] = all_rois[fg_inds]
    gt_assignment_pos = gt_assignment[fg_inds]

    # MP: compute diff to approximate bbox to ground truth
    bbox_target_data = _compute_targets(
        rois[:, 1:5], gt_boxes[gt_assignment[keep_inds], :4], labels)

    # MP: set the diff values in a matrix where each row corresponds to a foreground bbox
    #     and the values are stored starting at the index of the label.
    #     Therefore number of columns: 4*(no labels)
    #     The bg bboxes are also included in rows, but have all values equal to zero.
    bbox_targets, bbox_inside_weights = \
        _get_bbox_regression_labels(bbox_target_data, num_classes)

    '''
    # MP: printing and saving files
    print "overlaps with size {}: {}".format(overlaps.shape, overlaps)
    print "gt_assignment with size {}: {}".format(gt_assignment.shape, gt_assignment)
    print "max_overlaps with size{}: {}".format(max_overlaps.shape, max_overlaps)
    print "labels with size{}: {}".format(labels.shape, labels)
    print "bg_inds with size{}: {}".format(bg_inds.shape, bg_inds)
    print "bg_rois_per_this_image: {}".format(bg_rois_per_this_image)
    print "bg_inds with shape {}: {}".format(bg_inds.shape, bg_inds)
    print "fg_inds with size {}: {}".format(fg_inds.shape, fg_inds)
    print "labels with shape {}: {}".format(labels.shape,labels)
    print "rois wiht shape {}: {}".format(rois.shape, rois)
    print "rois_pos wiht shape {}: {}".format(rois_pos.shape, rois_pos)
    print "labels with shape {}: {}".format(labels.shape,labels)
    print "rois_pos wiht shape {}: {}".format(rois_pos.shape, rois_pos)
    print "gt_assignment_pos wiht shape {}: {}".format(gt_assignment_pos.shape, gt_assignment_pos)
    print "bbox_target_data wiht shape {}: {}".format(bbox_target_data.shape, bbox_target_data)
    print "diff: {}".format(rois_pos[:,:] + bbox_target_data[0:fg_inds.size,:])
    print "bbox_targets with size {}: {}".format(bbox_targets.shape, bbox_targets)
    print "bbox_inside_weights with size {}: {}".format(bbox_inside_weights.shape, bbox_inside_weights)

    np.savetxt('bbox_targets.txt', bbox_targets, delimiter=',')
    np.savetxt('bbox_inside_weights.txt', bbox_inside_weights, delimiter=',')
    '''

    return labels, rois, bbox_targets, bbox_inside_weights, gt_boxes[gt_assignment[keep_inds], :], rois_pos, gt_assignment_pos




#--------------------------------------------------------
# Faster R-CNN
