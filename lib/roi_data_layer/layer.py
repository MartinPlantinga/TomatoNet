# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""The data layer used during training to train a Fast R-CNN network.

RoIDataLayer implements a Caffe Python layer.
"""

import caffe
from fast_rcnn.config import cfg
from roi_data_layer.minibatch import get_minibatch
import numpy as np
import yaml
from multiprocessing import Process, Queue
import ipdb

class RoIDataLayer(caffe.Layer):
    """Fast R-CNN data layer used for training."""

    def _shuffle_roidb_inds(self):
        """Randomly permute the training roidb."""
        #if cfg.TRAIN.ASPECT_GROUPING:
        if cfg.TRAIN.ASPECT_GROUPING and cfg.TRAIN.USE_FLIPPED: #avoid error 'ValueError: total size of new array must be unchanged' in inds = np.reshape(inds, (-1, 2))
            widths = np.array([r['width'] for r in self._roidb])
            heights = np.array([r['height'] for r in self._roidb])
            horz = (widths >= heights)
            vert = np.logical_not(horz)
            horz_inds = np.where(horz)[0]
            vert_inds = np.where(vert)[0]
            inds = np.hstack((
                np.random.permutation(horz_inds),
                np.random.permutation(vert_inds)))
            inds = np.reshape(inds, (-1, 2))
            row_perm = np.random.permutation(np.arange(inds.shape[0]))
            inds = np.reshape(inds[row_perm, :], (-1,))
            self._perm = inds
        else:
            self._perm = np.random.permutation(np.arange(len(self._roidb)))
        self._cur = 0

    def _get_next_minibatch_inds(self):
        """Return the roidb indices for the next minibatch."""
        if self._cur + cfg.TRAIN.IMS_PER_BATCH >= len(self._roidb):
            self._shuffle_roidb_inds()

        db_inds = self._perm[self._cur:self._cur + cfg.TRAIN.IMS_PER_BATCH]
        self._cur += cfg.TRAIN.IMS_PER_BATCH
        return db_inds

    def _get_next_minibatch(self):
        """Return the blobs to be used for the next minibatch.

        If cfg.TRAIN.USE_PREFETCH is True, then blobs will be computed in a
        separate process and made available through self._blob_queue.
        """
        if cfg.TRAIN.USE_PREFETCH:
            return self._blob_queue.get()
        else:
            db_inds = self._get_next_minibatch_inds()
            minibatch_db = [self._roidb[i] for i in db_inds]
            return get_minibatch(minibatch_db, self._num_classes)

    def set_roidb(self, roidb):
        """Set the roidb to be used by this layer during training."""
        self._roidb = roidb
        self._shuffle_roidb_inds()
        if cfg.TRAIN.USE_PREFETCH:
            self._blob_queue = Queue(10)
            self._prefetch_process = BlobFetcher(self._blob_queue,
                                                 self._roidb,
                                                 self._num_classes)
            self._prefetch_process.start()
            # Terminate the child process when the parent exists
            def cleanup():
                print 'Terminating BlobFetcher'
                self._prefetch_process.terminate()
                self._prefetch_process.join()
            import atexit
            atexit.register(cleanup)

    def setup(self, bottom, top):
        """Setup the RoIDataLayer."""

        # parse the layer parameter string, which must be valid YAML
        layer_params = yaml.load(self.param_str_)

        self._num_classes = layer_params['num_classes']

        self._name_to_top_map = {}

        # data blob: holds a batch of N images, each with 3 channels
        idx = 0
        top[idx].reshape(cfg.TRAIN.IMS_PER_BATCH,
                        3,
                        max(cfg.TRAIN.SCALES),
                        cfg.TRAIN.MAX_SIZE) #cfg.TRAIN.SCALES = 600. cfg.TRAIN.MAX_SIZE = 1000 -> 1,3,600,1000
        self._name_to_top_map['data'] = idx
        idx += 1

        if cfg.TRAIN.HAS_RPN:
            top[idx].reshape(1, 3) #matrix (1,3)
            self._name_to_top_map['im_info'] = idx
            idx += 1
            top[idx].reshape(1, 4) #matrix (1,4)??? when get_minibatch then gt_boxes is matrix (1,5):  1:4 is boxes cordinate,  5 is gt_class
            self._name_to_top_map['gt_boxes'] = idx
            idx += 1
            if cfg.TRAIN.MASK_REG:
                top[idx].reshape(1, 2) #image index and obj index of segmentation mask
                self._name_to_top_map['seg_mask_inds'] = idx
                idx += 1

                top[idx].reshape(1, 1)
                self._name_to_top_map['flipped'] = idx
                idx += 1

        else: # not using RPN
            # rois blob: holds R regions of interest, each is a 5-tuple
            # (n, x1, y1, x2, y2) specifying an image batch index n and a
            # rectangle (x1, y1, x2, y2)
            top[idx].reshape(1, 5)
            self._name_to_top_map['rois'] = idx
            idx += 1

            # labels blob: R categorical labels in [0, ..., K] for K foreground
            # classes plus background
            top[idx].reshape(1)
            self._name_to_top_map['labels'] = idx
            idx += 1

            if cfg.TRAIN.BBOX_REG:
                # bbox_targets blob: R bounding-box regression targets with 4
                # targets per class
                top[idx].reshape(1, self._num_classes * 4)
                self._name_to_top_map['bbox_targets'] = idx
                idx += 1

                # bbox_inside_weights blob: At most 4 targets per roi are active;
                # thisbinary vector sepcifies the subset of active targets
                top[idx].reshape(1, self._num_classes * 4)
                self._name_to_top_map['bbox_inside_weights'] = idx
                idx += 1

                top[idx].reshape(1, self._num_classes * 4)
                self._name_to_top_map['bbox_outside_weights'] = idx
                idx += 1

        print 'RoiDataLayer: name_to_top:', self._name_to_top_map
        assert len(top) == len(self._name_to_top_map)

    def forward(self, bottom, top):
        """Get blobs and copy them into this layer's top blob vector."""
        blobs = self._get_next_minibatch()

	'''
	# MP: printing
	print "=========================================================="
	#print "blobs with shape: {}".format(blobs.shape)
	print "blobs['gt_boxes'] with shape {}: {}".format(blobs['gt_boxes'].shape, blobs['gt_boxes'])
	print "blobs['seg_mask_inds'] with shape {}: {}".format(blobs['seg_mask_inds'].shape, blobs['seg_mask_inds'])
	print "blobs['flipped'] with shape {}: {}".format(blobs['flipped'].shape, blobs['flipped'])
	print "blobs['im_info'] with shape {}: {}".format(blobs['im_info'].shape, blobs['im_info'])
	print "blobs['data'] with shape {}: {}".format(blobs['data'].shape, blobs['data'])
	print "=========================================================="\
	'''
        ###################Print blobs####################################
        # if cfg.TRAIN.MASK_REG:
        #     m_gt_boxes = blobs['gt_boxes']
        #     m_seg_mask_inds = blobs['seg_mask_inds']
        #     f = open("/home/tdo/Software/FRCN_ROOT/tools/roi_data_layer_forward.txt", "w")
        #     for ix, m_gt_box in enumerate(m_gt_boxes):
        #         x1 = m_gt_box[0]
        #         y1 = m_gt_box[1]
        #         x2 = m_gt_box[2]
        #         y2 = m_gt_box[3]
        #         cl = m_gt_box[4]
        #         m_seg_mask_ind = m_seg_mask_inds[ix]
        #         im_id = m_seg_mask_ind[0]
        #         mask_id = m_seg_mask_ind[1]
        #         f.write("GT box: "+str(x1)+" "+str(y1)+" "+str(x2)+" "+str(y2)+". Class: "+str(cl)+". Im id: "+str(im_id)+". Mask id: "+str(mask_id)+"\n")
        #     f.write("\n")
        #     f.close()
        ##############################################################
        for blob_name, blob in blobs.iteritems():
            top_ind = self._name_to_top_map[blob_name]
            # Reshape net's input blobs
            top[top_ind].reshape(*(blob.shape))
            # Copy data into net's input blobs
            top[top_ind].data[...] = blob.astype(np.float32, copy=False)
            '''
            if blob_name == 'gt_boxes':
                print "BOUNDING BOXES\n {}".format(blob)
            if blob_name == 'seg_mask_inds':
                print "IMAGE NUMBER: {}".format(blob[0][0])
                if blob[0][0] == 530390:
                    ipdb.set_trace()
            '''

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass

class BlobFetcher(Process):
    """Experimental class for prefetching blobs in a separate process."""
    def __init__(self, queue, roidb, num_classes):
        super(BlobFetcher, self).__init__()
        self._queue = queue
        self._roidb = roidb
        self._num_classes = num_classes
        self._perm = None
        self._cur = 0
        self._shuffle_roidb_inds()
        # fix the random seed for reproducibility
        np.random.seed(cfg.RNG_SEED)

    def _shuffle_roidb_inds(self):
        """Randomly permute the training roidb."""
        # TODO(rbg): remove duplicated code
        self._perm = np.random.permutation(np.arange(len(self._roidb)))
        self._cur = 0

    def _get_next_minibatch_inds(self):
        """Return the roidb indices for the next minibatch."""
        # TODO(rbg): remove duplicated code
        if self._cur + cfg.TRAIN.IMS_PER_BATCH >= len(self._roidb):
            self._shuffle_roidb_inds()

        db_inds = self._perm[self._cur:self._cur + cfg.TRAIN.IMS_PER_BATCH]
        self._cur += cfg.TRAIN.IMS_PER_BATCH
        return db_inds

    def run(self):
        print 'BlobFetcher started'
        while True:
            db_inds = self._get_next_minibatch_inds() #get mini roidb, voi batch index la db_inds
            minibatch_db = [self._roidb[i] for i in db_inds] #minibatch_db: chua cac roidb[i]
            blobs = get_minibatch(minibatch_db, self._num_classes) #lay cac field cua roidb[i] ra va dat vao blobs
            self._queue.put(blobs)
