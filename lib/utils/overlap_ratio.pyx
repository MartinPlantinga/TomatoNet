# --------------------------------------------------------
# overlap_ratio.pyx
# Computes ratio between:
# 1. overlap between roi_bbox and gt_bbox
# 2. union of roi_bbox and gt_bbox
# Written by Martin Plantinga
# --------------------------------------------------------

cimport cython
import numpy as np
cimport numpy as np

DTYPE = np.float
ctypedef np.float_t DTYPE_t

def bbox_overlaps(np.ndarray[DTYPE_t, ndim=2] roi_boxes, np.ndarray[DTYPE_t, ndim=2] gt_boxes):
	cdef unsigned int N = roi_boxes.shape[0]
	cdef unsigned int K = gt_boxes.shape[0]
	cdef np.ndarray[DTYPE_t, ndim=2] overlap_ratio = np.zeros((N, K), dtype=DTYPE)
	cdef DTYPE_t iw, ih, box_area, roi_area
	cdef DTYPE_t u_area, o_area
	cdef DTYPE_t x1k, y1k, x2k, y2k, x1n, x2n, y1n, y2n
	cdef unsigned int k, n
	for k in range(K):
		x1k = gt_boxes[k, 0]
		x2k = gt_boxes[k, 2]
		y1k = gt_boxes[k, 1]
		y2k = gt_boxes[k, 3]
		box_area = ( (x2k - x1k + 1) * (y2k-y1k + 1) )
		for n in range(N):
			x1n = roi_boxes[n, 0]
			x2n = roi_boxes[n, 2]
			y1n = roi_boxes[n, 1]
			y2n = roi_boxes[n, 3]
			iw = (min(x2n, x2k) - max(x1n, x1k) + 1 )
			if iw > 0:
				ih = ( min(y2n, y2k) - max(y1n, y1k) + 1 )
				if ih > 0:
					roi_area = ( (x2n - x1n + 1) * (y2n - y1n + 1))
					o_area = (iw * ih)
					u_area = float( roi_area + box_area - o_area)
					overlap_ratio[n, k] = o_area / u_area
	return overlap_ratio
