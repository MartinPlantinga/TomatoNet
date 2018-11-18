"""
See README.md for installation instructions before running.
Demo script to perform affordace detection from images
"""
# MP: import caffe takes place in _init_paths
import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect2
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer

import numpy as np
import os, cv2
import argparse

import caffe
import ipdb


CONF_THRESHOLD = 0.8
good_range = 0.005
mask_threshold = 0.5


# MP: define paths for network, architecture and images
cwd = os.getcwd()
root_path = os.path.abspath(os.path.join(cwd, os.pardir))  # get parent path
img_folder = cwd + '/imgs_tomato'
prototxt = root_path + '/models/tomato_db/VGG16/mask_rcnn_end2end/test_3classes.prototxt'

caffemodel = os.path.join(root_path, 'pretrained/tomato_net_iteration_70000.caffemodel')

if not os.path.isfile(caffemodel):
	raise IOError(('{:s} not found.\n').format(caffemodel))
print 'Mask R-CNN architecture: ', prototxt
print 'Mask R-CNN trained weigths and biases: ', caffemodel


OBJ_CLASSES = ('__background__', 'unripe', 'ripe')
# NOTE: BGR order, not RGB
background = [200, 222, 250]
c1 = [0,0,205] # unripe tomato
c2 = [34,139,34] # ripe tomato
mask_colors = np.array([background, c1, c2])
label_colors = mask_colors

# MP: make box colors equal to label_colors.
box_colors = label_colors

def reset_mask_ids(mask, before_uni_ids):
    # reset ID mask values from [0, 1, 4] to [0, 1, 2] to resize later
    counter = 0
    for id in before_uni_ids:
        mask[mask == id] = counter
        counter += 1

    return mask



def convert_mask_to_original_ids_manual(mask, original_uni_ids):
    #TODO: speed up!!!
    temp_mask = np.copy(mask) # create temp mask to do np.around()
    temp_mask = np.around(temp_mask, decimals=0)  # round 1.6 -> 2., 1.1 -> 1.
    current_uni_ids = np.unique(temp_mask)

    out_mask = np.full(mask.shape, 0, 'float32')

    mh, mw = mask.shape
    for i in range(mh-1):
        for j in range(mw-1):
            for k in range(1, len(current_uni_ids)):
                if mask[i][j] > (current_uni_ids[k] - good_range) and mask[i][j] < (current_uni_ids[k] + good_range):
                    out_mask[i][j] = original_uni_ids[k]
                    #mask[i][j] = current_uni_ids[k]

#     const = 0.005
#     out_mask = original_uni_ids[(np.abs(mask - original_uni_ids[:,None,None]) < const).argmax(0)]

    #return mask
    return out_mask

def draw_arrow(image, p, q, color, arrow_magnitude, thickness, line_type, shift):
    # draw arrow tail
    cv2.line(image, p, q, color, thickness, line_type, shift)
    # calc angle of the arrow
    angle = np.arctan2(p[1]-q[1], p[0]-q[0])
    # starting point of first line of arrow head
    p = (int(q[0] + arrow_magnitude * np.cos(angle + np.pi/4)),
    int(q[1] + arrow_magnitude * np.sin(angle + np.pi/4)))
    # draw first half of arrow head
    cv2.line(image, p, q, color, thickness, line_type, shift)
    # starting point of second line of arrow head
    p = (int(q[0] + arrow_magnitude * np.cos(angle - np.pi/4)),
    int(q[1] + arrow_magnitude * np.sin(angle - np.pi/4)))
    # draw second half of arrow head
    cv2.line(image, p, q, color, thickness, line_type, shift)

def draw_reg_text(img, obj_info):
    #print 'tbd'

    obj_id = obj_info[0]
    cfd = obj_info[1]
    xmin = obj_info[2]
    ymin = obj_info[3]
    xmax = obj_info[4]
    ymax = obj_info[5]

    draw_arrow(img, (xmin, ymin), (xmax, ymin), box_colors[obj_id], 0, 5, 8, 0)
    draw_arrow(img, (xmax, ymin), (xmax, ymax), box_colors[obj_id], 0, 5, 8, 0)
    draw_arrow(img, (xmax, ymax), (xmin, ymax), box_colors[obj_id], 0, 5, 8, 0)
    draw_arrow(img, (xmin, ymax), (xmin, ymin), box_colors[obj_id], 0, 5, 8, 0)

    # put text
    txt_obj = OBJ_CLASSES[obj_id] + ' ' + str(cfd)
    cv2.putText(img, txt_obj, (xmin, ymin-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1) # draw with red
    #cv2.putText(img, txt_obj, (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 1, box_colors[obj_id], 2)

#     # draw center
#     center_x = (xmax - xmin)/2 + xmin
#     center_y = (ymax - ymin)/2 + ymin
#     cv2.circle(img,(center_x, center_y), 3, (0, 255, 0), -1)

    return img



def visualize_bbox_mask(im, rois_final, rois_class_score, rois_class_ind, masks, original_h, original_w, im_name, thresh):

    if rois_final.shape[0] == 0:
        print 'No detected box at all!'
        return
    inds = np.where(rois_class_score[:, -1] >= thresh)[0]
    if len(inds) == 0:
        print 'No detected box with probality > thresh = ', thresh, '-- Choossing highest confidence bounding box.'
        inds = [np.argmax(rois_class_score)]
        max_conf = np.max(rois_class_score)
        if max_conf < 0.001:
            return  ## confidence is < 0.001 -- no good box --> must return

#filter based on threshold
    rois_final = rois_final[inds, :]
    rois_class_score = rois_class_score[inds,:]
    rois_class_ind = rois_class_ind[inds,:]
    masks = masks[inds, :, :, :]

    num_boxes = rois_final.shape[0]

    list_bboxes = []

    mask_total = np.zeros((original_h, original_w), dtype = np.float64)

    # MP: slight modification
    for i in range(num_boxes):

        # MP: determine the class, its bbox and corresponding score
        class_id = int(rois_class_ind[i,0])
        score = rois_class_score[i,0]
        bbox = rois_final[i, 1:5]
        x1 = int(round(bbox[0]))
        y1 = int(round(bbox[1]))
        x2 = int(round(bbox[2]))
        y2 = int(round(bbox[3]))
        curr_box = [class_id, score, x1, y1, x2, y2]
        list_bboxes.append(curr_box)

        h = y2 - y1
        w = x2 - x1

        mask = masks[i, :, :, :]
        #for line in range(mask2print.shape[0]):
        #   print "{}".format(mask2print[line,:])
        # MP: THIS HAS TO BE CHANGED IN MASK RCNN. WE JUST WANT TO CHOOSE THE MASK THAT CORRESPONDS TO THE HIGHEST BBOX SCORE.
        # MP: mask consists of 10 layers of 244x244 arrays, one layer for each mask label. The pixel values of each layer are the probability that the pixel corresponds to the mask label. np.argmax(mask, axis=0) takes for each of the 10 layers, the pixel that has the highest probability and assigns it the layer number.

        # MP:   choose the mask that corresponds to the bounding box
        #       only choose pixels with probability larger than 0.5
        mask = mask[class_id,:,:]
        mask[mask < mask_threshold] = 0
        mask[mask >= mask_threshold] = class_id
        mask = mask.astype(np.uint8)

        # mask = np.argmax(mask, axis=0)

        original_uni_ids = np.unique(mask)
        # sort before_uni_ids and reset [0, 1, 7] to [0, 1, 2]
        original_uni_ids.sort()
        mask = reset_mask_ids(mask, original_uni_ids)
        mask = cv2.resize(mask.astype('float'), (int(w), int(h)), interpolation=cv2.INTER_LINEAR)

        # MP: go back from [0, 1, 2] to [0, 1, 7]
        mask = convert_mask_to_original_ids_manual(mask, original_uni_ids)

        mask_total[y1:y2, x1:x2] += mask


    mask_total = mask_total.astype(np.uint8)
    mask_total_rgb = mask_colors.take(mask_total, axis = 0).astype(np.uint8)

    for bbox in list_bboxes:
        print 'box: ', bbox
        img_out = draw_reg_text(im, bbox)

    img_out_hsv = cv2.cvtColor(img_out, cv2.COLOR_BGR2HSV)
    mask_total_hsv = cv2.cvtColor(mask_total_rgb, cv2.COLOR_RGB2HSV)

    img_out_hsv[...,0] = mask_total_hsv[...,0]
    img_out_hsv[...,1] = mask_total_hsv[...,1] *0.6

    img_out = cv2.cvtColor(img_out_hsv, cv2.COLOR_HSV2RGB)


    img_name = ''
    cv2.namedWindow(img_name)        # Create a named window
    cv2.moveWindow(img_name, 40,30)  # Move it to (40,30)
    cv2.imshow('Obj detection', img_out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def run_tomato_net(net,im):

    # MP: original image height and width
    original_h, original_w, _ = im.shape

    # START TIMING
    timer = Timer()
    timer.tic()

    blobs = {'data': None}
    blobs['data'], im_scales = _get_image_blob(im)
    blobs['im_info'] = np.array([[blobs['data'].shape[2], blobs['data'].shape[3], im_scales[0]]], dtype = np.float32)
    #blobs['data'].astype(np.float32, copy=False)
    #blobs['im_info'].astype(np.float32, copy=False)


    # FORWARD PASS
    net.blobs['data'].reshape(*(blobs['data'].shape))
    net.blobs['im_info'].reshape(*(blobs['im_info'].shape))

    blobs_out = net.forward(**blobs)
    rois = net.blobs['rois'].data.copy()

    rois_class_score = blobs_out['rois_class_score']
    rois_class_ind = blobs_out['rois_class_ind']
    rois_final = blobs_out['rois_final']
    masks = blobs_out['mask_prob'] # shape of masks_out blob: (n, 10, 244, 244), where n is the number of bboxes and 10 is the number of mask classes.

    timer.toc()
    # STOP TIMER


    print ('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, rois_final.shape[0])

    # Visualize detections for each class
    visualize_bbox_mask(im,
		   rois_final,
		   rois_class_score,
		   rois_class_ind,
		   masks,
		   original_h,
		   original_w,
		   im_name,
		   thresh=CONF_THRESHOLD)

def _get_image_blob(im):
    im_orig = im.astype(np.float32, copy=True)
    im_orig -= cfg.PIXEL_MEANS

    im_shape = im_orig.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])

    processed_ims = []
    im_scale_factors = []


    im_scale_min = float(cfg.TEST.SCALES[0]) / float(im_size_min)
    im_scale_max = float(cfg.TEST.MAX_SIZE) / float(im_size_max)
    im_scale = min(im_scale_min, im_scale_max)

    im = cv2.resize(im_orig,
                    None,
                    None,
                    fx=im_scale,
                    fy=im_scale,
                    interpolation = cv2.INTER_LINEAR)
    processed_ims.append(im)
    im_scale_factors.append(im_scale)
    num_images = 1
    blob = np.zeros((num_images, im.shape[0], im.shape[1], im.shape[2]), dtype = np.float32)
    blob[0, 0:im.shape[0], 0:im.shape[1], :] = im
    channel_swap = (0, 3, 1, 2)
    blob = blob.transpose(channel_swap)
    return blob, np.array(im_scale_factors)

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='TomatoNet demo')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')

    args = parser.parse_args()

    return args

if __name__ == '__main__':

    args = parse_args()

    if args.cpu_mode:
        caffe.set_mode_cpu()
        print "Mode is CPU"
    else:
        caffe.set_device(args.gpu_id)
        caffe.set_mode_gpu()
        cfg.GPU_ID = args.gpu_id


    # load network
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)
    print '\n\nLoaded network {:s}'.format(caffemodel)

    list_test_img = os.walk(img_folder).next()[2]

    # run detection for each image
    for idx, im_name in enumerate(list_test_img):
        im = cv2.imread(img_folder + '/' + im_name)
	print 'Current idx: ', idx, ' / ', len(list_test_img)
        print 'Current img: ', im_name
        run_tomato_net(net, im)
