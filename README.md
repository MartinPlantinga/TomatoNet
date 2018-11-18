# TomatoNet

This repository is based on the paper [Mask R-CNN](https://arxiv.org/abs/1703.06870) (He et al.). The network allows to detect instances of tomato trusses. The dataset created for this network can be downloaded from [TomatoDB](https://mega.nz/#F!JUVTGYqA!OwLxr_GFgQpyYC2Jal7zJw). There are 3 classes that the network detects, namely: background, ripe tomato truss and unripe tomato truss. Due to the fact that the dataset contains mostly unripe tomato trusses, the network doesn't isn't able to detect these trusses well and classifies them as unripe tomato trusses.



### Contents
1. [Requirements](#requirements)
2. [Installation](#installation)
3. [Demo](#demo)
4. [Training](#training)
5. [Differences with Mask R-CNN](#differences-with-mask-r-cnn)
6. [Dataset](#dataset)
7. [Acknowledgements](#acknowledgements)


### Requirements
1. Caffe ***with support for python layers*** (see [Installation](#installation)).
2. Hardware:
	- To train a full TomatoNet, you'll need a GPU with ~11GB (e.g. Titan, K20, K40, Tesla, ...).
	- To test a full TomatoNet, you'll need ~6GB GPU.

### Installation
1. Clone the TomatoNet repository into your `$TomatoNet_ROOT` folder.
2. Build `Caffe` and `pycaffe`:
    ```bash
    cd $TomatoNet_ROOT/caffe-tomato-net`
    # Proceed as in the Caffe installation instructions: http://caffe.berkeleyvision.org/installation.html
    # Make sure to have 'WITH_PYTHON_LAYER := 1' uncommented in `Makefile.config`.
    make -j8
    make pycaffe
    ```
    Note that `-j8` indicates that the make process will use 8 cores/threads. This may be different for your processor, to find out the number for your system, run the `nproc` command.

3. Build the Cython modules:
    ```bash
    cd $TomatoNet_ROOT/lib`
    make
    ```

4. Download pretrained weights ([Mega Drive](https://mega.nz/#F!gMERwa4Q!MTC_OIpZpnjGLEqFhOvwYQ)). These weights have been trained on the TomatoDB dataset:
    - Extract the file you downloaded to `$TomatoNet_ROOT`
    - Make sure you have the caffemodel file like this: `'$TomatoNet_ROOT/pretrained/tomato_net_iteration_70000.caffemodel`

### Demo
After successfully completing installation, you'll be ready to run the demo.
1. Export pycaffe path:
    ```bash
    export PYTHONPATH=$TomatoNet_ROOT/caffe-tomato-net/python:$PYTHONPATH
    ```
2. Demo with unseen images:
    ```bash
    cd $TomatoNet_ROOT/tools
    python tomato_inference.py
    ```

### Training
Training can be done with the TomatoNet dataset:
1. Download the dataset from [TomatoDB](https://mega.nz/#F!JUVTGYqA!OwLxr_GFgQpyYC2Jal7zJw). Read more about the dataset under [Dataset](#dataset).
2. Create a `data` directory in `$TomatoNet_ROOT`, i.e. `$TomatoNet_ROOT/data/`.
3. Extract the dataset in `$TomatoNet_ROOT/data/`, i.e. `$TomatoNet_ROOT/data/TomatoDB`
2. Train TomatoNet:
    ```bash
    cd $TomatoNet_ROOT
    ./experiments/scripts/mask_rcnn_end2end.sh [GPU_ID] [NET] [Dataset] [--set ...]
    # e.g.: ./experiments/scripts/mask_rcnn_end2end.sh 0 VGG16 TomatoDB
    	```

### Differences with Mask R-CNN
1. TomatoNet is based on AffordanceNet, where multiple classes are annotated within the same  have multiple classes inside each instance.
2. The backbone is a VGG16 network

### Dataset
The TomatoNet dataset has been created with the [Photoshop Annotation Plugin](https://github.com/MartinPlantinga/Photoshop-Annotation-Plugin).

The `*.rdb` files in the TomatoDB dataset contain Numpy arrays that have the following key-value pairs per entry:
```python
{
	'boxes': array([[x1,  y1, x2, y2]], dtype=int64),
	'flipped': True/False,
	'gt_classes': array([...], dtype=int64),
	'gt_overlaps': sparse matrix,
	'image': path to image,
	'height': image height in pixels,
	'width': image width in pixels,
	'max_classes': array([...], dtype=int64),
	'max_overlaps': array([...]),
	'seg_areas': array([...], dtype=int64),
	'seg_mask_inds': array([[...,...]], dtype=int64) # checkout proposal_target_layer.py to see how it links to the segmentation mask.
 }
 ```

The file can be opened as follows:
```python
rdb = np.load('filename.rdb')
```
and can be saved as follows:
```python
with open('filename.rdb','wb') as outfile:
    np.save(outfile, rdb)
```

### Acknowledgements
This repo used source code from [Faster-RCNN](https://github.com/rbgirshick/py-faster-rcnn) and [Affordance Net](https://github.com/nqanh/affordance-net) and is based on the paper [Mask R-CNN](https://arxiv.org/abs/1703.06870) (He et al.).
