#!/bin/bash
# Usage:
# ./experiments/scripts/mask_rcnn_e2e.sh GPU NET DATASET [options args to {train,test}_net.py]
# DATASET is either pascal_voc or coco.
#
# Example:
# ./experiments/scripts/mask_rcnn_e2e.sh 0 VGG16 TomatoDS \
#   --set EXP_DIR foobar RNG_SEED 42 TRAIN.SCALES "[400, 500, 600, 700]"

set -x  # Exit immediately if a command exits with a non-zero status.
set -e  # Print commands and their arguments as they are executed.

export PYTHONUNBUFFERED="True"

GPU_ID=$1
NET=$2
NET_lc=${NET,,}
DATASET=$3

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:3:$len}
EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}

case $DATASET in
  pascal_voc)
    TRAIN_IMDB="voc_2012_train"
    TEST_IMDB="voc_2012_val"
    PT_DIR="pascal_voc"
    SUB_DIR="faster_rcnn_end2end"
    ITERS=1
    ;;
  coco)
    TRAIN_IMDB="coco_2014_train"
    TEST_IMDB="coco_2014_minival"
    PT_DIR="coco"
    SUB_DIR="mask_rcnn_end2end"
    ITERS=1
    ;;
  TomatoDB)
    TRAIN_IMDB="voc_2012_train"
    TEST_IMDB="voc_2012_val"
    PT_DIR="tomato_db"
    SUB_DIR="mask_rcnn_end2end"
    ITERS=1
    ;;
    *)
    echo "No dataset given"
    exit
    ;;

esac

LOG="experiments/logs/mask_rcnn_${NET}_${PT_DIR}.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

time ./tools/train_net.py --gpu ${GPU_ID} \
  --solver models/${PT_DIR}/${NET}/${SUB_DIR}/solver.prototxt \
  --weights data/imagenet_models/${NET}.v2.caffemodel \
  --imdb ${TRAIN_IMDB} \
  --iters ${ITERS} \
  --cfg experiments/cfgs/mask_rcnn_end2end.yml \
  ${EXTRA_ARGS}
