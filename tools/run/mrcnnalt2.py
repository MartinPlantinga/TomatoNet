import os
import subprocess
import sys
from time import localtime, strftime

pwd = os.curdir
root_dir = pwd + './../'

weights_path = '{}data/imagenet_models/VGG16.v2.caffemodel'.format(root_dir)
cfg_path = '{}experiments/cfgs/mask_rcnn_alt_opt.yml'.format(root_dir)
log_file="{}experiments/logs/mask_rcnn_alt_opt_{}".format(root_dir, strftime("%d-%m-%Y_%H_%M", localtime()))
#print log_file

exec_log_file = "exec &> >(tee -a \"{}\")".format(log_file)
#echo Logging output to "$LOG"
#os.system(exec &> >(tee -a "$LOG")
exec_python = "python ../train_mask_rcnn_alt_opt.py --gpu 0 --net_name 'VGG16' --weights {} --imdb 'voc_2012_train' --cfg {}".format(weights_path, cfg_path) 
exec_all = "'/bin/bash -c {}' ; {}".format(exec_log_file, exec_python)
#os.system(exec_all)
print exec_all
output_contents = subprocess.check_output(exec_python, shell=True)
#print output_contents

f = open(log_file, 'w')
f.write(output_contents)
f.close
