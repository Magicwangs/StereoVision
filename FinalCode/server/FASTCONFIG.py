#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  4 09:37:28 2017

@author: magic
"""
#import tensorflow as tf
import os


CONFIG = 'Network Configs'

#FLAGS = tf.app.flags.FLAGS

# Basic model parameters.
#tf.app.flags.DEFINE_integer('batchSize', 128,
#                                                """Number of images to process in a batch.""")
#tf.app.flags.DEFINE_string('data_dir', '/home/magic/StereoVision/Code/KITTI/KITTI2012/data_stereo_flow',
#                                           """Path to the KITTI data directory.""")
data_dir = '/home/xhq/magic/KITTI_Data'
data2012_dir = data_dir + '/KITTI2012'
data2015_dir = data_dir + '/KITTI2015'

## check dir
try:
    if not os.path.exists(data_dir):
        print "KITTI Data Dir is not exist"
    if not os.path.exists(data2012_dir):
        print "KITTI 2012 Data Dir is not exist"
    if not os.path.exists(data2015_dir):
        print "KITTI 2015 Data Dir is not exist"
except:
    print ("Data Dir  ERROR")
    exit()

## Global constants describing KITTIPatch data 
dataset_neg_low = 4
dataset_neg_high = 10
dataset_pos = 1
input_patch_size = (9,9)
inter_len = 4

batchSize = 128

train_Data = data2015_dir + '/training'

test_Data = data2012_dir + '/testing'

patch_Dir = './FastKITTIPatch'
log_Dir = './save/0329/log'
save_Dir = './save/0329'
top_Dir = './save'
disp_Dir = './disp/'
try:
    if not os.path.exists(disp_Dir):
        os.mkdir(disp_Dir)
        os.mkdir(disp_Dir + 'rightdisp')
        os.mkdir(disp_Dir + 'leftdisp')
    if not os.path.exists(patch_Dir):
        os.mkdir(patch_Dir)
    if not os.path.exists(top_Dir):
        os.mkdir(top_Dir)    
    if not os.path.exists(save_Dir):
        os.mkdir(save_Dir)
    if not os.path.exists(log_Dir):
        os.mkdir(log_Dir)
        os.mkdir(log_Dir+'/train')
        os.mkdir(log_Dir+'/test')
except:
    print ("Creat Patch Dir  ERROR")
    exit()


## Constants describing the training process.
num_conv_feature_maps = 64
num_fc_units = 384
learning_rate = 0.002
momentum = 0.5
weight_decay = 0.001*5

epochs = 20
iteration = 550000














