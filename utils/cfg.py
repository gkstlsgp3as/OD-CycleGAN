# -*- coding: utf-8 -*-
'''
@Time          : 2020/05/06 21:05
@Author        : Tianxiaomo
@File          : Cfg.py
@Noice         :
@Modificattion :
    @Author    :
    @Time      :
    @Detail    :

'''
import os
from easydict import EasyDict


_BASE_DIR = os.path.dirname(os.path.abspath(__file__))

Cfg = EasyDict()

Cfg.use_darknet_cfg = False
Cfg.cfgfile = os.path.join(_BASE_DIR, 'cfg', 'yolov4-custom.cfg')
Cfg.outepsg = 4326

Cfg.batch = 2 #2
Cfg.subdivisions = 1 # mini-batch = batch / subdivisions - 1

# 학습 이미지 크기 #32의 배수
Cfg.width = 640 #512 608
Cfg.height = 640 #512 608
Cfg.channels = 3
Cfg.momentum = 0.949
Cfg.decay = 0.0005
Cfg.stride = 32
Cfg.overlap = 200

# Classification

# RGB
Cfg.color = [[255,0,0], [0,0,255], [0,255,0], [255,0,255]] 
# ship, 'trn', 'veh', 'ped' => red, blue, green, purple

Cfg.angle = 0
Cfg.saturation = 1.5
Cfg.exposure = 1.5
Cfg.hue = .1

Cfg.learning_rate = 0.001                       #0.01 0.001
Cfg.burn_in = 1000
Cfg.max_batches = 18000                         #classes * 2000
Cfg.steps = [14400, 16200]                      # max_batches * 0.8, 0.9
Cfg.policy = Cfg.steps
Cfg.scales = .1, .1                             # steps[0] = learning_rate * .1...

Cfg.cutmix = 0
Cfg.mosaic = 1                                  # 4개 이미지를 1개로 합성

Cfg.letter_box = 0
Cfg.jitter = .2                                 # 학습 이미지 크기 및 ratio 변환
Cfg.track = 0
Cfg.w = Cfg.width
Cfg.h = Cfg.height
Cfg.flip = 1                                    # 좌우 반전
Cfg.blur = 0
Cfg.gaussian = 1
Cfg.boxes = 500                                 # 최대 검출 개수

Cfg.classes = 3

Cfg.train_label = os.path.join(_BASE_DIR, 'data', 'div_40.txt')
Cfg.val_label = os.path.join(_BASE_DIR, 'data' ,'div_40.txt')
Cfg.TRAIN_OPTIMIZER = 'adam'

if Cfg.mosaic and Cfg.cutmix:
    Cfg.mixup = 4
elif Cfg.cutmix:
    Cfg.mixup = 2
elif Cfg.mosaic:
    Cfg.mixup = 3

Cfg.checkpoints = os.path.join(_BASE_DIR, 'checkpoints')
Cfg.TRAIN_TENSORBOARD_DIR = os.path.join(_BASE_DIR, 'log')

Cfg.iou_type = 'ciou'  # 'iou', 'giou', 'diou', 'ciou', 'gaussian'
Cfg.img_mode = 'vh+vv' #  'grayscale', 'vv^2+vh^2', 'vv*vh' or 'org'
# Additional Input information and variables
# Name of each image: S1, CSK, K5, ICEYE
# Number of satellite image band: 1 or 3
Cfg.Satellite = 'S1'
Cfg.Satelliteband = 3
Cfg.division = 40 # K5: 10, S1: 15, 20 15

# New Band Test(True=1, False=0)
Cfg.NewTest = 0

Cfg.TRAIN_EPOCHS = 300
Cfg.export = 150

Cfg.scorethresh = 0.2
Cfg.inputsize = 1088

Cfg.calib = 1

# Confined MinMax value for normalization of input images
Cfg.minTh = 30
Cfg.maxTh = 230
# Bands
Cfg.min = [0, 0, 0]
Cfg.max = [0.15, 0.5, 50]
 # 0.15 150 / 0.5 200 / 50 250

# Directory of the training and test dataset
# S1 20m multi
#Cfg.train_img_path='/data/objdt/Sentinel_multi_train_dataset1/'
#Cfg.train_txt_path='/data/objdt/Sentinel_multi_train_dataset1/*.txt'
#Cfg.test_img_path='/data/objdt/Sentinel_multi_test_dataset1/'
#Cfg.test_txt_path='/data/objdt/Sentinel_multi_test_dataset1/*.txt'

# S1 20m
Cfg.img_path='/data/BRIDGE/BRIDGE_sentinel1_10m'

#Cfg.train_img_path='/data/objdt/Sentinel_test_dataset/'

Cfg.train_txt_path='./data/labels/sentinel/train/*.txt'
Cfg.valid_txt_path='./data/labels/sentinel/valid/*.txt'
Cfg.test_txt_path='./data/labels/sentinel/test/*.txt'

# train and val data as 1) directory: path/images/, 2) file: path/images.txt, or 3) list: [path1/images/, path2/images/]
Cfg.train='./data/org/sentinel/horz/train.txt'  # 2978 images
Cfg.valid='./data/org/sentinel/horz/valid.txt'  # 235 images
Cfg.test='./data/org/sentinel/horz/test.txt'  # 235 images

# number of classes
Cfg.nc=3
Cfg.size=5

# class names
Cfg.names=[ 'trn', 'veh', 'ped']

# ICEYE 1m multi(Before vs After Refocusing)
#Cfg.train_img_path='/data/objdt/ICEYE_multi_train_dataset_re/'
#Cfg.train_txt_path='/data/objdt/ICEYE_multi_train_dataset_re/*.txt'
#Cfg.test_img_path='/data/objdt/ICEYE_multi_test_dataset_re/'
#Cfg.test_txt_path='/data/objdt/ICEYE_multi_test_dataset_re/*.txt'

# Complete DB(K5, CSK, TDM, ICEYE) 1m
#Cfg.train_img_path='/data/objdt/1m_train_dataset/'
#Cfg.train_txt_path='/data/objdt/1m_train_dataset/*.txt'
#Cfg.test_img_path='/data/objdt/1m_test_dataset/'
#Cfg.test_txt_path='/data/objdt/1m_test_dataset/*.txt'

# Complete DB(K5, ICEYE) 0.3m
#Cfg.train_img_path='/data/objdt/0.3m_train_dataset/'
#Cfg.train_txt_path='/data/objdt/0.3m_train_dataset/*.txt'
#Cfg.test_img_path='/data/objdt/0.3m_test_dataset/'
#Cfg.test_txt_path='/data/objdt/0.3m_test_dataset/*.txt'








