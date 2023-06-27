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

Cfg.batch = 2 #2
Cfg.subdivisions = 1 # mini-batch = batch / subdivisions - 1

# 학습 이미지 크기 #32의 배수
Cfg.width = 608 #512 608
Cfg.height = 608 #512 608
Cfg.channels = 3
Cfg.momentum = 0.949
Cfg.decay = 0.0005

# Classification
Cfg.angle = 0

Cfg.saturation = 1.5
Cfg.exposure = 1.5
Cfg.hue = .1

Cfg.learning_rate = 0.001 #0.01 0.001
Cfg.burn_in = 1000
Cfg.max_batches = 18000 #classes * 2000
Cfg.steps = [14400, 16200] # max_batches * 0.8, 0.9
Cfg.policy = Cfg.steps
# steps[0] = learning_rate * .1...
Cfg.scales = .1, .1

Cfg.cutmix = 0
Cfg.mosaic = 1 # 4개 이미지를 1개로 합성

Cfg.letter_box = 0
Cfg.jitter = .2 # 학습 이미지 크기 및 ratio 변환
Cfg.classes = 10 #2 #1
Cfg.track = 0
Cfg.w = Cfg.width
Cfg.h = Cfg.height
Cfg.flip = 1 # 좌우 반전
Cfg.blur = 0
Cfg.gaussian = 1
Cfg.boxes = 500  # 최대 검출 개수


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

Cfg.iou_type = 'iou'  # 'giou', 'diou', 'ciou'

# Additional Input information and variables
# Name of each image: S1, CSK, K5, ICEYE
# Number of satellite image band: 1 or 3
Cfg.Satellite = 'S1'
Cfg.Satelliteband = 3
Cfg.division = 15 

# New Band Test(True=1, False=0)
Cfg.NewTest = 0

Cfg.TRAIN_EPOCHS = 200
Cfg.export = 100

Cfg.scorethresh = 0.2
Cfg.inputsize = 1088

Cfg.calib = 1

# Confined MinMax value for normalization of input images
# Band 1
Cfg.min1 = 0
Cfg.max1 = 0.15 # 0.15 150

# Band 2
Cfg.min2 = 0
Cfg.max2 = 0.5 # 0.5 200

# Band 3
Cfg.min3 = 0
Cfg.max3 = 50 #50 250

# Directory of the training and test dataset
# S1 20m multi
#Cfg.train_img_path='/data/objdt/Sentinel_multi_train_dataset/'
#Cfg.train_txt_path='/data/objdt/Sentinel_multi_train_dataset/*.txt'
#Cfg.test_img_path='/data/objdt/Sentinel_multi_test_dataset/'
#Cfg.test_txt_path='/data/objdt/Sentinel_multi_test_dataset/*.txt'

# S1 20m Wind Turbine
Cfg.train_img_path='/data/windturbin/SARDATA/trainDataset/'
Cfg.train_txt_path='/data/windturbin/SARDATA/trainDataset/txt_data/*.txt'
Cfg.test_img_path='/data/windturbin/SARDATA/testDatset/'
Cfg.test_txt_path='/data/windturbin/SARDATA/testDatset/txt_data/*.txt'









