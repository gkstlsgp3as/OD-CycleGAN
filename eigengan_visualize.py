#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 16:20:11 2023

@author: shhan
"""

import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
import torch    
import cv2
import numpy as np
import requests
import torchvision.transforms as transforms
from pytorch_grad_cam.eigen_cam import EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, scale_cam_image
from PIL import Image
import os
import yaml
import easydict

from utils.general import check_file, non_max_suppression, polygon_non_max_suppression
from models.yolo_arch2 import Model
from utils.torch_utils import intersect_dicts, TracedModel
from models.experimental import attempt_load
from utils.datasets import letterbox
from models.gan import create_model

device = torch.device('cuda')
COLORS = np.random.uniform(0, 255, size=(80, 3))

import numpy as np
import torch
import ttach as tta
from typing import Callable, List, Tuple
from pytorch_grad_cam.base_cam import BaseCAM
from pytorch_grad_cam.activations_and_gradients import ActivationsAndGradients
from pytorch_grad_cam.utils.svd_on_activations import get_2d_projection
from pytorch_grad_cam.utils.image import scale_cam_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
class BaseCAM2(BaseCAM):
    def __init__(self,
                 model: torch.nn.Module,
                 target_layers: List[torch.nn.Module],
                 use_cuda: bool = False,
                 reshape_transform: Callable = None,
                 compute_input_gradient: bool = False,
                 uses_gradients: bool = True) -> None:
        self.model = model.eval()
        self.target_layers = target_layers
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()
        self.reshape_transform = reshape_transform
        self.compute_input_gradient = compute_input_gradient
        self.uses_gradients = uses_gradients
        self.activations_and_grads = ActivationsAndGradients(
            self.model, target_layers, reshape_transform)
    
    def forward(self,
                input_tensor: torch.Tensor,
                targets: List[torch.nn.Module],
                eigen_smooth: bool = False) -> np.ndarray:

        if self.cuda:
            input_tensor = input_tensor.cuda()

        if self.compute_input_gradient:
            input_tensor = torch.autograd.Variable(input_tensor,
                                                   requires_grad=True)

        outputs = self.activations_and_grads(input_tensor)[0]
        if targets is None:
            target_categories = np.argmax(outputs.cpu().data.numpy(), axis=-1)
            targets = [ClassifierOutputTarget(
                category) for category in target_categories]

        if self.uses_gradients:
            self.model.zero_grad()
            loss = sum([target(output)
                       for target, output in zip(targets, outputs)])
            loss.backward(retain_graph=True)
        cam_per_layer = self.compute_cam_per_layer(input_tensor,
                                                   targets,
                                                   eigen_smooth)
        return self.aggregate_multi_layers(cam_per_layer)

class EigenCAM(BaseCAM2):
    def __init__(self, model, target_layers, use_cuda=False,
                 reshape_transform=None):
        super(EigenCAM, self).__init__(model,
                                       target_layers,
                                       use_cuda,
                                       reshape_transform,
                                       uses_gradients=False)

    def get_cam_image(self,
                      input_tensor,
                      target_layer,
                      target_category,
                      activations,
                      grads,
                      eigen_smooth):
        return get_2d_projection(activations)
    
def normalize_img(img0, imgsz=640, stride=32):
    '''
    img = letterbox(img0, imgsz, stride)[0]
    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416; already applied
    img = np.ascontiguousarray(img)

    # Gaussian Blur
    #if Cfg.kernel > 0:
    #    img = cv2.GaussianBlur(img, (Cfg.kernel,Cfg.kernel), 0)
            
    img = torch.from_numpy(img).to(device)
    img = img / 255.0  # 0 - 255 to 0.0 - 1.0   
    
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    
    return img
        '''
    img = letterbox(img0, imgsz, stride)[0]
    # Convert
    img = np.float32(img) #*255 #/ 255 
    tensor = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416; already applied
    tensor = np.ascontiguousarray(tensor)

    # Gaussian Blur
    #if Cfg.kernel > 0:
    #    img = cv2.GaussianBlur(img, (Cfg.kernel,Cfg.kernel), 0)
    tensor = torch.from_numpy(tensor).to(device)
    
    if tensor.ndimension() == 3:
        tensor = tensor.unsqueeze(0)
    
    return img, tensor

def parse_detections(pred):
    pred = non_max_suppression(pred, conf_thres=0.005, iou_thres=0.3)
    
    cfg = './data/sentinel.yaml'; cfg  = check_file(cfg)
    with open(cfg) as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader) 
    
    boxes, colors, names = [], [], []
    for i, det in enumerate(pred):  # detections per image
        if len(det):

            # Write results
            for *xyxy, conf, cls in reversed(det):
                category = int(cls)
                color = COLORS[category]
                name = cfg['names'][category]
                box = np.array([el.cpu() for el in xyxy], dtype=np.int32)
                
                boxes.append(box)
                colors.append(color)
                names.append(name)
                
    return boxes, colors, names

def polygon_parse_detections(pred):
    pred =  polygon_non_max_suppression(pred, conf_thres=0.005, iou_thres=0.3)
    
    cfg = './data/sentinel.yaml'; cfg  = check_file(cfg)
    with open(cfg) as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader) 
    
    boxes, colors, names = [], [], []
    for i, det in enumerate(pred):  # detections per image
        if len(det):

            # Write results
            for *xyxyxyxy, conf, cls in reversed(det):
                category = int(cls)
                color = COLORS[category]
                name = cfg['names'][category]
                box = np.array([el.cpu() for el in xyxyxyxy], dtype=np.int32)
                
                boxes.append(box)
                colors.append(color)
                names.append(name)
                
    return boxes, colors, names

def draw_detections(boxes, colors, names, img):
    for box, color, name in zip(boxes, colors, names):
        xmin, ymin, xmax, ymax = box
        cv2.rectangle(
            img,
            (xmin, ymin),
            (xmax, ymax),
            color, 
            1)

        #cv2.putText(img, name, (xmin, ymin - 5),
        #            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2,
        #            lineType=cv2.LINE_AA)
    return img

def polygon_draw_detections(boxes, colors, names, img):
    from utils.general import order_corners
    for box, color, name in zip(boxes, colors, names):
        cv2.polylines(img, pts=[box.reshape(-1,2)], isClosed=True, color=color, thickness=1, lineType=cv2.LINE_AA)
    
        #cv2.putText(img, name, (xmin, ymin - 5),
        #            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2,
        #            lineType=cv2.LINE_AA)
    return img

def renormalize_cam_in_bounding_boxes(boxes, colors, names, image_float_np, grayscale_cam):
    """Normalize the CAM to be in the range [0, 1] 
    inside every bounding boxes, and zero outside of the bounding boxes. """
    renormalized_cam = np.zeros(grayscale_cam.shape, dtype=np.float32)
    for x1, y1, x2, y2 in boxes:
        renormalized_cam[y1:y2, x1:x2] = scale_cam_image(grayscale_cam[y1:y2, x1:x2].copy())    
    renormalized_cam = scale_cam_image(renormalized_cam)
    eigencam_image_renormalized = show_cam_on_image(image_float_np, renormalized_cam, use_rgb=True)
    image_with_bounding_boxes = draw_detections(boxes, colors, names, eigencam_image_renormalized)
    return image_with_bounding_boxes

def get_target_layers(model, layer_name):
    
    target_layers = []
    for name, layer in model.named_children():
        if name == '':
            continue
        target_layers.append(layer)
        if name == layer_name:
            print('return!')
            return target_layers
            
    return target_layers

def get_layer_names(model):
    layer_names = {}
    for name, layer in model.named_modules():
        print(name)
        if (name.endswith('conv')) or ('.m.' in name):
            layer_names[name] = layer
        
    return layer_names

def load_image(path):
    # loads 1 image from dataset, returns img, original hw, resized hw
    from utils.cfg import Cfg
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)  # BGR
    assert img is not None, 'Image Not Found ' + path
    h0, w0 = img.shape[:2]  # orig hw
    r = 1  # ratio
    if r != 1:  # if sizes are not equal
        img = cv2.resize(img, (int(w0 * r), int(h0 * r)),
                             interpolation=cv2.INTER_AREA if r < 1 and not self.augment else cv2.INTER_LINEAR)
    if Cfg.img_mode != 'org':
        if Cfg.img_mode == 'vv+vh':
            newimg = img[...,1]+img[...,2]
        elif Cfg.img_mode == 'grayscale':
            newimg = 0.229*img[...,0]+0.587*img[...,1]+0.114*img[...,2]
        elif Cfg.img_mode == 'vv^2+vh^2':
            newimg = img[...,1]**2 +img[...,2]**2
        elif Cfg.img_mode == 'vv*vh':
            newimg = img[...,1]*img[...,2]
        img = np.dstack((newimg, newimg, newimg))
        if img.max() != 0:
            img = (img - img.min()) / (img.max() - img.min())
                
        return img, (h0, w0), img.shape[:2]  # img, hw_original, hw_resized

            
path = './inference/visualize/'
dstpath = './inference/eigenCAM/'
fl = os.listdir(path)
fl = [f for f in fl if f.endswith('tif')]
weights = './runs/train/exp406_batch16_epoch300_subset_sar_smoothing0.1_multiscaleFalse/weights/best.pt'
hyp = './data/hyp.scratch.custom.yaml'; hyp  = check_file(hyp)
with open(hyp) as f:
    hyp = yaml.load(f, Loader=yaml.SafeLoader) 
    
model = attempt_load(weights, map_location=device) 
#model = TracedModel(model, device, 640)
model.eval()
model.cuda()

opt = easydict.EasyDict({
    'model': 'cycle_gan',
    'dataset_mode': 'aligned',
    'input_nc': 3,
    'output_nc': 3, 
    'no_dropout': True,
    'ngf': 64, 
    'ndf': 64, 
    'netD': 'basic',
    'netG': 'resnet_9blocks',
    'n_layers_D': 3,
    'norm': 'instance',
    'init_type': 'normal',
    'direction': 'AtoB',
    'isTrain': False,
    'device': '0,1,2,3',
    'checkpoints_dir': '/data/BRIDGE/OD-CycleGAN/checkpoints/sar_cyclegan/',
    'exp_name': 'sar',
    'preprocess': 'resize_and_crop',
    'init_gain': 0.02,
    'load_iter': 0,
    'epoch': 'latest',
    'verbose': False
})
# setup GPU for gan_model
str_ids = opt.device.split(',')
opt.lst_ids = []
for str_id in str_ids:
    id = int(str_id)
    if id >= 0:
        opt.lst_ids.append(id)
if len(opt.lst_ids) > 0:
    torch.cuda.set_device(opt.lst_ids[0])
opt.device = torch.device('cuda:{}'.format(opt.lst_ids[0])) if opt.lst_ids else torch.device('cpu') 

gan_model = create_model(opt)    
gan_model.setup(opt)

layer_names = get_layer_names(model)

for f in fl:
    #Image.open(path+fl[-1]) np.array(Image.open(path+fl[-1]))#
    img =  load_image(path+f)[0]
    rgb_img = img.copy(); rgb_img = np.array(rgb_img*255, np.uint8)
    #img, tensor = normalize_img(img)
    Image.fromarray(np.uint8(img*255))
    
    gan = gan_model.gan_infer(img)
    
    input_img = torch.cat((torch.from_numpy(img.transpose(2,0,1)).unsqueeze(0), torch.from_numpy(gan.transpose(2,0,1)).unsqueeze(0)), dim=1).to(opt.device)
    pred = model(input_img)[0]
    boxes, colors, names = polygon_parse_detections(pred)
    bgr_img = np.flip(rgb_img, axis=2)
    detections = polygon_draw_detections(boxes, colors, names, bgr_img.copy())
    Image.fromarray(detections)
    Image.fromarray(detections).save(path+f.replace('.tif','_detections.png'))

    for name in layer_names.keys():
        target_layers = [layer_names[name]]
        cam = EigenCAM(model, target_layers, use_cuda=False)
        
        grayscale_cam = cam(input_img)[0, :, :]
        #Image.fromarray(np.uint8(grayscale_cam*255), 'L')
        cam_image = show_cam_on_image(np.flip(img, axis=2), grayscale_cam, use_rgb=True)
        #Image.fromarray(cam_image)
        cam_name = f.replace('.tif','_cam_'+name+'.png')
        Image.fromarray(cam_image).save(dstpath+cam_name)
        print('Saved:'+cam_name)
        
        #renormalized_cam_image = renormalize_cam_in_bounding_boxes(boxes, colors, names, img, grayscale_cam)
        #Image.fromarray(renormalized_cam_image)
        
        #Image.fromarray(np.hstack((rgb_img, cam_image, renormalized_cam_image)))
        
#target_layers = [layer_names['model.7.conv']]#get_target_layers(model)
#target_layers = [model.model[7].conv]
#target_layers = get_target_layers(model, 'model.28.conv')