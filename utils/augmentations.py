#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 13:50:09 2023

@author: shhan
"""
import cv2
import numpy as np
import random
import math

from utils.general import xyxyxyxyn2xyxyxyxy, xywhn2xyxy, xyn2xy, resample_segments, segment2box, segments2boxes, colorstr

import logging

class Albumentations:
    # Polygon YOLOv5 Albumentations class (optional, only used if package is installed)
    def __init__(self):
        self.transform = None
        try:
            import albumentations as A

            self.transform = A.Compose([
                A.MedianBlur(p=0.05),
                A.ToGray(p=0.1),
                A.RandomBrightnessContrast(p=0.35),
                A.CLAHE(p=0.2),
                A.InvertImg(p=0.3)],)
                # Not support for any position change to image

            logging.info(colorstr('albumentations: ') + ', '.join(f'{x}' for x in self.transform.transforms if x.p))
        except ImportError:  # package not installed, skip
            pass
        except Exception as e:
            logging.info(colorstr('albumentations: ') + f'{e}')

    def __call__(self, im, p=1.0):
        if self.transform and random.random() < p:
            new = self.transform(image=im)  # transformed
            im = new['image']
        return im

# Ancillary functions --------------------------------------------------------------------------------------------------------------------
def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # for rectangular inference, pad with gray(RGB: 114,114,114) pixels
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    
def random_perspective(img, targets=(), segments=(), degrees=10, translate=.1, scale=.1, shear=10, perspective=0.0,
                       border=(0, 0)):
    # torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-10, 10))
    # targets = [cls, xyxy]

    height = img.shape[0] + border[0] * 2  # shape(h,w,c)
    width = img.shape[1] + border[1] * 2

    # Center
    C = np.eye(3)
    C[0, 2] = -img.shape[1] / 2  # x translation (pixels)
    C[1, 2] = -img.shape[0] / 2  # y translation (pixels)

    # Perspective
    P = np.eye(3)
    P[2, 0] = random.uniform(-perspective, perspective)  # x perspective (about y)
    P[2, 1] = random.uniform(-perspective, perspective)  # y perspective (about x)

    # Rotation and Scale
    R = np.eye(3)
    a = random.uniform(-degrees, degrees)
    # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
    s = random.uniform(1 - scale, 1 + scale)
    # s = 2 ** random.uniform(-scale, scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

    # Translation
    T = np.eye(3)
    T[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * width  # x translation (pixels)
    T[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * height  # y translation (pixels)

    # Combined rotation matrix
    M = T @ S @ R @ P @ C  # order of operations (right to left) is IMPORTANT
    if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():  # image changed
        if perspective:
            img = cv2.warpPerspective(img, M, dsize=(width, height), borderValue=(114, 114, 114))
        else:  # affine
            img = cv2.warpAffine(img, M[:2], dsize=(width, height), borderValue=(114, 114, 114))

    # Visualize
    # import matplotlib.pyplot as plt
    # ax = plt.subplots(1, 2, figsize=(12, 6))[1].ravel()
    # ax[0].imshow(img[:, :, ::-1])  # base
    # ax[1].imshow(img2[:, :, ::-1])  # warped

    # Transform label coordinates
    n = len(targets)
    if n:
        use_segments = any(x.any() for x in segments)
        new = np.zeros((n, 4))
        if use_segments:  # warp segments
            segments = resample_segments(segments)  # upsample
            for i, segment in enumerate(segments):
                xy = np.ones((len(segment), 3))
                xy[:, :2] = segment
                xy = xy @ M.T  # transform
                xy = xy[:, :2] / xy[:, 2:3] if perspective else xy[:, :2]  # perspective rescale or affine

                # clip
                new[i] = segment2box(xy, width, height)

        else:  # warp boxes
            xy = np.ones((n * 4, 3))
            xy[:, :2] = targets[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
            xy = xy @ M.T  # transform
            xy = (xy[:, :2] / xy[:, 2:3] if perspective else xy[:, :2]).reshape(n, 8)  # perspective rescale or affine

            # create new boxes
            x = xy[:, [0, 2, 4, 6]]
            y = xy[:, [1, 3, 5, 7]]
                        
            new = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

            # clip
            new[:, [0, 2]] = new[:, [0, 2]].clip(0, width)
            new[:, [1, 3]] = new[:, [1, 3]].clip(0, height)

        # filter candidates
        i = box_candidates(box1=targets[:, 1:5].T * s, box2=new.T, area_thr=0.01 if use_segments else 0.10)
        targets = targets[i]
        targets[:, 1:5] = new[i]

    return img, targets

def box_candidates(box1, box2, wh_thr=2, ar_thr=20, area_thr=0.1, eps=1e-16):  # box1(4,n), box2(4,n)
    # Compute candidate boxes: box1 before augment, box2 after augment, wh_thr (pixels), aspect_ratio_thr, area_ratio
    w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
    w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
    ar = np.maximum(w2 / (h2 + eps), h2 / (w2 + eps))  # aspect ratio
    return (w2 > wh_thr) & (h2 > wh_thr) & (w2 * h2 / (w1 * h1 + eps) > area_thr) & (ar < ar_thr)  # candidates


# Ancillary functions with polygon anchor boxes-------------------------------------------------------------------------------------------
def polygon_box_candidates(box1, box2, wh_thr=3, ar_thr=20, area_thr=0.1, eps=1e-16):
    """
        box1(8,n), box2(8,n)
        Use the minimum bounding box as the approximation to polygon
        Compute candidate boxes: box1 before augment, box2 after augment, wh_thr (pixels), aspect_ratio_thr, area_ratio
    """
    w1, h1 = box1[0::2].max(axis=0)-box1[0::2].min(axis=0), box1[1::2].max(axis=0)-box1[1::2].min(axis=0) # start:end:step
    w2, h2 = box2[0::2].max(axis=0)-box2[0::2].min(axis=0), box2[1::2].max(axis=0)-box2[1::2].min(axis=0)
    ar = np.maximum(w2 / (h2 + eps), h2 / (w2 + eps))  # aspect ratio
    return (w2 > wh_thr) & (h2 > wh_thr) & (w2 * h2 / (w1 * h1 + eps) > area_thr) & (ar < ar_thr)  # candidates

def polygon_random_perspective(img, targets=(), segments=(), degrees=10, translate=.1, scale=.1, shear=10, perspective=0.0,
                       border=(0, 0), mosaic=False):
    """
        torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-10, 10))
        targets = [cls, xyxyxyxy]
    """
#     To restrict the polygon boxes within images
#     def restrict(img, new, shape0, padding=(0, 0, 0, 0)):
#         height, width = shape0
#         top0, bottom0, left0, right0 = np.ceil(padding[0]), np.floor(padding[1]), np.floor(padding[2]), np.ceil(padding[3])
#         # keep the original shape of image
#         if (height/width) < ((height+bottom0+top0)/(width+left0+right0)):
#             dw = int((height+bottom0+top0)/height*width)-(width+left0+right0)
#             top, bottom, left, right = map(int, (top0, bottom0, left0+dw/2, right0+dw/2))
#         else:
#             dh = int((width+left0+right0)*height/width)-(height+bottom0+top0)
#             top, bottom, left, right = map(int, (top0+dh/2, bottom0+dh/2, left0, right0))
#         img = cv2.copyMakeBorder(img, bottom, top, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
#         img = cv2.resize(img, (width, height), interpolation=cv2.INTER_LINEAR)
#         w_r, h_r = width/(width+left+right), height/(height+bottom+top)
#         new[:, 0::2] = (new[:, 0::2]+left)*w_r
#         new[:, 1::2] = (new[:, 1::2]+bottom)*h_r
#         return img, new
        
    height = img.shape[0] + border[0] * 2  # shape(h,w,c)
    width = img.shape[1] + border[1] * 2

    # Center
    C = np.eye(3)
    C[0, 2] = -img.shape[1] / 2  # x translation (pixels)
    C[1, 2] = -img.shape[0] / 2  # y translation (pixels)

    # Perspective
    P = np.eye(3)
    P[2, 0] = random.uniform(-perspective, perspective)  # x perspective (about y)
    P[2, 1] = random.uniform(-perspective, perspective)  # y perspective (about x)

    # Rotation and Scale
    R = np.eye(3)
    a = random.uniform(-degrees, degrees)
    # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
    s = random.uniform(1 - scale, 1 + scale)
    # s = 2 ** random.uniform(-scale, scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

    # Translation
    T = np.eye(3)
    T[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * width  # x translation (pixels)
    T[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * height  # y translation (pixels)

    # Combined rotation matrix
    M = T @ S @ R @ P @ C  # order of operations (right to left) is IMPORTANT
    image_transformed = False

    # Transform label coordinates
    n = len(targets)
    if n:
        # if using segments: please use general.py::polygon_segment2box
        # segment is unnormalized np.array([[(x1, y1), (x2, y2), ...], ...])
        # targets is unnormalized np.array([[class id, x1, y1, x2, y2, ...], ...])
        new = np.zeros((n, 8))
        xy = np.ones((n * 4, 3))
        xy[:, :2] = targets[:, 1:].reshape(n * 4, 2)
        xy = xy @ M.T  # transform
        new = (xy[:, :2] / xy[:, 2:3] if perspective else xy[:, :2]).reshape(n, 8)  # perspective rescale or affine
        
        if not mosaic:
            # Compute Top, Bottom, Left, Right Padding to Include Polygon Boxes inside Image
            top = max(new[:, 1::2].max().item()-height, 0)
            bottom = abs(min(new[:, 1::2].min().item(), 0))
            left = abs(min(new[:, 0::2].min().item(), 0))
            right = max(new[:, 0::2].max().item()-width, 0)
            
            R2 = np.eye(3)
            r = min(height/(height+top+bottom), width/(width+left+right))
            R2[:2] = cv2.getRotationMatrix2D(angle=0., center=(0, 0), scale=r)
            M2 = T @ S @ R @ R2 @ P @ C  # order of operations (right to left) is IMPORTANT
            
            if (border[0] != 0) or (border[1] != 0) or (M2 != np.eye(3)).any():  # image changed
                if perspective:
                    img = cv2.warpPerspective(img, M2, dsize=(width, height), borderValue=(114, 114, 114))
                else:  # affine
                    img = cv2.warpAffine(img, M2[:2], dsize=(width, height), borderValue=(114, 114, 114))
                image_transformed = True
                new = np.zeros((n, 8))
                xy = np.ones((n * 4, 3))
                xy[:, :2] = targets[:, 1:].reshape(n * 4, 2)
                xy = xy @ M2.T  # transform
                new = (xy[:, :2] / xy[:, 2:3] if perspective else xy[:, :2]).reshape(n, 8)  # perspective rescale or affine
            # img, new = restrict(img, new, (height, width), (top, bottom, left, right))
        
        # Use the following two lines can result in slightly tilting for few labels.
        # new[:, 0::2] = new[:, 0::2].clip(0., width)
        # new[:, 1::2] = new[:, 1::2].clip(0., height)
        # If use following codes instead, can mitigate tilting problems, but result in few label exceeding problems.
        cx, cy = new[:, 0::2].mean(-1), new[:, 1::2].mean(-1)
        new[(cx>width)|(cx<-0.)|(cy>height)|(cy<-0.)] = 0.
        
        # filter candidates
        # 0.1 for axis-aligned rectangle, 0.01 for segmentation, so choose intermediate 0.08
        i = polygon_box_candidates(box1=targets[:, 1:].T * s, box2=new.T, area_thr=0.08) 
        targets = targets[i]
        targets[:, 1:] = new[i]
        
    if not image_transformed:
        M = T @ S @ R @ P @ C  # order of operations (right to left) is IMPORTANT
        if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():  # image changed
            if perspective:
                img = cv2.warpPerspective(img, M, dsize=(width, height), borderValue=(114, 114, 114))
            else:  # affine
                img = cv2.warpAffine(img, M[:2], dsize=(width, height), borderValue=(114, 114, 114))
            image_transformed = True
        
    return img, targets
## Augmentation ----------------------------------------------------------------------------------------------------------


    return img, ratio, (dw, dh)

def augment_hsv(img, hgain=0.5, sgain=0.5, vgain=0.5):
    r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
    hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    dtype = img.dtype  # uint8

    x = np.arange(0, 256, dtype=np.int16)
    # In OpenCV, H: 0-180, S: 0-255, V: 0-255
    lut_hue = ((x * r[0]) % 180).astype(dtype) # color type; originally, 0-360º but only be represented to 180 wrt 8 bit
    lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
    lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

    img_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val))).astype(dtype) # LUT = look up table -> ref: https://mrsnake.tistory.com/142
    cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)  # no return needed



# Ancillary functions with polygon anchor boxes-------------------------------------------------------------------------------------------


