# YOLOR general utils

import glob
import logging
import math
import os
import platform
import random
import re
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torchvision
import yaml
import shapely
import shapely.geos

from utils.google_utils import gsutil_getsize
from utils.metrics import fitness
from utils.torch_utils import init_torch_seeds
from utils.cfg import Cfg

pi = 3.141592

# Settings
torch.set_printoptions(linewidth=320, precision=5, profile='long')
np.set_printoptions(linewidth=320, formatter={'float_kind': '{:11.5g}'.format})  # format short g, %precision=5
pd.options.display.max_columns = 10
cv2.setNumThreads(0)  # prevent OpenCV from multithreading (incompatible with PyTorch DataLoader)
os.environ['NUMEXPR_MAX_THREADS'] = str(min(os.cpu_count(), 8))  # NumExpr max threads

def set_logging(rank=-1):
    logging.basicConfig(
        format="%(message)s",
        level=logging.INFO if rank in [-1, 0] else logging.WARN)


def init_seeds(seed=0):
    # Initialize random number generator (RNG) seeds
    random.seed(seed)
    np.random.seed(seed)
    init_torch_seeds(seed)


def get_latest_run(search_dir='.'):
    # Return path to most recent 'last.pt' in /runs (i.e. to --resume from)
    last_list = glob.glob(f'{search_dir}/**/last*.pt', recursive=True)
    return max(last_list, key=os.path.getctime) if last_list else ''


def isdocker():
    # Is environment a Docker container
    return Path('/workspace').exists()  # or Path('/.dockerenv').exists()


def emojis(str=''):
    # Return platform-dependent emoji-safe version of string
    return str.encode().decode('ascii', 'ignore') if platform.system() == 'Windows' else str

def check_name(name):
    status = True
    splits = name.split('_')
    if len(splits[0]) != 5:
        status = False
        return status
    if splits[1][:3] != 'RGB':
        status = False
        return status
    if len(splits[1][3:]) !=6:
        status = False
        return status

    return status

def check_online():
    # Check internet connectivity
    import socket
    try:
        socket.create_connection(("1.1.1.1", 443), 5)  # check host accesability
        return True
    except OSError:
        return False


def check_git_status():
    # Recommend 'git pull' if code is out of date
    print(colorstr('github: '), end='')
    try:
        assert Path('.git').exists(), 'skipping check (not a git repository)'
        assert not isdocker(), 'skipping check (Docker image)'
        assert check_online(), 'skipping check (offline)'

        cmd = 'git fetch && git config --get remote.origin.url'
        url = subprocess.check_output(cmd, shell=True).decode().strip().rstrip('.git')  # github repo url
        branch = subprocess.check_output('git rev-parse --abbrev-ref HEAD', shell=True).decode().strip()  # checked out
        n = int(subprocess.check_output(f'git rev-list {branch}..origin/master --count', shell=True))  # commits behind
        if n > 0:
            s = f"⚠️ WARNING: code is out of date by {n} commit{'s' * (n > 1)}. " \
                f"Use 'git pull' to update or 'git clone {url}' to download latest."
        else:
            s = f'up to date with {url} ✅'
        print(emojis(s))  # emoji-safe
    except Exception as e:
        print(e)


def check_requirements(requirements='requirements.txt', exclude=()):
    # Check installed dependencies meet requirements (pass *.txt file or list of packages)
    import pkg_resources as pkg
    prefix = colorstr('red', 'bold', 'requirements:')
    if isinstance(requirements, (str, Path)):  # requirements.txt file
        file = Path(requirements)
        if not file.exists():
            print(f"{prefix} {file.resolve()} not found, check failed.")
            return
        requirements = [f'{x.name}{x.specifier}' for x in pkg.parse_requirements(file.open()) if x.name not in exclude]
    else:  # list or tuple of packages
        requirements = [x for x in requirements if x not in exclude]

    n = 0  # number of packages updates
    for r in requirements:
        try:
            pkg.require(r)
        except Exception as e:  # DistributionNotFound or VersionConflict if requirements not met
            n += 1
            print(f"{prefix} {e.req} not found and is required by YOLOR, attempting auto-update...")
            print(subprocess.check_output(f"pip install '{e.req}'", shell=True).decode())

    if n:  # if packages updated
        source = file.resolve() if 'file' in locals() else requirements
        s = f"{prefix} {n} package{'s' * (n > 1)} updated per {source}\n" \
            f"{prefix} ⚠️ {colorstr('bold', 'Restart runtime or rerun command for updates to take effect')}\n"
        print(emojis(s))  # emoji-safe


def check_img_size(img_size, s=32):
    # Verify img_size is a multiple of stride s
    if img_size < s:
        print("Error: Img size should be bigger than the stride")
        return -1
    
    new_size = make_divisible(img_size, int(s))  # ceil gs-multiple
    if new_size != img_size:
        print('WARNING: --img-size %g must be multiple of max stride %g, updating to %g' % (img_size, s, new_size))
    
    return new_size


def check_imshow():
    # Check if environment supports image displays
    try:
        assert not isdocker(), 'cv2.imshow() is disabled in Docker environments'
        cv2.imshow('test', np.zeros((1, 1, 3)))
        cv2.waitKey(1)
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        return True
    except Exception as e:
        print(f'WARNING: Environment does not support cv2.imshow() or PIL Image.show() image displays\n{e}')
        return False


def check_file(file):
    # Search for file if not found
    if Path(file).is_file() or file == '':
        return file
    else:
        files = glob.glob('./**/' + file, recursive=True)  # find file
        assert len(files), f'File Not Found: {file}'  # assert file was found
        assert len(files) == 1, f"Multiple files match '{file}', specify exact path: {files}"  # assert unique
        return files[0]  # return file


def check_dataset(dict):
    # Download dataset if not found locally
    val, s = dict.get('val'), dict.get('download')
    if val and len(val):
        val = [Path(x).resolve() for x in (val if isinstance(val, list) else [val])]  # val path
        if not all(x.exists() for x in val):
            print('\nWARNING: Dataset not found, nonexistent paths: %s' % [str(x) for x in val if not x.exists()])
            if s and len(s):  # download script
                print('Downloading %s ...' % s)
                if s.startswith('http') and s.endswith('.zip'):  # URL
                    f = Path(s).name  # filename
                    torch.hub.download_url_to_file(s, f)
                    r = os.system('unzip -q %s -d ../ && rm %s' % (f, f))  # unzip
                else:  # bash script
                    r = os.system(s)
                print('Dataset autodownload %s\n' % ('success' if r == 0 else 'failure'))  # analyze return value
            else:
                raise Exception('Dataset not found.')


def make_divisible(x, divisor):
    # Returns x evenly divisible by divisor
    return math.ceil(x / divisor) * divisor


def clean_str(s):
    # Cleans a string by replacing special characters with underscore _
    return re.sub(pattern="[|@#!¡·$€%&()=?¿^*;:,¨´><+]", repl="_", string=s)


def one_cycle(y1=0.0, y2=1.0, steps=100):
    # lambda function for sinusoidal ramp from y1 to y2
    return lambda x: ((1 - math.cos(x * math.pi / steps)) / 2) * (y2 - y1) + y1


def colorstr(*input):
    # Colors a string https://en.wikipedia.org/wiki/ANSI_escape_code, i.e.  colorstr('blue', 'hello world')
    *args, string = input if len(input) > 1 else ('blue', 'bold', input[0])  # color arguments, string
    colors = {'black': '\033[30m',  # basic colors
              'red': '\033[31m',
              'green': '\033[32m',
              'yellow': '\033[33m',
              'blue': '\033[34m',
              'magenta': '\033[35m',
              'cyan': '\033[36m',
              'white': '\033[37m',
              'bright_black': '\033[90m',  # bright colors
              'bright_red': '\033[91m',
              'bright_green': '\033[92m',
              'bright_yellow': '\033[93m',
              'bright_blue': '\033[94m',
              'bright_magenta': '\033[95m',
              'bright_cyan': '\033[96m',
              'bright_white': '\033[97m',
              'end': '\033[0m',  # misc
              'bold': '\033[1m',
              'underline': '\033[4m'}
    return ''.join(colors[x] for x in args) + f'{string}' + colors['end']


def labels_to_class_weights(labels, nc=80):
    # Get class weights (inverse frequency) from training labels
    if labels[0] is None:  # no labels loaded
        return torch.Tensor()

    labels = np.concatenate(labels, 0)  # labels.shape = (866643, 5) for COCO
    classes = labels[:, 0].astype(np.int32)  # labels = [class xywh]
    weights = np.bincount(classes, minlength=nc)  # occurrences per class

    # Prepend gridpoint count (for uCE training)
    # gpi = ((320 / 32 * np.array([1, 2, 4])) ** 2 * 3).sum()  # gridpoints per image
    # weights = np.hstack([gpi * len(labels)  - weights.sum() * 9, weights * 9]) ** 0.5  # prepend gridpoints to start

    weights[weights == 0] = 1  # replace empty bins with 1
    weights = 1 / weights  # number of targets per class
    weights /= weights.sum()  # normalize
    return torch.from_numpy(weights)


def labels_to_image_weights(labels, nc=80, class_weights=np.ones(80)):
    # Produces image weights based on class_weights and image contents
    class_counts = np.array([np.bincount(x[:, 0].astype(np.int32), minlength=nc) for x in labels])
    image_weights = (class_weights.reshape(1, nc) * class_counts).sum(1)
    # index = random.choices(range(n), weights=image_weights, k=1)  # weight image sample
    return image_weights


def coco80_to_coco91_class():  # converts 80-index (val2014) to 91-index (paper)
    # https://tech.amikelive.com/node-718/what-object-categories-labels-are-in-coco-dataset/
    # a = np.loadtxt('data/coco.names', dtype='str', delimiter='\n')
    # b = np.loadtxt('data/coco_paper.names', dtype='str', delimiter='\n')
    # x1 = [list(a[i] == b).index(True) + 1 for i in range(80)]  # darknet to coco
    # x2 = [list(b[i] == a).index(True) if any(b[i] == a) else None for i in range(91)]  # coco to darknet
    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34,
         35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
         64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]
    return x

def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y

'''
def xyxyxyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = torch.zeros(4)#x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = (max(x[0::2])-min(x[0::2])) / 2  # x center
    y[:, 1] = (max(x[1::2])-min(x[1::2])) / 2  # y center
    y[:, 2] = max(x[0::2])-min(x[0::2])  # width
    y[:, 3] = max(x[1::2])-min(x[1::2])  # height
    return y
'''
def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def xywhn2xyxy(x, w=640, h=640, padw=0, padh=0):
    # Convert nx4 boxes from [x, y, w, h] normalized to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = w * (x[:, 0] - x[:, 2] / 2) + padw  # top left x
    y[:, 1] = h * (x[:, 1] - x[:, 3] / 2) + padh  # top left y
    y[:, 2] = w * (x[:, 0] + x[:, 2] / 2) + padw  # bottom right x
    y[:, 3] = h * (x[:, 1] + x[:, 3] / 2) + padh  # bottom right y
    return y


def xyn2xy(x, w=640, h=640, padw=0, padh=0):
    # Convert normalized segments into pixel segments, shape (n,2)
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = w * x[:, 0] + padw  # top left x
    y[:, 1] = h * x[:, 1] + padh  # top left y
    return y


def segment2box(segment, width=640, height=640):
    # Convert 1 segment label to 1 box label, applying inside-image constraint, i.e. (xy1, xy2, ...) to (xyxy)
    x, y = segment.T  # segment xy
    inside = (x >= 0) & (y >= 0) & (x <= width) & (y <= height)
    x, y, = x[inside], y[inside]
    return np.array([x.min(), y.min(), x.max(), y.max()]) if any(x) else np.zeros((1, 4))  # xyxy


def segments2boxes(segments):
    # Convert segment labels to box labels, i.e. (cls, xy1, xy2, ...) to (cls, xywh)
    boxes = []
    for s in segments:
        x, y = s.T  # segment xy
        boxes.append([x.min(), y.min(), x.max(), y.max()])  # cls, xyxy
    return xyxy2xywh(np.array(boxes))  # cls, xywh


def resample_segments(segments, n=1000):
    # Up-sample an (n,2) segment
    for i, s in enumerate(segments):
        x = np.linspace(0, len(s) - 1, n)
        xp = np.arange(len(s))
        segments[i] = np.concatenate([np.interp(x, xp, s[:, i]) for i in range(2)]).reshape(2, -1).T  # segment xy
    return segments


def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords

# gwang add
def scale_polys(img1_shape, polys, img0_shape, ratio_pad=None):
    # ratio_pad: [(h_raw, w_raw), (hw_ratios, wh_paddings)]
    # Rescale coords (xyxyxyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = resized / raw
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0] # h_ratios
        pad = ratio_pad[1] # wh_paddings

    polys[:, [0, 2, 4, 6]] -= pad[0]  # x padding
    polys[:, [1, 3, 5, 7]] -= pad[1]  # y padding
    polys[:, :8] /= gain # Rescale poly shape to img0_shape
    #clip_polys(polys, img0_shape)
    return polys

def clip_coords(boxes, img_shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    boxes[:, 0].clamp_(0, img_shape[1])  # x1
    boxes[:, 1].clamp_(0, img_shape[0])  # y1
    boxes[:, 2].clamp_(0, img_shape[1])  # x2
    boxes[:, 3].clamp_(0, img_shape[0])  # y2


def bbox_iou(box1, box2, x1y1x2y2=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
    # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4
    box2 = box2.T

    # Get the coordinates of bounding boxes
    if x1y1x2y2:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    else:  # transform from xywh to xyxy
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    union = w1 * h1 + w2 * h2 - inter + eps

    iou = inter / union

    if GIoU or DIoU or CIoU:
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex (smallest enclosing box) width
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
        if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 +
                    (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center distance squared
            if DIoU:
                return iou - rho2 / c2  # DIoU
            elif CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / (h2 + eps)) - torch.atan(w1 / (h1 + eps)), 2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha)  # CIoU
        else:  # GIoU https://arxiv.org/pdf/1902.09630.pdf
            c_area = cw * ch + eps  # convex area
            return iou - (c_area - union) / c_area  # GIoU
    else:
        return iou  # IoU




def bbox_alpha_iou(box1, box2, x1y1x2y2=False, GIoU=False, DIoU=False, CIoU=False, alpha=2, eps=1e-9):
    # Returns tsqrt_he IoU of box1 to box2. box1 is 4, box2 is nx4
    box2 = box2.T

    # Get the coordinates of bounding boxes
    if x1y1x2y2:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    else:  # transform from xywh to xyxy
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    union = w1 * h1 + w2 * h2 - inter + eps

    # change iou into pow(iou+eps)
    # iou = inter / union
    iou = torch.pow(inter/union + eps, alpha)
    # beta = 2 * alpha
    if GIoU or DIoU or CIoU:
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex (smallest enclosing box) width
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
        if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = (cw ** 2 + ch ** 2) ** alpha + eps  # convex diagonal
            rho_x = torch.abs(b2_x1 + b2_x2 - b1_x1 - b1_x2)
            rho_y = torch.abs(b2_y1 + b2_y2 - b1_y1 - b1_y2)
            rho2 = ((rho_x ** 2 + rho_y ** 2) / 4) ** alpha  # center distance
            if DIoU:
                return iou - rho2 / c2  # DIoU
            elif CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                with torch.no_grad():
                    alpha_ciou = v / ((1 + eps) - inter / union + v)
                # return iou - (rho2 / c2 + v * alpha_ciou)  # CIoU
                return iou - (rho2 / c2 + torch.pow(v * alpha_ciou + eps, alpha))  # CIoU
        else:  # GIoU https://arxiv.org/pdf/1902.09630.pdf
            # c_area = cw * ch + eps  # convex area
            # return iou - (c_area - union) / c_area  # GIoU
            c_area = torch.max(cw * ch + eps, union) # convex area
            return iou - torch.pow((c_area - union) / c_area + eps, alpha)  # GIoU
    else:
        return iou # torch.log(iou+eps) or iou


def box_iou(box1, box2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)


def wh_iou(wh1, wh2):
    # Returns the nxm IoU matrix. wh1 is nx2, wh2 is mx2
    wh1 = wh1[:, None]  # [N,1,2]
    wh2 = wh2[None]  # [1,M,2]
    inter = torch.min(wh1, wh2).prod(2)  # [N,M]
    return inter / (wh1.prod(2) + wh2.prod(2) - inter)  # iou = inter / (area1 + area2 - inter)


def box_giou(box1, box2):
    """
    Return generalized intersection-over-union (Jaccard index) between two sets of boxes.
    Both sets of boxes are expected to be in ``(x1, y1, x2, y2)`` format with
    ``0 <= x1 < x2`` and ``0 <= y1 < y2``.
    Args:
        boxes1 (Tensor[N, 4]): first set of boxes
        boxes2 (Tensor[M, 4]): second set of boxes
    Returns:
        Tensor[N, M]: the NxM matrix containing the pairwise generalized IoU values
        for every element in boxes1 and boxes2
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)
    
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    union = (area1[:, None] + area2 - inter)

    iou = inter / union

    lti = torch.min(box1[:, None, :2], box2[:, :2])
    rbi = torch.max(box1[:, None, 2:], box2[:, 2:])

    whi = (rbi - lti).clamp(min=0)  # [N,M,2]
    areai = whi[:, :, 0] * whi[:, :, 1]

    return iou - (areai - union) / areai


def box_ciou(box1, box2, eps: float = 1e-7):
    """
    Return complete intersection-over-union (Jaccard index) between two sets of boxes.
    Both sets of boxes are expected to be in ``(x1, y1, x2, y2)`` format with
    ``0 <= x1 < x2`` and ``0 <= y1 < y2``.
    Args:
        boxes1 (Tensor[N, 4]): first set of boxes
        boxes2 (Tensor[M, 4]): second set of boxes
        eps (float, optional): small number to prevent division by zero. Default: 1e-7
    Returns:
        Tensor[N, M]: the NxM matrix containing the pairwise complete IoU values
        for every element in boxes1 and boxes2
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)
    
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    union = (area1[:, None] + area2 - inter)

    iou = inter / union

    lti = torch.min(box1[:, None, :2], box2[:, :2])
    rbi = torch.max(box1[:, None, 2:], box2[:, 2:])

    whi = (rbi - lti).clamp(min=0)  # [N,M,2]
    diagonal_distance_squared = (whi[:, :, 0] ** 2) + (whi[:, :, 1] ** 2) + eps

    # centers of boxes
    x_p = (box1[:, None, 0] + box1[:, None, 2]) / 2
    y_p = (box1[:, None, 1] + box1[:, None, 3]) / 2
    x_g = (box2[:, 0] + box2[:, 2]) / 2
    y_g = (box2[:, 1] + box2[:, 3]) / 2
    # The distance between boxes' centers squared.
    centers_distance_squared = (x_p - x_g) ** 2 + (y_p - y_g) ** 2

    w_pred = box1[:, None, 2] - box1[:, None, 0]
    h_pred = box1[:, None, 3] - box1[:, None, 1]

    w_gt = box2[:, 2] - box2[:, 0]
    h_gt = box2[:, 3] - box2[:, 1]

    v = (4 / (torch.pi ** 2)) * torch.pow((torch.atan(w_gt / h_gt) - torch.atan(w_pred / h_pred)), 2)
    with torch.no_grad():
        alpha = v / (1 - iou + v + eps)
    return iou - (centers_distance_squared / diagonal_distance_squared) - alpha * v


def box_diou(box1, box2, eps: float = 1e-7):
    """
    Return distance intersection-over-union (Jaccard index) between two sets of boxes.
    Both sets of boxes are expected to be in ``(x1, y1, x2, y2)`` format with
    ``0 <= x1 < x2`` and ``0 <= y1 < y2``.
    Args:
        boxes1 (Tensor[N, 4]): first set of boxes
        boxes2 (Tensor[M, 4]): second set of boxes
        eps (float, optional): small number to prevent division by zero. Default: 1e-7
    Returns:
        Tensor[N, M]: the NxM matrix containing the pairwise distance IoU values
        for every element in boxes1 and boxes2
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)
    
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    union = (area1[:, None] + area2 - inter)

    iou = inter / union

    lti = torch.min(box1[:, None, :2], box2[:, :2])
    rbi = torch.max(box1[:, None, 2:], box2[:, 2:])

    whi = (rbi - lti).clamp(min=0)  # [N,M,2]
    diagonal_distance_squared = (whi[:, :, 0] ** 2) + (whi[:, :, 1] ** 2) + eps

    # centers of boxes
    x_p = (box1[:, None, 0] + box1[:, None, 2]) / 2
    y_p = (box1[:, None, 1] + box1[:, None, 3]) / 2
    x_g = (box2[:, 0] + box2[:, 2]) / 2
    y_g = (box2[:, 1] + box2[:, 3]) / 2
    # The distance between boxes' centers squared.
    centers_distance_squared = (x_p - x_g) ** 2 + (y_p - y_g) ** 2

    # The distance IoU is the IoU penalized by a normalized
    # distance between boxes' centers squared.
    return iou - (centers_distance_squared / diagonal_distance_squared)


def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, multi_label=False,
                        labels=(), max_det=300):
    """Runs Non-Maximum Suppression (NMS) on inference results

    Returns:
         list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """

    nc = prediction.shape[2] - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Settings
    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
    max_det = 300  # maximum number of detections per image
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 10.0  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    t = time.time()
    output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            l = labels[xi]
            v = torch.zeros((len(l), nc + 5), device=x.device)
            v[:, :4] = l[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(l)), l[:, 0].long() + 5] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        if nc == 1:
            x[:, 5:] = x[:, 4:5] # for models with one class, cls_loss is 0 and cls_conf is always 0.5,
                                 # so there is no need to multiplicate.
        else:
            x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            print(f'WARNING: NMS time limit {time_limit}s exceeded')
            break  # time limit exceeded

    return output


def non_max_suppression_kpt(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, multi_label=False,
                        labels=(), kpt_label=False, nc=None, nkpt=None):
    """Runs Non-Maximum Suppression (NMS) on inference results

    Returns:
         list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """
    if nc is None:
        nc = prediction.shape[2] - 5  if not kpt_label else prediction.shape[2] - 56 # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Settings
    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
    max_det = 300  # maximum number of detections per image
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 10.0  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    t = time.time()
    output = [torch.zeros((0,6), device=prediction.device)] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            l = labels[xi]
            v = torch.zeros((len(l), nc + 5), device=x.device)
            v[:, :4] = l[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(l)), l[:, 0].long() + 5] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:5+nc] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:  # best class only
            if not kpt_label:
                conf, j = x[:, 5:].max(1, keepdim=True)
                x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]
            else:
                kpts = x[:, 6:]
                conf, j = x[:, 5:6].max(1, keepdim=True)
                x = torch.cat((box, conf, j.float(), kpts), 1)[conf.view(-1) > conf_thres]


        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            print(f'WARNING: NMS time limit {time_limit}s exceeded')
            break  # time limit exceeded

    return output

def save_one_box(xyxy, im, file='image.jpg', gain=1.02, pad=10, square=False, BGR=False, save=True):
    # Save image crop as {file} with crop size multiple {gain} and {pad} pixels. Save and/or return crop
    xyxy = torch.tensor(xyxy).view(-1, 4)
    b = xyxy2xywh(xyxy)  # boxes
    if square:
        b[:, 2:] = b[:, 2:].max(1)[0].unsqueeze(1)  # attempt rectangle to square
    b[:, 2:] = b[:, 2:] * gain + pad  # box wh * gain + pad
    xyxy = xywh2xyxy(b).long()
    clip_coords(xyxy, im.shape)
    crop = im[int(xyxy[0, 1]):int(xyxy[0, 3]), int(xyxy[0, 0]):int(xyxy[0, 2]), ::(1 if BGR else -1)]
    if save:
        cv2.imwrite(str(increment_path(file, mkdir=True).with_suffix('.jpg')), crop)
    return 

def strip_optimizer(f='best.pt', s=''):  # from utils.general import *; strip_optimizer()
    # Strip optimizer from 'f' to finalize training, optionally save as 's'
    x = torch.load(f, map_location=torch.device('cpu'))
    if x.get('ema'):
        x['model'] = x['ema']  # replace model with ema
    for k in 'optimizer', 'training_results', 'wandb_id', 'ema', 'updates':  # keys
        x[k] = None
    x['epoch'] = -1
    x['model'].half()  # to FP16
    for p in x['model'].parameters():
        p.requires_grad = False
    torch.save(x, s or f)
    mb = os.path.getsize(s or f) / 1E6  # filesize
    print(f"Optimizer stripped from {f},{(' saved as %s,' % s) if s else ''} {mb:.1f}MB")


def print_mutation(hyp, results, yaml_file='hyp_evolved.yaml', bucket=''):
    # Print mutation results to evolve.txt (for use with train.py --evolve)
    a = '%10s' * len(hyp) % tuple(hyp.keys())  # hyperparam keys
    b = '%10.3g' * len(hyp) % tuple(hyp.values())  # hyperparam values
    c = '%10.4g' * len(results) % results  # results (P, R, mAP@0.5, mAP@0.5:0.95, val_losses x 3)
    print('\n%s\n%s\nEvolved fitness: %s\n' % (a, b, c))

    if bucket:
        url = 'gs://%s/evolve.txt' % bucket
        if gsutil_getsize(url) > (os.path.getsize('evolve.txt') if os.path.exists('evolve.txt') else 0):
            os.system('gsutil cp %s .' % url)  # download evolve.txt if larger than local

    with open('evolve.txt', 'a') as f:  # append result
        f.write(c + b + '\n')
    x = np.unique(np.loadtxt('evolve.txt', ndmin=2), axis=0)  # load unique rows
    x = x[np.argsort(-fitness(x))]  # sort
    np.savetxt('evolve.txt', x, '%10.3g')  # save sort by fitness

    # Save yaml
    for i, k in enumerate(hyp.keys()):
        hyp[k] = float(x[0, i + 7])
    with open(yaml_file, 'w') as f:
        results = tuple(x[0, :7])
        c = '%10.4g' * len(results) % results  # results (P, R, mAP@0.5, mAP@0.5:0.95, val_losses x 3)
        f.write('# Hyperparameter Evolution Results\n# Generations: %g\n# Metrics: ' % len(x) + c + '\n\n')
        yaml.dump(hyp, f, sort_keys=False)

    if bucket:
        os.system('gsutil cp evolve.txt %s gs://%s' % (yaml_file, bucket))  # upload


def apply_classifier(x, model, img, im0):
    # applies a second stage classifier to yolo outputs
    im0 = [im0] if isinstance(im0, np.ndarray) else im0
    for i, d in enumerate(x):  # per image
        if d is not None and len(d):
            d = d.clone()

            # Reshape and pad cutouts
            b = xyxy2xywh(d[:, :4])  # boxes
            b[:, 2:] = b[:, 2:].max(1)[0].unsqueeze(1)  # rectangle to square
            b[:, 2:] = b[:, 2:] * 1.3 + 30  # pad
            d[:, :4] = xywh2xyxy(b).long()

            # Rescale boxes from img_size to im0 size
            scale_coords(img.shape[2:], d[:, :4], im0[i].shape)

            # Classes
            pred_cls1 = d[:, 5].long()
            ims = []
            for j, a in enumerate(d):  # per item
                cutout = im0[i][int(a[1]):int(a[3]), int(a[0]):int(a[2])]
                im = cv2.resize(cutout, (224, 224))  # BGR
                # cv2.imwrite('test%i.jpg' % j, cutout)

                im = im[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
                im = np.ascontiguousarray(im, dtype=np.float32)  # uint8 to float32
                im /= 255.0  # 0 - 255 to 0.0 - 1.0
                ims.append(im)

            pred_cls2 = model(torch.Tensor(ims).to(d.device)).argmax(1)  # classifier prediction
            x[i] = x[i][pred_cls1 == pred_cls2]  # retain matching class detections

    return x


def increment_path(path, exist_ok=True, sep=''):
    # Increment path, i.e. runs/exp --> runs/exp{sep}0, runs/exp{sep}1 etc.
    path = Path(path)  # os-agnostic
    
    if (path.exists() and exist_ok): #or (not path.exists()):
        return str(path)
    else:
        dirs = glob.glob(f"{path}{sep}*")  # similar paths
        matches = [re.search(rf"%s{sep}(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]  # indices
        n = max(i) + 1 if i else 2  # increment number

        return f"{path}{sep}{n}"  # update path

# Ancillary functions with polygon anchor boxes-------------------------------------------------------------------------------------------

def xyxyxyxyn2xyxyxyxy(x, w=640, h=640, padw=0, padh=0):
    # Convert normalized xyxyxyxy or segments into pixel xyxyxyxy or segments
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0::2] = w * x[:, 0::2] + padw  # all x
    y[:, 1::2] = h * x[:, 1::2] + padh  # all y
    return y

def polygon_inter_union(boxes1, boxes2):
    """
        Reference: https://github.com/ming71/yolov3-polygon/blob/master/utils/utils.py ;
        iou computation (polygon) with cpu;
        Boxes have shape nx8 and Anchors have mx8;
        Return intersection and union of boxes[i, :] and anchors[j, :] with shape of (n, m).
    """
    #from shapely.validation import make_valid
    n, m = boxes1.shape[0], boxes2.shape[0]
    inter = torch.zeros(n, m)
    union = torch.zeros(n, m)
    for i in range(n):
        polygon1 = shapely.geometry.Polygon(np.array(boxes1[i, :].view(4,2).cpu())).convex_hull
        for j in range(m):
            polygon2 = shapely.geometry.Polygon(np.array(boxes2[j, :].view(4,2).cpu())).convex_hull
            if polygon1.intersects(polygon2):
                try:
                    inter[i, j] = polygon1.intersection(polygon2).area
                    union[i, j] = polygon1.union(polygon2).area
                except shapely.geos.TopologicalError:
                    print('shapely.geos.TopologicalError occured')
    return inter, union

def polygon_b_inter_union(boxes1, boxes2):
    """
        iou computation (polygon) with cpu for class Polygon_ComputeLoss in loss.py;
        Boxes and Anchors having the same shape: nx8;
        Return intersection and union of boxes[i, :] and anchors[i, :] with shape of (n, ).
    """
    #from shapely.validation import make_valid
    n = boxes1.shape[0]
    inter = torch.zeros(n,)
    union = torch.zeros(n,)
    boxes1 = boxes1.detach().cpu(); boxes2 = boxes2.detach().cpu()
    
    for i in range(n):
        polygon1 = shapely.geometry.Polygon(np.array(boxes1[i, :].view(4,2))).convex_hull
        polygon2 = shapely.geometry.Polygon(np.array(boxes2[i, :].view(4,2))).convex_hull
        if polygon1.intersects(polygon2):
            try:
                inter[i] = polygon1.intersection(polygon2).area
                union[i] = polygon1.union(polygon2).area
            except shapely.geos.TopologicalError:
                print('shapely.geos.TopologicalError occured')
    return inter, union

def polygon_clip_coords(boxes, img_shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    boxes[:, 0::2].clamp_(0, img_shape[1])  # x1x2x3x4
    boxes[:, 1::2].clamp_(0, img_shape[0])  # y1y2y3y4

def polygon_scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxyxyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, 0::2] -= pad[0]  # x padding
    coords[:, 1::2] -= pad[1]  # y padding
    coords[:, :8] /= gain
    polygon_clip_coords(coords, img0_shape)  # inplace operation
    return coords

def polygon_bbox_iou(boxes1, boxes2, iou_type='ciou', eps=1e-7, ordered=False, constant=12.8, device=None):
    """
        Compute iou of polygon boxes for class Polygon_ComputeLoss in loss.py via cpu or cuda;
        For cuda code, please refer to files in ./iou_cuda
    """
    # For testing this function, please use ordered=False
    if not ordered:
        boxes1, boxes2 = order_corners(boxes1.clone()), order_corners(boxes2.clone())
    else:
        boxes1, boxes2 = boxes1.clone(), boxes2.clone()
    
    if boxes1.is_cuda:
        #boxes1_ = boxes1.float().contiguous()
        #boxes2_ = boxes2.float().contiguous()
        # using shapely (cpu) to compute
        if iou_type != 'gaussian':
            inter, union = polygon_b_inter_union(boxes1, boxes2)
    
    if iou_type != 'gaussian':
        union += eps
        iou = inter / union
        iou[torch.isnan(inter)] = 0.0
        iou[torch.logical_and(torch.isnan(inter), torch.isnan(union))] = 1.0
        iou[torch.isnan(iou)] = 0.0

    GIoU=False; DIoU=False; CIoU=False; GaIoU=False
    if iou_type == 'giou':
        GIoU = True
    elif iou_type == 'diou':
        DIoU = True
    elif iou_type == 'ciou':
        CIoU = True  
    elif iou_type == 'gaussian':
        GaIoU = True  

    if GIoU or DIoU or CIoU or GaIoU:
        # minimum bounding box of boxes1 and boxes2
        b1_x1, b1_x2 = boxes1[:, 0::2].min(dim=1)[0], boxes1[:, 0::2].max(dim=1)[0] # n,
        b1_y1, b1_y2 = boxes1[:, 1::2].min(dim=1)[0], boxes1[:, 1::2].max(dim=1)[0] # n,
        b2_x1, b2_x2 = boxes2[:, 0::2].min(dim=1)[0], boxes2[:, 0::2].max(dim=1)[0] # n,
        b2_y1, b2_y2 = boxes2[:, 1::2].min(dim=1)[0], boxes2[:, 1::2].max(dim=1)[0] # n,
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex (smallest enclosing box) width
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
        if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 +
                    (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center distance squared
            if DIoU:
                iou -= rho2 / c2  # DIoU
            elif CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                w2, h2 = b2_x2-b2_x1, b2_y2-b2_y1+eps
                w1, h1 = b1_x2-b1_x1, b1_y2-b1_y1+eps
                v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                
                with torch.no_grad():
                    iou = iou.to(device)
                    alpha = v / (v - iou + (1 + eps))
                iou -= (rho2 / c2 + v * alpha)  # CIoU
        elif GaIoU: 
            mins1 = torch.cat([boxes1[:, 0::2].min(dim=1).values.unsqueeze(0), boxes1[:, 1::2].min(dim=1).values.unsqueeze(0)], dim=0)
            maxs1 = torch.cat([boxes1[:, 0::2].max(dim=1).values.unsqueeze(0), boxes1[:, 1::2].max(dim=1).values.unsqueeze(0)], dim=0)
            mins2 = torch.cat([boxes2[:, 0::2].min(dim=1).values.unsqueeze(0), boxes2[:, 1::2].min(dim=1).values.unsqueeze(0)], dim=0)
            maxs2 = torch.cat([boxes2[:, 0::2].max(dim=1).values.unsqueeze(0), boxes2[:, 1::2].max(dim=1).values.unsqueeze(0)], dim=0)

            center1 = (mins1+maxs1)/2; center2 = (mins2+maxs2)/2
            dists = (center1 - center2)**2; center_distance = dists[0,:]+dists[1,:] # dist = (x1-x2)**2 + (y1-y2)**2
            
            w2, h2 = b2_x2-b2_x1, b2_y2-b2_y1+eps
            w1, h1 = b1_x2-b1_x1, b1_y2-b1_y1+eps
            wh_distance = ((w1 - w2) ** 2 + (h1 - h2) ** 2) / 4

            wassersteins = torch.sqrt(center_distance + wh_distance)
            normalized_wasserstein = torch.exp(-wassersteins/constant)
                
            iou = normalized_wasserstein
        else:  # GIoU https://arxiv.org/pdf/1902.09630.pdf
            c_area = cw * ch + eps  # convex area
            iou -= (c_area - union) / c_area  # GIoU
    return iou  # IoU


def polygon_box_iou(boxes1, boxes2, iou_type='iou', eps=1e-7, ordered=False, constant=12.8):
    """
        Compute iou of polygon boxes via cpu or cuda;
        For cuda code, please refer to files in ./iou_cuda
        Returns the IoU of shape (n, m) between boxes1 and boxes2. boxes1 is nx8, boxes2 is mx8
    """
    import time 
    start_time = time.time()
    # For testing this function, please use ordered=False
    if not ordered:
        boxes1, boxes2 = order_corners(boxes1.clone().cpu()), order_corners(boxes2.clone().cpu())
    else:
        boxes1, boxes2 = boxes1.clone().cpu(), boxes2.clone().cpu()
        
    # using shapely (cpu) to compute
    
    inter, union = polygon_inter_union(boxes1, boxes2)
    union += eps
    iou = inter / union
    iou[torch.isnan(inter)] = 0.0
    iou[torch.logical_and(torch.isnan(inter), torch.isnan(union))] = 1.0
    iou[torch.isnan(iou)] = 0.0

    GIoU=False; DIoU=False; CIoU=False; GaIoU=False
    if iou_type == 'giou':
        GIoU = True
    elif iou_type == 'diou':
        DIoU = True
    elif iou_type == 'ciou':
        CIoU = True  
    elif iou_type == 'gaussian':
        GaIoU = True  
    if GIoU or DIoU or CIoU or GaIoU:
        # minimum bounding box of boxes1 and boxes2
        b1_x1, b1_x2 = boxes1[:, 0::2].min(dim=1)[0], boxes1[:, 0::2].max(dim=1)[0] # 1xn
        b1_y1, b1_y2 = boxes1[:, 1::2].min(dim=1)[0], boxes1[:, 1::2].max(dim=1)[0] # 1xn
        b2_x1, b2_x2 = boxes2[:, 0::2].min(dim=1)[0], boxes2[:, 0::2].max(dim=1)[0] # 1xm
        b2_y1, b2_y2 = boxes2[:, 1::2].min(dim=1)[0], boxes2[:, 1::2].max(dim=1)[0] # 1xm
        for i in range(boxes1.shape[0]):
            cw = torch.max(b1_x2[i], b2_x2) - torch.min(b1_x1[i], b2_x1)  # convex (smallest enclosing box) width
            ch = torch.max(b1_y2[i], b2_y2) - torch.min(b1_y1[i], b2_y1)  # convex height
            if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
                c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
                rho2 = ((b2_x1 + b2_x2 - b1_x1[i] - b1_x2[i]) ** 2 +
                        (b2_y1 + b2_y2 - b1_y1[i] - b1_y2[i]) ** 2) / 4  # center distance squared
                if DIoU:
                    iou[i, :] -= rho2 / c2  # DIoU
                elif CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                    w2, h2 = b2_x2-b2_x1, b2_y2-b2_y1+eps
                    w1, h1 = b1_x2[i]-b1_x1[i], b1_y2[i]-b1_y1[i]+eps
                    v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                    with torch.no_grad():
                        alpha = v / (v - iou[i, :] + (1 + eps))
                    iou[i, :] -= (rho2 / c2 + v * alpha)  # CIoU
            elif GaIoU:
                mins1 = torch.cat([boxes1[:, 0::2].min(dim=1).values.unsqueeze(0), boxes1[:, 1::2].min(dim=1).values.unsqueeze(0)], dim=0)
                maxs1 = torch.cat([boxes1[:, 0::2].max(dim=1).values.unsqueeze(0), boxes1[:, 1::2].max(dim=1).values.unsqueeze(0)], dim=0)
                mins2 = torch.cat([boxes2[:, 0::2].min(dim=1).values.unsqueeze(0), boxes2[:, 1::2].min(dim=1).values.unsqueeze(0)], dim=0)
                maxs2 = torch.cat([boxes2[:, 0::2].max(dim=1).values.unsqueeze(0), boxes2[:, 1::2].max(dim=1).values.unsqueeze(0)], dim=0)

                center1 = (mins1+maxs1)/2; center2 = (mins2+maxs2)/2
                dists = (center1 - center2)**2; center_distance = dists[0,:]+dists[1,:] # dist = (x1-x2)**2 + (y1-y2)**2
                
                w2, h2 = b2_x2-b2_x1, b2_y2-b2_y1+eps
                w1, h1 = b1_x2[i]-b1_x1[i], b1_y2[i]-b1_y1[i]+eps
                wh_distance = ((w1 - w2) ** 2 + (h1 - h2) ** 2) / 4

                wassersteins = torch.sqrt(center_distance + wh_distance)
                normalized_wasserstein = torch.exp(-wassersteins/constant)
                
                iou = normalized_wasserstein
            else:  # GIoU https://arxiv.org/pdf/1902.09630.pdf
                c_area = cw * ch + eps  # convex area
                iou[i, :] -= (c_area - union[i, :]) / c_area  # GIoU

        #print(time.time()-start_time)
    return iou  # IoU

def polygon_non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, multi_label=False,
                        labels=(), max_det=300):
    """
        Runs Non-Maximum Suppression (NMS) on inference results for polygon boxes
        Returns:  list of detections, on (n,10) tensor per image [xyxyxyxy, conf, cls]
    """
    # prediction has the shape of (bs, all potential anchors, 89)
    assert not agnostic, "polygon does not support agnostic"
    nc = prediction.shape[2] - 9  # number of classes
    xc = prediction[..., 8] > conf_thres  # confidence candidates

    # Checks
    assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
    assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'

    # Settings
    min_wh, max_wh = 3, 4096  # (pixels) minimum and maximum box width and height
    max_nms = 30000  # maximum number of boxes into polygon_nms_kernel, can increase this value
    time_limit = 10.0  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    t = time.time()
    output = [torch.zeros((0, 10), device=prediction.device)] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        x = x[xc[xi]]  # confidence
    
        if labels and len(labels[xi]):
            l = labels[xi]
            v = torch.zeros((len(l), nc + 9), device=x.device)
            v[:, :8] = l[:, 1:9]  # box
            v[:, 8] = 1.0  # conf
            v[range(len(l)), l[:, 0].long() + 9] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain, process next image
        if not x.shape[0]:
            continue
        # Compute conf
        x[:, 9:] *= x[:, 8:9]  # conf = obj_conf * cls_conf

        # Box (x1, y1, x2, y2, x3, y3, x4, y4)
        box = x[:, :8].clone()

        # Detections matrix nx10 (xyxyxyxy, conf, cls)
        # Transfer sigmoid probabilities of classes (e.g. three classes [0.567, 0.907, 0.01]) to selected classes (1.0)
        if multi_label:
            i, j = (x[:, 9:] > conf_thres).nonzero(as_tuple=False).T
            # concat satisfied boxes (multi-label-enabled) along 0 dimension
            x = torch.cat((box[i], x[i, j + 9, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = x[:, 9:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 9:10] == torch.tensor(classes, device=x.device)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            x = x[x[:, 8].argsort(descending=True)[:max_nms]]  # sort by confidence

        # Polygon NMS does not support Batch NMS and Agnostic
        # x is the sorted predictions with boxes x[:, :8], confidence x[:, 8], class x[:, 9]         
        # cannot use torchvision.ops.nms, which only deals with axis-aligned boxes
        i = polygon_nms_kernel(x, iou_thres)  # polygon-NMS
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            boxes = x[:, :8]
            # update boxes as boxes(i,8) = weights(i,n) * polygon boxes(n,8)
            iou = polygon_box_iou(boxes[i], boxes, iou_type=Cfg.iou_type) > iou_thres  # iou matrix
            weights = iou * x[:, 8][None]  # polygon box weights
            x[i, :8] = torch.mm(weights, x[:, :8]).float() / weights.sum(1, keepdim=True)  # merged polygon boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy
        output[xi] = x[i]

        if (time.time() - t) > time_limit:
            print(f'WARNING: NMS time limit {time_limit}s exceeded')
            break  # time limit exceeded

    return output

def polygon_nms_kernel(x, iou_thres):
    """
        non maximum suppression kernel for polygon-enabled boxes
        x is the prediction with boxes x[:, :8], confidence x[:, 8], class x[:, 9] 
        Return the selected indices
    """
    
    unique_labels = x[:, 9].unique()
    _, scores_sort_index = torch.sort(x[:, 8], descending=True)
    x = x[scores_sort_index]
    x[:, :8] = order_corners(x[:, :8])
    indices = scores_sort_index
    selected_indices = []
    start_time=time.time()
    #breakpoint()
    # Iterate through all predicted classes
    for unique_label in unique_labels:
        x_ = x[x[:, 9]==unique_label]
        indices_ = indices[x[:, 9]==unique_label]
        
        while x_.shape[0]:
            
            # Save the indice with the highest confidence
            selected_indices.append(indices_[0])
            if len(x_) == 1: break
            # Compute the IOUs for all other the polygon boxes
            iou = polygon_box_iou(x_[0:1, :8], x_[1:, :8], ordered=True, iou_type=Cfg.iou_type).view(-1)
            #print(len(x_))
            # Remove overlapping detections with IoU >= NMS threshold
            x_ = x_[1:][iou < iou_thres]
            indices_ = indices_[1:][iou < iou_thres]
    #print("whole processing..", time.time()-start_time)
    return torch.LongTensor(selected_indices)

def order_corners(boxes):
    """
        Return sorted corners for loss.py::class Polygon_ComputeLoss::build_targets
        Sorted corners have the following restrictions: 
                                y3, y4 >= y1, y2; x1 <= x2; x4 <= x3
    """
    
    if boxes.shape[0] == 0:
        return torch.empty(0, 8, device=boxes.device)
    boxes = boxes.view(-1, 4, 2)
    x = boxes[..., 0]
    y = boxes[..., 1]
    y_sorted, y_indices = torch.sort(y) # sort y
    idx = torch.arange(0, y.shape[0], dtype=torch.long, device=boxes.device)
    complete_idx = idx[:, None].repeat(1, 4)
    x_sorted = x[complete_idx, y_indices]
    x_sorted[:, :2], x_bottom_indices = torch.sort(x_sorted[:, :2])
    x_sorted[:, 2:4], x_top_indices = torch.sort(x_sorted[:, 2:4], descending=True)
    y_sorted[idx, :2] = y_sorted[idx, :2][complete_idx[:, :2], x_bottom_indices]
    y_sorted[idx, 2:4] = y_sorted[idx, 2:4][complete_idx[:, 2:4], x_top_indices]
    
    # prevent the ambiguous case when the diagonal of the quadrilateral is parallel to the x-axis
    special = (y_sorted[:, 1] == y_sorted[:, 2]) & (x_sorted[:, 1] > x_sorted[:, 2])
    if idx[special].shape[0] != 0:
        x_sorted_1 = x_sorted[idx[special], 1].clone()
        x_sorted[idx[special], 1] = x_sorted[idx[special], 2]
        x_sorted[idx[special], 2] = x_sorted_1
    return torch.stack((x_sorted, y_sorted), dim=2).view(-1, 8).contiguous()