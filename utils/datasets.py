# Dataset utils and dataloaders

import glob
import math
import os
import random
import logging
from itertools import repeat
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ExifTags
from torch.utils.data import Dataset
from tqdm import tqdm
from multiprocessing.pool import ThreadPool, Pool

from utils.general import check_requirements, check_file, check_dataset, xyxy2xywh, xywh2xyxy, xywhn2xyxy, xyn2xy, \
    xyxyxyxyn2xyxyxyxy, segment2box, segments2boxes, resample_segments, clean_str, colorstr
from utils.torch_utils import torch_distributed_zero_first

from osgeo import gdal, ogr, osr
from utils.cfg import Cfg

# Parameters
help_url = 'https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data'
img_formats = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo']  # acceptable image suffixes
vid_formats = ['mov', 'avi', 'mp4', 'mpg', 'mpeg', 'm4v', 'wmv', 'mkv']  # acceptable video suffixes
logger = logging.getLogger(__name__)

num_threads = min(8, os.cpu_count())  # number of multiprocessing threads

# Get orientation exif tag
for orientation in ExifTags.TAGS.keys():
    if ExifTags.TAGS[orientation] == 'Orientation':
        break


def get_hash(files):
    # Returns a single hash value of a list of files
    return sum(os.path.getsize(f) for f in files if os.path.isfile(f))


def exif_size(img):
    # Returns exif-corrected PIL size
    s = img.size  # (width, height)
    try:
        rotation = dict(img._getexif().items())[orientation]
        if rotation == 6:  # rotation 270
            s = (s[1], s[0])
        elif rotation == 8:  # rotation 90
            s = (s[1], s[0])
    except:
        pass

    return s

def create_dataloader(path, imgsz, batch_size, stride, opt, hyp=None, augment=False, cache=False, pad=0.0, rect=False,
                      rank=-1, world_size=1, workers=8, image_weights=False, quad=False, prefix='', polygon=False):
    # Make sure only the first process in DDP process the dataset first, and the following others can use the cache
    with torch_distributed_zero_first(rank):
        LoadModels = LoadImagesAndLabels if not polygon else Polygon_LoadImagesAndLabels
        dataset = LoadModels(path, imgsz, batch_size,
                                      augment=augment,  # augment images
                                      hyp=hyp,  # augmentation hyperparameters
                                      rect=rect,  # rectangular training
                                      cache_images=cache,
                                      single_cls=opt.single_cls,
                                      stride=int(stride),
                                      pad=pad,
                                      image_weights=image_weights,
                                      prefix=prefix,
                                      yaml_file=path.split('/')[3]+'.yaml')

    batch_size = min(batch_size, len(dataset))
    nw = min([os.cpu_count() // world_size, batch_size if batch_size > 1 else 0, workers])  # number of workers
    sampler = torch.utils.data.distributed.DistributedSampler(dataset) if rank != -1 else None
    loader = torch.utils.data.DataLoader if image_weights else InfiniteDataLoader
    # Use torch.utils.data.DataLoader() if dataset.properties will update during training else InfiniteDataLoader()
    dataloader = loader(dataset,
                        batch_size=batch_size,
                        num_workers=nw,
                        sampler=sampler,
                        pin_memory=True,
                        collate_fn=LoadImagesAndLabels.collate_fn4 if quad else LoadImagesAndLabels.collate_fn)
    return dataloader, dataset


class InfiniteDataLoader(torch.utils.data.dataloader.DataLoader):
    """ Dataloader that reuses workers

    Uses same syntax as vanilla DataLoader
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, 'batch_sampler', _RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler(object):
    """ Sampler that repeats forever

    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


class LoadImages:  # for inference
    def __init__(self, path, img_size=640, stride=32):
        p = str(Path(path).absolute())  # os-agnostic absolute path
        if '*' in p:
            files = sorted(glob.glob(p, recursive=True))  # glob
        elif os.path.isdir(p):
            files = sorted(glob.glob(os.path.join(p, '*.*')))  # dir
        elif os.path.isfile(p):
            files = [p]  # files
        else:
            raise Exception(f'ERROR: {p} does not exist')

        images = [x for x in files if x.split('.')[-1].lower() in img_formats]
        ni = len(images)

        self.img_size = img_size
        self.stride = stride
        self.files = images 
        self.nf = ni   # number of files
        self.mode = 'image'
        
        assert self.nf > 0, f'No images found in {p}. ' \
                            f'Supported formats are:\nimages: {img_formats}\n'

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count == self.nf:
            raise StopIteration
        path = self.files[self.count]

        # Read image
        self.count += 1
        img0 = cv2.imread(path, cv2.IMREAD_UNCHANGED)  # BGR
        assert img0 is not None, 'Image Not Found ' + path
        #print(f'image {self.count}/{self.nf} {path}: ', end='')

        # Padded resize
        img = letterbox(img0, self.img_size, stride=self.stride)[0]

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        return path, img, img0

    def __len__(self):
        return self.nf  # number of files



# gwang add
# labels --> labelTxt
def img2label_paths(img_paths):
    # Define label paths as a function of image paths

    sa, sb = os.sep + 'images' + os.sep, os.sep + 'labels' + os.sep  # /images/, /labels/ substrings
    return ['txt'.join(x.replace(sa, sb, 1).rsplit(x.split('.')[-1], 1)) for x in img_paths]

class LoadSAR:
    def __init__(self, path, img_size=640, stride=32, save_dir=''):
        
        
        p = str(Path(path).absolute())  # os-agnostic absolute path
        if '*' in p:
            files = sorted(glob.glob(p, recursive=True))  # glob
        elif os.path.isdir(p):
            files = sorted(glob.glob(os.path.join(p, '*.*')))  # dir
        elif os.path.isfile(p):
            files = [p]  # files
        else:
            raise Exception(f'ERROR: {p} does not exist')

        images = [x for x in files if x.split('.')[-1].lower() in img_formats]
        ni = len(images)
        
        self.img_size = img_size
        self.stride = stride
        self.files = images
        self.nf = ni # number of files
        self.mode = 'image'
        self.save_dir = save_dir

        self.point = ogr.Geometry(ogr.wkbPoint) 
        self.inSpatialRef = osr.SpatialReference() 
        self.inSpatialRef = osr.SpatialReference() 
        self.coordTransform = ''
        self.epsg = ''

        assert self.nf > 0, f'No images found in {p}. ' \
                            f'Supported formats are:\nimages: {img_formats}'
    def __iter__(self):
        self.count = 0
        return self
                            
    def __next__(self):
        from utils.cfg import Cfg
        from utils.gdal_preprocess import band_to_input, division_testset
        
        if self.count == self.nf:
            raise StopIteration
        path = self.files[self.count]

        # Read image
        self.count += 1
        
        geotiffile = gdal.Open(path) 
        
        projection = geotiffile.GetProjection() 
        geotransform = geotiffile.GetGeoTransform() 

        proj = osr.SpatialReference(wkt=projection)
        print("original epsg: ", proj.GetAttrValue('AUTHORITY',1))
        self.epsg = proj.GetAttrValue('AUTHORITY',1)

        self.inSpatialRef.ImportFromEPSG(int(self.epsg)) 
        self.outSpatialRef = osr.SpatialReference() 
        self.outSpatialRef.ImportFromEPSG(Cfg.outepsg) 
        self.coordTransform = osr.CoordinateTransformation(self.inSpatialRef, self.outSpatialRef) #transform: 3857 -> 4326
 
        # RGB로 변환
        if Cfg.NewTest == 0:
            input_band = band_to_input(path, Cfg.Satelliteband)
        else:
            input_band = band_to_input(path, Cfg.Satelliteband, True)

        #rgb_band = band_to_rgb(img_path,bandnumber)

        #height, width = rgb_band.shape[:2]
        
        # 모델 추론을 위한 전처리
        # BGR --> RGB
        #input_band = input_band.transpose(2, 0, 1) 
        #rgb_band = cv2.cvtColor(rgb_band, cv2.COLOR_BGR2RGB)

        div_img_list, div_coord = division_testset(input_band=input_band, img_size=self.img_size)
        
        return path, input_band, div_img_list, div_coord, input_band.shape, projection, geotransform
        
    def __len__(self):
        return self.nf  # number of files
    
    def change_projection(self, proj):
        self.outSpatialRef.ImportFromEPSG(proj) 
        coordTransform = osr.CoordinateTransformation(self.inSpatialRef, self.outSpatialRef) #transform: 3857 -> 4326

        return coordTransform

class LoadImagesAndLabels(Dataset):  # for training/testing
    def __init__(self, path, img_size=640, batch_size=16, augment=False, hyp=None, rect=False, image_weights=False,
                 cache_images=False, single_cls=True, stride=32, pad=0.0, prefix='', yaml_file='sentinel.yaml'):
        self.img_size = img_size
        self.augment = augment
        self.hyp = hyp
        self.image_weights = image_weights
        self.rect = False if image_weights else rect
        self.mosaic = self.augment and not self.rect  # load 4 images at a time into a mosaic (only during training)
        self.mosaic_border = [-img_size // 2, -img_size // 2]
        self.stride = stride
        self.path = path   
        self.albumentations = Albumentations() if augment else None
        
        # gwang add
        import yaml
        with open('./data/{}'.format(yaml_file), errors='ignore') as f:
            data = yaml.safe_load(f)
        self.cls_names = data['names']
        try:
            f = []  # image files
            for p in path if isinstance(path, list) else [path]:
                p = Path(p)  # os-agnostic
                if p.is_dir():  # dir
                    f += glob.glob(str(p / '**' / '*.*'), recursive=True)
                    # f = list(p.rglob('**/*.*'))  # pathlib
                elif p.is_file():  # file
                    with open(p, 'r') as t:
                        t = t.read().strip().splitlines()
                        parent = str(p.parent) + os.sep
                        sat_type = str(p.parent).split('/')[2]+os.sep
                        f += [x.replace('./', parent) if x.startswith('./') else x for x in t]  # local to global path
                        # f += [p.parent / x.lstrip(os.sep) for x in t]  # local to global path (pathlib)
                else:
                    raise Exception(f'{prefix}{p} does not exist')
            #f = [l+'.tif' for l in f]
            
            self.img_files = sorted(['./data/images/'+sat_type+path.split('/')[-2]+'/'+path.split('/')[-1][:-4]+'/'+x+'.tif' for x in f])
            #print(self.img_files)
            #sorted([x.replace('/', os.sep) for x in f if x.split('.')[-1].lower() in img_formats])
            
            # self.img_files = sorted([x for x in f if x.suffix[1:].lower() in img_formats])  # pathlib
            assert self.img_files, f'{prefix}No images found'
            
        except Exception as e:
            raise Exception(f'{prefix}Error loading data from {path}: {e}\nSee {help_url}')
        
        
        # Check cache
        self.label_files = img2label_paths(self.img_files)  # labels
        
        cache_path = (p if p.is_file() else Path(self.label_files[0]).parent).with_suffix('.cache')  # cached labels
        
        if cache_path.is_file():
            cache, exists = torch.load(cache_path), True  # load
            #if cache['hash'] != get_hash(self.label_files + self.img_files) or 'version' not in cache:  # changed
            #    cache, exists = self.cache_labels(cache_path, prefix), False  # re-cache
        else:
            
            cache, exists = self.cache_labels(cache_path, prefix), False  # cache
        
        # Display cache
        nf, nm, ne, nc, n = cache.pop('results')  # found, missing, empty, corrupted, total
        if exists:
            d = f"Scanning '{cache_path}' images and labels... {nf} found, {nm} missing, {ne} empty, {nc} corrupted"
            tqdm(None, desc=prefix + d, total=n, initial=n)  # display cache results
        assert nf > 0 or not augment, f'{prefix}No labels in {cache_path}. Can not train without labels. See {help_url}'

        # Read cache
        cache.pop('hash')  # remove hash
        cache.pop('version')  # remove version
        labels, shapes, self.segments = zip(*cache.values())
        self.labels = list(labels)
        self.shapes = np.array(shapes, dtype=np.float64)
        self.img_files = list(cache.keys())  # update
        self.label_files = img2label_paths(cache.keys())  # update
        if single_cls:
            for x in self.labels:
                x[:, 0] = 0
        
        n = len(shapes)  # number of images
        bi = np.floor(np.arange(n) / batch_size).astype(np.int64)  # batch index
        nb = bi[-1] + 1  # number of batches
        self.batch = bi  # batch index of image
        self.n = n
        self.indices = range(n)
              

        # Rectangular Training
        if self.rect:
            # Sort by aspect ratio
            s = self.shapes  # wh
            ar = s[:, 1] / s[:, 0]  # aspect ratio
            irect = ar.argsort()
            self.img_files = [self.img_files[i] for i in irect]
            self.label_files = [self.label_files[i] for i in irect]
            self.labels = [self.labels[i] for i in irect]
            self.shapes = s[irect]  # wh
            ar = ar[irect]

            # Set training image shapes
            shapes = [[1, 1]] * nb
            for i in range(nb):
                ari = ar[bi == i]
                mini, maxi = ari.min(), ari.max()
                if maxi < 1:
                    shapes[i] = [maxi, 1]
                elif mini > 1:
                    shapes[i] = [1, 1 / mini]

            self.batch_shapes = np.ceil(np.array(shapes) * img_size / stride + pad).astype(np.int64) * stride

        # Cache images into memory for faster training (WARNING: large datasets may exceed system RAM)
        self.imgs = [None] * n
        # self.imgs, self.img_npy = [None] * n, [None] * n
        if cache_images:
            if cache_images == 'disk':
                self.im_cache_dir = Path(Path(self.img_files[0]).parent.as_posix() + '_npy')
                self.img_npy = [self.im_cache_dir / Path(f).with_suffix('.npy').name for f in self.img_files]
                self.im_cache_dir.mkdir(parents=True, exist_ok=True)
            gb = 0  # Gigabytes of cached images
            self.img_hw0, self.img_hw = [None] * n, [None] * n
            results = ThreadPool(8).imap(lambda x: load_image(*x), zip(repeat(self), range(n)))
            pbar = tqdm(enumerate(results), total=n)
            for i, x in pbar:
                if cache_images == 'disk':
                    if not self.img_npy[i].exists():
                        np.save(self.img_npy[i].as_posix(), x[0])
                    gb += self.img_npy[i].stat().st_size
                else:
                    self.imgs[i], self.img_hw0[i], self.img_hw[i] = x
                    gb += self.imgs[i].nbytes
                pbar.desc = f'{prefix}Caching images ({gb / 1E9:.1f}GB)'
            pbar.close()
            
    def cache_labels(self, path=Path('./labels.cache'), prefix=''):
        
        # Cache dataset labels, check images and read shapes
        x = {}  # dict
        nm, nf, ne, nc = 0, 0, 0, 0  # number missing, found, empty, duplicate
        pbar = tqdm(zip(self.img_files, self.label_files), desc='Scanning images', total=len(self.img_files))

        for i, (im_file, lb_file) in enumerate(pbar):
            try:
                # verify images
                im = cv2.imread(im_file, cv2.IMREAD_UNCHANGED) # read as is
                #im.verify()  # PIL verify
                if im is None:
                    raise Exception
                #shape = exif_size(im)  # image size
                shape = im.shape[:2]
                segments = []  # instance segments
                assert (shape[0] > 9) & (shape[1] > 9), f'image size {shape} <10 pixels'
                assert im_file.split('/')[-1].split('.')[-1] in img_formats, f'invalid image format {im.format}'

                # verify labels
                if os.path.isfile(lb_file):
                    nf += 1  # label found
                    with open(lb_file, 'r') as f:
                        l = [x.split() for x in f.read().strip().splitlines()]
                        # labels = [x.split() for x in f.read().strip().splitlines()]
                        
                        if any([len(x) > 8 for x in l]):  # is segment
                        # shh add
                             segments = [np.array(x[1:], dtype=np.float32).reshape(-1, 2) for x in l]  # (cls, xy1...)
                             # gwang add
                             classes = np.array([self.cls_names.index(x[0]) for x in l], dtype=np.float32)
                        #     segments = [np.array(x[:8], dtype=np.float32).reshape(-1, 2) for x in l]  # (cls, xy1...)
                             l = np.concatenate((classes.reshape(-1, 1), segments2boxes(segments)), 1)  # (cls, xywh)
                        else:
                            # shh add
                            l_ = []
                            for label in l:
                                cls_id = self.cls_names.index(label[0])
                                #xywh = xyxyxyxy2xywh(label[1:])
                                l_.append(np.concatenate((cls_id, label[1:]), axis=None))
                            l = np.array(l_, dtype=np.float32)
                        l = np.array(l, dtype=np.float32)
                    
                    if len(l):
                    #nl = len(l)
                    #if nl:
                        assert l.shape[1] == 5, 'labels require 5 columns each'
                        assert (l >= 0).all(), 'negative labels'
                        #assert (l[:, 1:] <= 1).all(), 'non-normalized or out of bounds coordinate labels'

                        # shhan add
                        if (l < 0).any(): # unnormalized labels
                            l = np.array([label for label in l if (label[1:] >= 0).all()])
                            if l.shape[0]==0:
                                l = np.zeros((0,5), dtype=np.float32)
                        # shhan add
                        if (l[:, 1:] > 1).any(): # unbounded labels
                            l = np.array([label for label in l if (label[1:] <= 1).all()])
                            if l.shape[0]==0:
                                l = np.zeros((0,5), dtype=np.float32)
                                
                        if np.unique(l, axis=0).shape[0] == l.shape[0]: 
                            l = np.unique(l, axis=0) # duplicate labels

                        # filter out 3 pixels
                        original_wh = [max(np.array([lb[1::2].max()-lb[1::2].min(),
                                         lb[2::2].max()-lb[2::2].min()])) for lb in l]
                        wh0 = np.multiply(original_wh, shape[0])  # wh of minimum outter bounding box
                        l = np.array([l[i] for i in range(len(l)) if wh0[i] >= 1.0])
                        if l.shape[0]==0:
                            l = np.zeros((0,9), dtype=np.float32)
                    else:
                        ne += 1  # label empty
                        l = np.zeros((0, 5), dtype=np.float32)
                        #l = np.zeros((0, 9), dtype=np.float32)

                else:
                    nm += 1  # label missing
                    l = np.zeros((0, 5), dtype=np.float32)
                    #l = np.zeros((0, 9), dtype=np.float32)

                x[im_file] = [l, shape, segments]
                
            except Exception as e:
                nc += 1
                print(f'{prefix}WARNING: Ignoring corrupted image and/or label {im_file}: {e}')

            pbar.desc = f"{prefix}Scanning '{path.parent / path.stem}' images and labels... " \
                        f"{nf} found, {nm} missing, {ne} empty, {nc} corrupted"
        pbar.close()

        if nf == 0:
            print(f'{prefix}WARNING: No labels found in {path}. See {help_url}')

        x['hash'] = get_hash(self.label_files + self.img_files)
        x['results'] = nf, nm, ne, nc, i + 1
        x['version'] = 0.1  # cache version
        torch.save(x, path)  # save for next time
        logging.info(f'{prefix}New cache created: {path}')

        return x

    def __len__(self):
        return len(self.img_files)

    # def __iter__(self):
    #     self.count = -1
    #     print('ran dataset iter')
    #     #self.shuffled_vector = np.random.permutation(self.nF) if self.augment else np.arange(self.nF)
    #     return self

    def __getitem__(self, index):
        index = self.indices[index]  # linear, shuffled, or image_weights

        hyp = self.hyp
        mosaic = self.mosaic and random.random() < hyp['mosaic']
        if mosaic:
            # Load mosaic
            if random.random() < 0.8:
                img, labels = load_mosaic(self, index)
            else:
                img, labels = load_mosaic9(self, index)
            shapes = None
            # MixUp https://arxiv.org/pdf/1710.09412.pdf
            if random.random() < hyp['mixup']:
                if random.random() < 0.8:
                    img2, labels2 = load_mosaic(self, random.randint(0, len(self.labels) - 1))
                else:
                    img2, labels2 = load_mosaic9(self, random.randint(0, len(self.labels) - 1))
                r = np.random.beta(8.0, 8.0)  # mixup ratio, alpha=beta=8.0
                img = (img * r + img2 * (1 - r)).astype(img.dtype) # original: np.uint8
                labels = np.concatenate((labels, labels2), 0)
        else:
            # Load image
            # gwang add
            img, (h0, w0), (h, w) = load_image(self, index)
            #img, (h0, w0), (h, w), img_label = load_image_label(self, index)

            # Letterbox
            shape = self.batch_shapes[self.batch[index]] if self.rect else self.img_size  # final letterboxed shape
            img, ratio, pad = letterbox(img, shape, auto=False, scaleup=self.augment)
            shapes = (h0, w0), ((h / h0, w / w0), pad)  # for COCO mAP rescaling

            labels = self.labels[index].copy()
            if labels.size:  # normalized xywh to pixel xyxy format
                # gwang add
                # labels[:, 1:] = xywhn2xyxy(labels[:, 1:], ratio[0] * w, ratio[1] * h, padw=pad[0], padh=pad[1])
                labels[:, 1:] = xywhn2xyxy(labels[:, 1:], ratio[0] * w, ratio[1] * h, padw=pad[0], padh=pad[1])
                    #labels[:, [1, 3, 5, 7]] = img_label[:, [1, 3, 5, 7]] * ratio[0] + pad[0] # x1 x2 x3 x4
                    #labels[:, [2, 4, 6, 8]] = img_label[:, [2, 4, 6, 8]] * ratio[1] + pad[1] # y1 y2 y3 y4

        if self.augment:
            # Augment imagespace
            if not mosaic:
                # gwang, fix the random_perspective
                img, labels = random_perspective(img, labels,
                                                 degrees=hyp['degrees'],
                                                 translate=hyp['translate'],
                                                 scale=hyp['scale'],
                                                 shear=hyp['shear'],
                                                 perspective=hyp['perspective'])
            
            #img, labels = self.albumentations(img, labels)

            # Augment colorspace
            #img = augment_hsv(img, hgain=hyp['hsv_h'], sgain=hyp['hsv_s'], vgain=hyp['hsv_v'])
            # Apply cutouts
            # if random.random() < 0.9:
            #     labels = cutout(img, labels)
            
            #if random.random() < hyp['paste_in']:
            #    sample_labels, sample_images, sample_masks = [], [], [] 
            #    while len(sample_labels) < 30:
            #        sample_labels_, sample_images_, sample_masks_ = load_samples(self, random.randint(0, len(self.labels) - 1))
            #        sample_labels += sample_labels_
            #        sample_images += sample_images_
            #        sample_masks += sample_masks_
            #        #print(len(sample_labels))
            #        if len(sample_labels) == 0:
            #            break
            #    labels = pastein(img, labels, sample_labels, sample_images, sample_masks)
        
        nl = len(labels)  # number of labels
        if nl:
            labels[:, 1:5] = xyxy2xywh(labels[:, 1:5])  # convert xyxy to xywh
            labels[:, [2, 4]] /= img.shape[0]  # normalized height 0-1
            labels[:, [1, 3]] /= img.shape[1]  # normalized width 0-1

        if self.augment:
            # flip up-down
            if random.random() < hyp['flipud']:
                img = np.flipud(img)
                if nl:
                    labels[:, 2] = 1 - labels[:, 2]
            # flip left-right
            if random.random() < hyp['fliplr']:
                img = np.fliplr(img)
                if nl:
                    labels[:, 1] = 1 - labels[:, 1]
        labels_out = torch.zeros((nl, 6))

        if nl: 
            labels_out[:, 1:] = torch.from_numpy(labels)
            #labels_out[:, 1:] = torch.from_numpy(labels_obb)
        #########
        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416 
        img = np.ascontiguousarray(img)
        return torch.from_numpy(img), labels_out, self.img_files[index], shapes

    @staticmethod
    def collate_fn(batch):
        img, label, path, shapes = zip(*batch)  # transposed
        for i, l in enumerate(label):
            l[:, 0] = i  # add target image index for build_targets()
        return torch.stack(img, 0), torch.cat(label, 0), path, shapes

    @staticmethod
    def collate_fn4(batch):
        img, label, path, shapes = zip(*batch)  # transposed
        n = len(shapes) // 4
        img4, label4, path4, shapes4 = [], [], path[:n], shapes[:n]

        ho = torch.tensor([[0., 0, 0, 1, 0, 0]])
        wo = torch.tensor([[0., 0, 1, 0, 0, 0]])
        s = torch.tensor([[1, 1, .5, .5, .5, .5]])  # scale
        for i in range(n):  # zidane torch.zeros(16,3,720,1280)  # BCHW
            i *= 4
            if random.random() < 0.5:
                im = F.interpolate(img[i].unsqueeze(0).float(), scale_factor=2., mode='bilinear', align_corners=False)[
                    0].type(img[i].type())
                l = label[i]
            else:
                im = torch.cat((torch.cat((img[i], img[i + 1]), 1), torch.cat((img[i + 2], img[i + 3]), 1)), 2)
                l = torch.cat((label[i], label[i + 1] + ho, label[i + 2] + wo, label[i + 3] + ho + wo), 0) * s
            img4.append(im)
            label4.append(l)

        for i, l in enumerate(label4):
            l[:, 0] = i  # add target image index for build_targets()

        return torch.stack(img4, 0), torch.cat(label4, 0), path4, shapes4

    def load_image(self, index):
        # loads 1 image from dataset, returns img, original hw, resized hw
        img = self.imgs[index]
        if img is None:  # not cached
            path = self.img_files[index]
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)  # BGR
            assert img is not None, 'Image Not Found ' + path
            h0, w0 = img.shape[:2]  # orig hw
            r = self.img_size / max(h0, w0)  # ratio
            if r != 1:  # if sizes are not equal
                img = cv2.resize(img, (int(w0 * r), int(h0 * r)),
                                 interpolation=cv2.INTER_AREA if r < 1 and not self.augment else cv2.INTER_LINEAR)
            return img, (h0, w0), img.shape[:2]  # img, hw_original, hw_resized
        else:
            return self.imgs[index], self.img_hw0[index], self.img_hw[index]  # img, hw_original, hw_resized

# gwang add
def load_image_label(self, i):
    # loads 1 image from dataset index 'i', returns im, original hw, resized hw
    im = self.imgs[i]
    label = self.labels[i].copy() # labels (array): (num_gt_perimg, [cls_id, poly])
    # print('i', i)
    # print('self.img_npy', self.img_npy)
    if im is None:  # not cached in ram
        npy = self.img_npy[i]
        # npy = None
        if npy and npy.exists():  # load npy
            im = np.load(npy)
        else:  # read image
            path = self.img_files[i]
            im = cv2.imread(path, cv2.IMREAD_UNCHANGED)  # BGR
            assert im is not None, f'Image Not Found {path}'
        h0, w0 = im.shape[:2]  # orig hw
        r = self.img_size / max(h0, w0)  # ratio
        if r != 1:  # if sizes are not equal
            im = cv2.resize(im, (int(w0 * r), int(h0 * r)),
                            interpolation=cv2.INTER_AREA if r < 1 and not self.augment else cv2.INTER_LINEAR)
            label[:, 1:] *= r
        return im, (h0, w0), im.shape[:2], label  # im, hw_original, hw_resized, resized_label
    else:
        return self.imgs[i], self.img_hw0[i], self.img_hw[i], self.labels[i]  # im, hw_original, hw_resized, resized_label    
    
def load_mosaic(self, index):
    # loads images in a 4-mosaic

    labels4, segments4 = [], []
    s = self.img_size
    yc, xc = [int(random.uniform(-x, 2 * s + x)) for x in self.mosaic_border]  # mosaic center x, y
    indices = [index] + random.choices(self.indices, k=3)  # 3 additional image indices
    for i, index in enumerate(indices):
        # Load image
        img, _, (h, w) = load_image(self, index)

        # place img in img4
        if i == 0:  # top left
            img4 = np.full((s * 2, s * 2, img.shape[2]), 0.45, dtype=img.dtype)  # base image with 4 tiles; original: np.uint8
            x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
        elif i == 1:  # top right
            x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
            x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
        elif i == 2:  # bottom left
            x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
        elif i == 3:  # bottom right
            x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
            x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

        img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
        padw = x1a - x1b
        padh = y1a - y1b

        # Labels
        labels, segments = self.labels[index].copy(), self.segments[index].copy()
        if labels.size:
            labels[:, 1:] = xywhn2xyxy(labels[:, 1:], w, h, padw, padh)  # normalized xywh to pixel xyxy format
            segments = [xyn2xy(x, w, h, padw, padh) for x in segments]
        labels4.append(labels)
        segments4.extend(segments)

    # Concat/clip labels
    labels4 = np.concatenate(labels4, 0)
    for x in (labels4[:, 1:], *segments4):
        np.clip(x, 0, 2 * s, out=x)  # clip when using random_perspective()
    # img4, labels4 = replicate(img4, labels4)  # replicate

    # Augment
    img4, labels4 = random_perspective(img4, labels4, segments4,
                                       degrees=self.hyp['degrees'],
                                       translate=self.hyp['translate'],
                                       scale=self.hyp['scale'],
                                       shear=self.hyp['shear'],
                                       perspective=self.hyp['perspective'],
                                       border=self.mosaic_border)  # border to remove

    return img4, labels4


def load_mosaic9(self, index):
    # loads images in a 9-mosaic

    labels9, segments9 = [], []
    s = self.img_size
    indices = [index] + random.choices(self.indices, k=8)  # 8 additional image indices
    for i, index in enumerate(indices):
        # Load image
        img, _, (h, w) = load_image(self, index)

        # place img in img9
        if i == 0:  # center
            img9 = np.full((s * 3, s * 3, img.shape[2]), 0.45, dtype=img.dtype)  # base image with 4 tiles; original: np.uint8
            h0, w0 = h, w
            c = s, s, s + w, s + h  # xmin, ymin, xmax, ymax (base) coordinates
        elif i == 1:  # top
            c = s, s - h, s + w, s
        elif i == 2:  # top right
            c = s + wp, s - h, s + wp + w, s
        elif i == 3:  # right
            c = s + w0, s, s + w0 + w, s + h
        elif i == 4:  # bottom right
            c = s + w0, s + hp, s + w0 + w, s + hp + h
        elif i == 5:  # bottom
            c = s + w0 - w, s + h0, s + w0, s + h0 + h
        elif i == 6:  # bottom left
            c = s + w0 - wp - w, s + h0, s + w0 - wp, s + h0 + h
        elif i == 7:  # left
            c = s - w, s + h0 - h, s, s + h0
        elif i == 8:  # top left
            c = s - w, s + h0 - hp - h, s, s + h0 - hp

        padx, pady = c[:2]
        x1, y1, x2, y2 = [max(x, 0) for x in c]  # allocate coords

        # Labels
        labels, segments = self.labels[index].copy(), self.segments[index].copy()
        if labels.size:
            labels[:, 1:] = xywhn2xyxy(labels[:, 1:], w, h, padx, pady)  # normalized xywh to pixel xyxy format
            segments = [xyn2xy(x, w, h, padx, pady) for x in segments]
        labels9.append(labels)
        segments9.extend(segments)

        # Image
        img9[y1:y2, x1:x2] = img[y1 - pady:, x1 - padx:]  # img9[ymin:ymax, xmin:xmax]
        hp, wp = h, w  # height, width previous

    # Offset
    yc, xc = [int(random.uniform(0, s)) for _ in self.mosaic_border]  # mosaic center x, y
    img9 = img9[yc:yc + 2 * s, xc:xc + 2 * s]

    # Concat/clip labels
    labels9 = np.concatenate(labels9, 0)
    labels9[:, [1, 3]] -= xc
    labels9[:, [2, 4]] -= yc
    c = np.array([xc, yc])  # centers
    segments9 = [x - c for x in segments9]

    for x in (labels9[:, 1:], *segments9):
        np.clip(x, 0, 2 * s, out=x)  # clip when using random_perspective()
    # img9, labels9 = replicate(img9, labels9)  # replicate

    # Augment
    img9, labels9 = random_perspective(img9, labels9, segments9,
                                       degrees=self.hyp['degrees'],
                                       translate=self.hyp['translate'],
                                       scale=self.hyp['scale'],
                                       shear=self.hyp['shear'],
                                       perspective=self.hyp['perspective'],
                                       border=self.mosaic_border)  # border to remove

    return img9, labels9

    

## Polygon bbox-related ancilary functions ------------------------------------------------------
class Polygon_LoadImagesAndLabels(Dataset):  # for training/testing
    """
        Polygon_LoadImagesAndLabels for polygon boxes
    """
    def __init__(self, path, img_size=640, batch_size=16, augment=False, hyp=None, rect=False, image_weights=False,
                 cache_images=False, single_cls=False, stride=32, pad=0.0, prefix='', yaml_file = 'sentinel.yaml'):
        self.img_size = img_size
        self.augment = augment
        self.hyp = hyp
        self.image_weights = image_weights
        self.rect = False if image_weights else rect
        self.mosaic = self.augment and not self.rect  # load 4 images at a time into a mosaic (only during training)
        self.mosaic_border = [-img_size // 2, -img_size // 2]
        self.stride = stride
        self.path = path
        
        import yaml
        with open('./data/{}'.format(yaml_file), errors='ignore') as f:
            data = yaml.safe_load(f)
        
        self.cls_names = data['names']
        # albumentation
        self.albumentations = Albumentations() if augment else None

        try:
            f = []  # image files
            for p in path if isinstance(path, list) else [path]:
                p = Path(p)  # os-agnostic
                if p.is_dir():  # dir
                    f += glob.glob(str(p / '**' / '*.*'), recursive=True)
                    # f = list(p.rglob('**/*.*'))  # pathlib
                elif p.is_file():  # file
                    with open(p, 'r') as t:
                        t = t.read().strip().splitlines()
                        parent = str(p.parent) + os.sep
                        sat_type = str(p.parent).split('/')[2]+os.sep
                        f += [x.replace('./', parent) if x.startswith('./') else x for x in t]  # local to global path
                        # f += [p.parent / x.lstrip(os.sep) for x in t]  # local to global path (pathlib)
                else:
                    raise Exception(f'{prefix}{p} does not exist')
            self.img_files = sorted(['./data/images/'+sat_type+path.split('/')[-2]+'/'+path.split('/')[-1][:-4]+'/'+x+'.tif' for x in f])
            #self.img_files = sorted(['./'+parent+'images/'+path.split('/')[-1][:-4]+'/'+x+'.tif' for x in f])
            
            #self.img_files = sorted([x.replace('/', os.sep) for x in f if x.split('.')[-1].lower() in img_formats])
            # self.img_files = sorted([x for x in f if x.suffix[1:].lower() in img_formats])  # pathlib
            assert self.img_files, f'{prefix}No images found'
        except Exception as e:
            raise Exception(f'{prefix}Error loading data from {path}: {e}\nSee {help_url}')
        
        # Check cache
        self.label_files = img2label_paths(self.img_files)  # labels
        cache_path = (p if p.is_file() else Path(self.label_files[0]).parent).with_suffix('.cache')  # cached labels
        if cache_path.is_file():
            cache, exists = torch.load(cache_path), True  # load
            if cache.get('hash') != get_hash(self.label_files + self.img_files):
                cache, exists = self.cache_labels(cache_path, prefix), False  # re-cache
        else:
            print("Not exist")
            cache, exists = self.cache_labels(cache_path, prefix), False  # cache

        # Display cache
        nf, nm, ne, nc, n = cache.pop('results')  # found, missing, empty, corrupted, total
        
        if exists:
            d = f"Scanning '{cache_path}' images and labels... {nf} found, {nm} missing, {ne} empty, {nc} corrupted"
            tqdm(None, desc=prefix + d, total=n, initial=n)  # display cache results
        assert nf > 0 or not augment, f'{prefix}No labels in {cache_path}. Can not train without labels. See {help_url}'

        # Read cache
        cache.pop('hash')  # remove hash
        cache.pop('version')  # remove version
        labels, shapes, self.segments = zip(*cache.values())
        self.labels = list(labels)
        self.shapes = np.array(shapes, dtype=np.float64)
        self.img_files = list(cache.keys())  # update
        self.label_files = img2label_paths(cache.keys())  # update
        if single_cls:
            for x in self.labels:
                x[:, 0] = 0

        n = len(shapes)  # number of images
        bi = np.floor(np.arange(n) / batch_size).astype(np.int64)  # batch index
        nb = bi[-1] + 1  # number of batches
        self.batch = bi  # batch index of image
        self.n = n
        self.indices = range(n)

        # Rectangular Training
        if self.rect:
            # Sort by aspect ratio
            s = self.shapes  # image wh
            ar = s[:, 1] / s[:, 0]  # aspect ratio
            irect = ar.argsort()
            self.img_files = [self.img_files[i] for i in irect]
            self.label_files = [self.label_files[i] for i in irect]
            self.labels = [self.labels[i] for i in irect]
            self.shapes = s[irect]  # wh
            ar = ar[irect]

            # Set training image shapes
            shapes = [[1, 1]] * nb
            for i in range(nb):
                ari = ar[bi == i]
                mini, maxi = ari.min(), ari.max()
                if maxi < 1:
                    shapes[i] = [maxi, 1]
                elif mini > 1:
                    shapes[i] = [1, 1 / mini]

            self.batch_shapes = np.ceil(np.array(shapes) * img_size / stride + pad).astype(np.int64) * stride

        # Cache images into memory for faster training (WARNING: large datasets may exceed system RAM)
        self.imgs = [None] * n
        if cache_images:
            gb = 0  # Gigabytes of cached images
            self.img_hw0, self.img_hw = [None] * n, [None] * n
            results = ThreadPool(num_threads).imap(lambda x: load_image(*x), zip(repeat(self), range(n)))
            pbar = tqdm(enumerate(results), total=n)
            for i, x in pbar:
                self.imgs[i], self.img_hw0[i], self.img_hw[i] = x  # img, hw_original, hw_resized = load_image(self, i)
                gb += self.imgs[i].nbytes
                pbar.desc = f'{prefix}Caching images ({gb / 1E9:.1f}GB)'
            pbar.close()

    def cache_labels(self, path=Path('./labels.cache'), prefix=''):
        # Cache dataset labels, check images and read shapes
        x = {}  # dict
        nm, nf, ne, nc = 0, 0, 0, 0  # number missing, found, empty, corrupt
        pbar = tqdm(zip(self.img_files, self.label_files), desc='Scanning images', total=len(self.img_files))
        for i, (im_file, lb_file) in enumerate(pbar):
            try:
                # verify images
                #im = Image.open(im_file)
                #im.verify()  # PIL verify
                im = cv2.imread(im_file, cv2.IMREAD_UNCHANGED)
                if im is None:
                    raise Exception
                #shape = exif_size(im)  # image size
                shape = im.shape[:2]
                segments = []  # instance segments
                assert (shape[0] > 9) & (shape[1] > 9), f'image size {shape} <10 pixels'
                assert im_file.split('/')[-1].split('.')[-1] in img_formats, f'invalid image format {im.format}'
            
                # verify labels
                if os.path.isfile(lb_file):
                    
                    nf += 1  # label found
                    with open(lb_file, 'r') as f:
                        # l = [x.split() for x in f.read().strip().splitlines()]
                        labels = [x.split() for x in f.read().strip().splitlines()]
                        
                        # shh add
                        l_ = []
                        for label in labels:
                            cls_id = self.cls_names.index(label[0])
                            l_.append(np.concatenate((cls_id, label[1:]), axis=None))
                        l = np.array(l_, dtype=np.float32)
                    
                    nl = len(l)
                    if nl:
                        # assert l.shape[1] == 5, 'labels require 5 columns each'
                        assert len(label) == 9, f'Yolov7-OBB labels require 9 columns, {len(label)} columns detected'
                        
                        # shhan add
                        if (l < 0).any(): # unnormalized labels
                            l = np.array([label for label in l if (label[1:] >= 0).all()])
                            if l.shape[0]==0:
                                l = np.zeros((0,9), dtype=np.float32)
                        # shhan add
                        if (l[:, 1:] > 1).any(): # unbounded labels
                            l = np.array([label for label in l if (label[1:] <= 1).all()])
                            if l.shape[0]==0:
                                l = np.zeros((0,9), dtype=np.float32)
                                
                        if np.unique(l, axis=0).shape[0] == l.shape[0]: 
                            l = np.unique(l, axis=0) # duplicate labels
                        
                        # filter out 3 pixels
                        original_wh = [max(np.array([lb[1::2].max()-lb[1::2].min(),
                                         lb[2::2].max()-lb[2::2].min()])) for lb in l]
                        wh0 = np.multiply(original_wh, shape[0])  # wh of minimum outter bounding box
                        l = np.array([l[i] for i in range(len(l)) if wh0[i] >= 3.0])
                        if l.shape[0]==0:
                            l = np.zeros((0,9), dtype=np.float32)
                    else:
                        ne += 1  # label empty
                        # l = np.zeros((0, 5), dtype=np.float32)
                        l = np.zeros((0, 9), dtype=np.float32)

                else:
                    nm += 1  # label missing
                    # l = np.zeros((0, 5), dtype=np.float32)
                    l = np.zeros((0, 9), dtype=np.float32)

                x[im_file] = [l, shape, segments]
            except Exception as e:
                nc += 1
                print(f'{prefix}WARNING: Ignoring corrupted image and/or label {im_file}: {e}')
            
            pbar.desc = f"{prefix}Scanning '{path.parent / path.stem}' images and labels... " \
                    f"{nf} found, {nm} missing, {ne} empty, {nc} corrupted"
        pbar.close()
       
        if nf == 0:
            print(f'{prefix}WARNING: No labels found in {path}. See {help_url}')

        x['hash'] = get_hash(self.label_files + self.img_files)
        x['results'] = nf, nm, ne, nc, i + 1
        x['version'] = 0.1  # cache version
        torch.save(x, path)  # save for next time
        logging.info(f'{prefix}New cache created: {path}')
       
        return x

    def __len__(self):
        return len(self.img_files)

    # def __iter__(self):
    #     self.count = -1
    #     print('ran dataset iter')
    #     #self.shuffled_vector = np.random.permutation(self.nF) if self.augment else np.arange(self.nF)
    #     return self
    
    def __getitem__(self, index):
        index = self.indices[index]  # linear, shuffled, or image_weights

        hyp = self.hyp
        mosaic = self.mosaic and random.random() < hyp['mosaic']
        if mosaic:
            # Load mosaic
            if random.random() < 0.8:
                img, labels = polygon_load_mosaic(self, index)
            else:
                img, labels = polygon_load_mosaic9(self, index)
            shapes = None
            # MixUp https://arxiv.org/pdf/1710.09412.pdf
            if random.random() < hyp['mixup']:
                if random.random() < 0.8:
                    img2, labels2 = polygon_load_mosaic(self, random.randint(0, len(self.labels) - 1))
                else:
                    img2, labels2 = polygon_load_mosaic9(self, random.randint(0, len(self.labels) - 1))

                r = np.random.beta(32.0, 32.0)  # mixup ratio, alpha=beta=32.0
                img = (img * r + img2 * (1 - r)).astype(img.dtype) # original: np.uint8
                labels = np.concatenate((labels, labels2), 0)
        else:
            # Load image
            img, (h0, w0), (h, w) = load_image(self, index)
            # Letterbox
            shape = self.batch_shapes[self.batch[index]] if self.rect else self.img_size  # final letterboxed shape
            img, ratio, pad = letterbox(img, shape, auto=False, scaleup=self.augment)
            shapes = (h0, w0), ((h / h0, w / w0), pad)  # for COCO mAP rescaling
            labels = self.labels[index].copy() 
            if labels.size:  # normalized format to pixel xyxyxyxy format
                labels[:, 1:] = xyxyxyxyn2xyxyxyxy(labels[:, 1:], ratio[0] * w, ratio[1] * h, padw=pad[0], padh=pad[1])

        if self.augment:
            # Augment imagespace
            if not mosaic:
                img, labels = polygon_random_perspective(img, labels,
                                                         degrees=hyp['degrees'],
                                                         translate=hyp['translate'],
                                                         scale=hyp['scale'],
                                                         shear=hyp['shear'],
                                                         perspective=hyp['perspective'])
            # Augment colorspace
            #img = augment_hsv(img, hgain=hyp['hsv_h'], sgain=hyp['hsv_s'], vgain=hyp['hsv_v'])
            # Polygon does not support cutouts
        nL = len(labels)  # number of labels
        if nL:
            labels[:, 2::2] /= img.shape[0]  # normalized height 0-1
            labels[:, 1::2] /= img.shape[1]  # normalized width 0-1

        if self.augment:
            # albumentation
            img = self.albumentations(img)
            # flip up-down for all y
            if random.random() < hyp['flipud']:
                img = np.flipud(img)
                if nL:
                    labels[:, 2::2] = 1 - labels[:, 2::2]
            # flip left-right for all x
            if random.random() < hyp['fliplr']:
                img = np.fliplr(img)
                if nL:
                    labels[:, 1::2] = 1 - labels[:, 1::2]
        # original label shape is (nL, 9), add one column for target image index for build_targets()
        labels_out = torch.zeros((nL, 10))
        if nL:
            labels_out[:, 1:] = torch.from_numpy(labels)

        # Convert
        img = img.transpose((2, 0, 1))[::-1]  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)
        return torch.from_numpy(img), labels_out, self.img_files[index], shapes

    # Polygon does not support collate_fn4
    @staticmethod
    def collate_fn(batch):
        img, label, path, shapes = zip(*batch)  # transposed
        for i, l in enumerate(label):
            l[:, 0] = i  # add target image index for build_targets()
        return torch.stack(img, 0), torch.cat(label, 0), path, shapes
    
def polygon_load_mosaic(self, index):
    # loads images in a 4-mosaic, with polygon boxes
    labels4, segments4 = [], []
    s = self.img_size
    yc, xc = [int(random.uniform(-x, 2 * s + x)) for x in self.mosaic_border]  # mosaic center x, y
    indices = [index] + random.choices(self.indices, k=3)  # 3 additional image indices
    for i, index in enumerate(indices):
        # Load image
        img, _, (h, w) = load_image(self, index)

        # place img in img4
        if i == 0:  # top left
            img4 = np.full((s * 2, s * 2, img.shape[2]), 0.45, dtype=img.dtype)  # base image with 4 tiles; original: np.uint8
            x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
        elif i == 1:  # top right
            x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
            x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
        elif i == 2:  # bottom left
            x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
        elif i == 3:  # bottom right
            x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
            x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

        img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
        padw = x1a - x1b
        padh = y1a - y1b

        # Labels
        labels, segments = self.labels[index].copy(), self.segments[index].copy()
        if labels.size:
            # from normalized, unpadded xyxyxyxy to pixel, padded xyxyxyxy format
            labels[:, 1:] = xyxyxyxyn2xyxyxyxy(labels[:, 1:], w, h, padw, padh)
            segments = [xyxyxyxyn2xyxyxyxy(x, w, h, padw, padh) for x in segments]
        labels4.append(labels)
        segments4.extend(segments)

    # Concat/clip labels
    labels4 = np.concatenate(labels4, 0)
    for x in (labels4[:, 1:], *segments4):
        np.clip(x, 0, 2 * s, out=x)  # inplace clip when using polygon_random_perspective()

    # Augment
    img4, labels4 = polygon_random_perspective(img4, labels4, segments4,
                                               degrees=self.hyp['degrees'],
                                               translate=self.hyp['translate'],
                                               scale=self.hyp['scale'],
                                               shear=self.hyp['shear'],
                                               perspective=self.hyp['perspective'],
                                               border=self.mosaic_border,
                                               mosaic=True)  # border to remove

    return img4, labels4

def polygon_load_mosaic9(self, index):
    # loads images in a 9-mosaic, with polygon boxes

    labels9, segments9 = [], []
    s = self.img_size
    indices = [index] + random.choices(self.indices, k=8)  # 8 additional image indices
    for i, index in enumerate(indices):
        # Load image
        img, _, (h, w) = load_image(self, index)

        # place img in img9
        if i == 0:  # center
            img9 = np.full((s * 3, s * 3, img.shape[2]), 0.45, dtype=img.dtype)  # base image with 4 tiles; original: np.uint8
            h0, w0 = h, w
            c = s, s, s + w, s + h  # xmin, ymin, xmax, ymax (base) coordinates
        elif i == 1:  # top
            c = s, s - h, s + w, s
        elif i == 2:  # top right
            c = s + wp, s - h, s + wp + w, s
        elif i == 3:  # right
            c = s + w0, s, s + w0 + w, s + h
        elif i == 4:  # bottom right
            c = s + w0, s + hp, s + w0 + w, s + hp + h
        elif i == 5:  # bottom
            c = s + w0 - w, s + h0, s + w0, s + h0 + h
        elif i == 6:  # bottom left
            c = s + w0 - wp - w, s + h0, s + w0 - wp, s + h0 + h
        elif i == 7:  # left
            c = s - w, s + h0 - h, s, s + h0
        elif i == 8:  # top left
            c = s - w, s + h0 - hp - h, s, s + h0 - hp

        padx, pady = c[:2]
        x1, y1, x2, y2 = [max(x, 0) for x in c]  # allocate coords

        # Labels
        labels, segments = self.labels[index].copy(), self.segments[index].copy()
        if labels.size:
            # from normalized, unpadded xyxyxyxy to pixel, padded xyxyxyxy format
            labels[:, 1:] = xyxyxyxyn2xyxyxyxy(labels[:, 1:], w, h, padx, pady)  
            segments = [xyxyxyxyn2xyxyxyxy(x, w, h, padx, pady) for x in segments]
        labels9.append(labels)
        segments9.extend(segments)

        # Image
        img9[y1:y2, x1:x2] = img[y1 - pady:, x1 - padx:]  # img9[ymin:ymax, xmin:xmax]
        hp, wp = h, w  # height, width previous

    # Offset
    yc, xc = [int(random.uniform(0, s)) for _ in self.mosaic_border]  # mosaic center x, y
    img9 = img9[yc:yc + 2 * s, xc:xc + 2 * s]

    # Concat/clip labels
    labels9 = np.concatenate(labels9, 0)
    labels9[:, 1::2] -= xc
    labels9[:, 2::2] -= yc
    c = np.array([xc, yc])  # centers
    segments9 = [x - c for x in segments9]

    for x in (labels9[:, 1:], *segments9):
        np.clip(x, 0, 2 * s, out=x)  # inplace clip when using polygon_random_perspective()
    # Augment
    img9, labels9 = polygon_random_perspective(img9, labels9, segments9,
                                               degrees=self.hyp['degrees'],
                                               translate=self.hyp['translate'],
                                               scale=self.hyp['scale'],
                                               shear=self.hyp['shear'],
                                               perspective=self.hyp['perspective'],
                                               border=self.mosaic_border,
                                               mosaic=True)  # border to remove

    return img9, labels9   

def polygon_verify_image_label(params):
    # Verify one image-label pair
    im_file, lb_file, prefix = params
    nm, nf, ne, nc = 0, 0, 0, 0  # number missing, found, empty, corrupt
    try:
        # verify images
        im = Image.open(im_file)
        im.verify()  # PIL verify
        shape = exif_size(im)  # image size
        segments = []  # instance segments
        assert (shape[0] > 9) & (shape[1] > 9), f'image size {shape} <10 pixels'
        assert im.format.lower() in img_formats, f'invalid image format {im.format}'

        # verify labels
        if os.path.isfile(lb_file):
            nf = 1  # label found
            l = np.loadtxt(lb_file, dtype=np.float32)
            if len(l.shape)==1: l = l[None, :]
            segments = [x[1:].reshape(-1, 2) for x in l]  # ((x1, y1), (x2, y2), ...)
            if len(l):
                assert l.shape[1] == 9, 'labels require 9 columns each'
                # Common out following lines to enable: polygon corners can be out of images
                # assert (l >= 0).all(), 'negative labels'
                # assert (l[:, 1:] <= 1).all(), 'non-normalized or out of bounds coordinate labels'
                assert np.unique(l, axis=0).shape[0] == l.shape[0], 'duplicate labels'
            else:
                ne = 1  # label empty
                l = np.zeros((0, 9), dtype=np.float32)
        else:
            nm = 1  # label missing
            l = np.zeros((0, 9), dtype=np.float32)
        return im_file, l, shape, segments, nm, nf, ne, nc
    except Exception as e:
        nc = 1
        logging.info(f'{prefix}WARNING: Ignoring corrupted image and/or label {im_file}: {e}')
        return [None] * 4 + [nm, nf, ne, nc]
    

## anxiliary functions
def load_image(self, index):
    # loads 1 image from dataset, returns img, original hw, resized hw
    
    img = self.imgs[index]
    if img is None:  # not cached
        path = self.img_files[index]
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)  # BGR
        assert img is not None, 'Image Not Found ' + path
        h0, w0 = img.shape[:2]  # orig hw
        r = self.img_size / max(h0, w0)  # ratio
        if r != 1:  # if sizes are not equal
            img = cv2.resize(img, (int(w0 * r), int(h0 * r)),
                             interpolation=cv2.INTER_AREA if r < 1 and not self.augment else cv2.INTER_LINEAR)
        return img, (h0, w0), img.shape[:2]  # img, hw_original, hw_resized
    else:
        return self.imgs[index], self.img_hw0[index], self.img_hw[index]  # img, hw_original, hw_resized

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

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
                A.InvertImg(p=0.3)],
                A.augmentations.transforms.GaussNoise (var_limit=(10.0, 50.0), mean=0, per_channel=True, 
                                                       always_apply=False, p=0.5),
                A.transforms.RandomGamma (gamma_limit=(80, 120), eps=None, always_apply=False, p=0.5),
                )
                #transforms.RandomApply([AddGaussianNoise(0., 1.)], p=0.5))
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
def letterbox(img, new_shape=(640, 640), color=(0.45, 0.45, 0.45), auto=True, scaleFill=False, scaleup=True, stride=32):
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
    return img, ratio, (dw, dh)
    
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
            img = cv2.warpPerspective(img, M, dsize=(width, height), borderValue=(0.45, 0.45, 0.45))
        else:  # affine
            img = cv2.warpAffine(img, M[:2], dsize=(width, height), borderValue=(0.45, 0.45, 0.45))

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
                    img = cv2.warpPerspective(img, M2, dsize=(width, height), borderValue=(0.45, 0.45, 0.45))
                else:  # affine
                    img = cv2.warpAffine(img, M2[:2], dsize=(width, height), borderValue=(0.45, 0.45, 0.45))
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
                img = cv2.warpPerspective(img, M, dsize=(width, height), borderValue=(0.45, 0.45, 0.45))
            else:  # affine
                img = cv2.warpAffine(img, M[:2], dsize=(width, height), borderValue=(0.45, 0.45, 0.45))
            image_transformed = True
        
    return img, targets
## Augmentation ----------------------------------------------------------------------------------------------------------

def augment_hsv(img, hgain=0.5, sgain=0.5, vgain=0.5):
    rnd = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
    if img.dtype == np.uint8:
        hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
        dtype = img.dtype  # org: uint8; 

        x = np.arange(0, 256, dtype=rnd.dtype) # original: int16
        # In OpenCV, H: 0-180, S: 0-255, V: 0-255
        lut_hue = ((x * rnd[0]) % 180).astype(dtype) # color type; originally, 0-360º but only be represented to 180 wrt 8 bit
        lut_sat = np.clip(x * rnd[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * rnd[2], 0, 255).astype(dtype)
        
        # LUT only supports 8-bit LUTs
        img_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val))) # LUT = look up table -> ref: https://mrsnake.tistory.com/142
        img = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR) # no need to return
    else: # float
        hue, sat, val = cv2.split(cv2.cvtColor(np.array(img, dtype=np.uint8), cv2.COLOR_BGR2HSV))
        dtype = img.dtype  # org: uint8; now: float32

        x = np.arange(0, 256, dtype=rnd.dtype) # original: int16
        # In OpenCV, H: 0-180, S: 0-255, V: 0-255
        lut_hue = ((x * rnd[0]) % 180).astype(dtype) # color type; originally, 0-360º but only be represented to 180 wrt 8 bit
        lut_sat = np.clip(x * rnd[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * rnd[2], 0, 255).astype(dtype)
        
        # LUT only supports 8-bit LUTs
        img_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val))) # LUT = look up table -> ref: https://mrsnake.tistory.com/142
        img = cv2.cvtColor(img_hsv/255, cv2.COLOR_HSV2BGR)*255
    return img  

# Ancillary functions with polygon anchor boxes-------------------------------------------------------------------------------------------



