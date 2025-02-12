import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import easydict

from models.experimental import attempt_load
from utils.datasets import LoadImages, LoadSAR
from utils.general import check_img_size, check_requirements, check_imshow, apply_classifier, \
    xyxy2xywh, strip_optimizer, set_logging, check_file, check_name

from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel

from utils.plots import Annotator, set_colors
from models.gan import create_model

@torch.no_grad()
def detect(weights='yolov7.pt',  # model.pt path(s)
           source='data/images',  # file/dir/URL/glob, 0 for webcam
           img_size=640,  # inference size (pixels)
           conf_thres=0.25,  # confidence threshold
           iou_thres=0.45,  # NMS IOU threshold
           max_det=1000,  # maximum detections per image
           device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
           view_img=False,  # show results
           save_txt=False,  # save results to *.txt
           save_conf=False,  # save confidences in --save-txt labels
           save_crop=False,  # save cropped prediction boxes
           nosave=False,  # do not save images/videos
           classes=None,  # filter by class: --class 0, or --class 0 2 3
           agnostic_nms=False,  # class-agnostic NMS
           augment=False,  # augmented inference
           update=False,  # update all models
           project='runs/detect',  # save results to project/name
           name='exp',  # save results to project/name
           exist_ok=False,  # existing project/name ok, do not increment
           line_thickness=1,  # bounding box thickness (pixels)
           hide_labels=False,  # hide labels
           hide_conf=False,  # hide confidences
           half=False,  # use FP16 half-precision inference
           no_trace=True,
           polygon=False,
           proj='4326'
           ):
    
    # Load GAN model
    opt_gan = easydict.EasyDict({
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
        'device': device,
        'checkpoints_dir': '/data/BRIDGE/OD-CycleGAN/checkpoints/sar_cyclegan/',
        'exp_name': 'sar',
        'preprocess': 'resize_and_crop',
        'init_gain': 0.02,
        'load_iter': 0,
        'epoch': 'latest',
        'verbose': False
    })
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    trace = not no_trace

    # Directories
    #save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    save_dir = Path(project, exist_ok=exist_ok)  # define the project
    (save_dir).mkdir(parents=True, exist_ok=True)  # make dir
    #(save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model

    
    # setup GPU for gan_model
    str_ids = opt_gan.device.split(',')
    opt_gan.lst_ids = []
    for str_id in str_ids:
        id = int(str_id)
        if id >= 0:
            opt_gan.lst_ids.append(id)
    #if len(opt_gan.lst_ids) > 0:
    #    torch.cuda.set_device(opt_gan.lst_ids[0])
    opt_gan.device = device#torch.device('cuda:{}'.format(opt.lst_ids[0])) if opt.lst_ids else torch.device('cpu') 
    gan_model = create_model(opt_gan)    
    gan_model.setup(opt_gan)
    stride = int(model.stride.max())  # model stride
    img_size = check_img_size(img_size, s=stride)  # check img_size
    if img_size == -1:
        quit()

    if trace:
        model = TracedModel(model, device, img_size)
        # https://happy-jihye.github.io/dl/torch-2/

    if half:
        model.half()  # to FP16 https://hoya012.github.io/blog/Mixed-Precision-Training/
    
    import numpy as np
    from utils.cfg import Cfg
    from utils.datasets import letterbox
    import os
    
    # Get datasets
    img_formats = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo']  # acceptable image suffixes

    files = os.listdir(source)
    image_ext = np.unique(np.array([x.split('.')[-1].lower() for x in files if x.split('.')[-1].lower() in img_formats]))

    if image_ext in ['tif', 'tiff']:
        dataset = LoadSAR(source, img_size=img_size, stride=stride, save_dir=save_dir, gan_model=gan_model)
    elif image_ext in ['jpg', 'jpeg', 'png']:
        dataset = LoadImages(source, img_size=img_size, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = set_colors(names, target=name.split('_')[0])

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, img_size, img_size).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = img_size
    old_img_b = 3

    # Import utilities 
    '''
    if opt.polygon:
        from utils.general import polygon_non_max_suppression, polygon_scale_coords, polygon_plot_one_box   
    else:
        from utils.general import non_max_suppression, scale_coords, plot_one_box
    NonMax = polygon_non_max_suppression if polygon else non_max_suppression
    ScaleCoords = polygon_scale_coords if polygon else scale_coords
    Plot = polygon_plot_one_box if polygon else plot_one_box
    '''
    from utils.general import polygon_non_max_suppression, polygon_scale_coords
    from utils.plots import polygon_plot_one_box, plot_one_box
    NonMax = polygon_non_max_suppression # needs to append 'if polygon else non_max_suppression' with horiziontal poly model and its weights
    ScaleCoords = polygon_scale_coords

    t0 = time.time()
    for path, input_band, div_img_list, div_coord, shapes, projection, geotransform in dataset:  
        # 테스트 이미지를 1/div_num 만큼 width, height를 분할하고, 크롭된 이미지와 위치좌표를 반환
        p = Path(path)
        
        with open(save_dir / (p.stem + '_' + name + '.txt'), 'w') as f:
            if polygon:
                f.write('ImageName,Lon,Lat,Class,Size,X1,Y1,X2,Y2,X3,Y3,X4,Y4\n')
            else:
                f.write('ImageName,Lon,Lat,Class,Size,X,Y,W,H\n')
        b1_image = np.dstack((input_band[:,:,1]*255, input_band[:,:,1]*255, input_band[:,:,1]*255))
        SLC = True if p.stem.split('_')[2] == 'SLC' else False
        if not SLC:
            xoff, ca, cb, yoff, cd, ce = geotransform
        tot_det = torch.Tensor().to(device)
        for d_id, img0 in enumerate(div_img_list):
            # 원본 이미지 좌표로 변환하기 위해 분활 좌표를 저장
            div_x, div_y = div_coord[d_id][0], div_coord[d_id][1]
            #img = letterbox(img0, img_size, stride=Cfg.stride)[0]
            # Convert
            img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)
            img = img0[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416; already applied
            img = np.ascontiguousarray(img)
                            
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            #img /= 255.0  # 0 - 255 to 0.0 - 1.0
            
            if img.ndimension() == 3:
                img = img.unsqueeze(0)
    
            # Warmup
            if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
                old_img_b = img.shape[0]
                old_img_h = img.shape[2]
                old_img_w = img.shape[3]
                for i in range(3):
                    model(img, augment=augment)[0]
            # Inference
            t1 = time_synchronized()
            #with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
            pred = model(img, augment=augment)[0]
            t2 = time_synchronized()
            
            pred[:,:,:8:2] = pred[:,:,:8:2] + div_coord[d_id][2]
            pred[:,:,1:8:2] = pred[:,:,1:8:2] + div_coord[d_id][3]
            
            tot_det = torch.cat([tot_det, pred])
        
        # Apply NMS
        max_det = 1000
        tot_det = NonMax(tot_det, conf_thres, iou_thres, classes=classes, agnostic=agnostic_nms, max_det=max_det)
        t3 = time_synchronized()    
        
        # Process detections
        for d_id, dets in enumerate(tot_det):  # detections per image
            s, im0 = '', img0
            
            save_path = str(save_dir / p.name)#.replace('.tif','_{}.tif'.format(d_id)))  # img.jpg
            if len(dets)==0:
                continue
            
            # Rescale boxes from img_size to im0 size
            #detn = det.clone()
            # Print results
            for c in dets[:, -1].unique():
                n = (dets[:, -1] == c).sum()  # detections per class
                s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                 
            # Write results 
            for *xyxyxyxy, conf, cls in reversed(dets): # det per crop
                if save_txt:  # Write to file
                    # get lon lat of the box center
                    if not SLC:
                        x1 = min(xyxyxyxy[0::2]); x2 = max(xyxyxyxy[0::2]) # top-left
                        y1 = min(xyxyxyxy[1::2]); y2 = max(xyxyxyxy[1::2]) # bottom-right

                        lon = ca * (x1+x2)/2 + cb * (y1+y2)/2 + xoff
                        lat = cd * (x2+x2)/2 + ce * (y1+y2)/2 + yoff
                    
                        if (dataset.epsg != proj): 
                            # when the input coordinate system is different to dst coordinate system
                            dataset.point.AddPoint(float(lon), float(lat))
                            if proj != Cfg.outepsg: # if proj argument is different from cfg 
                                dataset.coordTransform = dataset.change_projection(int(proj))
                            dataset.point.Transform(dataset.coordTransform) # 3857 to 4326
                            
                            lat = dataset.point.GetX() # pop x
                            lon = dataset.point.GetY() # pop y
                    
                    with open(save_dir / (p.stem + '_' + name + '.txt'), 'a') as f:
                        if polygon:
                            if not SLC:
                                line = (str(Path(path).name.replace('.tif', '_' + name + '.tif')), lon, lat, cls, Cfg.size, *xyxyxyxy, conf) if save_conf \
                                    else (str(Path(path).name.replace('.tif', '_' + name + '.tif')), lon, lat, cls, Cfg.size, *xyxyxyxy)
                                f.write(('%s,' % line[0] + '%.14g,'*2 % line[1:3] + '%d,'*2 % line[3:5] + \
                                        '%.14g,'*(len(line)-5) % line[5:])[:-1]+'\n')
                            else:
                                line = (str(Path(path).name.replace('.tif', '_' + name + '.tif')), cls, Cfg.size, *xyxyxyxy, conf) if save_conf \
                                    else (str(Path(path).name.replace('.tif', '_' + name + '.tif')), cls, Cfg.size, *xyxyxyxy)     
                                f.write(('%s,' % line[0] + '%.14g,'*2 % line[1:3] + \
                                        '%.14g,'*(len(line)-3) % line[3:])[:-1]+'\n')                               
                            
                        else:
                            coord_xy = [x1, y1, x2-x1, y2-y1]
                            if not SLC:
                                
                                line = (str(Path(path).name.replace('.tif', '_' + name + '.tif')), lon, lat, cls, Cfg.size, *coord_xy, conf) if save_conf \
                                    else (path.split('/')[-1][:-4], lon, lat, cls, Cfg.size, *coord_xy)
                                f.write(('%s,' % line[0] + '%.14g,'*2 % line[1:3] + '%d,'*2 % line[3:5] + \
                                        '%.14g,'*(len(line)-5) % line[5:])[:-1]+'\n')
                            else:
                                line = (str(Path(path).name.replace('.tif', '_' + name + '.tif')), cls, Cfg.size, *coord_xy, conf) if save_conf \
                                    else (path.split('/')[-1][:-4], cls, Cfg.size, *coord_xy)
                                f.write(('%s,' % line[0] + '%.14g,'*2 % line[1:3] + \
                                        '%.14g,'*(len(line)-3) % line[3:])[:-1]+'\n')
                            
                        
                    if save_img or view_img:  # Add bbox to image
                    #    
                        label = None if hide_labels else (names[int(cls)] if hide_conf else f'{names[int(cls)]} {conf:.2f}')
                        
                        if polygon:
                            xyxyxyxy = [int(el) for el in xyxyxyxy]
                            polygon_plot_one_box(xyxyxyxy, b1_image, label=label, color=colors[int(cls)], line_thickness=1)
                            # plot bboxes only on band1
                        else:
                            xyxy = [x1, y1, x2, y2]
                            xyxy = [int(el) for el in xyxy]
                            plot_one_box(xyxy, b1_image, label=label, color=colors[int(cls)], line_thickness=1)
                            # plot bboxes only on band1

                # Show results
                if view_img:
                    cv2.imshow(str(p), im0)
                    cv2.waitKey(1)  # 1 millisecond

        # Save results (image with detections)
        if save_img or view_img:
            from osgeo import gdal 
            import zipfile
            #from tifffile import imwrite

            
            save_path = str(save_dir / Path(path).name.replace('.tif', '_' + name + '.tif'))
            print(f" The image with the result is saved in: {save_path}")
            
            b1_image = np.array(b1_image, dtype=np.uint8)
            cv2.imwrite(save_path, b1_image)   # find what to plot   
            #imwrite(save_path, b1_image)
            
            if not SLC: # slc는 geotransform 안함. 
                outfile = gdal.Open(save_path, gdal.GA_Update) 
                outfile.SetGeoTransform(geotransform) # 
                outfile.SetProjection(projection) # gdal에서 원하는 좌표계로 변형
                outfile.FlushCache() # 저장

            zip_file = save_path.replace('tif', 'zip')
            with zipfile.ZipFile(zip_file, 'a') as z:
                for ext in ['tif','txt']:
                    z.write(save_path.replace('tif',ext), compress_type = zipfile.ZIP_DEFLATED, arcname=os.path.basename(save_path.replace('tif',ext)))
                z.close()

            print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')
                    
        if save_txt or save_img:
            s = f"\n{len(list(save_dir.glob('*.txt')))} labels saved to {save_dir}" if save_txt else ''
            #print(f"Results saved to {save_dir}{s}")
    
    # print time (inference + NMS)
    print(f'Done. ({time.time() - t0:.3f}s)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='poly_yolo.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='./inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.01, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='output/', help='save results to project/name')
    parser.add_argument('--name', default='BRIDG_RGB000102', help='save results as defined name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    parser.add_argument('--polygon', action='store_true', help='enable polygon anchor boxes')
    parser.add_argument('--proj', default='4326', help='define the projection coordinates to transform')

    # gwang add
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')

    opt = parser.parse_args()
    
    status = check_name(opt.name)
    if status == False:
        print("Invalid file name: "+opt.name+". It should have the format [TARGT]_RGB[######]")
        quit()

    opt.weights = check_file(opt.weights[0])
    #opt.weights = '/data/BRIDGE/yolo-rotate/runs/train/exp51_batch16_epoch300_imgsize640_lr0.001_lrf0.1_smoothing0.1_multiscaleFalse_3pixel/weights/best.pt'
    
    print(opt)
    #check_requirements(exclude=('pycocotools', 'thop'))

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov7.pt']:
                detect(**vars(opt))
                strip_optimizer(opt.weights)
        else:
            detect(**vars(opt))
