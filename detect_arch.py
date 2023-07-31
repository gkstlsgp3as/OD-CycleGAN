import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadImages, LoadSAR
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, polygon_non_max_suppression, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path, polygon_scale_coords
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel

from utils.general import non_max_suppression_obb, scale_polys, save_one_box
from utils.plots import Annotator, colors, polygon_plot_one_box, plot_one_box

@torch.no_grad()
def detect(weights='yolov7.pt',  # model.pt path(s)
           source='data/images',  # file/dir/URL/glob, 0 for webcam
           imgsz=640,  # inference size (pixels)
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
           satellite=False
           ):
    polygon=True; satellite=True
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    trace = not no_trace

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    if trace:
        model = TracedModel(model, device, opt.imgsz)

    if half:
        model.half()  # to FP16
    
    if satellite and polygon:
        import numpy as np
        from utils.cfg import Cfg
        from utils.datasets import letterbox
        
        dataset = LoadSAR(source, img_size=imgsz, stride=stride, save_dir=save_dir)
        # Get names and colors
        names = model.module.names if hasattr(model, 'module') else model.names
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
    
        # Run inference
        if device.type != 'cpu':
            model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
        old_img_w = old_img_h = imgsz
        old_img_b = 3
    
        t0 = time.time()
        for path, rgb_band, div_img_list, div_coord, shapes in dataset:  
            # 테스트 이미지를 1/div_num 만큼 width, height를 분할하고, 크롭된 이미지와 위치좌표를 반환
            for d_id, img0 in enumerate(div_img_list):
                
                # 원본 이미지 좌표로 변환하기 위해 분활 좌표를 저장
                div_x, div_y = div_coord[d_id][0], div_coord[d_id][1]
                img = letterbox(img0, imgsz, stride=Cfg.stride)[0]
                # Convert
                img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416; already applied
                img = np.ascontiguousarray(img)
    
                # Gaussian Blur
                #if Cfg.kernel > 0:
                #    img = cv2.GaussianBlur(img, (Cfg.kernel,Cfg.kernel), 0)
                           
                img = torch.from_numpy(img).to(device)
                img = img.half() if half else img.float()  # uint8 to fp16/32
                img /= 255.0  # 0 - 255 to 0.0 - 1.0
                if img.ndimension() == 3:
                    img = img.unsqueeze(0)
        
                # Warmup
                if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
                    old_img_b = img.shape[0]
                    old_img_h = img.shape[2]
                    old_img_w = img.shape[3]
                    for i in range(3):
                        model(img, augment=opt.augment)[0]
        
                # Inference
                t1 = time_synchronized()
                #with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
                pred = model(img, augment=opt.augment)[0]
                #if (pred[...,8] > 0.001).any():
                #    breakpoint()
                t2 = time_synchronized()
        
                # Apply NMS
                max_det = 1000
                pred = polygon_non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms, max_det=max_det)
                                
                t3 = time_synchronized()
        
                #print('\n\n', pred[0].shape, '\n\n')
                #print(pred)
                # Process detections
                for i, det in enumerate(pred):  # detections per image
                    p, s, im0 = path, '', img0
        
                    p = Path(p)  # to Path
                    save_path = str(save_dir / p.name)#.replace('.tif','_{}.tif'.format(d_id)))  # img.jpg
                    txt_path = str(save_dir / 'labels' / p.stem) # img.txt
                    gn = torch.tensor(im0.shape)[[1, 0, 1, 0, 1, 0, 1, 0]]  # normalization gain whwh
                    if len(det):
                        # Rescale boxes from img_size to im0 size
                        #detn = det.clone()
                        det[:, :8] = polygon_scale_coords(img.shape[2:], det[:, :8], im0.shape).round() #xyxyxyxy
        
                        # Print results
                        for c in det[:, -1].unique():
                            n = (det[:, -1] == c).sum()  # detections per class
                            s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
        
                        # Write results
                        for *xyxyxyxy, conf, cls in reversed(det):
                            if save_txt:  # Write to file
                                xyxyxyxyn = ((torch.tensor(xyxyxyxy).view(1, 8)) / gn).view(-1).tolist()
                                line = (cls, *xyxyxyxy, conf) if save_conf else (cls, *xyxyxyxy)  # label format
                                with open(save_dir / 'labels' / (p.stem + '.txt'), 'a') as f:
                                    f.write(('%g ' * len(line)).rstrip() % line + '\n')
                               
                                    
                                if save_img or view_img:  # Add bbox to image
                                #    
                                    label = None if hide_labels else (names[int(cls)] if hide_conf else f'{names[int(cls)]} {conf:.2f}')
                                    #label = f'{names[int(cls)]} {conf:.2f}'
                                #    #plot_xywh_box(xywh, im0, label=label, color=colors[int(cls)], line_thickness=1)
                                   # polygon_plot_one_box(torch.tensor(xyxyxyxy).cpu().numpy(), im0, label=label, color=colors(c, True), line_thickness=line_thickness)
                                    print(label)
                                    #polygon_plot_one_box(*xyxyxyxy, rgb_band, label=label, color=colors[int(cls)], line_thickness=1)
    
    
                    # Stream results
                    if view_img:
                        cv2.imshow(str(p), im0)
                        cv2.waitKey(1)  # 1 millisecond
        
            # Save results (image with detections)
            if save_img or view_img:
                 save_path = str(save_dir / Path(path).name)
            
                 print(f" The image with the result is saved in: {save_path}")
                 cv2.imwrite(save_path, cv2.cvtColor(rgb_band, cv2.COLOR_RGB2GRAY))   # find what to plot             
                 # Print time (inference + NMS)
                 print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')
                        
            if save_txt or save_img:
                s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
                #print(f"Results saved to {save_dir}{s}")
        
        print(f'Done. ({time.time() - t0:.3f}s)')

    else:
        # Set Dataloader
        dataset = LoadImages(source, img_size=imgsz, stride=stride)
    
        # Get names and colors
        names = model.module.names if hasattr(model, 'module') else model.names
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
    
        # Run inference
        if device.type != 'cpu':
            model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
        old_img_w = old_img_h = imgsz
        old_img_b = 1
    
        t0 = time.time()
        
        for path, img, im0s in dataset:
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)
    
            # Warmup
            if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
                old_img_b = img.shape[0]
                old_img_h = img.shape[2]
                old_img_w = img.shape[3]
                for i in range(3):
                    model(img, augment=opt.augment)[0]
    
            # Inference
            t1 = time_synchronized()
            pred = model(img, augment=opt.augment)[0]
            t2 = time_synchronized()
    
            # Apply NMS
            # pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
            # gwang add
            max_det = 1000
            if polygon: 
                pred = polygon_non_max_suppression(pred, opt.conf_thres, opt.iou_thres, opt.classes, opt.agnostic_nms, multi_label=True, max_det=max_det)
            else: 
                pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
                
            t3 = time_synchronized()
    
    
            # Process detections
            for i, det in enumerate(pred):  # detections per image
                #pred_poly = rbox2poly(det[:, :5]) # (n, [x1 y1 x2 y2 x3 y3 x4 y4])
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)
    
                p = Path(p)  # to Path
                save_path = str(save_dir / p.name)  # img.jpg
                txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
                s += '%gx%g ' % img.shape[2:]  # print string
                if polygon:
                    gn = torch.tensor(im0.shape)[[1, 0, 1, 0, 1, 0, 1, 0]]  # normalization gain xyxyxyxy
                else:
                    gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                
                if not polygon: imc = im0.copy() if save_crop else im0  # for save_crop
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    # det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                    
                    # Rescale boxes from img_size to im0 size
                    if polygon:
                        det[:, :8] = polygon_scale_coords(img.shape[2:], det[:, :8], im0.shape).round()
                    else:
                        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                    
                    if polygon:
                        # Write results
                        for *xyxyxyxy, conf, cls in reversed(det):
                            if save_txt:  # Write to file
                                xyxyxyxyn = (torch.tensor(xyxyxyxy).view(1, 8) / gn).view(-1).tolist()  # normalized xyxyxyxy
                                line = (cls, *xyxyxyxyn, conf) if save_conf else (cls, *xyxyxyxyn)  # label format
                                with open(txt_path + '.txt', 'a') as f:
                                    f.write(('%g ' * len(line)).rstrip() % line + '\n')
        
                            if save_img or view_img:  # Add bbox to image
                                c = int(cls)  # integer class
                                label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                                polygon_plot_one_box(torch.tensor(xyxyxyxy).cpu().numpy(), im0, label=label, color=colors(c, True), line_thickness=line_thickness)
                    else:
                        # Write results
                        for *xyxy, conf, cls in reversed(det):
                            if save_txt:  # Write to file
                                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                                line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                                with open(txt_path + '.txt', 'a') as f:
                                    f.write(('%g ' * len(line)).rstrip() % line + '\n')
        
                            if save_img or save_crop or view_img:  # Add bbox to image
                                c = int(cls)  # integer class
                                label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                                plot_one_box(xyxy, im0, label=label, color=colors(c, True), line_thickness=line_thickness)
                                if save_crop:
                                    save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)
        
    
                # Print time (inference + NMS)
                print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')
    
                # Stream results
                if view_img:
                    cv2.imshow(str(p), im0)
                    cv2.waitKey(1)  # 1 millisecond
    
                # Save results (image with detections)
                if save_img:
                    if dataset.mode == 'image':
                        cv2.imwrite(save_path, im0)
                        print(f" The image with the result is saved in: {save_path}")
    
        if save_txt or save_img:
            s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
            #print(f"Results saved to {save_dir}{s}")
    
        print(f'Done. ({time.time() - t0:.3f}s)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='poly_yolo.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='./inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--imgsz', type=int, default=640, help='inference size (pixels)')
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
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    parser.add_argument('--polygon', action='store_true', help='enable polygon anchor boxes')
    parser.add_argument('--satellite', action='store_true', help='enable detection for satellite images')
    # gwang add
    
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')

    opt = parser.parse_args()
    opt.polygon=True; opt.satellite=True
    print(opt)
    #check_requirements(exclude=('pycocotools', 'thop'))

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov7.pt']:
                detect(**vars(opt))
                strip_optimizer(opt.weights)
        else:
            detect(**vars(opt))
