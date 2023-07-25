import numpy as np
from osgeo import gdal
import os
import cv2
import pandas as pd
import json
from tifffile import imwrite

def landmask(tif_name):
    from osgeo import gdal
    from osgeo import ogr
    from PIL import Image
    
    ras_ds = gdal.Open(tif_name, gdal.GA_ReadOnly)
    gt = ras_ds.GetGeoTransform()

    vecPath = "/data/BRIDGE/yolo-rotate/landmask/water_road/"
    vec_ds = ogr.Open(vecPath)
    lyr = vec_ds.GetLayer()

    filename='/data/BRIDGE/yolo-rotate/landmask/mask/temp.tif'
    drv_tiff = gdal.GetDriverByName("GTiff") 
    chn_ras_ds = drv_tiff.Create(filename, ras_ds.RasterXSize, ras_ds.RasterYSize, 1, gdal.GDT_Float32)
    chn_ras_ds.SetGeoTransform(gt)

    gdal.RasterizeLayer(chn_ras_ds, [1], lyr) 
    chn_ras_ds.GetRasterBand(1).SetNoDataValue(0.0) 
    chn_ras_ds = None

    raster = gdal.Open(filename)
    band_data = np.array(raster.GetRasterBand(1).ReadAsArray())
    return np.array(band_data, np.uint8)

def geotiffreadRef(tif_name):
    import gdal
    import numpy as np

    gdal.AllRegister()

    ds = gdal.Open(tif_name)
    gt = ds.GetGeoTransform()
    rows, cols = ds.RasterXSize, ds.RasterYSize

    gt = np.array(gt)
    gt[2] = gt[0] + gt[1] * (rows - 1)
    gt[4] = gt[3] + gt[5] * (cols - 1)

    gt.astype(np.double)

    return gt, rows, cols


def geographicToIntrinsic(tif_ref, lat, lon):
    import numpy as np
    from scipy.interpolate import interp1d

    max_lat = tif_ref[3]
    min_lat = tif_ref[4]
    max_lon = tif_ref[2]
    min_lon = tif_ref[0]
    space_lat = tif_ref[5]
    space_lon = tif_ref[1]

    num_lat = round(((max_lat - space_lat) - min_lat) / (-space_lat))
    num_lon = round(((max_lon + space_lon) - min_lon) / space_lon)

    lat_array = np.linspace(max_lat, min_lat, num_lat)
    lat_order = np.linspace(1, len(lat_array), len(lat_array))
    lon_array = np.linspace(min_lon, max_lon, num_lon)
    lon_order = np.linspace(1, len(lon_array), len(lon_array))

    lat_order = lat_order.astype(int)
    lon_order = lon_order.astype(int)

    try:
        lat_y = interp1d(lat_array, lat_order)
        y = lat_y(lat)
    except:
        lat_y = interp1d(lat_array, lat_order, fill_value='extrapolate')
        y = lat_y(lat)

    try:
        lon_x = interp1d(lon_array, lon_order)
        x = lon_x(lon)
    except:
        lon_x = interp1d(lon_array, lon_order, fill_value='extrapolate')
        x = lon_x(lon)

    # if len(lat)==1:
    #    y=float(y)

    # if len(lon)==1:
    #    x=float(x)

    return y, x

# Text file 형식으로 되어 있는 학습자료를 csv파일 형식으로 옮김
# Attention: file with 1 trainingData is unapprehendible
def concat_txt2csv(exportPath='./milestone/rgb_train.csv', typeNum=2):
    import glob
    import pandas as pd
    import numpy as np
    from utils.cfg import Cfg

    if typeNum==1:
        txt_path = Cfg.train_txt_path
    elif typeNum==2:
        txt_path = Cfg.valid_txt_path
    else:
        txt_path = Cfg.test_txt_path

    txt_name = glob.glob(txt_path)

    # Massive array to concatenate all data
    txt_data = np.empty((0, 9), int)
    txt_name_export = []

    for temp in txt_name:
        try:
            # Open assigned *.txt file
            #filedata = np.loadtxt(temp, 'str', delimiter=',')
            filedata = np.loadtxt(temp,dtype='str', delimiter=' ')

            # Assign array full of filename
            txt_name = temp.split('/')[-1]
            txt_name_temp = [txt_name.replace('txt','tif')] * len(filedata)
            
            # Insert data into txt_data folder
            txt_data = np.concatenate((txt_data, filedata), axis=0)
            txt_name_export = txt_name_export + txt_name_temp
        except:
            aaaaa=1


    # array export to rgb_train.csv
    df = pd.DataFrame(columns=['image', 'X1', 'Y1', 'X2', 'Y2', 'X3', 'Y3', 'X4', 'Y4', 'Class'])
    df['image'] = txt_name_export
    df['X1'] = txt_data[:, 1]
    df['Y1'] = txt_data[:, 2]
    df['X1'] = txt_data[:, 3]
    df['Y1'] = txt_data[:, 4]
    df['X1'] = txt_data[:, 5]
    df['Y1'] = txt_data[:, 6]
    df['X1'] = txt_data[:, 7]
    df['Y1'] = txt_data[:, 8]
    df['Class'] = txt_data[:, 0]

    df.to_csv(exportPath, index=False)
    

def split_set(img_size=640, datatype='sentinel', source='org', polygon=True):
    from utils.cfg import Cfg
    import pandas as pd
    import random
    
    import yaml
    with open('./data/{}.yaml'.format(datatype), errors='ignore') as f:
        data = yaml.safe_load(f)
    
    random.seed(1004)

    ## data split for train and valid data 
    fl_list = os.listdir(Cfg.img_path)
    tif_list = [fl for fl in fl_list if fl.endswith('tif')]
    random.shuffle(tif_list)
    
    valid_num = round(len(tif_list)*0.1)
    
    valid_set = tif_list[:valid_num]; test_set = tif_list[valid_num:valid_num*2]
    train_set = tif_list[valid_num*2:]
    
    ## image division for train and valid dataset
    origin_image_folder = Cfg.img_path
    #f = open('./data/div_{}.txt'.format(str(div_num)), 'w')
    
    division = division_set_poly if polygon else division_set

    division(valid_set, origin_image_folder, 'valid', datatype, img_size, source)
    division(train_set, origin_image_folder, 'train', datatype, img_size, source)
    division(test_set, origin_image_folder, 'test', datatype, img_size, source)
           
    for i, div_set in enumerate(['train','valid','test']):
        
        f = open(data[div_set],'w')
        
        img_path = './data/images/{}/{}/{}/'.format(datatype, source, div_set)
        fl_list = os.listdir(img_path); fl_list.sort()
        [f.write(l.replace('.tif','')+'\n') for l in fl_list]
        f.close() 
        
        concat_txt2csv(exportPath='./milestone/rgb_{}.csv'.format(div_set), typeNum=i+1)

# 이미지를 분할하고 각 이미지의 bbox 정보를 분활된 이미지에 맞게 변환
def division_set_poly(image_list, origin_image_folder, div_set, datatype='sentinel', img_size=640, source='org'):    
    from utils.cfg import Cfg
    from utils.general import order_corners
    import torch
    from utils.gdal_preprocess import line_detection
    import random
    
    print("Start dividing images for polygon\n\n")
    for i in image_list:
        last_name = i.split('_')[-1]
        image_path = os.path.join(origin_image_folder, i)
        
        bandnumber = Cfg.Satelliteband

        # RGB로 변환
        if Cfg.NewTest == 0:
            rgb_image = band_to_input(image_path, bandnumber)
        else:
            rgb_image = band_to_input(image_path, bandnumber,True)
        
        # x, y, w, h를 x1, y1, x2, y2로 변환
        image_df = pd.read_csv(image_path.replace("tif", "txt"), names=['class','X1','Y1','X2','Y2','X3','Y3','X4','Y4'])
        image_df['class'] = [col.lower().strip() for col in image_df['class'].to_numpy() if isinstance(col, str)]
        #image_df['X2'] = image_df['X'] + image_df['W']
        #image_df['Y2'] = image_df['Y'] + image_df['H']
        
        bboxes = image_df[['X1', 'Y1', 'X2', 'Y2', 'X3', 'Y3', 'X4', 'Y4', 'class']].to_numpy()
        coords = bboxes[:,:-1];  classes =  bboxes[:,-1]
        coords = order_corners(torch.Tensor(np.array(coords, dtype=np.float64)))      
        bboxes = np.hstack([np.array(coords), classes.reshape(-1,1)])
        # 분할 구간 설정
        h, w = rgb_image.shape[:2]

        hd = [x for x in range(0, h, img_size-Cfg.overlap)]
        wd = [x for x in range(0, w, img_size-Cfg.overlap)]
        hd[-1] = h - img_size; wd[-1] = w - img_size
        hd.sort(); wd.sort()
        minTh = np.mean(rgb_image[rgb_image>Cfg.minTh])
        #maxTh = np.mean(rgb_image[rgb_image>Cfg.maxTh])

        land_mask = landmask(image_path)     
        nl = 0  
        
        for h_id, div_h in enumerate(hd[:-1]):
            for w_id, div_w in enumerate(wd[:-1]):
                # 분할된 이미지의 좌표
                x1, y1 = div_w, div_h
                x2, y2 = div_w+img_size, div_h+img_size
                
                # 이미지 크롭
                crop = rgb_image[y1:y2, x1:x2]
                # Image.fromarray(crop).save('./milestone/line/crop_'+save_name.replace('tif','png'))
                
                #lines = line_detection(crop)
                # Image.fromarray(lines*255).save('./milestone/line/lines_10560_0_11200_640_9D5A.png')
                
                div_boxes = []
                save_name = str(x1) + '_' + str(y1) + '_' + str(x2) + '_' + str(y2) + '_' + last_name
                line = save_name
                
                save_txt_path = './data/labels/{}/{}/{}/'.format(datatype, source, div_set)
                save_img_path = './data/images/{}/{}/{}/'.format(datatype, source, div_set)
                
                f = open(os.path.join(save_txt_path, save_name.replace("tif","txt")), 'w')
                
                for b in bboxes:
                    # 현재 분할된 이미지의 x, y 구간 
                    
                    min_x = np.min(b[0:-1:2]); max_x = np.max(b[0:-1:2])
                    min_y = np.min(b[1:-1:2]); max_y = np.max(b[1:-1:2])
                    
                    if (x1 <= min_x <= x2) and (x1 <= max_x <= x2) and \
                        (y1 <= min_y <= y2) and (y1 <= max_y <= y2):

                        #b[:8] = np.multiply(b[:8],2)
                        #bb_bias = [-0.5, -0.5, +0.5, -0.5, +0.5, +0.5, -0.5, +0.5] # need to revise; invalid bboxes found 
                        #b[:8] = np.add(b[:8], bb_bias)
                        #min_x = np.min(b[0:-1:2]); max_x = np.max(b[0:-1:2])
                        #min_y = np.min(b[1:-1:2]); max_y = np.max(b[1:-1:2])
                        
                        # 모든 픽셀이 전부 1 (물) 이거나 0 (육지)이면 제외
                        if not ((land_mask[round(min_y):round(max_y), round(min_x):round(max_x)] > 0).all())\
                            and (not (land_mask[round(min_y):round(max_y), round(min_x):round(max_x)] == 0).all()): #and ((max_x - min_x) * (max_y - min_y) >= 3.0):
                            # 원본 bbox 좌표를 분할된 이미지 좌표로 변환 
                            dw = (x2-x1); dh = (y2-y1) #dw = (x2-x1)*2; dh = (y2-y1)*2
                                
                            image_coord = [(c-x1)/dw if i%2==0 else (c-y1)/dh for i,c in enumerate(b[:-1])]
                                
                            bbox = np.hstack([b[8], image_coord]) #cls, center_x, center_y, width, height
                            #print([b[4], centx, centy, (dx2-dx1), (dy2-dy1)])
                            #print(dw, dh)
                                
                            div_boxes.append(bbox)  
                            nl += 1          

                imwrite(os.path.join(save_img_path, save_name),crop)
                #cv2.imwrite(os.path.join(save_img_path, save_name), crop)
                if len(div_boxes) > 0:
                    print('saved: ',os.path.join(save_img_path, save_name))
                    
                    for d in div_boxes:
                        #class_name = 'ship' if strd[4]==0 else 'other'
                        f.write('%s %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f\n' % (d[0], float(d[1]), float(d[2]), \
                                                                                  float(d[3]), float(d[4]), float(d[5]), \
                                                                                      float(d[6]), float(d[7]), float(d[8])))
                    f.close()
                else: 
                    print('no bboxes found: ',os.path.join(save_txt_path, save_name))
                    f.close()
                    if random.random() > 0.01:
                        os.remove(os.path.join(save_img_path, save_name))
                        os.remove(os.path.join(save_txt_path, save_name.replace("tif","txt")))

        print('total number of labels found:', nl)
                
def division_set(image_list, origin_image_folder, div_set, datatype='sentinel', img_size=640, source='org'):    
    from utils.cfg import Cfg
    from utils.general import order_corners
    import torch
    from utils.gdal_preprocess import line_detection
    import random
    
    
    print("Start dividing images\n\n")
    for i in image_list:
        last_name = i.split('_')[-1]
        image_path = os.path.join(origin_image_folder, i)
        
        bandnumber = Cfg.Satelliteband

        # RGB로 변환
        if Cfg.NewTest == 0:
            rgb_image = band_to_input(image_path, bandnumber)
        else:
            rgb_image = band_to_input(image_path, bandnumber,True)
        
        # x, y, w, h를 x1, y1, x2, y2로 변환
        image_df = pd.read_csv(image_path.replace("tif", "txt"), names=['class','X1','Y1','X2','Y2','X3','Y3','X4','Y4'])
        image_df['class'] = [col.lower().strip() for col in image_df['class'].to_numpy() if isinstance(col, str)]
        #image_df['X2'] = image_df['X'] + image_df['W']
        #image_df['Y2'] = image_df['Y'] + image_df['H']
        
        bboxes = image_df[['X1', 'Y1', 'X2', 'Y2', 'X3', 'Y3', 'X4', 'Y4', 'class']].to_numpy()
        coords = bboxes[:,:-1]; coords = order_corners(torch.Tensor(np.array(coords, dtype=np.float64)))
        classes =  bboxes[:,-1]       
        bboxes = np.hstack([np.array(coords), classes.reshape(-1,1)])
        # 분할 구간 설정
        h, w = rgb_image.shape[:2]

        hd = [x for x in range(0, h, img_size-Cfg.overlap)]
        wd = [x for x in range(0, w, img_size-Cfg.overlap)]
        hd[-1] = h - img_size; wd[-1] = w - img_size
        hd.sort(); wd.sort()
        minTh = np.mean(rgb_image[rgb_image>Cfg.minTh])
        #maxTh = np.mean(rgb_image[rgb_image>Cfg.maxTh])

        land_mask = landmask(image_path) 
        nl = 0  
        
        for h_id, div_h in enumerate(hd[:-1]):
            for w_id, div_w in enumerate(wd[:-1]):
                # 분할된 이미지의 좌표
                x1, y1 = div_w, div_h
                x2, y2 = div_w+img_size, div_h+img_size
                
                # 이미지 크롭
                crop = rgb_image[y1:y2, x1:x2]
                
                div_boxes = []
                save_name = str(x1) + '_' + str(y1) + '_' + str(x2) + '_' + str(y2) + '_' + last_name
                
                save_txt_path = './data/labels/{}/{}/{}/'.format(datatype, source, div_set)
                save_img_path = './data/images/{}/{}/{}/'.format(datatype, source, div_set)
                
                f = open(os.path.join(save_txt_path, save_name.replace("tif","txt")), 'w')
                
                for b in bboxes:
                    
                    # 현재 분할된 이미지의 x, y 구간 
                    min_x = np.min(b[0:-1:2]); max_x = np.max(b[0:-1:2])
                    min_y = np.min(b[1:-1:2]); max_y = np.max(b[1:-1:2])
                    
                    if (x1 <= min_x <= x2) and (x1 <= max_x <= x2) and \
                        (y1 <= min_y <= y2) and (y1 <= max_y <= y2):
                        
                        #bb_bias = [-2, -2, +2, -2, +2, +2, -2, +2] 
                        #b[:8] = np.add(b[:8], bb_bias)
                        #min_x = np.min(b[0:-1:2]); max_x = np.max(b[0:-1:2])
                        #min_y = np.min(b[1:-1:2]); max_y = np.max(b[1:-1:2])
                        
                        # land_mask = 1: land, 0: ocean
                        if (not (land_mask[round(min_y):round(max_y), round(min_x):round(max_x)] > 0).all()) \
                            and (not (land_mask[round(min_y):round(max_y), round(min_x):round(max_x)] == 0).all()):# (max_x - min_x) * (max_y - min_y) >= 3.0 and 
                            # 원본 bbox 좌표를 분할된 이미지 좌표로 변환 
                            dw = (x2-x1); dh = (y2-y1) #dw = (x2-x1)*2; dh = (y2-y1)*2
                            
                            b = [(min_x+max_x)/2, (min_y+max_y)/2, max_x-min_x, max_y-min_y, b[-1]] # cx, cy, w, h
                            image_coord = [(c-x1)/dw if i%2==0 else (c-y1)/dh for i,c in enumerate(b[:-3])]
                            size = [c/dw if i%2==0 else c/dh for i,c in enumerate(b[2:4])]
                            
                            bbox = np.hstack([b[-1], image_coord, size]) #cls, center_x, center_y, width, height
                            #print([b[4], centx, centy, (dx2-dx1), (dy2-dy1)])
                            #print(dw, dh)
                            
                            div_boxes.append(bbox)  
                            nl += 1 

                #cv2.imwrite(os.path.join(save_img_path, save_name), crop)
                imwrite(os.path.join(save_img_path, save_name),crop)
                if len(div_boxes) > 0:
                    print('saved: ',os.path.join(save_img_path, save_name))
                    
                    for d in div_boxes:
                        #class_name = 'ship' if strd[4]==0 else 'other'
                        f.write('%s %.6f %.6f %.6f %.6f\n' % (d[0], float(d[1]), float(d[2]), float(d[3]), float(d[4])))
                    f.close()
                    print('total number of labels found:', nl)
                else: 
                    print('no bboxes found: ',os.path.join(save_txt_path, save_name))
                    f.close()
                    if random.random() > 0.01:
                        os.remove(os.path.join(save_img_path, save_name))
                        os.remove(os.path.join(save_txt_path, save_name.replace("tif","txt")))


def division_testset(input_band=None, img_size=640):
    img_list, div_coord = [], []
    
    # 분할 구간 설정
    h, w = input_band.shape[:2]

    hd = [x for x in range(0, h, img_size-200)]
    wd = [x for x in range(0, w, img_size-200)]
    hd[-1] = h - img_size; wd[-1] = w - img_size
    hd.sort(); wd.sort()
    
    for h_id, div_h in enumerate(hd[:-1]):
        for w_id, div_w in enumerate(wd[:-1]):
            # 분할된 이미지의 좌표
            x1, y1 = div_w, div_h
            x2, y2 = div_w+img_size, div_h+img_size

            dw = x2-x1; dh = y2-y1
            # Crop
            crop = input_band[y1:y2, x1:x2]
            img_list.append(crop)
            div_coord.append([dw, dh, div_w, div_h])

    return img_list, div_coord    
       
# band 1 ~ 3을 0 ~ 255 값을 갖는 rgb로 변환
def band_to_input(tif_path,bandnumber,partest=False):
    from utils.cfg import Cfg
    import numpy as np
    from sklearn import preprocessing

    raster = gdal.Open(tif_path)

    # transformation of 3-banded SAR image
    if bandnumber==3:
        bands = []
        for i in range(raster.RasterCount):
            band = raster.GetRasterBand(i+1)
            meta = band.GetMetadata()
            if band.GetMinimum() is None or band.GetMaximum()is None:
                band.ComputeStatistics(0)

            band_data = np.array(raster.GetRasterBand(i+1).ReadAsArray())

            max_num = Cfg.max[i]
            min_num = Cfg.min[i]

            # fill nan with neighbors
            mask = np.isnan(band_data)
            band_data[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), band_data[~mask])
            band_data[band_data > max_num] = max_num
            band_data[band_data < min_num] = min_num

            band_data = band_data * ((1 - min_num) / (max_num - min_num))
            #band_data = band_data * ((255 - min_num)/ (max_num - min_num))

            bands.append(band_data)

         # band 1, 2, 3을 RGB로 변환
        rgb = np.dstack((bands[2], bands[1], bands[0]))

    # transformation of single-banded SAR image
    elif bandnumber==1:
        max_num = Cfg.max[0]
        min_num = Cfg.min[0]

        band_data1 = np.array(raster.GetRasterBand(1).ReadAsArray())
        # max_num = np.quantile(band_data1, 0.9, axis=None)
        # band_data1=band_data1/0.8191
        band_data1[band_data1 > max_num] = max_num
        band_data1[band_data1 < min_num] = min_num
        band_data1 = band_data1 * ((1 - min_num) / (max_num - min_num))

        rgb = np.zeros((band_data1.shape[0], band_data1.shape[1], 3))
        rgb = np.dstack((band_data1, band_data1, band_data1))
        #rgb = np.array(rgb, np.uint8)

    # transformation of double-banded SAR image(B1,B2,B2)
    elif bandnumber==2:
        max_num = Cfg.max[0]
        min_num = Cfg.min[0]

        band_data1 = np.array(raster.GetRasterBand(1).ReadAsArray())
        # max_num = np.quantile(band_data1, 0.9, axis=None)
        # band_data1=band_data1/0.8191
        band_data1[band_data1 > max_num] = max_num
        band_data1[band_data1 < min_num] = min_num
        band_data1 = band_data1 * ((1 - min_num) / (max_num - min_num))

        rgb = np.zeros((band_data1.shape[0], band_data1.shape[1], 3))
        rgb[:, :, 0] = band_data1

        # For Band2(Min/Max)
        max_num = Cfg.max[1]
        min_num = Cfg.min[1]

        band_data2 = np.array(raster.GetRasterBand(2).ReadAsArray())
        # max_num = np.quantile(band_data2, 0.9, axis=None)
        # band_data2 = band_data2 / 0.8191
        band_data2[band_data2 > max_num] = max_num
        band_data2[band_data2 < min_num] = min_num
        band_data2 = band_data2 * ((1 - min_num) / (max_num - min_num))

        rgb[:, :, 1] = band_data2
        rgb[:, :, 2] = band_data2

    elif bandnumber > 3:
        for bn in range(bandnumber):
            max_num = Cfg.max[bn]
            min_num = Cfg.min[bn]

            band_data = np.array(raster.GetRasterBand(bn+1).ReadAsArray())
            # max_num = np.quantile(band_data1, 0.9, axis=None)
            # band_data1=band_data1/0.8191
            band_data.clip(min_num, max_num)
            band_data = band_data * ((1 - min_num) / (max_num - min_num))

            rgb = np.zeros((band_data.shape[0], band_data.shape[1], bandnumber))
            rgb[:, :, bn] = band_data

    return rgb


# 외각 라인 검출(육지를 제거하기 위해)
def line_detection(input_array):
    input_image = np.array(input_array, np.uint8)
    # Image.fromarray(input_image*255).save('./milestone/line/grey_10560_0_11200_640_9D5A.png')

    # 비교적 잡음이 적은 band 1 영상에 대해 수행
    gray_image = input_image[:,:,2]
    # Image.fromarray(gray_image).save('./milestone/line/gray_'+save_name.replace('tif','png'))

    blur_image = cv2.medianBlur(gray_image, 5) 
    # Image.fromarray(blur_image).save('./milestone/line/blur_'+save_name.replace('tif','png'))


    # band 1 침식과정을 통해 흰색 노이즈 제거
    erode_image = cv2.erode(blur_image, (3,3), iterations=1)
    # Image.fromarray(erode_image).save('./milestone/line/erode_'+save_name.replace('tif','png'))

    # threshhold
    thr = 15
    ret, thresh = cv2.threshold(erode_image, thr, 1, 0)
    # Image.fromarray(thresh).save('./milestone/line/thres_'+save_name.replace('tif','png'))

    # 육지 정보를 저장할 이미지
    line_filter = np.zeros(input_array.shape[:2], np.uint8)
    
    # 외각 라인 검출
    try:
        ext_contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    except:
        _, ext_contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for c in ext_contours:
        # 각 라인들의 면적
        area = cv2.contourArea(c)
        # 면적이 600 이상일 경우 육지로 판단하고 해당 위치의 픽셀값을 1로
        # 600 미만일 경우 0
        if area >= 1:
            line_filter = cv2.drawContours(line_filter, [c], -1, 1, -1)
    
    return line_filter

# Corresponding function to geotiffread of MATLAB
def geotiffread(tif_name, num_band):
    import gdal
    import numpy as np

    gdal.AllRegister()

    ds = gdal.Open(tif_name)

    if num_band == 3:
        band1 = ds.GetRasterBand(1)
        arr1 = band1.ReadAsArray()
        band2 = ds.GetRasterBand(2)
        arr2 = band2.ReadAsArray()
        band3 = ds.GetRasterBand(3)
        arr3 = band3.ReadAsArray()

        cols, rows = arr1.shape

        arr = np.zeros((cols, rows, 3))
        arr[:, :, 0] = arr1
        arr[:, :, 1] = arr2
        arr[:, :, 2] = arr3

    elif num_band == 1:
        band1 = ds.GetRasterBand(1)
        arr = band1.ReadAsArray()

        cols, rows = arr.shape


    else:
        print('cannot open except number of band is 1 or 3')

    gt = ds.GetGeoTransform()
    gt = np.array(gt)
    gt[2] = gt[0] + gt[1] * (rows - 1)
    gt[4] = gt[3] + gt[5] * (cols - 1)

    gt.astype(np.double)

    return arr, gt

# Median filtering on Oversampled image(Especially K5 0.3m)
def median_filter(img, filter_size=(5,5), stride=1):
    import numpy as np
    from PIL import Image
    import matplotlib.pyplot as plt

    img_shape = np.shape
    result_shape=tuple(np.int64(np.array(img_shape)-np.array(filter_size))/stride+1)

    result=np.zeros(result_shape)
    for h in range(0,result_shape[0],stride):
        for w in range(0,result_shape[1],stride):
            tmp=img[h:h+filter_size[0],w:w+filter_size[1]]
            tmp=np.sort(tmp.ravel())
            result[h,w]=tmp[int(filter_size[0]*filter_size[1]/2)]

    return result

   


    

