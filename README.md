# SAR-EdgeYOLO: Robust Bridge Detection in Low-Resolution SAR via Edge Enhancement (IEEE Access)

[Shinhye Han](https://gkstlsgp3as.github.io/), [Juyoung Song](https://www.researchgate.net/profile/Juyoung-Song-3), Hwisong Kim, Duk-jin Kim

---
Bridge detection from Synthetic Aperture Radar (SAR) is of great significance for infrastructure management, disaster prevention, and navigation automation. Although high-resolution SAR is increasingly accessible, exploiting Sentinel-1 remains advantageous due to its global coverage and accessibility. However, low spatial resolution of 20 m in Sentinel-1 products poses challenges in detecting small and indistinct bridges. To enhance bridge detection in low-resolution Sentinel-1 imagery, we propose a novel architecture that incorporates CycleGAN as an edge detector. CycleGAN generates detailed boundaries using 5 m land use maps, with enhanced bridge saliency. Since bridges often connect distinct edges like riverbanks and roads, this edge information assists Poly-YOLOv8, our chosen detector, in accurately localizing bridges. Our approach then integrates edge information through feature fusion module and feature alignment loss. Accordingly, the proposed SAR-EdgeYOLO achieves a precision of 98.2%, recall of 91.7%, and mAP of 94.8% at IoU 0.5, marking improvements of 3.9, 3.8, and 3.1 percentage points respectively over the baseline. The results demonstrate that CycleGAN-aided edge extraction effectively addresses the limitations of low-resolution remote sensing data, improving bridge detection accuracy and multi-class localization. This research can further contribute to the advancement of time-series infrastructure monitoring with wide applicability and higher accuracy.

<img src = "https://github.com/user-attachments/assets/61d9c557-e2de-4f2e-b230-9f1d49f051cb" width="90%" height="90%" img align="center">
<img src = "https://github.com/user-attachments/assets/97c00a05-ccb5-4bb7-ae24-e388aa37eebf" width="90%" height="90%" img align="center">

## Requirements
* Python 3.9.19, Pytorch 1.11.0+cu113
* More detail (See [requirements.txt](requirements.txt))

```
conda create -n sardet python=3.9
conda activate sardet
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113

pip install -r requirements_4dsr.txt
```

### :point_right: Results
The aerial and Sentinel-1 images along with the detection results of challenging cases. Red boxes indicate train bridges, and green boxes specify vehicle bridges. 

<img src = "https://github.com/user-attachments/assets/b882449a-f320-481f-b561-123d39ff9c21" width="60%" height="60%"  img align="center">

### :airplane: Training
```
python -m torch.distributed.launch --nproc_per_node 1 --master_port 23330 train_fusion.py --cfg ./cfg/training/yolov8_attn_fus4_3x3.yaml --weights '' --batch-size 16 --img-size 640 --epochs 300 --data-file sentinel.yaml --hyp hyp.scratch.custom.yaml --adam --label-smoothing 0.1 --mode sar
```

### 🛩️ Testing
```
python test.py --weights ./runs/train/exp86_batch16_epoch300_imgsize640_lr0.001_lrf0.1_smoothing0.1_multiscaleFalse_wind+org+v5/weights/last.pt --data sentinel_wind.yaml --img-size 640 --iou-thres 0.3 --task valid --device 0 --conf-thres 0.01
```

### :rocket: Inference 
Reproduce the results for bridge detection from low-resolution Sentinel-1:
```
python detect_fusion_v2.py --weights {pretrained model path} --img-size 640  --conf 0.002 --source {input image path} --iou-thres 0.3 --save-txt --no-trace --output-format csv --output-band 3 --polygon --gan-weights {cycleGAN model weight}
```

### 📞: Contact
If you have any questions, please feel free to contact me via `sienna.shhan@gmail.com`.
For details see [SAREdgeYOLO_IEEEAccess_2024.pdf](https://github.com/user-attachments/files/17585284/SAREdgeYOLO_IEEEAccess_2024.pdf) published at IEEE Access in Oct. 2024. 
