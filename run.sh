#!/bin/sh
  
#SBATCH -J  OD-CycleGAN_adam_sam                       # 작업 이름
#SBATCH -o  out.train_ODCycleGAN.%j          # stdout 출력 파일 이름 (%j는 %jobId로 확장됨)
#SBATCH -e  err.train_ODCycleGAN.%j          # stderr 출력 파일 이름 (%j는 %jobId로 확장됨)
#SBATCH -p  gpu                             # 큐 또는 파티션 이름
#SBATCH -t  50:30:00                        # 최대 실행 시간 (hh:mm:ss) - 1.5시간
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --gres=gpu:4                        # 요청한 노드당 GPU 수

#module  purge
#module  load  ohpc  cuda/11.8

echo  $SLURM_SUBMIT_HOST
echo  $SLURM_JOB_NODELIST
echo  $SLURM_SUBMIT_DIR

echo  ### START ###

### cuda  test  ###

#/home/mario/cuda-samples/bin/x86_64/linux/release/nbody --benchmark --numdevices=4 -numbodies=10240000
source /home/shhan/.bashrc
conda activate yolo_obb

srun python train_attn.py --cfg poly_yolov8_attn_sam.yaml --resume 'runs/train/exp432_batch16_epoch300_subset_sar_smoothing0.1_multiscaleFalse/weights/best.pt' --img-size 640 --epochs 300 --gpu_ids 0,1,2,3 --data-file sentinel.yaml --hyp hyp.scratch.custom.yaml --polygon --label-smoothing 0.1 --model cycle_gan --dataset_mode aligned --dataroot  ./data/images/sentinel/gan  --batch-size 16 --checkpoints_dir /data/BRIDGE/OD-CycleGAN/checkpoints/sar_cyclegan/ --exp_name sar --fusion mid --adam 

date  ; echo  ##### END #####

