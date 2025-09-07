# CiSeg: Unsupervised Cross-Modality Adaptation for 3D Medical Image Segmentation via Causal Intervention



## 1. Data Preparation



- **MMWHS dataset**：[多模态全心分割挑战](https://zmiclab.github.io/zxh/0/mmwhs/)

- **Abdominal CT**: https://www.synapse.org/#!Synapse:syn3193805/wiki/217789
- **CHAOS MRI:**[Description - CHAOS - Grand Challenge](https://chaos.grand-challenge.org/Combined_Healthy_Abdominal_Organ_Segmentation/)

Our preprocessed data and weights are available at the following link.

[google](https://drive.google.com/drive/folders/1cjvVKq2lbjv4xkWZatKxNEpzrSd4LE96?usp=drive_link)

[Baidu Netdisk](通过网盘分享的文件：cardiac_abdominal
链接: https://pan.baidu.com/s/1yIainEESJZnIiPLHm4u16Q 提取码: 3bxh 
--来自百度网盘超级会员v6的分享)

[deeplabv2 pre-training weights](https://drive.google.com/drive/folders/1UFqj18A4vuoknldoqAkg9tx7S6CUjxRL)

The file structure should be:

```
  data
     cardiac
          tr_ct
           image
             ct_train_1001_image.nii.gz
             ....  
           label
             ct_train_1001_label.nii.gz
             ...
          tr_ct_slice
             ct_train_1001_image_slice_0.h5
             ct_train_1001_image_slice_1.h5
             ...
          tr_mr
            image
             mr_train_1001_image.nii.gz
             ... 
            label
             mr_train_1001_label.nii.gz
             ...
           tr_ct_slice
             mr_train_1001_image_slice_0.h5
             mr_train_1001_image_slice_1.h5
             ...
          val_ct
             image
               ...
             label
               ...
          val_ct_slice
             ct_train_1003_image_norm.h5
             ...
          val_mr
             image
               ...
             label
               ...
           val_mr_slice
             mr_train_1007_image_norm.h5
             ...
             
          tr_ct_slices.list
          tr_mr_slices.list
          val_ct_slices.list
          val_mr_slices.list
            
     abdominal
          tr_ct
           image
           label
          tr_ct_slice
          
          tr_mr
            image
            label
          tr_ct_slice
             
          val_ct
             image
             label 
          val_ct_slice
             
          val_mr
             image 
             label        
          val_mr_slice

          tr_ct_slices.list
          tr_mr_slices.list
          val_ct_slices.list
          val_mr_slices.list
```

### 1.1 Pre-processing the MMWHS dataset

```
python dataloaders/pre_mmwhs.py
python dataloaders/nii2h5.py
```

Or please refer to [GenericSSL](https://github.com/xmed-lab/GenericSSL/blob/main/code/data/preprocess_mmwhs.py).

### 1.2 Pre-processing the Abdominal dataset

```
python dataloaders/pre_abdominal.py
python dataloaders/nii2h5.py
```

## 2. Training/Testing (Full supervision)

```
eg：CT (Cardiac Dataset)
python train_fully_supervised.py
python test.py
```

## 3. Training/Testing (Ours)

```
eg：MRI2CT (Cardiac Dataset)
python train_CiSeg.py
python test_CiSeg.py
```

## 4. Other methods

[AdaOutput](https://github.com/wasidennis/AdaptSegNet)、[AdvEnt](https://github.com/valeoai/ADVENT)、 [CycleGAN](https://github.com/junyanz/CycleGAN)、 [SAFAv2](https://github.com/cchen-cc/SIFA#readme)、[MPSCL](https://github.com/TFboys-lzz/MPSCL)、

 [SE-ASA](https://github.com/fengweie/SE_ASA)、 [CPCL](https://cvlab.yonsei.ac.kr/projects/DASS/)、[MIC](https://github.com/lhoyer/MIC)、 [ASC](https://github.com/zihang-xu/ASC)、[MAAL](https://github.com/M4cheal/MAAL)、[DCLPS](https://github.com/taozh2017/DCLPS)

## 5. Acknowledgments

This code is mainly based on [SSL4MIS](https://github.com/HiLab-git/SSL4MIS) and [Learning-Debiased-Disentangled](https://github.com/kakaoenterprise/Learning-Debiased-Disentangled).

