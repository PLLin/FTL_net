# FTLNet_Pytorch

Pytorch codes for Feature Transfer Learning for Face Recognition with Under-Represented Data 

------

## 1. Introduction

- This repo is an unofficial implementation of Feature Transfer Learning for Face Recognition with Under-Represented Data [(paper)](http://cvlab.cse.msu.edu/pdfs/Yin_Yu_Sohn_Liu_Chandraker_CVPR2019.pdf)

------

## 2. Build/Run docker environment 

- Please build up docker environment and do everything in it.

```
  cd ./docker/
  ./build.sh
```

- When docker build is done, run docker to enter docker shell by

```
  cd ./docker/
  ./run.sh
```

------

## 3. Prepare Train and Verification Datasets

- download the refined emore dataset from [InsightFace_Pytorch](https://github.com/TreB1eN/InsightFace_Pytorch)

- after unzip the files to 'data' path, run :

```
  python prepare_data.py
```

------

## 4. Train

- Train for pretrained model

```
  python train_pre.py
```

- Train with Feature Transfer Learning

```
  python train_ftl.py
```

------

## 5. References 

- [deepinsight/insightface](https://github.com/deepinsight/insightface) 
- [InsightFace_Pytorch](https://github.com/TreB1eN/InsightFace_Pytorch)
