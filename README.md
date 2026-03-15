# Enhanced DETR Framework for Accurate Remote Sensing Object Detection
[![DOI](https://zenodo.org/badge/1162847535.svg)](https://doi.org/10.5281/zenodo.19036498)
> **Important Note:** > This repository contains the official implementation, source code, and dataset guidelines for the manuscript **"Enhanced DETR Framework for Accurate Remote Sensing Object Detection"**, which is currently submitted to ***The Visual Computer***. 
> 
> If you find this code, dataset, or our research helpful in your work, **please consider citing our relevant manuscript**. (Citation details will be updated upon publication).
---
# Usage

## Requirements

```bash
pip install -r requirements.txt
```

## Dataset Preparation

Download the following datasets and convert each to YOLO format:

| Dataset | Download |
|---------|----------|
| LEVIR | [Baidu Pan](http://pan.baidu.com/s/1geTwAVD) |
| HRSID | [GitHub](https://github.com/chaozhong2010/HRSID) |
| RSOD | [GitHub](https://github.com/RSIA-LIESMARS-WHU/RSOD-Dataset-) |

## Training

Edit `train.py` to set model config and dataset path, then run:

```bash
python train.py
```

## Validation

Edit `val.py` to set `model_path` and `data`, then run:

```bash
python val.py
```


