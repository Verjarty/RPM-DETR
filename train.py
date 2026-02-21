import os, torch, time, shutil
from ultralytics import RTDETR

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True


def trainFunc(modelYaml,name,dataPth):
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    
    model = RTDETR(r'ultralytics/cfg/models/rt-detr/'+modelYaml)
    if os.path.exists('runs/train/'+name):
        shutil.rmtree('runs/train/'+name)
    model.train(data=dataPth,
        imgsz=640,
        epochs=200,
        batch=6,
        seed=42,
        deterministic=False,
        warmup_epochs=2000,
        patience=20,
        optimizer="AdamW",
        lr0=0.0001,
        lrf=0.25,
        cos_lr=True,
        close_mosaic=80, 
        weight_decay=0.0001,

        hsv_h=0.01,
        hsv_s=0.4,
        hsv_v=0.3,
        degrees=0.0,
        flipud=0.5,
        fliplr=0.5,
        mosaic=0.5,
        mixup=0.5,
        translate=0.05,
        scale=0.5,

        workers=12,
        project='runs/train',
        name=name,
        verbose=True,
        exist_ok=True,
        cache='disk',
    )

if __name__ == '__main__':
    trainList=[
        ['RPM-DETR.yaml','results','dataset.yaml'],
    ]
    for arg in trainList:
        try:
            trainFunc(*arg)
        except Exception as e:
            error_message = str(e)
            print(error_message)