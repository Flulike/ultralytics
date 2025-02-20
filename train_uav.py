import warnings
import os
from pathlib import Path
from ultralytics import RTDETR
import torch

warnings.filterwarnings('ignore')


def check_path(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Path does not exist: {path}")


if __name__ == '__main__':
    torch.cuda.empty_cache()
    # 获取当前脚本所在的目录
    current_dir = Path(__file__).parent
    # 构建相对路径
    yaml_path = "/home/guo/ultralytics/ultralytics/cfg/datasets/_visdrone.yaml"
    check_path(yaml_path)
    model = RTDETR('ultralytics/cfg/models/uav-detr/uavdetr.yaml')
    model.train(data=str(yaml_path),
                cache=False,
                imgsz=640,
                epochs=36,
                batch=4,
                workers=8,
                device='3',
                # resume='', # last.pt path
                project='/home/guo/ultralytics/results/ultralytics/uavdetr',
                name='exp',
                patience = 40,
                )