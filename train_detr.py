from ultralytics import RTDETR

device = 2

# Load a model
model = RTDETR('/home/guo/ultralytics/ultralytics/cfg/models/uva_detr/uvadetr-r50.yaml')  # build a new model from YAML
# model = RTDETR("rtdetr-l.pt")  # load a pretrained model (recommended for training)
# model = RTDETR("rtdetr-l.yaml").load("rtdetr-l.pt")  # build from YAML and transfer weights

project = "/home/guo/results/uva_rtdetr/car"

# Train the model
results = model.train(data="/home/guo/ultralytics/ultralytics/cfg/datasets/carclass.yaml", epochs=100, imgsz=640, device=device, project=project, batch=8)