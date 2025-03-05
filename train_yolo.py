from ultralytics import YOLO

device = [2]

# Load a model
model = YOLO("yolov8l.yaml")  # build a new model from YAML
model = YOLO("yolov8l.pt")  # load a pretrained model (recommended for training)
model = YOLO("yolov8l.yaml").load("yolov8l.pt")  # build from YAML and transfer weights

project = "/home/guo/ultralytics/results/ultralytics/yolov8/l"

# Train the model
results = model.train(data="/home/guo/ultralytics/ultralytics/cfg/datasets/_visdrone.yaml", epochs=300, imgsz=640, device=device, project=project, batch=12, optimizer='SGD')