from ultralytics import YOLO

# Load a model
model = YOLO("YOLOv8x.pt")  # load a pretrained model (recommended for training)

# Use the model

results = model(source="cat-segmentation-YOLO\maine-coon-cat-photography-robert-sijka-67-57ad952ba9cac__880.jpg", show=False, conf=0.6, classes=15) # predict on an image

for result in results:
    with open("coordinategatto1.txt","w") as f:
        coordinate = (str(result.boxes.xywh[0]))
        coordinate = coordinate.replace("tensor(", "").replace(")", "")
        coordinate = coordinate.strip('[]')
        f.write(coordinate)
