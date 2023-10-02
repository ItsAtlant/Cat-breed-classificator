from ultralytics import YOLO

# Load a model
model = YOLO("YOLOv8x.pt")  # load a pretrained model (recommended for training)

# Use the model

results = model(source="dfa68c08-d00c-4d79-8d40-65f2334439d5.jpg", show=True, conf=0.6, classes=15) # predict on an image

for result in results:
    with open("coordinategatto1.txt","w") as f:
        coordinate = (str(result.boxes.xywh[0]))
        f.write(coordinate)
