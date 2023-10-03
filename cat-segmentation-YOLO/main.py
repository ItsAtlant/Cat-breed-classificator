from ultralytics import YOLO
import os


if not os.path.isfile("YOLOv8x.pt"):
    #TODO fare il download leggendo il file yolodownload.txt
    ...

# Load a model
model = YOLO("YOLOv8x.pt")  # load a pretrained model (recommended for training)


#TODO mettere i commenti e se si riesce pulire un po il codice
for (root, dirs, files) in os.walk('fotogatti', topdown=True):

    root_controllo = root.split("_")[-1]
    if root_controllo != "coordinate":
        for immagine in files:
            if not os.path.isdir(f"{root}_coordinate"):
                os.mkdir(f"{root}_coordinate")

            results = model(source=f"{root}/{immagine}", show=False,
                            conf=0.6, classes=15)  # predict on an image

            for result in results:
                immagine = immagine.split(".")[0]
                with open(f"{root}_coordinate/{immagine}.txt", "w") as f:
                    coordinate = (str(result.boxes.xywh[0]))
                    coordinate = coordinate.replace("tensor(", "").replace(")", "")
                    coordinate = coordinate.strip('[]')
                    f.write(coordinate)
