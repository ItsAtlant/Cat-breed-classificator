from ultralytics import YOLO
import os
import requests


#controlla se il modello esiste, in caso contrario lo scarica
if not os.path.isfile("YOLOv8x.pt"):
    URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x.pt"
    response = requests.get(URL)
    open("YOLOv8x.pt", "wb").write(response.content)


# Load a model
model = YOLO("YOLOv8x.pt")  # load a pretrained model (recommended for training)

count = 0
#TODO mettere i commenti e se si riesce pulire un po il codice
for (root, dirs, files) in os.walk('fotogatti', topdown=True):

    root_controllo = root.split("_")[-1]
    if root_controllo != "coordinate" and root_controllo != "fotogatti":
        nome_cartella = root_controllo.split("\\")[-1]
        with open("Tabella_razza_nome.txt","a") as f:
            #TODO controllare se la riga che si mette gia` esiste per evitare di riscrivere
            f.write(str(count)+","+nome_cartella+"\n")


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
                    f.write(str(count)+","+coordinate)
                    count = +1
