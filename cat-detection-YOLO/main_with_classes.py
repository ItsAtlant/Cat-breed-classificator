from ultralytics import YOLO
import os
import requests
from PIL import Image

# controlla se il modello esiste, in caso contrario lo scarica
if not os.path.isfile("YOLOv8x.pt"):
    URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x.pt"
    response = requests.get(URL)
    open("YOLOv8x.pt", "wb").write(response.content)

# Load a model
model = YOLO("YOLOv8x.pt")  # load a pretrained model (recommended for training)

count = 0
# TODO mettere i commenti e se si riesce pulire un po il codice
for (root, dirs, files) in os.walk('fotogatti', topdown=True):

    root_controllo = root.split("_")[-1]
    if root_controllo != "coordinate" and root_controllo != "fotogatti":
        nome_cartella = root_controllo.split("\\")[-1]
        with open("Tabella_razza_nome.txt", "a") as f:
            # TODO controllare se la riga che si mette gia` esiste per evitare di riscrivere
            f.write(str(count) + "," + nome_cartella + "\n")

        for immagine_name in files:
            # controllo se non ho gia` una cartella con le coordinate, nel caso la creo
            if not os.path.isdir(f"{root}_coordinate"):
                os.mkdir(f"{root}_coordinate")
            # resize dell'immagine
            img = Image.open(f"{root}/{immagine_name}", mode="r")
            output = img.resize((420, 420))
            output.save(f"{root}\\resized-{immagine_name}")
            # faccio la prediction usando il modello
            results = model(source=f"{root}/{immagine_name}", show=False,
                            conf=0.6, classes=15)  # predict on an image

            for result in results:
                immagine2_name = immagine_name.split(".")[0]  # prendo il nome dell'immagine_name
                try:
                    with open(f"{root}_coordinate/{immagine2_name}.txt",
                              "w") as f:  # apro il file di testo per le coodinate
                        coordinate = (str(result.boxes.xywh[0]))  # prendo le informazione , x,y,w,h
                        coordinate = coordinate.replace("tensor(", "").replace(")", "")
                        coordinate = coordinate.strip('[]')  # lavoro la stringa
                        f.write(str(count) + "," + coordinate)

                except:
                    os.remove(f"{root}\\{immagine_name}")

            os.remove(f"{root}\\resized-{immagine_name}")
            # C:\Users\david\Desktop\programming\Cat-breed-classificator\cat-detection-YOLO\fotogatti\American Shorthair\American-Shorthair-1000x600.png
