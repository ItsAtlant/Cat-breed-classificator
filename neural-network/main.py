import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import layers
import os
from PIL import Image

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def build_detector(input_shape=(420, 420, 3), num_classes=35):
    model = tf.keras.Sequential()

    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D(2, 2))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D(2, 2))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D(2, 2))

    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(num_classes + 4, activation='sigmoid'))

    return model


class DataLoader():
    def __init__(self, data_dir, batch_size):
        self.data_dir = data_dir
        self.batch_size = batch_size

        self.breed_dirs = [d for d in os.listdir(data_dir) if not d.endswith('_coordinate')]
        self.image_files = []
        for breed in self.breed_dirs:
            for file in os.listdir(os.path.join(data_dir, breed)):
                if file.endswith('.jpg'):
                    self.image_files.append(os.path.join(breed, file))

        self.current_idx = 0

    def __len__(self):
        return (len(self.image_files) + self.batch_size - 1) // self.batch_size

    def __getitem__(self, idx):
        start_idx = idx * self.batch_size
        end_idx = min(start_idx + self.batch_size, len(self.image_files))

        batch_images = []
        batch_labels = []
        batch_bboxes = []

        for i in range(start_idx, end_idx):
            img_path = self.image_files[i]
            breed = os.path.dirname(img_path)
            img_name = os.path.basename(img_path)
            txt_name = img_name.replace('.jpg', '.txt')

            img = Image.open(os.path.join(self.data_dir, img_path))
            img = np.array(img.resize((420, 420))) / 255.0
            batch_images.append(img)

            with open(os.path.join(self.data_dir, breed + '_coordinate', txt_name), 'r') as file:
                lines = file.readlines()
                for line in lines:
                    class_id, x, y, w, h = map(float, line.split(','))
                    batch_labels.append(class_id)
                    batch_bboxes.append([x, y, w, h])

        return np.array(batch_images), [np.array(batch_labels), np.array(batch_bboxes)]

    def __iter__(self):
        self.current_idx = 0
        return self

    def __next__(self):
        if self.current_idx >= len(self):
            raise StopIteration
        batch = self[self.current_idx]
        self.current_idx += 1
        return batch


model = build_detector()
model.compile(optimizer='adam', loss=['categorical_crossentropy', 'mse'], metrics=['accuracy'])

train_data = DataLoader(data_dir=r"C:\Users\david\Desktop\programming\Cat-breed-classificator\cat-detection-YOLO\fotogatti", batch_size=32)
history = model.fit(train_data, epochs=5)

model.save('model.keras')


def predict_and_draw(model, image_path, threshold=0.5):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    original_img = img.copy()
    img = cv2.resize(img, (420, 420)) / 255.0
    predictions = model.predict(np.expand_dims(img, axis=0))

    class_probs = predictions[0][0]
    bbox_coords = predictions[1][0] * 420

    predicted_class = np.argmax(class_probs)
    if class_probs[predicted_class] > threshold:
        x, y, w, h = bbox_coords
        top_left = (int(x - w / 2), int(y - h / 2))
        bottom_right = (int(x + w / 2), int(y + h / 2))
        color = (0, 255, 0)
        cv2.rectangle(original_img, top_left, bottom_right, color, 2)
        cv2.putText(original_img, f"Class: {predicted_class}", (top_left[0], top_left[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.imshow('Prediction', original_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


image_path = "provagatto.jpg"
predict_and_draw(model, image_path)