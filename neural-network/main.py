from keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense
from keras.models import Model

# Definisci l'input con shape flessibile (None, None, 3) per immagini RGB
input_img = Input(shape=(None, None, 3))

# Aggiungi uno o più layer convoluzionali e di pooling
x = Conv2D(64, (3, 3), activation='relu')(input_img)
x = MaxPooling2D((2, 2))(x)
# Aggiungi altri layer convoluzionali e di pooling se necessario

# Aggiungi uno strato di pooling globale
x = GlobalAveragePooling2D()(x)

# Aggiungi uno strato completamente connesso con 20 unità di output per le classi
output = Dense(20, activation='softmax')(x)

# Crea il modello
model = Model(inputs=input_img, outputs=output)

# Compila il modello
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Stampa una sintesi del modello
model.summary()