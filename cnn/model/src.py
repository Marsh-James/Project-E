from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Lambda
from sklearn.model_selection import train_test_split
from keras.preprocessing import image
from keras.backend import resize_images
from keras_preprocessing.image import ImageDataGenerator

import os
from PIL import Image
import numpy as np

train_data = []
train_labels = []

for (dirpath, dirnames, filenames) in os.walk("./data/"):
    images = [f for f in filenames if f.endswith(".jpg")]
    for imageFile in images:
        image = Image.open(os.path.join(dirpath, imageFile))
        image.convert("L")
        image.resize((256, 256), Image.ANTIALIAS)
        train_data.append(np.array(image))
        train_labels.append(os.path.basename(dirpath))

train_data = np.array(train_data)
train_labels = np.array(train_labels)
print(train_data)
print(train_labels)

x_train, y_train, x_validation, y_validation = train_test_split(
    train_data,
    train_labels,
    test_size=0.2,
    shuffle=True
)
#datagfen
model = Sequential()
#feedin
model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(256, 256, 1)))
model.add(Conv2D(32, kernel_size=3, activation='relu'))
model.add(Flatten())
model.add(Dense(15, activation='softmax'))

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train,
          validation_data=(x_validation, y_validation),
          epochs=15)