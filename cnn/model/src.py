from os import path

from keras.engine.saving import load_model
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Lambda, MaxPooling2D, Dropout
from sklearn.model_selection import train_test_split
from keras.preprocessing import image
from keras.backend import resize_images
from keras_preprocessing.image import ImageDataGenerator
import pandas as pd

import os
from PIL import Image
import numpy as np

# train_data = []
# train_labels = []

# for (dirpath, dirnames, filenames) in os.walk("../data/"):
#     images = [f for f in filenames if f.endswith(".jpg")]
#     for imageFile in images:
#         image = Image.open(os.path.join(dirpath, imageFile))
#         image.convert("L")
#         image.resize((256, 256), Image.ANTIALIAS)
#         train_data.append(np.array(image))
#         train_labels.append(os.path.basename(dirpath))

# train_data = np.array(train_data)
# train_labels = np.array(train_labels)
# print(train_data)
# print(train_labels)

# x_train, y_train, x_validation, y_validation = train_test_split(
#     train_data,
#     train_labels,
#     test_size=0.2,
#     shuffle=True
# )
# datagfen

train_datagenerator = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2)

train_data_gen = train_datagenerator.flow_from_directory(
    "../data/training/",
    target_size=(128,128),
    color_mode="grayscale",
    batch_size=32,
    subset="training")


validation_data_gen = train_datagenerator.flow_from_directory(
    "../data/training/",
    target_size=(128,128),
    color_mode="grayscale",
    batch_size=32,
    subset="validation")


def predictions():
    test_datagen = ImageDataGenerator(rescale=1. / 255)

    test_generator = test_datagen.flow_from_directory(
        "../data/testing/",
        target_size=(128, 128),
        color_mode="grayscale",
        batch_size=32,
        shuffle=False
    )

    test_generator.reset()

    model = load_model('model-v1.h5')

    pred = model.predict_generator(test_generator, verbose=1)
    predicted_class_indices = np.argmax(pred, axis=1)

    labels = (train_data_gen.class_indices)
    labels = dict((v, k) for k, v in labels.items())
    predictions = [labels[k] for k in predicted_class_indices]

    filenames = test_generator.filenames
    results = pd.DataFrame({"Filename": filenames,
                            "Predictions": predictions})
    results.to_csv("./predictions.csv", index=None, header=False)
    print(results)


if not path.exists("model-v1.h5"):

    model = Sequential()
    model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(128, 128, 1)))
    model.add(MaxPooling2D((2,2)))
    model.add(Conv2D(128, kernel_size=3, activation='relu'))
    model.add(MaxPooling2D((2,2)))
    model.add(Conv2D(256, kernel_size=3, activation='relu'))
    model.add(MaxPooling2D((2,2)))
    model.add(Dropout(0.2))
    model.add(Conv2D(512, kernel_size=3, activation='relu'))
    model.add(MaxPooling2D((2,2)))
    model.add(Dropout(0.3))
    model.add(Conv2D(768, kernel_size=3, activation='relu'))
    model.add(MaxPooling2D((2,2)))
    model.add(Dropout(0.4))
    model.add(Flatten())
    model.add(Dense(15, activation='softmax'))

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # model.fit(x_train, y_train,
    #           validation_data=(x_validation, y_validation),
    #           epochs=15)

    model.fit_generator(
        train_data_gen,
        steps_per_epoch=96,
        validation_data=validation_data_gen,
        validation_steps=32,
        epochs=15,
    )

    model.save('model-v1.h5')
    predictions()
else:
    predictions()
