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

'''
Generators are used to pull all training & validation data out from named files,
    where the file acts as the class for the image group. Flow_from_directory() allows
    formatting on-the-fly therefore completing all preprocessing.
'''
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

'''
Predictions method takes a trained model and an input folder to produce a txt output of
    filenames and predictions.
'''
def predictions():
    test_datagen = ImageDataGenerator(rescale=1. / 255)

    test_generator = test_datagen.flow_from_directory(
        "../data/testing/",
        target_size=(128, 128),
        color_mode="grayscale",
        batch_size=32,
        shuffle=False
    )
    # Generator might have data loaded from training, so we reset
    test_generator.reset()

    # Load in model to train from, irrespective if just trained or using old file
    model = load_model('model-v1.h5')

    # Prediction classes are pulled from the model
    pred = model.predict_generator(test_generator, verbose=1)
    predicted_class_indices = np.argmax(pred, axis=1)

    # Labels are numerical values, we need to find the original class names to give meaning to the predictions
    model_labels = (train_data_gen.class_indices)
    model_labels = dict((v, k) for k, v in model_labels.items())

    prediction = [model_labels[k] for k in predicted_class_indices]

    # Pair up prediction and file to produce output in console and in file.
    filenames = test_generator.filenames
    filenames = list(map(os.path.basename, filenames))
    output = pd.DataFrame({"File": filenames, "Prediction": prediction})
    output.to_csv("./run3.txt", index=None, header=False, sep=" ")
    print(output)

# If model already has been trained skip this part, if not, continue
if not path.exists("model-v1.h5"):

    '''
    Model consists of increasing convolution layers of increasing node size to extract features.
    Max pooling to decrease the resolution of the features extracted with dropout included to reduce overfitting.
    
    Final layer is flattened into a 15 node dense layer which corresponds to each prediction class
    '''
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

    model.fit_generator(
        train_data_gen,
        steps_per_epoch=96,
        validation_data=validation_data_gen,
        validation_steps=32,
        epochs=15,
    )

    # File saved for later use
    model.save('model-v1.h5')

    # Produce predictions on test set
    predictions()
else:
    predictions()
