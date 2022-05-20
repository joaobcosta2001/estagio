import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
from tensorflow.keras.applications import VGG16

num_classes = 8
input_shape = (128,128,3)

print("Importing data...","")
trainingData = tf.keras.preprocessing.image_dataset_from_directory(directory = "C:\estagio\\neural_networks\Projeto\modelo\\train",image_size=(224,224), label_mode = 'binary')
if trainingData == None:
    print("ERROR dataset not imported!")
print("Imported!")
base_model = VGG16(weights='imagenet', include_top=False)

for layer in base_model.layers[:14]:
    layer.trainable = False
for layer in base_model.layers[14:]:
    layer.trainable = True

model = keras.models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Flatten(),
    layers.Dense(1024,activation='relu'),
    layers.Dense(2,activation = 'softmax'),
    layers.Flatten()
])
model.summary()

batch_size = 128
epochs = 15

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
print("Model compiled")
model.fit(x=trainingData, batch_size=batch_size, epochs=epochs)

#score = model.evaluate(testingData, verbose=0)
#print("Test loss:", score[0])
#print("Test accuracy:", score[1])


model.save('.\modelo')