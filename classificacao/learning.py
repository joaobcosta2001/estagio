import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
from tensorflow.keras.applications import VGG16


#Este script treinara um modelo com base no VGG16 e recorrendo ao dataset presente em imageDir. Este modelo sera guardado numa pasta chamada "saved_model"


imageDir = "C:\estagio\\neural_networks\Projeto\classificacao\dataset\\fundo_branco_augmented"


num_classes = 3
input_shape = (128,128,3)

print("Importing data...","")
trainingData = tf.keras.preprocessing.image_dataset_from_directory(directory = imageDir,image_size=(224,224), label_mode = 'categorical')
if trainingData == None:
    print("ERROR dataset not imported!")
    exit()
print("Imported!")
base_model = VGG16(weights='imagenet', include_top=False,input_shape = (224,224,3))

for layer in base_model.layers[:14]:
    layer.trainable = False
for layer in base_model.layers[14:]:
    layer.trainable = True

model = keras.models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Flatten(),
    layers.Dropout(0.2),
    layers.Dense(1024,activation='relu'),
    layers.Dense(3,activation = 'softmax'),
    layers.Flatten()
])

batch_size = 128
epochs = 15

model.compile(loss="categorical_crossentropy", optimizer=keras.optimizers.SGD(learning_rate=0.001), metrics=["accuracy"])
print("Model compiled")
model.fit(x=trainingData, batch_size=batch_size, epochs=epochs)

#score = model.evaluate(testingData, verbose=0)
#print("Test loss:", score[0])
#print("Test accuracy:", score[1])


model.save('.\saved_model')