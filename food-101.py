# -*- coding: utf-8 -*-
"""
Created on Sat Apr  8 21:38:17 2023

@author: Sevendi Eldrige Rifki Poluan
"""


"""
    Import the necessary libraries
"""

import tensorflow as tf
import os
from sklearn.preprocessing import LabelEncoder 
import matplotlib.pyplot as plt 
import itertools
import numpy as np
import tensorflow_datasets as tfds

"""
    Load and preprocess the dataset in preparation for training and testing.
"""

CLASS_NAMES = os.listdir("food-101/images")
IMG_SIZE = [128, 128]
BATCH_SIZE = 32

training, testing = tf.keras.utils.image_dataset_from_directory("food-101/images",
                                                        class_names=CLASS_NAMES,
                                                        label_mode="int",
                                                        image_size=IMG_SIZE,
                                                        shuffle=True,
                                                        seed=77,
                                                        validation_split=0.2,
                                                        subset="both",
                                                        batch_size=None)
 

def func(img, lbl):
    # img = img * (1. / 255.)
    img = tf.cast(img, dtype=tf.float32)
    return img, lbl

train = training.map(func, tf.data.AUTOTUNE).batch(BATCH_SIZE).prefetch(buffer_size=tf.data.AUTOTUNE).cache() 
test = testing.map(func, tf.data.AUTOTUNE).batch(BATCH_SIZE).prefetch(buffer_size=tf.data.AUTOTUNE).cache()  
  
"""
    In this step, I will design a basic model architecture to train the dataset. I will employ transfer learning by utilizing a pre-trained model from the ImageNet dataset, specifically the ResNet101 model. To do this, I will freeze all the layers of the ResNet101 model up to the last 7 layers, allowing only the last 7 layers to be trainable. Here's how it will be implemented.

        In order to accelerate the training process, I have enabled mixed precision in this environment. This is possible because the GPU I am working on supports this feature, and I will make use of its resources.
        
        It's important to note that when using mixed precision, the output needs to be in tf.float32 format.
"""

tf.keras.mixed_precision.set_global_policy(policy="mixed_float16")  

ResNet101 = tf.keras.applications.ResNet101(include_top=False,
                                              input_shape=IMG_SIZE + [3],
                                              weights='imagenet',
                                              classes=1000)

for layer in ResNet101.layers[:-7]:
    layer.trainable = False

segment = tf.keras.Sequential([
    tf.keras.layers.RandomHeight(0.2),
    tf.keras.layers.RandomWidth(0.2),
    tf.keras.layers.RandomZoom(0.2),
    tf.keras.layers.RandomFlip("horizontal"),
])

inputs = tf.keras.layers.Input(shape=IMG_SIZE + [3])
x = segment(inputs)
x = ResNet101(x)
x = tf.keras.layers.GlobalAveragePooling2D(name="g_avg_pool")(x) 
x = tf.keras.layers.Dense(256, activation="relu")(x)
x = tf.keras.layers.Dropout(0.2)(x)
x = tf.keras.layers.Dense(len(CLASS_NAMES))(x)
outputs = tf.keras.layers.Activation(activation="softmax", dtype=tf.float32)(x)
model_basic = tf.keras.Model(inputs, outputs)

model_basic.summary()

"""
    Define callbacks for updating the learning rate, early stopping if the model does not show improvement, and saving the model if any improvement occurs.
"""

lr_callback = tf.keras.callbacks.ReduceLROnPlateau(monitor="loss", # quantity to be monitored.
                                                   factor=0.2, # factor by which the learning rate will be reduced. new_lr = lr * factor.
                                                   patience=5, # number of epochs with no improvement after which learning rate will be reduced.
                                                   min_lr=1e-7) # lower bound on the learning rate.

es_callback = tf.keras.callbacks.EarlyStopping(monitor="loss",
                                               patience=5, # Number of epochs with no improvement after which training will be stopped.
                                               start_from_epoch=5,
                                               mode="min") # training will stop when the quantity monitored has stopped decreasing

save_callback = tf.keras.callbacks.ModelCheckpoint(filepath="checkpoint/model-basic/ckpt-{epoch:02d}-{loss:.4f}.ckpt",
                                                   monitor="loss",
                                                   save_best_only=True,
                                                   save_weights_only=False)

"""
    Now let's proceed with training the model.
"""

model_basic.compile(loss="sparse_categorical_crossentropy",
                   optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                   metrics=["accuracy"])

model_basic.fit(train,
                epochs=50,
                validation_data=test,
                validation_steps=len(test),
                callbacks=[lr_callback, es_callback, save_callback])
 
