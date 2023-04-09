# -*- coding: utf-8 -*-
"""
Created on Sun Apr  9 20:14:08 2023 

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
import time

"""
    Load and preprocess the dataset in preparation for training and testing.
"""

CLASS_NAMES = os.listdir("food-101/images")
IMG_SIZE = [224, 224]
BATCH_SIZE = 16

training = tf.keras.preprocessing.image_dataset_from_directory("food-101/images",
                                                       class_names=CLASS_NAMES,
                                                       label_mode="int",
                                                       image_size=IMG_SIZE,
                                                       shuffle=True,
                                                       seed=42,
                                                       validation_split=0.25,
                                                       subset="training",
                                                       batch_size=None) 
validation = tf.keras.preprocessing.image_dataset_from_directory("food-101/images",
                                                       class_names=CLASS_NAMES,
                                                       label_mode="int",
                                                       image_size=IMG_SIZE,
                                                       shuffle=True,
                                                       seed=42,
                                                       validation_split=0.25,
                                                       subset="validation",
                                                       batch_size=None) 
 

def func(img, lbl):
    # img = img * (1. / 255.)
    # img = tf.image.resize(img, IMG_SIZE)
    img = tf.cast(img, dtype=tf.float32)
    return img, lbl

train = training.map(func, tf.data.AUTOTUNE).batch(BATCH_SIZE).prefetch(buffer_size=tf.data.AUTOTUNE).cache() 
test = validation.map(func, tf.data.AUTOTUNE).batch(BATCH_SIZE).prefetch(buffer_size=tf.data.AUTOTUNE)
  
"""
    In this step, I will design a basic model architecture to train the dataset. I will employ transfer learning by utilizing a pre-trained model from the ImageNet dataset, specifically the ResNet101 model. To do this, I will freeze all the layers of the ResNet101 model up to the last 7 layers, allowing only the last 7 layers to be trainable. Here's how it will be implemented.

        In order to accelerate the training process, I have enabled mixed precision in this environment. This is possible because the GPU I am working on supports this feature, and I will make use of its resources.
        
        It's important to note that when using mixed precision, the output needs to be in tf.float32 format.
""" 
  
def create_model():
    EfficientNet = tf.keras.applications.EfficientNetB0(include_top=False)
    
    for layer in EfficientNet.layers[:-7]:
        layer.trainable = False
        
    segment = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.2),
        tf.keras.layers.RandomHeight(0.2),
        tf.keras.layers.RandomWidth(0.2),
        tf.keras.layers.RandomZoom(0.2)
    ])
    
    inputs = tf.keras.layers.Input(shape=IMG_SIZE + [3])
    x = segment(inputs)
    x = EfficientNet(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D(name="g_avg_pool")(x)  
    x = tf.keras.layers.Dense(len(CLASS_NAMES))(x)
    outputs = tf.keras.layers.Activation(activation="softmax", dtype=tf.float32)(x)
    model_basic = tf.keras.Model(inputs, outputs)
    
    model_basic.compile(loss="sparse_categorical_crossentropy",
                   optimizer=tf.keras.optimizers.Adam(),
                   metrics=["accuracy"])
    
    return model_basic

model_basic = create_model()
model_basic.summary() 
 

"""
    Define callbacks for updating the learning rate and early stopping if the model does not show improvement.
    
        In this case, I have designed a custom function for saving weights to address some errors encountered in the original TensorFlow library that require fixing.
"""

class CustomLearningRateScheduler(tf.keras.callbacks.Callback):
    def __init__(self, factor, patience, min_lr):
        super(CustomLearningRateScheduler, self).__init__()
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.wait = 0
        self.best_val_loss = float('inf')

    def on_epoch_end(self, epoch, logs=None):
        
        # print('\rlr_logs: ', logs, '\r')
        print("\rlr_wait: ", self.wait, "\r")
        print("\rlr_current:", tf.keras.backend.get_value(self.model.optimizer.lr), "\r")
        
        current_val_loss = logs['val_loss'] 
        
        if current_val_loss < self.best_val_loss:
            self.best_val_loss = current_val_loss
            self.wait = 0
        else:
            self.wait += 1
            print("\rself.wait >= self.patience:", self.wait >= self.patience, "\r")
            if self.wait >= self.patience:
                old_lr = tf.keras.backend.get_value(self.model.optimizer.lr)
                new_lr = old_lr * self.factor
                
                print("\rnew_lr >= self.min_lr:", new_lr >= self.min_lr, " (updating ...)\r")
                if new_lr >= self.min_lr:
                    print(f'\rEpoch {epoch+1}: Learning rate reduced to {new_lr} \r')
                    tf.keras.backend.set_value(self.model.optimizer.lr, new_lr) 
                else:
                    print(f'\rEpoch {epoch+1}: Minimum learning rate reached\r')
                self.wait = 0
                
lr_callback = CustomLearningRateScheduler(factor=0.1, 
                                          patience=3,
                                          min_lr=1e-7) 

es_callback = tf.keras.callbacks.EarlyStopping(monitor="loss",
                                               patience=1, # Number of epochs with no improvement after which training will be stopped.
                                               start_from_epoch=1,
                                               mode="min") # training will stop when the quantity monitored has stopped decreasing

class ModelCheckpointCustom(tf.keras.callbacks.Callback):
    
    def __init__(self, model_path, save_best_only=False):
        super(ModelCheckpointCustom, self).__init__()
         
        self.model_path = model_path
        self.save_best_only = save_best_only
        self.best_val_loss = float('inf') 

    def on_epoch_end(self, epoch, logs=None):
        
        # print('\rlogs: ', logs, '\r')
        
        current_val_loss = 0 if logs['val_loss'] is None else logs['val_loss']
        
        # print('\rcurrent_val_loss: ', current_val_loss, '\r')
        # print('\rsave_best_only: ', self.save_best_only, '\r')
        # print('\rbest_val_loss: ', self.best_val_loss, '\r')
        # print('\rcurrent_val_loss < best_val_loss: ', current_val_loss < self.best_val_loss, '\r') 
        
        if self.save_best_only and current_val_loss < self.best_val_loss:
            print('\rSaving weights at (save_best_only=True): ', self.model_path, '\r')
            self.best_val_loss = current_val_loss 
            self.model.save_weights(self.model_path)
            print("\rModel has been saved!\r")
        else: 
            print("\rNot improving! Don't save it!\r")

save_callbacks = ModelCheckpointCustom(model_path='checkpoint/model-basic/', save_best_only=True) 

"""
    Now let's proceed with training the model.     
"""

start_time = time.perf_counter()
model_basic.fit(train,
                epochs=100,
                validation_data=test,
                validation_steps=len(test) * 0.25,
                callbacks=[save_callbacks, lr_callback, es_callback])

end_time = time.perf_counter()
print("Training time: ", end_time - start_time, 'seconds')

 
