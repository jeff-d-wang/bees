#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import shutil


# In[5]:


DIR = os.getcwd()

DATA = os.path.join(DIR, 'data')

IMG_PATH = os.path.join(DATA, 'bee_imgs')

print(IMG_PATH)


# In[6]:


data = pd.read_csv(os.path.join(DATA, 'bee_data.csv'))


# In[7]:


#print(data)

health_hash = {}

counter = 0
for i in data.health.unique():
    health_hash[i] = counter
    counter += 1

print(health_hash)
    
print(data[['file', 'health']])


# In[8]:


#print(list(data['health'].values))

y = [health_hash[x] for x in list(data['health'].values) if type(x) is str]

len(data.columns)

data.insert(len(data.columns)-1, "health_hash", y)

print(data)


# In[13]:


healthy_counter = 0
distressed_counter = 0

train_dir = os.path.join(IMG_PATH, 'train')
validation_dir = os.path.join(IMG_PATH, 'validation')

print(os.path.join(validation_dir, 'distressed'))

#for index, row in data.iterrows():
    #print(row['file'], row['health_hash'])
    #if(row['health_hash'] != 1):
        #if (distressed_counter % 5 == 0):
            #shutil.copy(os.path.join(IMG_PATH, row['file']), os.path.join(os.path.join(validation_dir, 'distressed'), row['file']))
        #else:
            #shutil.copy(os.path.join(IMG_PATH, row['file']), os.path.join(os.path.join(train_dir, 'distressed'), row['file']))
        #distressed_counter += 1
        #print(distressed_counter)
    #else:
        #if (healthy_counter % 5 == 0):
            #shutil.copy(os.path.join(IMG_PATH, row['file']), os.path.join(os.path.join(validation_dir, 'healthy'), row['file']))
        #else:
            #shutil.copy(os.path.join(IMG_PATH, row['file']), os.path.join(os.path.join(train_dir, 'healthy'), row['file']))
        #healthy_counter += 1
        #print(healthy_counter)


# In[14]:


train_healthy_dir = os.path.join(train_dir, 'healthy')  # directory with our training healthy pictures
train_distressed_dir = os.path.join(train_dir, 'distressed')  # directory with our training distressed pictures
validation_healthy_dir = os.path.join(validation_dir, 'healthy')  # directory with our validation healthy pictures
validation_distressed_dir = os.path.join(validation_dir, 'distressed')  # directory with our validation distressed pictures

num_hea_tr = len(os.listdir(train_healthy_dir))
num_dis_tr = len(os.listdir(train_distressed_dir))
num_hea_val = len(os.listdir(validation_healthy_dir))
num_dis_val = len(os.listdir(validation_distressed_dir))

total_train = num_dis_tr + num_hea_tr
total_val = num_dis_val + num_hea_val

print(total_train)
print(total_val)

print(validation_distressed_dir)


# In[15]:


batch_size = 128
epochs = 11
IMG_HEIGHT = 150
IMG_WIDTH = 150


# In[16]:


train_image_generator = ImageDataGenerator(rescale=1./255) # Generator for our training data
validation_image_generator = ImageDataGenerator(rescale=1./255) # Generator for our validation data


# In[17]:


train_data_gen = train_image_generator.flow_from_directory(batch_size=batch_size,
                                                           directory=train_dir,
                                                           shuffle=True,
                                                           target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                           class_mode='binary')


# In[18]:


val_data_gen = validation_image_generator.flow_from_directory(batch_size=batch_size,
                                                              directory=validation_dir,
                                                              target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                              class_mode='binary')


# In[19]:


sample_training_images, _ = next(train_data_gen)


# In[20]:


# This function will plot images in the form of a grid with 1 row and 5 columns where images are placed in each column.
def plotImages(images_arr):
    fig, axes = plt.subplots(1, 5, figsize=(20,20))
    axes = axes.flatten()
    for img, ax in zip( images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()


# In[21]:


image_gen_train = ImageDataGenerator(
                    rescale=1./255,
                    rotation_range=45,
                    width_shift_range=.15,
                    height_shift_range=.15,
                    horizontal_flip=True,
                    zoom_range=0.5
                    )


# In[22]:


train_data_gen = image_gen_train.flow_from_directory(batch_size=batch_size,
                                                     directory=train_dir,
                                                     shuffle=True,
                                                     target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                     class_mode='binary')


# In[23]:


augmented_images = [train_data_gen[0][0][0] for i in range(5)]
plotImages(augmented_images)


# In[24]:


image_gen_val = ImageDataGenerator(rescale=1./255)


# In[25]:


val_data_gen = image_gen_val.flow_from_directory(batch_size=batch_size,
                                                 directory=validation_dir,
                                                 target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                 class_mode='binary')


# In[26]:


model_new = Sequential([
    Conv2D(16, 3, padding='same', activation='relu', 
           input_shape=(IMG_HEIGHT, IMG_WIDTH ,3)),
    MaxPooling2D(),
    Dropout(0.2),
    Conv2D(32, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Conv2D(64, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Dropout(0.2),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(1, activation='sigmoid')
])


# In[27]:


model_new.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model_new.summary()


# In[28]:


history = model_new.fit_generator(
    train_data_gen,
    steps_per_epoch=total_train // batch_size,
    epochs=epochs,
    validation_data=val_data_gen,
    validation_steps=total_val // batch_size
)


# In[ ]:


acc = history.history['acc']
val_acc = history.history['val_acc']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()


# In[ ]:


nero = [train_data_gen[0][0][0] for i in range(5)]
#print(nero)
plotImages(nero)


# In[ ]:


giga = model_new.predict(val_data_gen)


# In[ ]:


print(giga)


# In[ ]:




