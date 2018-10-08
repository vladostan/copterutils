
# coding: utf-8

# In[1]:
import os
import matplotlib.pylab as plt
from glob import glob
import numpy as np
import datetime

# Get the date and time
now = datetime.datetime.now()
loggername = str(now).split(".")[0];

# Print stdout to file
import sys

class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout
        self.log = open('logs/{}.txt'.format(loggername), 'w')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  

    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        #you might want to specify some extra behavior here.
        pass    

#sys.stdout = Logger()

#sys.stdout = open('logs/{}'.format(loggername), 'w')

print('Date and time: {}\n'.format(loggername))

# In[ ]:
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
import tensorflow as tf

data_gen_args = dict(rotation_range = 15,
                     width_shift_range = 0.2,
                     height_shift_range = 0.2,
                     horizontal_flip = True,
                     validation_split = 0.2)

print("Keras augmentation used: {}\n".format(data_gen_args))

image_datagen = ImageDataGenerator(**data_gen_args)
mask_datagen = ImageDataGenerator(**data_gen_args)

seed = 1

image_generator = image_datagen.flow_from_directory(
    '/home/kenny/Desktop/copterutils/seg/data/ffdimages/',
    class_mode = None,
    batch_size = 1,
    seed = seed)

mask_generator = mask_datagen.flow_from_directory(
    '/home/kenny/Desktop/copterutils/seg/data/ffdlabels/',
    class_mode = None,
    batch_size = 1,
    seed = seed)

train_generator = zip(image_generator, mask_generator)

print("Keras image data format: {}\n".format(K.image_data_format()))

# In[ ]:
epochs = 1
batch_size = 1

print("Epochs: {}, batch size: {}\n".format(epochs,batch_size))

data_gen_args = dict(rotation_range = 15,
                     width_shift_range = 0.2,
                     height_shift_range = 0.2,
                     horizontal_flip = True)

print("Keras augmentation used: {}\n".format(data_gen_args))

image_datagen = ImageDataGenerator(**data_gen_args)
mask_datagen = ImageDataGenerator(**data_gen_args)

seed = 1
image_datagen.fit(x_train, augment=True, seed=seed)
mask_datagen.fit(y_train, augment=True, seed=seed)

image_generator = image_datagen.flow(x_train, seed=seed, batch_size=batch_size)
mask_generator = mask_datagen.flow(y_train, seed=seed, batch_size=batch_size)

train_generator = zip(image_generator, mask_generator)

test_datagen = ImageDataGenerator()
validation_generator = test_datagen.flow(x_test, y_test, batch_size=batch_size)

# Configure batch size and retrieve one batch of images
# for x_batch in image_datagen.flow(x_train, seed=seed, batch_size=9):
#     # Show 9 images
#     for i in range(0, 9):
#         plt.subplot(330 + 1 + i)
#         plt.imshow(x_batch[i])
#     # show the plot
#     plt.show()
#     break
    
# for y_batch in mask_datagen.flow(y_train, seed=seed, batch_size=9):
#     # Show 9 images
#     for i in range(0, 9):
#         plt.subplot(330 + 1 + i)
#         plt.imshow(y_batch[i,:,:,6])
#     # show the plot
#     plt.show()
#     break

# for x_batch, y_batch in datagen.flow(x_train, y_train, batch_size=9):
#     # Show 9 images
#     for i in range(0, 9):
#         plt.subplot(330 + 1 + i)
#         plt.imshow(y_batch[i])
#     # show the plot
#     plt.show()
#     break

# In[ ]:
# # Define model
# # U-Net
from models.Unet import unet

model = unet(input_size = x_train.shape[1:], n_classes=n_classes)

print("Model summary:")
model.summary()

# In[ ]:
from keras import optimizers

#model.compile(optimizer = 'adadelta', loss = 'categorical_crossentropy', metrics = ['accuracy'])

learning_rate = 1e-4
optimizer = optimizers.Adam(lr = learning_rate)
loss = 'categorical_crossentropy'
metrics = ['accuracy']

print("Optimizer: {}, learning rate: {}, loss: {}, metrics: {}\n".format(optimizer, learning_rate, loss, metrics))

model.compile(optimizer = optimizer, loss = loss, metrics = metrics)

# In[ ]:

def get_tf_session():
    gpu_options = tf.GPUOptions(allow_growth=True)
    return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
K.set_session(get_tf_session())

# In[ ]:

from keras import callbacks

model_checkpoint = callbacks.ModelCheckpoint('weights/{}.hdf5'.format(loggername), monitor='loss', verbose=1, save_best_only=True, save_weights_only=True)
tensor_board = callbacks.TensorBoard(log_dir='./tblogs')
reduce_lr = callbacks.ReduceLROnPlateau(monitor='loss', factor=0.2,
                              patience=5, verbose = 1, min_lr=1e-5)
csv_logger = callbacks.CSVLogger('logs/{}.log'.format(loggername))

callbacks = [model_checkpoint, tensor_board, reduce_lr, csv_logger]

print("Callbacks: {}\n".format(callbacks))

# In[ ]:


steps_per_epoch = len(x_train)//batch_size*2 #Twice dataset for training (leads more keras augmentation)
validation_steps = len(x_test)//batch_size

print("Steps per epoch: {}".format(steps_per_epoch))
print("Validation steps: {}\n".format(validation_steps))

print("Starting training...\n")
history = model.fit_generator(
    train_generator,
    validation_data = validation_generator,
    steps_per_epoch = steps_per_epoch,
    validation_steps = validation_steps,
    epochs = epochs,
    verbose = 1,
    class_weight = class_weighting,
    callbacks = callbacks
)
print("Finished training\n")
