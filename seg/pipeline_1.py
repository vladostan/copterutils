
# coding: utf-8

# In[20]:

import os
import matplotlib.pylab as plt
from glob import glob
import numpy as np


# # READ IMAGES AND MASKS

# In[21]:

PATH = os.path.abspath('data')

SOURCE_IMAGES = os.path.join(PATH, "images/resized")

images = glob(os.path.join(SOURCE_IMAGES, "*.jpg"))

SOURCE_LABELS = os.path.join(PATH, "labels/resized")

labels = glob(os.path.join(SOURCE_LABELS, "*.png"))


# In[23]:

images.sort()
labels.sort()


# In[24]:

print("Images num: {}".format(np.size(images)))
print("Labels num: {}".format(np.size(labels)))


# In[25]:

x = [] # images
y = [] # labels

for img in images:
    full_size_image = plt.imread(img)
    x.append(full_size_image)

for lbl in labels:
    full_size_label = plt.imread(lbl,0)
    y.append(full_size_label)

del(images, labels, full_size_image, full_size_label)

# In[26]:

x = np.asarray(x)
y = np.asarray(y)

# ### Class weighting

# In[29]:

n = np.bincount(y.reshape(y.shape[0]*y.shape[1]*y.shape[2]))


# In[31]:

cs = ['background','asphalt', 'building', 'forest', 'grass', 'ground', 'roadAsphalt', 'roadGround', 'water']


# In[33]:

cw = np.median(n)/n


# In[35]:

class_weights = dict(enumerate(cw))

# # Convert one mask to N masks

# In[36]:

n_images, h, w = y.shape
y = y.reshape([n_images, h*w])
n_classes = len(cs)
temp = np.zeros(np.append(y.shape,n_classes),dtype='int')


# In[37]:

for i in range (n_images):
    for j in range(h*w):
            temp[i,j,(y[i,j])] = 1


# In[38]:

y = temp.reshape([n_images,h,w,n_classes])
del(temp)


# In[41]:

x = x/255

# # Prepare for training

# In[45]:

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.25, random_state=1)

del(x,y)

# In[29]:

from keras.preprocessing.image import ImageDataGenerator
from keras import layers, models, optimizers
from keras import backend as K
import tensorflow as tf

# In[50]:

nb_train_samples = len(x_train)
nb_test_samples = len(x_test)
epochs = 2
batch_size = 1

# # Define model

# # U-Net

# In[48]:

import skimage.io as io
import skimage.transform as trans
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.layers.merge import concatenate
from keras.callbacks import ModelCheckpoint, LearningRateScheduler

def unet(pretrained_weights = None, input_size = (512,640,3), n_classes = 9):
    inputs = Input(input_size)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
    merge6 = concatenate([drop4,up6], axis = 3)
#     merge6 = merge([drop4,up6], mode = 'concat', concat_axis = 3)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([conv3,up7], axis = 3)
#     merge7 = merge([conv3,up7], mode = 'concat', concat_axis = 3)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv2,up8], axis = 3)
#     merge8 = merge([conv2,up8], mode = 'concat', concat_axis = 3)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv1,up9], axis = 3)
#     merge9 = merge([conv1,up9], mode = 'concat', concat_axis = 3)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    
    conv10 = Conv2D(n_classes, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv10 = Reshape((input_size[0]*input_size[1],n_classes))(conv10)
    conv10 = Activation('softmax')(conv10)
    
    model = Model(inputs = inputs, outputs = conv10)

    model.compile(optimizer = 'adadelta', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    
#     model.summary()

    if(pretrained_weights):
        model.load_weights(pretrained_weights)

    return model


# In[36]:

model = unet(input_size = (h,w,3), n_classes=9)


# In[38]:

y_train_flat = y_train.reshape(y_train.shape[0],y_train.shape[1]*y_train.shape[2],y_train.shape[3])
y_test_flat = y_test.reshape(y_test.shape[0],y_test.shape[1]*y_test.shape[2],y_test.shape[3])

del(y_train, y_test)

# In[40]:

def get_tf_session():
    gpu_options = tf.GPUOptions(allow_growth=True)
    return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
K.set_session(get_tf_session())

# In[41]:

class_weighting = np.fromiter(class_weights.values(), dtype=float)

# In[42]:

history = model.fit(x_train, y_train_flat, batch_size=batch_size, epochs=epochs, verbose=1, class_weight=class_weighting)

# In[42]:

with open('/trainHistoryDict', 'wb') as file_pi:
        pickle.dump(history.history, file_pi)

# In[67]:

model.save_weights('weights/trial.hdf5')


# In[43]:

score = model.evaluate(x_test, y_test_flat, batch_size = 1, verbose=1)
print('Test score:', score[0])
print('Test accuracy:', score[1])
