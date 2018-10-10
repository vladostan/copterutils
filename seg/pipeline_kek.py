
# coding: utf-8

# In[1]:


import os
import matplotlib.pylab as plt
from glob import glob
import numpy as np

# # READ IMAGES AND MASKS

# In[2]:


PATH = os.path.abspath('data')

SOURCE_IMAGES = os.path.join(PATH, "images/resized/*")

images = glob(os.path.join(SOURCE_IMAGES, "*.jpg"))

SOURCE_LABELS = os.path.join(PATH, "labels/resized/*")

labels = glob(os.path.join(SOURCE_LABELS, "*.png"))


# In[3]:


images.sort()
labels.sort()


# In[4]:


print(np.size(images))
print(np.size(labels))


# In[5]:


x = [] # images
y = [] # labels

for img in images:
    full_size_image = plt.imread(img)
    x.append(full_size_image)

for lbl in labels:
    full_size_label = plt.imread(lbl,0)
    y.append(full_size_label)


# In[6]:


x = np.asarray(x)
y = np.asarray(y)


# In[7]:


print(y.min())
print(y.max())
print(x.shape)
print(y.shape)


# ### Class weighting

# In[8]:


n = np.bincount(y.reshape(y.shape[0]*y.shape[1]*y.shape[2]))


# In[9]:


print(str(n/n.sum()*100) + " %")


# In[10]:


cs = ['background','asphalt', 'building', 'forest', 'grass', 'ground', 'roadAsphalt', 'roadGround', 'water']



# In[12]:

cw = np.median(n)/n


# In[14]:


class_weights = dict(enumerate(cw))
class_weighting = np.fromiter(class_weights.values(), dtype=float)
print(class_weighting)


# # Convert one mask to N classes masks

# In[15]:


#Convert batch of masks from 1 to N of classes
def one_to_n(y,n):
    n_images, h, w = y.shape
    y = y.reshape([n_images, h*w])
    temp = np.zeros(np.append(y.shape,n_classes),dtype='int')
    for i in range (n_images):
        for j in range(h*w):
            temp[i,j,(y[i,j])] = 1
    return temp.reshape([n_images,h,w,n])

def n_to_one(y):
    n_images, h, w, n = y.shape
    y = y.reshape([n_images, h*w, n])
    temp = np.zeros([n_images, h*w],dtype='int')
    for i in range (n_images):
        for j in range(h*w):
            temp[i,j] = np.argmax(y[i,j])
    return temp.reshape([n_images,h,w])


# In[16]:

n_classes = len(cs)
y = one_to_n(y,n_classes)
# y = n_to_one(y)
print(y.shape)


# In[18]:


x = x/255


# In[19]:


y = y.reshape([y.shape[0], y.shape[1]*y.shape[2],y.shape[3]])
y.shape


# # Prepare for training

# In[20]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=1)


# In[21]:


print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)


# In[22]:


from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
import tensorflow as tf


# In[23]:


K.image_data_format()


# In[24]:


img_height, img_width = x_train.shape[1:3]
nb_train_samples = len(x_train)
nb_test_samples = len(x_test)


# In[25]:


train_datagen = ImageDataGenerator(rotation_range=45,
                                   width_shift_range=0.25,
                                   height_shift_range=0.25,
                                   zoom_range=0.25,
                                   horizontal_flip=True, 
                                   vertical_flip=True,
                                   )
valtest_datagen = ImageDataGenerator()


# In[26]:


epochs = 50
batch_size = 2


# In[27]:


train_generator = train_datagen.flow(x_train, y_train, batch_size=batch_size)
valtest_generator = valtest_datagen.flow(x_test, y_test, batch_size=batch_size)


# # Define model

# # U-Net

# In[28]:
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.layers.merge import concatenate
from keras.callbacks import ModelCheckpoint, LearningRateScheduler

def unet(pretrained_weights = None, input_size = (256,320,3), n_classes = 9):
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
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    
    conv10 = Conv2D(n_classes, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv10 = Reshape((input_size[0]*input_size[1],n_classes))(conv10)
    conv10 = Activation('softmax')(conv10)
    
    model = Model(inputs = inputs, outputs = conv10)
    
    if(pretrained_weights):
        model.load_weights(pretrained_weights)

    return model


# In[29]:


model = unet(input_size = (img_height,img_width,3), n_classes=9)
model.compile(optimizer = 'adadelta', loss = 'categorical_crossentropy', metrics = ['accuracy'])
model.compile(optimizer = Adam(lr=0.005, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0), loss = 'categorical_crossentropy', metrics = ['accuracy'])


# In[30]:


model.summary()


# In[31]:


from keras.callbacks import ReduceLROnPlateau

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=10, min_lr=0.0001)

callbacks_list = [reduce_lr]


# In[32]:

def get_tf_session():
    gpu_options = tf.GPUOptions(allow_growth=True)
    return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
K.set_session(get_tf_session())


history = model.fit_generator(
    train_generator, 
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=valtest_generator,
    validation_steps=nb_test_samples // batch_size,
    verbose = 1,
    class_weight = class_weighting,
    callbacks=callbacks_list
)