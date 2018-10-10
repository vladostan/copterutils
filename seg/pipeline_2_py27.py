
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

sys.stdout = Logger()

#sys.stdout = open('logs/{}'.format(loggername), 'w')

print('Date and time: {}\n'.format(loggername))

# # READ IMAGES AND MASKS
# In[2]:
PATH = os.path.abspath('data')

SOURCE_IMAGES = [os.path.join(PATH, "images/ds1"), 
                 os.path.join(PATH, "images/ds1/albumentated")]

images = []
labels = []

for si in SOURCE_IMAGES:
    images.extend(glob(os.path.join(si, "*.jpg")))
    labels.extend(glob(os.path.join(si.replace("images","labels"), "*.png")))
    
print("Datasets used: {}\n".format(SOURCE_IMAGES))

images.sort()
labels.sort()

# In[4]:
print("Number of images: {}".format(np.size(images)))
print("Number of masks: {}\n".format(np.size(labels)))

# In[18]:
x = [] # images
y = [] # labels

for img in images:
    full_size_image = plt.imread(img)
    x.append(full_size_image)

for lbl in labels:
    full_size_label = plt.imread(lbl,0)
    y.append(full_size_label)
    
del(images, labels, full_size_image, full_size_label)

# In[19]:
x = np.asarray(x)
y = np.asarray(y)

# In[20]:
print("Y min: {}".format(y.min()))
print("Y max: {}\n".format(y.max()))
print("X shape: {}".format(x.shape))
print("Y shape: {}\n".format(y.shape))

# In[21]:
# Crop squared image
n_images, h, w = x.shape[0:3]

x = x[:,:,(w-h)//2:(w-(w-h)//2),:]
y = y[:,:,(w-h)//2:(w-(w-h)//2)]

n_images, h, w = x.shape[0:3]

print("Cropping squared images...")
print("X shape: {}".format(x.shape))
print("Y shape: {}\n".format(y.shape))

# # Visualize
# In[22]:
#fig, axes = plt.subplots(nrows = 1, ncols = 2)
#fig.set_size_inches(10,5)
#axes[0].imshow(x[0])
#axes[1].imshow(y[0])

# # Split images and masks into batches (optional)
# In[23]:
h_t, w_t = (512,512)
split_factor = 4 # We crop this number of smaller images out of one
x_t = np.zeros([n_images*split_factor, h_t, w_t, 3], dtype='uint8')
y_t = np.zeros([n_images*split_factor, h_t, w_t], dtype='uint8')

print("Cropping 512x512 batches...")

# In[25]:
for i in range(n_images):
    x_t[i] = x[i,:h//2,:w//2,:]
    x_t[n_images+i] = x[i,:h//2,w//2:w,:]
    x_t[n_images*2+i] = x[i,h//2:h,:w//2,:]
    x_t[n_images*3+i] = x[i,h//2:h,w//2:w,:]
    y_t[i] = y[i,:h//2,:w//2]
    y_t[n_images+i] = y[i,:h//2,w//2:w]
    y_t[n_images*2+i] = y[i,h//2:h,:w//2]
    y_t[n_images*3+i] = y[i,h//2:h,w//2:w]

x = x_t
y = y_t
del(x_t,y_t)

print("X shape: {}".format(x.shape))
print("Y shape: {}\n".format(y.shape))

# # Class weighting
# In[29]:
n = np.bincount(y.reshape(y.shape[0]*y.shape[1]*y.shape[2]))

cs = ['background', 'asphalt', 'building', 'forest', 'grass', 'ground', 'roadAsphalt', 'roadGround', 'water']

print("Classes: {}\n".format(cs))
print("Class distribution: {}\n".format(str(n/n.sum()*100) + " %"))

# In[32]:
#import seaborn as sns
#
#sns.barplot(x=cs, y=n)

# In[33]:
cw = np.median(n)/n

# In[34]:
#sns.barplot(x=cs, y=n*cw)

# In[35]:

class_weights = dict(enumerate(cw))
class_weighting = np.fromiter(class_weights.values(), dtype=float)
print("Class weights for balancing dataset: {}\n".format(class_weighting))

# Convert one mask to N classes masks
# In[37]:
n_classes = len(cs)

from keras.utils import to_categorical

y = to_categorical(y, num_classes=n_classes)
y = y.reshape(x.shape[:3] + (n_classes,))

print("One hot encoding Y matrix to categorical...")
print("Y shape: {}\n".format(y.shape))

# In[38]:
#nrows, ncols = 3,3
#fig, axes = plt.subplots(nrows=3, ncols=3)
#fig.set_size_inches(15,10)
#cl = 0
#im = 0
#for i in range(nrows):
#    for j in range(ncols):
#        axes[i,j].imshow(y[im,:,:,cl], cmap='gray')
#        axes[i,j].set_title(cs[cl])
#        axes[i,j].grid(None)
#        cl += 1
#     
#fig.tight_layout()

# In[39]:
x = np.float32(x/255.)
y = y.astype('int8')

print("X dtype after conversion: {}".format(x.dtype))
print("Y dtype after conversion: {}\n".format(y.dtype))

# # Prepare for training
# In[ ]:

from sklearn.model_selection import train_test_split

test_size = 0.2
print("Train/test split: {}/{}\n".format(1-test_size,test_size))
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=1)

del(x,y)

# In[ ]:

print("X_train shape: {}".format(x_train.shape))
print("Y_train shape: {}\n".format(y_train.shape))
print("X_test shape: {}".format(x_test.shape))
print("Y_test shape: {}\n".format(y_test.shape))

# In[ ]:
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
import tensorflow as tf

print("Keras image data format: {}\n".format(K.image_data_format()))

# In[ ]:
epochs = 100
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

from itertools import izip
train_generator = izip(image_generator, mask_generator)

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

learning_rate = 5e-5
optimizer = optimizers.Adam(lr = learning_rate)
loss = 'categorical_crossentropy'
metrics = ['accuracy']

print("Optimizer: {}, learning rate: {}, loss: {}, metrics: {}\n".format(optimizer, learning_rate, loss, metrics))

model.compile(optimizer = optimizer, loss = loss, metrics = metrics)

model.load_weights('weights/2018-10-09 11:20:32.hdf5')

# In[ ]:

def get_tf_session():
    gpu_options = tf.GPUOptions(allow_growth=True)
    return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
#K.set_session(get_tf_session())

# In[ ]:

from keras import callbacks

model_checkpoint = callbacks.ModelCheckpoint('weights/{}.hdf5'.format(loggername), monitor='loss', verbose=1, save_best_only=True, save_weights_only=True)
tensor_board = callbacks.TensorBoard(log_dir='./tblogs')
reduce_lr = callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5,
                              patience=2, verbose = 1, min_lr=1e-5)
csv_logger = callbacks.CSVLogger('logs/{}.log'.format(loggername))
early_stopper = callbacks.EarlyStopping(monitor='loss', min_delta = 0.0025, patience = 3, verbose = 1)

callbacks = [model_checkpoint, tensor_board, reduce_lr, csv_logger, early_stopper]

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
