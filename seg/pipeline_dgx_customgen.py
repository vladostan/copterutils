# coding: utf-8

# In[1]:
import os
import matplotlib.pylab as plt
from glob import glob
import numpy as np
import datetime

# Get the date and time
now = datetime.datetime.now()
loggername = str(now).split(".")[0]
loggername = loggername.replace(":","-")

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
		 os.path.join(PATH,"images/ds2")]

images = []
labels = []

for si in SOURCE_IMAGES:
    images.extend(glob(os.path.join(si, "*.jpg")))
    labels.extend(glob(os.path.join(si.replace("images","labels"), "*.png")))
    
print("Datasets used: {}\n".format(SOURCE_IMAGES))

images.sort()
labels.sort()

# In[]
def get_image(path):
    
    image = plt.imread(path)
    return(np.asarray(image))
    
print("Image dtype:{}\n".format(get_image(images[0]).dtype))

def get_label(path):

    label = plt.imread(path, 0)
    return(np.asarray(label))

print("Label dtype:{}\n".format(get_label(labels[0]).dtype))


# In[]
def preprocess_input(x):
    
    # Crop squared image
    h, w = x.shape[:2]

    x = x[:,(w-h)//2:(w-(w-h)//2),:]
    
    h, w = x.shape[:2]
    
    # Split images and masks into batches (optional)
    h_t, w_t = (512,512)
    split_factor = 4 # We crop this number of smaller images out of one
    x_t = np.zeros([split_factor, h_t, w_t, 3], dtype='uint8')
    
    x_t[0] = x[:h//2,:w//2,:]
    x_t[1] = x[:h//2,w//2:w,:]
    x_t[2] = x[h//2:h,:w//2,:]
    x_t[3] = x[h//2:h,w//2:w,:]
    
#     x_t = np.float32(x_t/255.)

    return(x_t)
    
x = get_image(images[0])
x = preprocess_input(x)
print("Preprocess input shape of output: {}".format(x.shape))

# In[]
def preprocess_output(y):
    
    # Crop squared image
    h, w = y.shape[:2]

    y = y[:,(w-h)//2:(w-(w-h)//2)]
    
    h, w = y.shape[:2]
    
    # Split images and masks into batches (optional)
    h_t, w_t = (512,512)
    split_factor = 4 # We crop this number of smaller images out of one
    y_t = np.zeros([split_factor, h_t, w_t], dtype='uint8')
    
    y_t[0] = y[:h//2,:w//2]
    y_t[1] = y[:h//2,w//2:w]
    y_t[2] = y[h//2:h,:w//2]
    y_t[3] = y[h//2:h,w//2:w]
    
#     y_t = to_categorical(y_t, num_classes=9)
#     y_t = y_t.reshape(y_t.shape[:3] + (9,))
#     y_t = y_t.astype('int8')

    return(y_t)

y = get_label(labels[0])
y = preprocess_output(y)
print("Preprocess output shape of output: {}".format(y.shape))

# # Prepare for training
# In[ ]:

from sklearn.model_selection import train_test_split

test_size = 0.2
images_train, images_test, labels_train, labels_test = train_test_split(images, labels, test_size=test_size, random_state=1)

print(len(images_train))
print(len(labels_train))
print(len(images_test))
print(len(labels_test))

# In[]: AUGMENTATIONS

from albumentations import (
    HorizontalFlip,
    OpticalDistortion,
    RandomSizedCrop,
    ShiftScaleRotate,
    OneOf,
    Compose,
    CLAHE,
    RandomContrast,
    RandomGamma
)

def augment_big(image, mask):

    original_height, original_width = image.shape[:2]
    
    aug = Compose([
        RandomSizedCrop(p=0.5, min_max_height=(original_height//2, original_height), height=original_height, width=original_width),
        OpticalDistortion(p=0.5, distort_limit=0.25, shift_limit=0.5),
        OneOf([
            CLAHE(p=1., clip_limit=4.),
            RandomContrast(p=1., limit=0.25),
            RandomGamma(p=1., gamma_limit=(50,200))
            ], p=0.5),
        ], p=0.5)

    augmented = aug(image=image, mask=mask)

    image_heavy = augmented['image']
    mask_heavy = augmented['mask']
    
    return image_heavy, mask_heavy

def augment_small(image, mask):

    original_height, original_width = image.shape[:2]
    
    aug = Compose([
        HorizontalFlip(p=0.5),
        ShiftScaleRotate(shift_limit=0.1, scale_limit=0, rotate_limit=20, p=0.5),
        ], p=0.5)

    augmented = aug(image=image, mask=mask)

    image_heavy = augmented['image']
    mask_heavy = augmented['mask']
    
    return image_heavy, mask_heavy

# In[]: CUSTOM GENERATORS
    
from keras.utils import to_categorical

def train_generator(images_path, labels_path, batch_size = 1):
    
    while True:
        ids = np.random.randint(0, len(images_path), batch_size)
        
        image_batch = np.take(images_path, ids)
        label_batch = np.take(labels_path, ids)
        
        batch_input = np.zeros([batch_size, 1024, 1280, 3], dtype='uint8')
        batch_output = np.zeros([batch_size, 1024, 1280], dtype='uint8') 

        # READ Images and masks:
        for i in range(len(image_batch)):
            inp = get_image(image_batch[i])
            batch_input[i] = inp
            outp = get_label(label_batch[i])
            batch_output[i] = outp

        # Albumentations augmentation:
        for i in range(len(batch_input)):
            batch_input[i], batch_output[i]  = augment_big(batch_input[i], batch_output[i])
        
        # Preprocess Images and masks:
        inp = []
        outp = []
        for i in range(len(batch_input)):
            inp.extend(preprocess_input(batch_input[i]))
            outp.extend(preprocess_output(batch_output[i]))
            
        inp = np.asarray(inp)
        outp = np.asarray(outp)

        # Return a tuple of (input,output) to feed the network
        ids = np.random.randint(0, batch_size*4, batch_size)
        
        batch_x = np.array(inp)
        batch_y = np.array(outp)
        
        batch_x = np.take(batch_x, ids, axis = 0)
        batch_y = np.take(batch_y, ids, axis = 0)
        
        out_x = np.zeros_like(batch_x)
        out_y = np.zeros_like(batch_y)
        
        # AUGMENT
        for i in range(len(batch_x)):
            image_heavy, mask_heavy  = augment_small(batch_x[i], batch_y[i])
            out_x[i] = image_heavy
            out_y[i] = mask_heavy
            
        out_x = np.float32(out_x/255.)
        
        out_y = to_categorical(out_y, num_classes=9)
        out_y = out_y.reshape(out_y.shape[:3] + (9,))
        out_y = out_y.astype('int8')
            
        yield(out_x, out_y)      
#         return(out_x, out_y)
        
def val_generator(images_path, labels_path, batch_size = 1):
    
    while True:
        ids = np.random.randint(0, len(images_path), batch_size)
        
        image_batch = np.take(images_path, ids)
        label_batch = np.take(labels_path, ids)
        
        batch_input = np.zeros([batch_size, 1024, 1280, 3], dtype='uint8')
        batch_output = np.zeros([batch_size, 1024, 1280], dtype='uint8') 

        # READ Images and masks:
        for i in range(len(image_batch)):
            inp = get_image(image_batch[i])
            batch_input[i] = inp
            outp = get_label(label_batch[i])
            batch_output[i] = outp
        
        # Preprocess Images and masks:
        inp = []
        outp = []
        for i in range(len(batch_input)):
            inp.extend(preprocess_input(batch_input[i]))
            outp.extend(preprocess_output(batch_output[i]))
            
        inp = np.asarray(inp)
        outp = np.asarray(outp)

        # Return a tuple of (input,output) to feed the network
        ids = np.random.randint(0, batch_size*4, batch_size)
        
        batch_x = np.array(inp)
        batch_y = np.array(outp)
        
        out_x = np.take(batch_x, ids, axis = 0)
        out_y = np.take(batch_y, ids, axis = 0)
            
        out_x = np.float32(out_x/255.)
        
        out_y = to_categorical(out_y, num_classes=9)
        out_y = out_y.reshape(out_y.shape[:3] + (9,))
        out_y = out_y.astype('int8')
            
        yield(out_x, out_y)      
#         return(out_x, out_y)

# In[ ]:
batch_size = 1

train_gen = train_generator(images_path=images_train, labels_path=labels_train, batch_size=batch_size)
val_gen = val_generator(images_path=images_test, labels_path=labels_test, batch_size=batch_size)

#print(train_gen[0].shape)
#print(train_gen[1].shape)
#print(val_gen[0].shape)
#print(val_gen[1].shape)

# In[ ]:
# # Define model
# # U-Net
from models.Unet import unet

model = unet(input_size = (512,512,3), n_classes=9)

print("Model summary:")
model.summary()

# In[ ]:
from keras import optimizers
from keras import backend as K
import tensorflow as tf

#model.compile(optimizer = 'adadelta', loss = 'categorical_crossentropy', metrics = ['accuracy'])

def jaccard_distance_loss(y_true, y_pred, smooth=100):
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return (1 - jac) * smooth

def dice_coef(y_true, y_pred, smooth=1):
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    return (2. * intersection + smooth) / (K.sum(K.square(y_true),-1) + K.sum(K.square(y_pred),-1) + smooth)

def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)

learning_rate = 1e-4
optimizer = optimizers.Adam(lr = learning_rate)
metrics = ['accuracy']
loss = 'categorical_crossentropy'

#loss = dice_coef_loss()

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
#tensor_board = callbacks.TensorBoard(log_dir='./tblogs')
reduce_lr = callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5,
                              patience=2, verbose = 1, min_lr=1e-5)
csv_logger = callbacks.CSVLogger('logs/{}.log'.format(loggername))
early_stopper = callbacks.EarlyStopping(monitor='loss', min_delta = 0.0025, patience = 3, verbose = 1)

callbacks = [model_checkpoint, reduce_lr, csv_logger, early_stopper]

print("Callbacks: {}\n".format(callbacks))

# In[ ]:
steps_per_epoch = len(images_train)//batch_size*4*2 # 4 crops * twice run
validation_steps = len(images_test)//batch_size*4 # 4 crops
epochs = 100

print("Steps per epoch: {}".format(steps_per_epoch))
print("Validation steps: {}\n".format(validation_steps))

print("Starting training...\n")
history = model.fit_generator(
    train_gen,
    validation_data = val_gen,
    steps_per_epoch = steps_per_epoch,
    validation_steps = validation_steps,
    epochs = epochs,
    verbose = 1,
    callbacks = callbacks
)
print("Finished training\n")
