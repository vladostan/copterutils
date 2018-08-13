import cv2
import re
import numpy as np
import time
from scipy.ndimage import morphology
from sklearn.model_selection import train_test_split
from skimage import io
from skimage.transform import resize

def imread_and_resize_cv2(e,im_shape,color=True):
    im  = cv2.imread(e) if color else cv2.imread(e,0)
    return cv2.resize(im,im_shape[::-1],interpolation=cv2.INTER_NEAREST)

def depthread_and_resize_cv2(e,im_shape,color=True):
    im = cv2.imread(e, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    return cv2.resize(im,im_shape[::-1],interpolation=cv2.INTER_NEAREST)

def imread_and_resize_sk(e,im_shape,color=True):
    im = io.imread(e, as_grey=not color)
    im = resize(im,output_shape=im_shape)
    # for cv2  compatibility
    return (im*255.).astype('uint8')


def in_filename(x,pat="kia?_*"):
    names = [e.split('/')[-1] for e in x]
    return [i for i,item in enumerate(names) if bool(re.match(pat,item)) ]

def expand(x,axis=-1):
    return np.expand_dims(x,axis=axis)

def squeeze(x,axis=-1):
    return np.squeeze(x,axis=axis)

def in_filename(x,pat='kia'):
    return [i for i,item in enumerate(x) if pat in item]

def binarize_mask(masks,target_label=7):
    bin_masks = masks.copy()
    bin_masks[bin_masks!=target_label]=0
    bin_masks[bin_masks==target_label]=1
    return bin_masks

def boundary_mask(mask):
    mask2 = morphology.distance_transform_edt(mask)
    mask2 = (mask2==1).astype('uint8')
    kernel = np.ones((7,7), np.uint8)
    mask2 = cv2.dilate(mask2.copy(), kernel, iterations=1)
    return (mask2 & mask)

def inference_time(model,X):
    start = time.clock() 
    _ = model.predict(X,batch_size=1,verbose=0)
    end = time.clock()
    t = end-start
    return [t,t/len(X)]