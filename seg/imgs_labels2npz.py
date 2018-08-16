
# coding: utf-8

import os
import matplotlib.pylab as plt
from glob import glob
import numpy as np


PATH = os.path.abspath('data')
SOURCE_IMAGES = os.path.join(PATH, "images")
images = glob(os.path.join(SOURCE_IMAGES, "*.jpg"))
SOURCE_LABELS = os.path.join(PATH, "labels")
labels = glob(os.path.join(SOURCE_LABELS, "*.png"))


images.sort()
labels.sort()


x = [] # images
y = [] # labels

for img in images:
    full_size_image = plt.imread(img)
    x.append(full_size_image)

for lbl in labels:
    full_size_label = plt.imread(lbl,0)
    y.append(full_size_label)
    

np.savez("data.npz",x=x,y=y)