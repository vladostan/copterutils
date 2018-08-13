import numpy as np
import nrrd
import scipy.misc as smp
from PIL import Image
from scipy import misc

data = nrrd.read("test.nrrd")[0]

red = np.empty([1024, 1280])
green = np.empty([1024, 1280])
blue = np.empty([1024, 1280])

types = np.zeros(12)

for i in range (0, 1024):
    for j in range (0, 1280):
        if data[i][j][0] == 1:
            red[i][j] = 255
            green[i][j] = 0
            blue[i][j] = 0
            types[0] += 1
        if data[i][j][0] == 2:
            red[i][j] = 0
            green[i][j] = 255
            blue[i][j] = 0
            types[1] += 1
        if data[i][j][0] == 3:
            red[i][j] = 0
            green[i][j] = 0
            blue[i][j] = 255
            types[2] += 1
        if data[i][j][0] == 4:
            red[i][j] = 255
            green[i][j] = 255
            blue[i][j] = 255
            types[3] += 1
        if data[i][j][0] == 5:
            red[i][j] = 150
            green[i][j] = 255
            blue[i][j] = 155
            types[4] += 1
        if data[i][j][0] == 6:
            red[i][j] = 165
            green[i][j] = 42
            blue[i][j] = 42
            types[5] += 1
        if data[i][j][0] == 7:
            red[i][j] = 100
            green[i][j] = 200
            blue[i][j] = 55
            types[6] += 1
        if data[i][j][0] == 8:
            red[i][j] = 255
            green[i][j] = 255
            blue[i][j] = 0
            types[7] += 1
        if data[i][j][0] == 9:
            red[i][j] = 0
            green[i][j] = 0
            blue[i][j] = 0
            types[8] += 1
        if data[i][j][0] == 10:
            red[i][j] = 255
            green[i][j] = 192
            blue[i][j] = 203
            types[9] += 1
        if data[i][j][0] == 11:
            red[i][j] = 0
            green[i][j] = 255
            blue[i][j] = 255
            types[10] += 1
        if data[i][j][0] == 12:
            red[i][j] = 165
            green[i][j] = 42
            blue[i][j] = 65
            types[11] += 1

image = np.concatenate([arr[np.newaxis] for arr in [red, green, blue]]).transpose([1, 2, 0])
print(image.shape)
print(types)
misc.imsave("my.jpg", image)
