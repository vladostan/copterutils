import numpy as np
import nrrd
import scipy.misc as smp
from PIL import Image
from scipy import misc
import cv2

data = nrrd.read("test.nrrd")[0].reshape(1024, 1280)
misc.imsave("my_bin.jpg", data)
# img = Image.fromarray(data, '1')
# img.save('my.png')

# red = np.empty([1024, 1280])
# green = np.empty([1024, 1280])
# blue = np.empty([1024, 1280])
#
# types = np.zeros(2)
#
# for i in range (0, 1024):
#     for j in range (0, 1280):
#         if data[i][j][0] == 1:
#             red[i][j] = 255
#             green[i][j] = 0
#             blue[i][j] = 0
#             types[1] += 1
#         else:
#             red[i][j] = 0
#             green[i][j] = 0
#             blue[i][j] = 0
#             types[0] += 1
#
# image = np.concatenate([arr[np.newaxis] for arr in [red, green, blue]]).transpose([1, 2, 0])
# cv2.GaussianBlur(image, (3, 3), sigmaX = 3)
# print(image.shape)
# print(types)
# misc.imsave("my_bin.jpg", image)
