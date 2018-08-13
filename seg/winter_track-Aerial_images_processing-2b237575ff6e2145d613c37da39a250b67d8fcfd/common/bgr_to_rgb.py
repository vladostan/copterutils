import cv2
import numpy as np
from scipy import misc
import glob
from PIL import Image

for file in glob.glob("images/*.jpg"):
    img = cv2.imread(file)
    data = np.asarray(img)
    img = Image.fromarray(np.roll(data, 0, axis=-1))
    misc.imsave(file, img)
    print(file.split("/")[1] + " saved")
    # for i in range(0, img.shape[0]):
    #     for j in range(0, img.shape[1]):
    #         buffer = img[i][j][0]
    #         img[i][j][0] = img[i][j][2]
    #         img[i][j][2] = img[i][j][0]
    # misc.imsave(file, img)
    # print(file + " saved")
