from imutils.perspective import four_point_transform
from imutils import contours
import numpy as np
import argparse
import imutils
import cv2

def auto_canny(image, sigma=0.50):
    # compute the median of the single channel pixel intensities
    v = np.median(image)

    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)


    # return the edged image
    return edged

# construct the argument parse and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--image", required=True,
#     help="path to the input image")
# args = vars(ap.parse_args())
image = cv2.imread("1529912145.jpg")
r = 500.0 / image.shape[1]
dim = (500, int(image.shape[0] * r))

# perform the actual resizing of the image and show it
#image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
equalized_img =  cv2.equalizeHist(gray)
#cv2.imshow('Equalized', equalized_img)
# cv2.waitKey(0)
blurred = cv2.GaussianBlur(equalized_img, (7, 7), 0)
# edged =cv2.Canny(equalized_img, 30, 160)
edged = auto_canny(blurred)
cv2.imwrite('edged.jpg', edged)
cv2.waitKey(0)
