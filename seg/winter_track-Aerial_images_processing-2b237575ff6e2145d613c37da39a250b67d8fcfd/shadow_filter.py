import cv2
import numpy as np
import glob

file = "1529912111.jpg"
img = cv2.imread(file)

for file in glob.glob("common/images/*"):
    img = cv2.imread(file, -1)
    rgb_planes = cv2.split(img)
    result_planes = []
    result_norm_planes = []
    for plane in rgb_planes:
        dilated_img = cv2.dilate(plane, np.ones((7,7), np.uint8))
        bg_img = cv2.medianBlur(dilated_img, 25)
        diff_img = 235 - cv2.absdiff(plane, bg_img)
        norm_img = cv2.normalize(diff_img, diff_img, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        result_planes.append(diff_img)
        result_norm_planes.append(norm_img)
    if len(result_planes) != 0:
        result = cv2.merge(result_planes)
        result_norm = cv2.merge(result_norm_planes)
        #cv2.imwrite('shadows_out.png', result)
        cv2.imwrite(file.split("/")[2], result_norm)
        print(result.shape)
