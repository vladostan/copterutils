#!/usr/bin/env python
import rospy
from std_msgs.msg import String
import cv2
from sensor_msgs.msg import CompressedImage
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import time
import keras
import tensorflow
import nn
import numpy as np
from scipy import misc
import time
import sys
from cv_bridge import CvBridge, CvBridgeError
import os

frames_ratio = 3
frames_counter = 0

path = sys.argv[1]

model = nn.get_custom_unet((256, 320, 3), bin=True)
model.load_weights(path + "weights_bin.h5")
model._make_predict_function()

def callback(data):
    # global frames_ratio
    # global frames_counter
    # if frames_counter % frames_ratio != 0:
    #     frames_counter += 1
    #     return
    # frames_counter += 1
    global path
    start = time.time()
    global model
    rospy.loginfo(rospy.get_caller_id() + "I heard")
    bridge = CvBridge()

    img = bridge.imgmsg_to_cv2(data, "bgr8")
    square_width = 256
    square_height = 320


    print("Predicting")
    start = time.time()
    squares_img_test = []
    start_point_row = 0
    start_point_col = 0
    for i in range (0, 4):
        for j in range (0, 4):
                squares_img_test.append(img[start_point_row:start_point_row + 256, start_point_col:start_point_col + 320])
                start_point_col += 320
        start_point_col = 0
        start_point_row += 256


    squares_img_test = np.concatenate([arr[np.newaxis] for arr in squares_img_test])
    res = model.predict(squares_img_test, batch_size=1)
    filename = "test.nrrd"

    test = np.zeros((1024, 1280, 1))
    m = 0
    n = 0
    for sqV in range (0, 4):
        for sqH in range (0, 4):
            for i in range(0, square_width):
                for j in range (0, square_height):
                    if res[sqV * 4 + sqH][i][j] > 0.5:
                    	test[i + n][j + m][0] = 1
            m += 320

        m = 0
        n += 256

    data = test.reshape(1024, 1280)
    name = str(time.time())
    misc.imsave(path + name + ".jpg", data)
    pub = rospy.Publisher('marked_images', Image)
    cv_image = np.asarray(cv2.imread(path + name + ".jpg"))
    bridge = CvBridge()
    image_message = bridge.cv2_to_imgmsg(cv_image, "bgr8")
    if sys.argv[2] != "1":
        os.remove(path + name + ".jpg")
    else:
        print("saved")
    print(res[0].shape)
    print("predicting finished")
    pub.publish(image_message)
    print(time.time() - start)

def listener():
     # In ROS, nodes are uniquely named. If two nodes with the same
     # node are launched, the previous one is kicked off. The
     # anonymous=True flag means that rospy will choose a unique
     # name for our 'listener' node so that multiple listeners can
     # run simultaneously.
    rospy.init_node('predict_bin', anonymous=True)

    rospy.Subscriber("img", Image, callback)

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

if __name__ == '__main__':
    print("start")
    listener()
