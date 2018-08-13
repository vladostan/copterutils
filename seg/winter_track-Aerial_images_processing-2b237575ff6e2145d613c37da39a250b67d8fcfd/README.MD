# U-net based road segmentation project for self-driving cars and drones

## Requirements:
	- Python 2.7
	- Tensorflow 1.9.0
		- install it by command pip install tensorflow
	- Keras 2.2.0
		- install it by command pip install keras
	- ROS Kinetic
	- pynrrd lib
		- install it by command pip install nrrd

## Project Structure:
- common
	- eval.py - ml evaluation metrics
	- lidar.py - lidar data  utils
	- nn.py - segmetation models
	- utils.py - varios utils for  data preprocessing
	- segmentation.py - multiclass segmentation learning
	- bin_segmentation.py - binary segmentation learning
	- predict.py - multicalss segmentation
	- predict_bin.py - binary segmentation
	- weights.h5 - weights for multiclass model
	- weights_bin.h5 - weights for binary model
- lidar_processing
	- kitti.ipynb
- misc
	- bird_view.ipynb - bird_view transform with pure tensorflow
- scripts
	- keras2tf.py - keras to tf models converter
- unet_experiments

- bin_segmentation - ROS package for saving marked images
	- how to use:
		1. Copy this folder to catkin workspace
		2. Execute catkin build
		2. Run roscore
		3. Start publish messages from pylon_camera_node_aca1300/image_raw/compressed
		4. Run launch file run.launch:
		 	- use roslaunch bin_segmentation run.launch to_save:=1 if you want to save marked images (imaegs will be saved in bin_segmentation/scripts)
		 	- use roslaunch bin_segmentation run.launch to_save:=0 if you don't want to save marked images
		5. Marked images will be posted to the node /marked_images
