import numpy as np 
import keras
from keras import Model
import nn
import cv2
import nrrd


model = nn.get_custom_unet((256, 320, 3))
model.load_weights("weights.h5")
testImg = np.asarray(cv2.imread("images/1529574467.jpg"))

square_width = 256
square_height = 320


print("Predicting")
squares_img_test = []
start_point_row = 0
start_point_col = 0
for i in range (0, 4):
	for j in range (0, 4):
		squares_img_test.append(testImg[start_point_row:start_point_row + 256, start_point_col:start_point_col + 320])
		print(squares_img_test[i + j].shape)
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
				for k in range (0, 12):
					if res[sqV * 4 + sqH][i][j][k] == max(res[sqV * 4 + sqH][i][j][0:12]):
						test[i + n][j + m][0] = k + 1
		m += 320

	m = 0
	n += 256
print(res)
nrrd.write(filename, (test))
#img = Image.fromarray(res[0][0].astype(int), 'I')
print(res[0].shape)
#img.save('test.png')
cv2.imwrite("test.jpg", testImg)
print("predicting finished")

