import urllib.request
import cv2
import numpy as np
import random
import hashlib
from keras.models import model_from_json
from keras.preprocessing import image
import os
import time

json_file = open("model-hand-digits.json", "r")
model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(model_json)
loaded_model.load_weights("model-hand-digits.h5")
print("Loaded model from disk")
font = cv2.FONT_HERSHEY_SIMPLEX 


address = "http://10.103.221.207:8080/shot.jpg"

dir_one = "/home/zerosec/Videos/datasets/validation/1"
dir_two = "/home/zerosec/Videos/datasets/validation/2"
dir_three = "/home/zerosec/Videos/datasets/validation/3"
dir_four = "/home/zerosec/Videos/datasets/validation/4"
dir_five = "/home/zerosec/Videos/datasets/validation/5"



def nothing(x):
    pass


cv2.namedWindow("Color Adjustments",cv2.WINDOW_NORMAL)
cv2.resizeWindow("Color Adjustments", (300, 300)) 
cv2.createTrackbar("Thresh", "Color Adjustments", 0, 255, nothing)

#COlor Detection Track

cv2.createTrackbar("Lower_H", "Color Adjustments", 0, 255, nothing)
cv2.createTrackbar("Lower_S", "Color Adjustments", 0, 255, nothing)
cv2.createTrackbar("Lower_V", "Color Adjustments", 0, 255, nothing)
cv2.createTrackbar("Upper_H", "Color Adjustments", 255, 255, nothing)
cv2.createTrackbar("Upper_S", "Color Adjustments", 255, 255, nothing)
cv2.createTrackbar("Upper_V", "Color Adjustments", 255, 255, nothing)

# Upper_V 192
# Lower_S 31
cond = 0
while True:
	imgResp = urllib.request.urlopen(address)  
	imgNp = np.array(bytearray(imgResp.read()),dtype=np.uint8)
	img = cv2.imdecode(imgNp,-1)

	cv2.rectangle(img, (230,5), (450,255), (255, 0, 0), 0)
	crop_image = hand_image = img[5:255, 230:450]
	crop_image = cv2.resize(crop_image, (64, 64))
	hsv = cv2.cvtColor(crop_image, cv2.COLOR_BGR2HSV)
	rgb = cv2.cvtColor(crop_image, cv2.COLOR_BGR2RGB)


	l_h = cv2.getTrackbarPos("Lower_H", "Color Adjustments")
	l_s = cv2.getTrackbarPos("Lower_S", "Color Adjustments")
	l_v = cv2.getTrackbarPos("Lower_V", "Color Adjustments")

	u_h = cv2.getTrackbarPos("Upper_H", "Color Adjustments")
	u_s = cv2.getTrackbarPos("Upper_S", "Color Adjustments")
	u_v = cv2.getTrackbarPos("Upper_V", "Color Adjustments")

	lower_bound = np.array([l_h, l_s, l_v])
	upper_bound = np.array([u_h, u_s, u_v])

	mask = cv2.inRange(hsv, lower_bound, upper_bound)
	filtr = cv2.bitwise_and(crop_image, crop_image, mask=mask)

	mask1  = cv2.bitwise_not(mask)
	m_g = cv2.getTrackbarPos("Thresh", "Color Adjustments") #getting track bar value
	ret,thresh = cv2.threshold(mask1,m_g,255,cv2.THRESH_BINARY_INV)

	test_image = np.expand_dims(crop_image, axis = 0)
	result = loaded_model.predict(test_image)
	answer = ""
	if int(round(result[0][0])) == 1: #One
		answer = "VolumeUp"
	elif int(round(result[0][1])) == 1: #Two
		answer = "VolumeDown"
	elif int(round(result[0][2])) == 1: #Three
		answer = "Playing"
	elif int(round(result[0][3])) == 1: #Four
		answer = "Stop"
	elif int(round(result[0][4])) == 1: #Five
		answer = "No Volume"
	else:
		answer = "Nothing"

	print(img.shape)
	imag = cv2.putText(img, answer, (250,200), font,1, (255, 0, 0), 2, cv2.LINE_AA) 
	cv2.imshow('Main Video',imag)
	cv2.imshow('Base Image',crop_image)
	cv2.imshow("Hand Image", hand_image)
	id_value = random.randint(0,999999999)
	id_value = hashlib.md5(f'{str(id_value)}'.encode("utf-8")) 
	id_value = id_value.digest()
	interrupt = cv2.waitKey(10)
	if interrupt & 0xFF == ord('q'):
		break
	if interrupt & 0xFF == ord('z'):
		cv2.imwrite(dir_one+"/"+str(id_value)+".jpg", crop_image)
		print("Picture One Saved")
	if interrupt & 0xFF == ord('x'):
		cv2.imwrite(dir_two+"/"+str(id_value)+".jpg", crop_image)
		print("Picture Two Saved")
	if interrupt & 0xFF == ord('c'):
		cv2.imwrite(dir_three+"/"+str(id_value)+".jpg", crop_image)
		print("Picture Three Saved")
	if interrupt & 0xFF == ord('v'):
		cv2.imwrite(dir_four+"/"+str(id_value)+".jpg", crop_image)
		print("Picture Four Saved")
	if interrupt & 0xFF == ord('b'):
		cv2.imwrite(dir_five+"/"+str(id_value)+".jpg", crop_image)
		print("Picture Five Saved")
    	

cv2.destroyAllWindows()

