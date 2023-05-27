# To Capture Frame
import cv2
import tensorflow as tf
# To process image array
import numpy as np


# import the tensorflow modules and load the model
model=tf.keras.models.load_model("keras_model.h5")


# Attaching Cam indexed as 0, with the application software
camera = cv2.VideoCapture(0)

# Infinite loop
while True:

	suc,img=camera.read()


	img=cv2.resize(img,(224,224))
	# expand dimensions
	test_img=np.array(img,dtype=np.float32)
	test_img=np.expand_dims(test_img,axis=0)
	# normalization
	normal_img=test_img/255.0

	# prediction model
	prediction=model.predict(normal_img)
	print("prediction is: ",prediction)

	cv2.imshow("the box",img)

	key = cv2.waitKey(1)
	if key == (32):
	    break
    
camera.release()

# close the open window
cv2.destroyAllWindows()
