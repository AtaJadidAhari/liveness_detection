from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import argparse
import imutils
import pickle
import time
import cv2
import os

probability_to_detect = 0.5
# load our serialized face detector from disk
print("loading face detector...")
protoPath = "./face_detector/deploy.prototxt"
modelPath = "./face_detector/res10_300x300_ssd_iter_140000.caffemodel"
	
net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

# load the liveness detector model and label encoder from disk
print("[INFO] loading liveness detector...")
liveness_detector_model_path = "./liveness.model"
model = load_model(liveness_detector_model_path)

label_path = './le.pickle'
le = pickle.loads(open(label_path, "rb").read())

# initialize the video stream and allow the camera sensor to warmup
print("[INFO] starting video stream...")

cap = cv2.VideoCapture(0)
time.sleep(2.0)


while True:
	
	ret, frame = cap.read()
	frame = imutils.resize(frame, width=600)

	
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
		(300, 300), (104.0, 177.0, 123.0))

	
	net.setInput(blob)
	detections = net.forward()

	# loop over faces
	for i in range(0, detections.shape[2]):
		
		confidence = detections[0, 0, i, 2]

		# filter out weak detections
		if confidence > probability_to_detect:
			
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# ensure the detected bounding box does fall outside the
			# dimensions of the frame
			startX = max(0, startX)
			startY = max(0, startY)
			endX = min(w, endX)
			endY = min(h, endY)

			
			face = frame[startY:endY, startX:endX]
			face = cv2.resize(face, (32, 32))
			face = face.astype("float") / 255.0
			face = img_to_array(face)
			face = np.expand_dims(face, axis=0)

			
			#passing faces to liveness_model to determine if the face is "real" or "fake"
			preds = model.predict(face)[0]
			j = np.argmax(preds)
			label = le.classes_[j]
			

			#drawing boxes with labels
			label = "{}: {:.4f}".format(label, preds[j])
			cv2.putText(frame, label, (startX, startY - 10),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
			cv2.rectangle(frame, (startX, startY), (endX, endY),
				(0, 0, 255), 2)

	
	cv2.imshow("Frame", frame)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break


cap.release()
cv2.destroyAllWindows()
