import os
from PIL import Image
import numpy as np
import cv2
import pickle

Base_Dir = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(Base_Dir, "Images")
face_cascade = cv2.CascadeClassifier('HaarCascade/haarcascade_frontalface_default.xml')
recoganiser = cv2.face.LBPHFaceRecognizer_create()

current_ID = 0
label_IDS = {}
y_lebels = []
x_train = []

for root, dirs, files in os.walk(image_dir):
	for file in files:
		if(file.endswith("png") or file.endswith("jpg") or file.endswith("JPG")):
			path = os.path.join(root, file)
			label = os.path.basename(os.path.dirname(path)).replace(" ","_").lower()
			# print(label)
			# print(path)
			if not label in label_IDS:
				label_IDS[label]= current_ID;
				current_ID+=1

			

			id_ = label_IDS[label]
			# x_train.append(path) #Verfy the image and convert into gray and numpy array
			# y_lebels.append(label) # some number for our labels

			pil_image = Image.open(path).convert('L')
			size = (600,600)
			final_image = pil_image.resize(size, Image.ANTIALIAS)
			image_array = np.array(pil_image, 'uint8')
			faces = face_cascade.detectMultiScale(image_array, 2, 7)
			for(x, y, w, h) in faces:
				roi = image_array[y:y+h, x:x+w]
				x_train.append(roi)
				y_lebels.append(id_)


# print(y_lebels)
# print(x_train)

with open("lebels.pickle", "wb") as f:
	pickle.dump(label_IDS, f)

recoganiser.train(x_train, np.array(y_lebels))
recoganiser.save("trainer.yml")

