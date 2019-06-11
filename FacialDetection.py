#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pickle


face_cascade =  cv2.CascadeClassifier('HaarCascade/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('HaarCascade/haarcascade_eye.xml')

# smile_cascade = cv2.CascadeClassifier('Nariz.xml')

recoganiser = cv2.face.LBPHFaceRecognizer_create()
recoganiser.read('trainer.yml')

labels = {}
with open("lebels.pickle", "rb") as f:
    temp_labels = pickle.load(f)
    labels = {v:k for k,v in temp_labels.items()}

cap = cv2.VideoCapture(1)

while 1:
    (_, frame) = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 2, 7)
    for (x, y, w, h) in faces:

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0xFF, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]

        # facial Recoganizer

        id_, conf = recoganiser.predict(roi_gray) # lower the confidence the better
        print(conf)


        if(conf<=80):
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = labels[id_]
            cv2.putText(frame, name, (x,y), font, 1,(0,255,255),2, cv2.LINE_AA)

        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0,
                          0xFF, 0), 2)

    cv2.imshow('frame', frame)

    k = cv2.waitKey(5) & 0xFF
    if k == ord('q'):
        break

cv2.destroyAllWindows()
cap.release()
