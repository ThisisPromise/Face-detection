

import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


video = cv2.VideoCapture(0)

while True:
    success_video, frame = video.read()

    gray_scale_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    face = face_cascade.detectMultiScale(gray_scale_image, 1.1, 4)

    for (x, y, w, h) in face:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

    cv2.imshow('image', frame)
    cv2.waitKey(1)


