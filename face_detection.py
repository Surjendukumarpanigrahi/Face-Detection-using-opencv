import cv2
from random import randrange

#load some pre trained data fo frontal from open cv(haae cascade algorithim)
trained_face_data=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#img=cv2.imread('rdj.jpg')

#must convert to grayscale
webcam=cv2.VideoCapture(0)

while True:
    
    
    sucessful_frame_read,frame=webcam.read()
    
    grayscaled_img=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    
    

    #detect faces
    face_coordinates=trained_face_data.detectMultiScale(grayscaled_img)


    for [x,y,w,h] in face_coordinates:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(randrange(256),randrange(256),randrange(256)),8)


    cv2.imshow('face',frame)
    key= cv2.waitKey(1)
    
    
    if key==81 or key==113:
        break