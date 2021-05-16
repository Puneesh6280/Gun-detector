# -*- coding: utf-8 -*-
"""
Created on Sun May 16 13:28:36 2021

@author: DELL
"""

# OpenCV Python program to detect cars in video frame 
# import libraries of python OpenCV  
import cv2 
   
# capture frames from a video 
cap = cv2.VideoCapture(0) 
   
# Trained XML classifiers describes some features of some object we want to detect 
car_cascade = cv2.CascadeClassifier('cascade_gun.xml') 
   
# loop runs if capturing has been initialized. 
while True: 
    # reads frames from a video 
    ret, frames = cap.read() 
       
    # convert to gray scale of each frames 
    gray = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY) 
       
   
    # Detects cars of different sizes in the input image 
    cars = car_cascade.detectMultiScale(gray, 1.1, 35) 
       
    # To draw a rectangle in each cars 
    for (x,y,w,h) in cars: 
        cv2.rectangle(frames,(x,y),(x+w,y+h),(0,0,255),0) 
    cv2.imshow("cap", frames) 
       
    # Wait for Esc key to stop 
    if cv2.waitKey(33) == 27: 
        break
   
# De-allocate any associated memory usage 
cv2.destroyAllWindows() 