import cv2
import numpy as np
from matplotlib import pyplot as plt
from imutils import perspective
from imutils import contours
import imutils

import random as rng

## Read
##"C:/Users/israe/Downloads/frames_ml_trim7s/frame%d.jpg" % count, image)  

img = cv2.imread("C:/Users/israe/Downloads/frames_ml_trim7s/frame200.jpg")

## convert to hsv
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

## mask of green (36,25,25) ~ (86, 255,255)
# mask = cv2.inRange(hsv, (36, 25, 25), (86, 255,255))
mask = cv2.inRange(hsv, (36, 40, 40), (77, 255,255))
#thresh , im_bw = cv2.threshold(hsv,(36, 40, 40), (77, 255,255),0)
## slice the green
imask = mask>0
green = np.zeros_like(img, np.uint8)
green[imask] = img[imask]

## save 
#cv2.imwrite("green.png", green)


new_green = cv2.cvtColor(green, cv2.COLOR_RGB2GRAY)
#cv2.imshow("green", cv2.resize(new_green,None,fx=0.75, fy=0.75))
#cv2.waitKey(0)

thresh, im_bw = cv2.threshold(new_green,180,255, 0)
#cv2.imshow("white", cv2.resize(im_bw,None,fx=0.75, fy=0.75))
#cv2.waitKey(0)



kernel = np.ones((5,5),np.uint8)
dilation = cv2.dilate(im_bw,kernel,iterations = 2)

erode = cv2.erode(dilation,None,iterations = 1) 
#cv2.imshow("eroded", cv2.resize(erode,None,fx=0.75, fy=0.75))
#cv2.waitKey(0)

kernel = np.ones((45,45),np.uint8)
close = cv2.morphologyEx(erode, cv2.MORPH_CLOSE, kernel)


contours, hierarchy = cv2.findContours(close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

minArea = 20 * 20 
# draw the contours with area bigger than a minimum and that is almost rectangular 
counter = 1
for cnt in contours:
  x,y,w,h = cv2.boundingRect(cnt)
  area = cv2.contourArea(cnt)
  if area > (w*h*.60) and area > minArea:
    #original = cv2.rectangle(img, (x,y),(x+w,y+h), (0,0,255), 3)
    
    #cv2.circle(img, (int(x), int(y)), 9, (255,0,0), -1)
    
    #cv2.circle(img, (int(x+w), int(y+h)), 9, (255,0,0), -1)
    cv2.circle(img, (int(x+(w/2)), int(y+(h/2) )   ), 9, (255,0,0), -1)
    print("bbox: ",x,y,x+w,y+h)
    counter = counter +1 

cv2.imshow("image",cv2.resize( img,None,fx=0.75,fy=0.75))

cv2.waitKey(0)
