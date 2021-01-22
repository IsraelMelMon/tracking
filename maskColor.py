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
cv2.imshow("green", cv2.resize(new_green,None,fx=0.75, fy=0.75))
cv2.waitKey(0)

thresh, im_bw = cv2.threshold(new_green,180,255, 0)
cv2.imshow("white", cv2.resize(im_bw,None,fx=0.75, fy=0.75))
cv2.waitKey(0)



kernel = np.ones((5,5),np.uint8)
dilation = cv2.dilate(im_bw,kernel,iterations = 2)

erode = cv2.erode(dilation,None,iterations = 1) 
cv2.imshow("eroded", cv2.resize(erode,None,fx=0.75, fy=0.75))
cv2.waitKey(0)

kernel = np.ones((45,45),np.uint8)
close = cv2.morphologyEx(erode, cv2.MORPH_CLOSE, kernel)


contours, hierarchy = cv2.findContours(close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

minArea = 20 * 20 
# draw the contours with area bigger than a minimum and that is almost rectangular 
for cnt in contours:
  x,y,w,h = cv2.boundingRect(cnt)
  area = cv2.contourArea(cnt)
  if area > (w*h*.60) and area > minArea:
    original = cv2.rectangle(img, (x,y),(x+w,y+h), (0,0,255), 3)
    print("bbox: ",x,y,x+w,y+h)
cv2.imshow("image",cv2.resize( img,None,fx=0.75,fy=0.75))

cv2.waitKey(0)
cv2.destroyAllWindows()
#print(cnts)
#cnts = cnts[0] if imutils.is_cv2() else cnts[1]
#contours_poly = [None]*len(contours) = boundRect = radius
#print(list( enumerate(cnts)))

"""

for i, c in enumerate(cnts):
    contours_poly[i] = cv2.approxPolyDP(c, 3, True)
    boundRect[i] = cv2.boundingRect(contours_poly[i])
    centers[i], radius[i] = cv2.minEnclosingCircle(contours_poly[i])
"""
drawing = np.zeros((dilation.shape[0], dilation.shape[1], 3), dtype=np.uint8)

for i in range(len(contours)):
        color = (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256))
        cv2.drawContours(drawing, contours_poly, i, color)
        cv2.rectangle(drawing, (int(boundRect[i][0]), int(boundRect[i][1])), \
          (int(boundRect[i][0]+boundRect[i][2]), int(boundRect[i][1]+boundRect[i][3])), color, 2)
        cv2.circle(drawing, (int(centers[i][0]), int(centers[i][1])), int(radius[i]), color, 2)
cv2.imshow('Contours', drawing)
cv2.waitKey(0)

"""
print(len(cnts))
cnt = cnts[1]
a = img.copy()
cv2.drawContours(a, [cnt], 2, (0,0,255), 5)
cv2.imshow("Image", cv2.resize(a,None, fx=0.5, fy=0.5)) 

cv2.waitKey(0)
"""

#for c in cnts:
    # This is to ignore that small hair countour which is not big enough
    #if cv2.contourArea(c) < 500:
    #    continue
#print(cnts[0][0])
# compute the rotated bounding box of the contour
"""box = cv2.minAreaRect(cnts[0][0])
box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
box = np.array(box, dtype="int")

# order the points in the contour such that they appear
# in top-left, top-right, bottom-right, and bottom-left
# order, then draw the outline of the rotated bounding
# box
box = perspective.order_points(box)
# draw the contours on the image
orig = img.copy()
cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 5)

# loop over the original points
for (xA, yA) in list(box):
    # draw circles corresponding to the current points and
    cv2.circle(orig, (int(xA), int(yA)), 9, (0,0,255), -1)
    cv2.putText(orig, "({},{})".format(xA, yA), (int(xA - 50), int(yA - 10) - 20),
        cv2.FONT_HERSHEY_SIMPLEX, 1.8, (255,0,0), 5)

    # show the output image, resize it as per your requirements
    cv2.imshow("Image", cv2.resize(orig,(800,600))) 

cv2.waitKey(0)


"""