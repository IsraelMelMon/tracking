import cv2
import numpy as np
from matplotlib import pyplot as plt
from imutils import perspective
from imutils import contours
import imutils
from imutils.video import VideoStream, FPS
import time
import random as rng
from scipy.spatial import distance as dist
from centroidTracker import CentroidTracker
from imutils.video import FileVideoStream
## Read
#"C:/Users/israe/Downloads/frames_ml_trim7s/frame%d.jpg" % count, image)  
#stream = cv2.VideoCapture('C:/Users/israe/Downloads/tracking/trim7s.mp4')
#success,image = vidcap.read()
#fvs = FileVideoStream('C:/Users/israe/detectVideo_Output.avi').start()

ct = CentroidTracker()
fvs = FileVideoStream('C:/Users/israe/detectVideo_Output.avi').start()
time.sleep(2.0)
H,W = None, None
fps = FPS().start()
#cv2.destroyAllWindows()
#imgNew = fvs.read()
#img = cv2.imread("C:/Users/israe/Downloads/frames_ml_trim7s/frame200.jpg")
#fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
#name = '/CountdetectVideo_Output_' + args.method + '.avi'
#videoOut = cv2.VideoWriter('CountdetectVideo_Output.mp4',fourcc, 20.0, (imgNew.shape[1], imgNew.shape[0]))
counter = 1
totalObjects = []
lastNumberItem = []
frameText = []
while fvs.more():
    """
    (grabbed, frame) = stream.read()
    
    
    if not grabbed:
        print("[INFO]: End of video frames")
        break

    
    if frame.all()==None:
        assert grabbed==None, "Video is in invalid directory or name is invalid"
    """
    img =fvs.read()
    ## convert to hsv
    #img = frame
    if img is None:
        break
    
    img = img[:,:1000]
    #second and first img = img[100:,:960]
    
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
    erode = cv2.dilate(im_bw,kernel,iterations = 2)

    erode = cv2.erode(erode,None,iterations = 1) 
    #cv2.imshow("eroded", cv2.resize(erode,None,fx=0.75, fy=0.75))
    #cv2.waitKey(0)

    kernel = np.ones((45,45),np.uint8)
    close = cv2.morphologyEx(erode, cv2.MORPH_CLOSE, kernel)


    contours, hierarchy = cv2.findContours(close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    minArea = 20 * 20 
    # draw the contours with area bigger than a minimum and that is almost rectangular 
    
    rects = []
    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        if area > (w*h*.60) and area > minArea:
            #original = cv2.rectangle(img, (x,y),(x+w,y+h), (0,0,255), 3)
            bbox = (x,y,x+w,y+h)
            #cv2.circle(img, (int(x), int(y)), 9, (255,0,0), -1)
            rects.append(bbox)
            #cv2.circle(img, (int(x+w), int(y+h)), 9, (255,0,0), -1)
            cv2.circle(img, (int(x+(w/2)), int(y+(h/2) )   ), 9, (255,0,0), -1)
            #print("bbox: ",x,y,x+w,y+h)
            #counter = counter +1 
 
                #rects = listOfbboxes
    #print("boundingBoxes: ", rects)
    rects = np.array(rects)
    objects = ct.update(rects)
	# loop over the tracked objects
    #if frameText != []:
    #    del frameText
    for (objectID, centroid) in objects.items():
        # draw both the ID of the object and the centroid of the
        # object on the output frame
        text = "{}".format(objectID+1)
        centroidText = cv2.putText(img, text, (centroid[0] - 10, centroid[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        #frameText = cv2.putText(img, text, (100, 100),
        #    cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 255, 0), 2)
        lastNumberItem.append(int(text))
    #print(lastNumberItem)
    if len(lastNumberItem) != 0 :
    # choose to display only the first class
            cv2.putText(img, str(max(lastNumberItem)), (150,150) ,cv2.FONT_HERSHEY_SIMPLEX, 2, [255, 255, 
            0], thickness=3)

    img = imutils.resize(img, width=800)
    cv2.imshow("Frame", img)
    #cv2.imwrite("Frame{}".format(counter)+".jpg",img)
    cv2.waitKey(1)
    
    key = cv2.waitKey(1) & 0xFF
    fps.update()
    counter = counter + 1
    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break
    
    
    fps.update()
    
# eam.stop()
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
cv2.destroyAllWindows()

fvs.stop()  
#videoOut.release()