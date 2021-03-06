#makeMovie

import cv2
import os

image_folder = 'frames'
video_name = 'videoCollected.avi'

fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

video = cv2.VideoWriter(video_name, fourcc, 30, (width,height))

for image in images:
    video.write(cv2.imread(os.path.join(image_folder, image)))

cv2.destroyAllWindows()
video.release()