import cv2
vidcap = cv2.VideoCapture('C:/Users/israe/Downloads/trim7s.mp4')
success,image = vidcap.read()
count = 0
while success:
  cv2.imwrite("C:/Users/israe/Downloads/frames_ml_trim7s/frame%d.jpg"  \ 
                % count, image)     # save frame as JPEG file      
  success,image = vidcap.read()
  print('Read a new frame: ', success)
  count += 1