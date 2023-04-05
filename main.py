import cv2
import datetime
import numpy as np
from trackchar import MatchP1

def convert_frame_to_timestamp(frame_count, fps):
  seconds = int(frame_count)/float(fps)
  timestamp = str(datetime.timedelta(seconds=seconds))
  return timestamp

def check_round(frame_gray):
  round1 = cv2.imread('./ref/rounds/round1.jpg',0)
  w, h =  round1.shape[::-1]
  crop_frame = frame_gray[205:442, 742:842]
  cv2.imwrite('check_round_test.jpg',crop_frame)  
  # threshold = 0.8
  # w, h =  round1.shape[::-1]
  # res = cv2.matchTemplate(frame_gray, round1,cv2.TM_CCOEFF_NORMED)
  # loc = np.where(res >= threshold)
  # if loc[0].size > 0:
  #   check_round(frame_gray)

def find_round(frame_count, frame_gray, frame):
  threshold = 0.8
  round = cv2.imread('./ref/rounds/round.jpg',0)
  w, h =  round.shape[::-1]
  res = cv2.matchTemplate(frame_gray, round,cv2.TM_CCOEFF_NORMED)
  loc = np.where(res >= threshold)
  for pt in zip(*loc[::-1]):
    cv2.rectangle(frame, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)
  if loc[0].size > 0:
    check_round(frame_gray)

if __name__ == '__main__':
  cap = cv2.VideoCapture('./match1.mp4')
  if not cap.isOpened():
    print('error opening video')
    exit(0)

  video_fps = cap.get(cv2.CAP_PROP_FPS)
  frame_count = 0
  print('Video Dimensions: ', cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

  round1_frame = 0
  current_round = None

  while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
      break

    frame_count += 1
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if find_round(frame_count, frame_gray, frame):
      round1_frame = find_round(frame_count, frame_gray, frame)

    if (round1_frame > 0 and frame_count == round1_frame+40):
      cv2.imwrite('round1.jpg',frame)  
      break
      
    # MatchP1(frame_gray, frame)

    cv2.imshow('detected',frame)

    if cv2.waitKey(1) == ord('q'):
      break
  
  print(convert_frame_to_timestamp(round1_frame,video_fps))
  cap.release()
  cv2.destroyAllWindows()
  round1 = cv2.imread('round1.jpg',0)
  MatchP1(round1,round1)
