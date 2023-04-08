import cv2
import datetime
import numpy as np
from trackchar import MatchP1

def convert_frame_to_timestamp(frame_count, fps):
  seconds = int(frame_count)/float(fps)
  timestamp = str(datetime.timedelta(seconds=seconds))
  return timestamp

def check_round(frame_gray):
  crop_frame = frame_gray[205:442, 742:842]
  round1 = cv2.imread('./ref/rounds/round1.jpg',0)
  threshold = 0.8
  res_round1 = cv2.matchTemplate(crop_frame, round1,cv2.TM_CCOEFF_NORMED)
  loc_round1 = np.where(res_round1 >= threshold)
  if loc_round1[0].size > 0:
    return 1
  round2 = cv2.imread('./ref/rounds/round2.jpg',0)
  res_round2 = cv2.matchTemplate(crop_frame, round2,cv2.TM_CCOEFF_NORMED)
  loc_round2 = np.where(res_round2 >= threshold)
  if loc_round2[0].size > 0:
    return 2
  crop_frame_final = frame_gray[202:332, 482:810]
  round3 = cv2.imread('./ref/rounds/round3.jpg',0)
  res_round3 = cv2.matchTemplate(crop_frame_final, round3,cv2.TM_CCOEFF_NORMED)
  loc_round3 = np.where(res_round3 >= threshold)
  if loc_round3[0].size > 0:
    return 3

def find_round(frame_count, frame_gray, frame):
  threshold = 0.8
  round = cv2.imread('./ref/rounds/round.jpg',0)
  w, h =  round.shape[::-1]
  res = cv2.matchTemplate(frame_gray, round,cv2.TM_CCOEFF_NORMED)
  loc = np.where(res >= threshold)
  for pt in zip(*loc[::-1]):
    cv2.rectangle(frame, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)
  if loc[0].size > 0:
    return [check_round(frame_gray), frame_count]
  
def check_ko(frame):
  crop_frame = frame[255:515, 457:580]
  ko = cv2.imread('./ref/rounds/ko.jpg',0)
  threshold = 0.75
  res_ko = cv2.matchTemplate(crop_frame, ko,cv2.TM_CCOEFF_NORMED)
  loc_ko = np.where(res_ko >= threshold)
  if loc_ko[0].size > 0:
    cv2.imwrite('ko.jpg',frame)  
    print('ko found')

if __name__ == '__main__':
  cap = cv2.VideoCapture('./match3.mp4')
  if not cap.isOpened():
    print('error opening video')
    exit(0)

  video_fps = cap.get(cv2.CAP_PROP_FPS)
  frame_count = 0
  print('Video Dimensions: ', cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

  round1_frame = 0
  round2_frame = 0
  round3_frame = 0

  current_round = None

  while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
      break

    frame_count += 1
    
    if (frame_count % 2) == 0:
      frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
      if find_round(frame_count, frame_gray, frame):
        if find_round(frame_count, frame_gray, frame)[0] == 1:
          round1_frame = find_round(frame_count, frame_gray, frame)[1]
        if find_round(frame_count, frame_gray, frame)[0] == 2:
          round2_frame = find_round(frame_count, frame_gray, frame)[1]
        if find_round(frame_count, frame_gray, frame)[0] == 3:
          round3_frame = find_round(frame_count, frame_gray, frame)[1]

      if (round3_frame != 0):
        check_ko(frame)

      if (round1_frame > 0 and frame_count == round1_frame+40):
        cv2.imwrite('round1.jpg',frame)  
        # break

      cv2.imshow('detected',frame)

    if cv2.waitKey(1) == ord('q'):
      break
  
  print('round1:',convert_frame_to_timestamp(round1_frame,video_fps))
  print('round2:',convert_frame_to_timestamp(round2_frame,video_fps))
  print('round3:',convert_frame_to_timestamp(round3_frame,video_fps))

  cap.release()
  cv2.destroyAllWindows()
  round1 = cv2.imread('round1.jpg',0)
  MatchP1(round1,round1)
