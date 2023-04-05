import cv2
import numpy as np
import os.path
from config import chars

w = 135
h = 60
threshold = 0.8

def MatchP1(frame_gray, frame):
  P1_current_char = []
  P2_current_char = []

  for char in chars:
    if os.path.exists(f'ref/CharsP/{char[1]}P1.png'):
      imgP1 = cv2.imread(f'ref/CharsP/{char[1]}P1.png',0)
      w, h =  imgP1.shape[::-1]
      res = cv2.matchTemplate(frame_gray, imgP1,cv2.TM_CCOEFF_NORMED)
      threshold = 0.8
      loc = np.where(res >= threshold)
      for pt in zip(*loc[::-1]):
        cv2.rectangle(frame, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)
      if loc[0].size > 0:
        P1_current_char.append(char[1])

    if os.path.exists(f'ref/CharsP/{char[1]}P2.png'):
      imgP2 = cv2.imread(f'ref/CharsP/{char[1]}P2.png',0)
      w, h =  imgP2.shape[::-1]
      res = cv2.matchTemplate(frame_gray, imgP2,cv2.TM_CCOEFF_NORMED)
      threshold = 0.8
      loc = np.where(res >= threshold)
      for pt in zip(*loc[::-1]):
        cv2.rectangle(frame, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)
      if loc[0].size > 0:
        P2_current_char.append(char[1])
      
  P1_current_char = list(dict.fromkeys(P1_current_char))
  P2_current_char = list(dict.fromkeys(P2_current_char))
  print(P1_current_char,'vs',P2_current_char )

  # return P1_current_char