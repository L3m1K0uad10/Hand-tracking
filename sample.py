import time

import cv2
import mediapipe as mp

from module import HandDetector 


pTime = 0 
cTime = 0 
cap = cv2.VideoCapture(0)

detector = HandDetector()

while True:
    success, img = cap.read()
    img = detector.find_hands(img)
    lm_list = detector.find_position(img, draw = False)
    if len(lm_list) != 0:
        print(lm_list[4])
    
    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
    
    cv2.imshow("Image", img)
    cv2.waitKey(1)