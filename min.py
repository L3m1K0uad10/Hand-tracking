import time

import cv2
import mediapipe as mp



cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands # formality before using the model
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils # formality before using the model in order to draw the landmarks in the loop at line 27



while True:
    success, img = cap.read()

    # send our RGB image to the model
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)   # convert to RGB
    results = hands.process(imgRGB) # process and send the image to the model
    #print(results)
    #print(results.multi_hand_landmarks)

    # open the object that we receive from the model and extract the infos within
    # we can have multiple hands in the image so we need to loop through the hands and extract the them one by one
    if results.multi_hand_landmarks:
        for hand_marks in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(img, hand_marks) # draw the landmarks on the hand

    cv2.imshow("Image", img)

    cv2.waitKey(1)
