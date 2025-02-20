import time

import cv2
import mediapipe as mp



cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands # formality before using the model
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils # formality before using the model in order to draw the landmarks in the loop at line 27


pTime = 0 # previous time
cTime = 0 # current time

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
            # get the id number and get also the landmarks info(x, y, z) of the hand
            for id, lm in enumerate(hand_marks.landmark):
                #print(id, lm)
                height, width, channel = img.shape
                channel_x, channel_y = int(lm.x * width), int(lm.y * height)
                print(id, channel_x, channel_y)
                # if id == 0:
                cv2.circle(img, (channel_x, channel_y), 15, (255, 0, 255), cv2.FILLED)

            mpDraw.draw_landmarks(img, hand_marks, mpHands.HAND_CONNECTIONS) # draw the landmarks on the hand
                                                   # draw the connection by using mpHands.HAND_CONNECTIONS
    
    # calculate the frame rate
    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime

    # display the frame rate on the screen
    cv2.putText(
        img, 
        str(int(fps)), 
        (10, 70), 
        cv2.FONT_HERSHEY_PLAIN, 
        3, 
        (255, 0, 255), 
        3
    )
    
    cv2.imshow("Image", img)

    cv2.waitKey(1)
