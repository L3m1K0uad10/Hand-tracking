import time

import cv2
import mediapipe as mp



class HandDetector():
    def __init__(self, mode = False, max_hands = 2, model_complexity = 1, detection_confidence = 0.5, tracking_confidence = 0.5):
        self.mode = mode 
        self.max_hands = max_hands
        self.model_complexity = model_complexity
        self.detection_confidence = detection_confidence
        self.tracking_confidence = tracking_confidence

        self.mpHands = mp.solutions.hands 
        self.hands = self.mpHands.Hands(
            self.mode, 
            self.max_hands, 
            self.model_complexity,
            self.detection_confidence, 
            self.tracking_confidence
        )
        self.mpDraw = mp.solutions.drawing_utils 

    def find_hands(self, img, draw = True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)   # convert to RGB
        self.results = self.hands.process(imgRGB) # process and send the image to the model

        if self.results.multi_hand_landmarks:
            for hand_marks in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, hand_marks, self.mpHands.HAND_CONNECTIONS) 
    
        return img
    
    def find_position(self, img, hand_number = 0, draw = True):
        lm_list = []

        if self.results.multi_hand_landmarks:
            my_hand = self.results.multi_hand_landmarks[hand_number]
        
            for id, lm in enumerate(my_hand.landmark):
                height, width, channel = img.shape
                channel_x, channel_y = int(lm.x * width), int(lm.y * height)
                lm_list.append([id, channel_x, channel_y])
                if draw:
                    cv2.circle(img, (channel_x, channel_y), 15, (255, 0, 255), cv2.FILLED)
        
        return lm_list
    


def main():
    pTime = 0 
    cTime = 0 
    cap = cv2.VideoCapture(0)

    detector = HandDetector()

    while True:
        success, img = cap.read()
        img = detector.find_hands(img)
        lm_list = detector.find_position(img)
        if len(lm_list) != 0:
            print(lm_list[4])
        
        cTime = time.time()
        fps = 1/(cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
        
        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()