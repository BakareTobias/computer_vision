import cv2
import mediapipe as mp
import time



class HandDetector():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5, pTime=0, cTime=0):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.pTime = pTime
        self.cTime = cTime

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands()
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True, bothHands=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
                    if not bothHands:
                        break
        return img

    def findPosition(self, img, handNo=0, draw=True):
        landmark_coordinates = {}
        if self.results.multi_hand_landmarks:
            try: 
                myHand = self.results.multi_hand_landmarks[handNo]
                for id, lm in enumerate(myHand.landmark):
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    landmark_coordinates[id] = (cx, cy)
                    if draw:
                            cv2.circle(img, (cx, cy), 6, (255, 122, 211), cv2.FILLED)
            except IndexError:
                return None
        return landmark_coordinates

    def displayFPS(self, img):
        self.cTime = time.time()
        fps = 1/(self.cTime-self.pTime)
        self.pTime = self.cTime

        cv2.putText(img, f'FPS: {int(fps)}', (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

        return img


