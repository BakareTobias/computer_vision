import logging

import cv2
import mediapipe as mp
import time
import pandas as pd



class HandDetector():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.7, trackCon=0.5, pTime=0, cTime=0):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.pTime = pTime
        self.cTime = cTime

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode = self.mode,
            max_num_hands = self.maxHands,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon
        )
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
                logging.error("Using handNo=1 will fail when there is only one hand detected")
                return None
        else:
            landmark_coordinates = None
        return landmark_coordinates

    def normalizeLandmarks(self,landmark_coordinates):
        normalized_landmarks = []
        for key in landmark_coordinates:
            #each landmark recomputed as position relative to landmark_0
            # flipped y axis(it has 0 at the top of window)
            x = landmark_coordinates[key][0] - landmark_coordinates[0][0]
            y = -1 * (landmark_coordinates[key][1] - landmark_coordinates[0][1]) 
            
            
            try:#skip edge cases that cause division by zero error
                #each landmark coordinate normalized according to palm width(5-17)
                standard_hand_width =  abs(landmark_coordinates[5][0] - landmark_coordinates[17][0])
                x /= standard_hand_width
                # and palm height(0-5)
                standard_hand_height = abs(landmark_coordinates[0][1] - landmark_coordinates[5][1])
                y /= standard_hand_height

                x = round(x,6)
                y = round(y,6)

                normalized_landmarks.append(x)
                normalized_landmarks.append(y)

            except ZeroDivisionError as e:
                logging.info(f"Skipping edge case due to {e}")
                normalized_landmarks = None
                break
        if normalized_landmarks:
            normalized_landmarks = pd.DataFrame(normalized_landmarks).T
        return normalized_landmarks

    def displayFPS(self, img):
        self.cTime = time.time()
        fps = 1/(self.cTime-self.pTime)
        self.pTime = self.cTime

        cv2.putText(img, f'FPS: {int(fps)}', (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

        return img


