import modules.hand_tracking_module as htm
import cv2
import pyautogui as pag
import numpy as np
from pynput.mouse import Button, Controller
import math

mouse = Controller()



def main():
    #setup webcam capture and hand detector
    cap = cv2.VideoCapture(0)#using webcam no 0
    cap.set(1, 640)  # Width
    cap.set(4, 480)  # Height

    #callback to hand detector module
    hand_detector = htm.HandDetector()
    frame_counter = 0
    while True:
        #setup image capture from webcam
        success, img = cap.read()#reading the image from webcam
        img = hand_detector.findHands(img,bothHands=False,draw=True)

        #only one hand is tracked for this program
        hand0_landmark_coordinates = hand_detector.findPosition(img, handNo=0, draw=False)

        if hand0_landmark_coordinates:

            #fingertip landmark tracking
            x8, y8 = hand0_landmark_coordinates[8]
            x5,y5 = hand0_landmark_coordinates[5]

            if frame_counter % 7 == 0: 
            
                #if thumb tip above landmark 0, scroll up, if below scroll down
                if y5 > y8:
                    cv2.putText(img, "Scroll down", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    mouse.scroll(0,-1)
                elif y5 < y8:
                    cv2.putText(img, "Scroll up", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    mouse.scroll(0,1)
            
        #display result
        img = cv2.flip(img,2)#flipped on x axis so finger movements, camera output, and cursor movement all align
        hand_detector.displayFPS(img)

        #cv2.imshow("Image", img)
        #cv2.waitKey(1)#waiting for 1 millisecond before showing the next frame
        frame_counter += 1

        
        
    


if __name__ == "__main__":
    main()
