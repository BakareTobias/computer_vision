import modules.hand_tracking_module as htm
import cv2
import pyautogui as pag
import numpy as np
from pynput.mouse import Button, Controller
import math

mouse = Controller()

screen_width, screen_height = pag.size()  # Get the screen size

#lists used to monitor click gesture state over consecutive frames. By waiting x frames before executing
#the program avoids misfires from mediapipe noise
#initialized to a value larger than the click threshold
inf = math.inf
left_click_test = [inf,inf,inf]
right_click_test = [inf,inf,inf]

#smoothing values are mouse positions averaged over consecutive frames to create a smoother movement pattern
#initialized to the minimum valid position on a screen
mouse_smoothingx = [0,0,0]
mouse_smoothingy = [0,0,0]

#when the distance between two landmarks(in pixels) is less than the click_threshold, a click gesture is registered
click_threshold = 40

#a scaling factor is used to make the virtual trackpad in the same ratio as the original screen. this ensure all pixels on the screen are mapped
scaling_factor = 8

#virtual trackpad dimensions
rectangle_top_left = (50, 250)
rectangle_height = int(screen_height/scaling_factor)
rectangle_width = int(screen_width/scaling_factor)
rectangle_bottom_right = (rectangle_top_left[0] + rectangle_width, rectangle_top_left[1] + rectangle_height)

#standard Kalman filter params
measured = []
predicted = []
dr_frame = np.zeros((400, 400, 3), np.uint8)
input = np.array((2, 1), np.float32)
output = np.zeros((2, 1), np.float32)

#kalman filter initialization
kalman_fil = cv2.KalmanFilter(4, 2)
kalman_fil.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
kalman_fil.transitionMatrix = np.array(
    [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32
)
kalman_fil.processNoiseCov = (
    np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
    * 0.03
)

def main():
    #setup webcam capture and hand detector
    cap = cv2.VideoCapture(0)#using webcam no 0
    cap.set(1, 640)  # Width
    cap.set(4, 480)  # Height

    #callback to hand detector module
    hand_detector = htm.HandDetector()

    while True:
        #setup image capture from webcam
        success, img = cap.read()#reading the image from webcam
        img = hand_detector.findHands(img,bothHands=False,draw=False)

        #only one hand is tracked for this program
        hand0_landmark_coordinates = hand_detector.findPosition(img, handNo=0, draw=False)

        if hand0_landmark_coordinates:

            #fingertip landmark tracking
            x8, y8 = hand0_landmark_coordinates[8]
            x12,y12 = hand0_landmark_coordinates[12]
            x16,y16 = hand0_landmark_coordinates[16]
            
            #computing distances between fingertips of index and middle finger, and middle finger and ring finger respectively
            distance1 = ((x12 - x8) ** 2 + (y12 - y8) ** 2) ** 0.5
            distance2 = ((x16 - x12) ** 2 + (y16 - y12) ** 2) ** 0.5

            #lists are updated to track only 3 most recent values
            left_click_test.append(distance1)
            left_click_test.pop(0)
            
            right_click_test.append(distance2)
            right_click_test.pop(0)

            #smoothing by averaging past 3 frames to reduce jitter
            mouse_smoothingx.append(x8)
            mouse_smoothingx.pop(0)
            mouse_smoothedx = sum(mouse_smoothingx)/len(mouse_smoothingx)

            mouse_smoothingy.append(y8)
            mouse_smoothingy.pop(0)
            mouse_smoothedy = sum(mouse_smoothingy)/len(mouse_smoothingy)

            #converting smoothed values into numpy array form
            mp = np.array([[np.float32(mouse_smoothedx)], [np.float32(mouse_smoothedy)]])
            measured.append((x8, y8))
            
            #check if index fingertip is within the virtual trackpad/rectangle area 
            if rectangle_top_left[0] <= x8 <= rectangle_bottom_right[0] and rectangle_top_left[1] <= y8 <= rectangle_bottom_right[1]:
                cv2.circle(img, (x8, y8), 10, (0, 255, 0), cv2.FILLED)  # Draw a green circle following landmark 8

                #run kalman filter
                kalman_fil.correct(mp)
                tp = kalman_fil.predict()

                mouse_kalmanx = tp[0]
                mouse_kalmany = tp[1]
                
                #scaling positions back to screen size
                x8_mapped = int(screen_width) - ((mouse_kalmanx - rectangle_top_left[0]) * scaling_factor) #flipped x axis
                y8_mapped = ((mouse_kalmany - rectangle_top_left[1]) * scaling_factor)

                #without this line, the program fails when the mouse is at corners/edges
                pag.FAILSAFE = False
                mouse.position = (x8_mapped,y8_mapped)
            
                #check for right click
                if (16 in hand0_landmark_coordinates) and (12 in hand0_landmark_coordinates) and (8 in hand0_landmark_coordinates): #16 is tip of middle finger               
                    
                    #if the distances between the landmarks are below established threshold for 3 consecutive frames, we can consider it a right click gesture
                    if all(right_click_test[i] < click_threshold for i in range(len(right_click_test))) and all(left_click_test[i] < click_threshold for i in range(len(left_click_test))):
                        cv2.putText(img, "Right click Detected", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        #pag.click(button='right')
                        mouse.press(Button.right)
                        mouse.release(Button.right)
                
                    #check for left click
                    #the left click works like a normal mouse in that holding the gesture will turn a regular click into a hold(for dragging operations)
                    #if the distances between the landmarks are below established threshold for 3 consecutive frames, we can consider it a left click gesture                    
                    elif all(left_click_test[i] < click_threshold for i in range(len(left_click_test))):
                        cv2.putText(img, "Left click Detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        mouse.press(Button.left)
                        mouse.position = (x8_mapped,y8_mapped)
                    
                    #releasing the gesture returns to regular cursor mode
                    elif not all(left_click_test[i] < click_threshold for i in range(len(left_click_test))):
                        mouse.release(Button.left)
                    
                    
            
        #the virtual trackpad
        cv2.rectangle(img, rectangle_top_left, rectangle_bottom_right, (255, 0, 255), 2)  # Draw a rectangle around the webcam feed

        #display result
        img = cv2.flip(img,2)#flipped on x axis so finger movements, camera output, and cursor movement all align
        hand_detector.displayFPS(img)

        cv2.imshow("Image", img)
        cv2.waitKey(1)#waiting for 1 millisecond before showing the next frame

        
        
    


if __name__ == "__main__":
    main()
