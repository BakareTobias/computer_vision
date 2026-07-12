import modules.hand_tracking_module as htm
import cv2
import pyautogui as pag


#key variables
vol_step = 1 #how much to change the volume per increment
distance_threshold = 15 #how far apart fingers need to be to trigger volume change
palm_base_threshold = 280  #how high palm should be to trigger vol control

def main():
    cap = cv2.VideoCapture(0)#using webcam no 0
    hand_detector = htm.HandDetector()

    while True:
        #setup image capture from webcam
        success, img = cap.read()#reading the image from webcam
        img = hand_detector.findHands(img,bothHands=False)

        #track hand landmarks and get their coordinates
        hand0_landmark_coordinates = hand_detector.findPosition(img, handNo=0, draw=False)
        #hand1_landmark_coordinates = hand_detector.findPosition(img, handNo=1, draw=False)
        
        img = hand_detector.displayFPS(img)

        
        if hand0_landmark_coordinates:
            #use the base of the palm (landmark 0) as switch for activating the volume control. 
            # If the palm is above a certain threshold, the volume control will be deactivated.     
            #print(f"Landmark 0 coordinates: {hand0_landmark_coordinates[0]}")
            palm_base_y = hand0_landmark_coordinates[0][1]
            if palm_base_y > palm_base_threshold:
                pass
            else:
                #compute distance between landmarks 4 and 8 if both are detected
                if 4 in hand0_landmark_coordinates and 8 in hand0_landmark_coordinates:  
                    cv2.line(img, hand0_landmark_coordinates[4], hand0_landmark_coordinates[8], (255, 0, 255), 3)

                    x4,y4 = hand0_landmark_coordinates[4]
                    x8,y8 = hand0_landmark_coordinates[8]
                    distance = ((x8 - x4) ** 2 + (y8 - y4) ** 2) ** 0.5
                    #print(f"Distance between landmark 4 and 8: {distance:.2f} pixels")

                    if distance > distance_threshold:
                        pag.press('volumeup')  # Simulate volume up key press
                    elif distance < distance_threshold:
                        pag.press('volumedown')  # Simulate volume down key press



                
                     
        #cv2.imshow("Image", img)
        #cv2.waitKey(1)#waiting for 1 millisecond before showing the next frame

        
        
    


if __name__ == "__main__":
    main()
