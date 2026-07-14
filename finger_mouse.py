import modules.hand_tracking_module as htm
import cv2
import pyautogui as pag

screen_width, screen_height = pag.size()  # Get the screen size
#print(f"Screen size: {screen_width}x{screen_height}")
left_click_test = [100,100,100]
right_click_test = [100,100,100]

rectangle_top_left = (50, 200)
rectangle_height = 160
rectangle_width = 300
rectangle_bottom_right = (rectangle_top_left[0] + rectangle_width, rectangle_top_left[1] + rectangle_height)

def main():
    #setup webcam capture and hand detector
    cap = cv2.VideoCapture(0)#using webcam no 0
    cap.set(1, 640)  # Width
    cap.set(4, 480)  # Height
    hand_detector = htm.HandDetector()

    while True:
        #setup image capture from webcam
        success, img = cap.read()#reading the image from webcam
        img = hand_detector.findHands(img,bothHands=False)

        hand0_landmark_coordinates = hand_detector.findPosition(img, handNo=0, draw=False)

        if hand0_landmark_coordinates:

            x8, y8 = hand0_landmark_coordinates[8]
            x12,y12 = hand0_landmark_coordinates[12]
            x16,y16 = hand0_landmark_coordinates[16]

            distance1 = ((x12 - x8) ** 2 + (y12 - y8) ** 2) ** 0.5
            distance2 = ((x16 - x12) ** 2 + (y16 - y12) ** 2) ** 0.5

            left_click_test.append(distance1)
            left_click_test.pop(0)
            
            right_click_test.append(distance2)
            right_click_test.pop(0)
            
            
            #check if index fingertip is within the rectangle area 
            if rectangle_top_left[0] <= x8 <= rectangle_bottom_right[0] and rectangle_top_left[1] <= y8 <= rectangle_bottom_right[1]:
                cv2.circle(img, (x8, y8), 10, (0, 255, 0), cv2.FILLED)  # Draw a green circle at landmark 8
            
                #check for right click
                if (16 in hand0_landmark_coordinates) and (12 in hand0_landmark_coordinates) and (8 in hand0_landmark_coordinates): #16 is tip of middle finger               
                    
                    #if the distances between the landmarks are below a certain threshold for 3 consecutive frames, we can consider it a right click gesture
                    if all(right_click_test[i] < 20 for i in range(len(right_click_test))) and all(left_click_test[i] < 20 for i in range(len(left_click_test))):
                        cv2.putText(img, "Right click Detected", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                    #check for left click
                    #if the distances between the landmarks are below a certain threshold for 3 consecutive frames, we can consider it a left click gesture                    
                    elif all(left_click_test[i] < 20 for i in range(len(left_click_test))):
                        cv2.putText(img, "Left click Detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                

            

        cv2.rectangle(img, rectangle_top_left, rectangle_bottom_right, (255, 0, 255), 2)  # Draw a rectangle around the webcam feed

        #display result
        cv2.imshow("Image", img)
        cv2.waitKey(1)#waiting for 1 millisecond before showing the next frame

        
        
    


if __name__ == "__main__":
    main()
