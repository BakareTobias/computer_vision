import modules.hand_tracking_module as htm
import cv2
from pynput.mouse import Button, Controller
import pickle
import pandas as pd

mouse = Controller()



def main():
    #setup webcam capture and hand detector
    cap = cv2.VideoCapture(0)#using webcam no 0
    cap.set(1, 640)  # Width
    cap.set(4, 480)  # Height

    with open('ML_pipeline/models/random_forest_06.pkl', 'rb') as f:
        log_reg = pickle.load(f)

    classes = ['peace','high_five','sixer',"thumbs_up",'f_sign','take_the_l','pinch']


    #callback to hand detector module
    hand_detector = htm.HandDetector()
    frame_counter = 0
    while True:
        #setup image capture from webcam
        success, img = cap.read()#reading the image from webcam
        img = hand_detector.findHands(img,bothHands=False,draw=True)

        #only one hand is tracked for this program
        hand0_landmark_coordinates = hand_detector.findPosition(img, handNo=0, draw=False)
        dataset_instance = []
        for key in hand0_landmark_coordinates:
            

            #2 a. each landmark recomputed as position relative to landmark_0
            # flipped y axis(it has 0 at the top of window)
            x = hand0_landmark_coordinates[key][0] - hand0_landmark_coordinates[0][0]
            y = -1 * (hand0_landmark_coordinates[key][1] - hand0_landmark_coordinates[0][1]) 
            
            #2 b. each landmark coordinate normalized according to palm width(5-17)
            standard_hand_width =  abs(hand0_landmark_coordinates[5][0] - hand0_landmark_coordinates[17][0])
            try:#skip edge cases that cause division by zero error
                x /= standard_hand_width
                # and palm height(0-5)
                standard_hand_height = abs(hand0_landmark_coordinates[0][1] - hand0_landmark_coordinates[5][1])
                y /= standard_hand_height

                x = round(x,6)
                y = round(y,6)

                dataset_instance.append(x)
                dataset_instance.append(y)

            except ZeroDivisionError:
                print('Zero error')
                break
            
        try:#skip edge cases that cause division by zero error

            dataset_instance = pd.DataFrame(dataset_instance).T
            gesture_detected = log_reg.predict(dataset_instance)
            proba = log_reg.predict_proba(dataset_instance)
            proba = proba[0][gesture_detected[0]]
            #cv2.putText(img, f"{classes[gesture_detected[0]]} {proba[0][gesture_detected[0]]}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            if proba > 0.5:
                cv2.putText(img, f"{classes[gesture_detected[0]]}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                if frame_counter % 5 == 0: 
            
                    #if thumb tip above landmark 0, scroll up, if below scroll down
                    if classes[gesture_detected[0]] == 'peace':
                        cv2.putText(img, "Scroll down", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        mouse.scroll(0,-1)
                    elif classes[gesture_detected[0]] == 'take_the_l':
                        cv2.putText(img, "Scroll up", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        mouse.scroll(0,1)
            else:
                cv2.putText(img, "Unrecognized", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        except ValueError:
            pass    
        #display result
        img = cv2.flip(img,2)#flipped on x axis so finger movements, camera output, and cursor movement all align
        hand_detector.displayFPS(img)

        #cv2.imshow("Image", img)
        #cv2.waitKey(1)#waiting for 1 millisecond before showing the next frame
        frame_counter += 1

        
        
    


if __name__ == "__main__":
    main()
