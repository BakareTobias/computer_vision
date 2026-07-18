import modules.hand_tracking_module as htm
import cv2
import numpy as np


#1. start media pipe and hand detection with landmarks
def main(label):
    #setup webcam capture and hand detector
    cap = cv2.VideoCapture(0)#using webcam no 0
    cap.set(1, 640)  # Width
    cap.set(4, 480)  # Height

    #callback to hand detector module
    hand_detector = htm.HandDetector()

    counter = 0

    #create label.csv
    
    while True:
        #setup image capture from webcam
        success, img = cap.read()#reading the image from webcam
        img = hand_detector.findHands(img,bothHands=False,draw=False)

        #only one hand is tracked for this program
        hand0_landmark_coordinates = hand_detector.findPosition(img, handNo=0, draw=False)

        #2. every Xms, capture the landmarks
        if hand0_landmark_coordinates:
            dataset_instance = []
            for key in hand0_landmark_coordinates:
                dataset_instance.append(hand0_landmark_coordinates[key][0])
                dataset_instance.append(hand0_landmark_coordinates[key][1])
            dataset_instance.append(label)
            print(dataset_instance) 
            print()
            print()


            #3. assign label
            #4. add to training set

            #5. increase counter
            counter +=1
            #6. if counter = 100, stop loop
            if counter == 100:
                break

    #7. save dataset to ML_pipeline/datasets



if __name__ == "__main__":
    main(label='peace')
