import modules.hand_tracking_module as htm
import cv2
import os
import csv


path = 'ML_pipeline/datasets'
#1. start media pipe and hand detection with landmarks
def main(label,new_file):
    #setup webcam capture and hand detector
    cap = cv2.VideoCapture(0)#using webcam no 0
    cap.set(1, 640)  # Width
    cap.set(4, 480)  # Height

    #callback to hand detector module
    hand_detector = htm.HandDetector()

    counter = 0

    #create label.csv
    file = label + ".csv"

    #define headers
    headers = ['x0', 'y0', 'x1', 'y1', 'x2', 'y2', 'x3', 'y3', 'x4', 'y4', 'x5', 'y5', 'x6', 'y6', 'x7', 'y7', 'x8', 'y8', 'x9', 'y9', 'x10', 'y10', 'x11', 'y11', 'x12', 'y12', 'x13', 'y13', 'x14', 'y14', 'x15', 'y15', 'x16', 'y16', 'x17', 'y17', 'x18', 'y18', 'x19', 'y19', 'x20', 'y20', 'label']
    if new_file:
        #create csv file, add headers
        with open(os.path.join(path, file), 'w', newline="") as f:
            writer = csv.writer(f)
            writer.writerow( headers)
            
    while True:
        #setup image capture from webcam
        success, img = cap.read()#reading the image from webcam
        img = hand_detector.findHands(img,bothHands=False,draw=True)

        #only one hand is tracked for this program
        hand0_landmark_coordinates = hand_detector.findPosition(img, handNo=0, draw=False)
        
        #2. every X frames, capture the landmarks
        if hand0_landmark_coordinates:
            if (counter % 20) == 0:

                dataset_instance = []
                for key in hand0_landmark_coordinates:
                    #2 a. each landmark recomputed as position relative to landmark_0
                    # flipped y axis(it has 0 at the top of window)
                    x = hand0_landmark_coordinates[key][0] - hand0_landmark_coordinates[0][0]
                    y = -1 * (hand0_landmark_coordinates[key][1] - hand0_landmark_coordinates[0][1]) 
                    
                    #2 b. each landmark coordinate normalized according to palm width(5-17)
                    standard_hand_width =  abs(hand0_landmark_coordinates[5][0] - hand0_landmark_coordinates[17][0])
                    x /= standard_hand_width
                    # and palm height(0-5)
                    standard_hand_height = abs(hand0_landmark_coordinates[0][1] - hand0_landmark_coordinates[5][1])
                    y /= standard_hand_height

                    x = round(x,6)
                    y = round(y,6)

                    dataset_instance.append(x)
                    dataset_instance.append(y)

                #3. assign label         
                dataset_instance.append(label)
                
                #4. add to training set
                #file = 'peace.csv'
                with open(os.path.join(path, file), 'a', newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(dataset_instance)

            #5. increase counter
            counter +=1
      

        #7. save dataset to ML_pipeline/datasets
        cv2.imshow("Image", img)
        cv2.waitKey(1)#waiting for 1 millisecond before showing the next frame



if __name__ == "__main__":
    main(label='high_five',new_file=False)
