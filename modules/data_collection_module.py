import numpy as np
import pandas as pd

import modules.hand_tracking_module as htm
import cv2
import os
import csv
import logging


class DataCollection():
    def __init__(self ):
        self.hand_detector = htm.HandDetector()


    #capture static frames from webcam or video feed
    def capture_data(self,
                    label,
                    new_file=True,
                    capture_rate = 15,
                    bothHands= False,
                    path_to_datasource_file=None, 
                    path_to_destination_folder=None,
                    ):

        if not bothHands:
            headers = ['x0', 'y0', 'x1', 'y1', 'x2', 'y2', 'x3', 'y3', 'x4', 'y4', 'x5', 'y5', 'x6', 'y6', 'x7', 'y7', 'x8', 'y8', 'x9', 'y9', 'x10', 'y10', 'x11', 'y11', 'x12', 'y12', 'x13', 'y13', 'x14', 'y14', 'x15', 'y15', 'x16', 'y16', 'x17', 'y17', 'x18', 'y18', 'x19', 'y19', 'x20', 'y20', 'label']

        elif bothHands: 
            headers = [ 'h0x0', 'h0y0', 'h0x1', 'h0y1', 'h0x2', 'h0y2', 'h0x3', 'h0y3', 'h0x4', 'h0y4', 'h0x5', 'h0y5', 'h0x6', 'h0y6', 'h0x7', 'h0y7',
                        'h0x8', 'h0y8', 'h0x9', 'h0y9', 'h0x10', 'h0y10', 'h0x11', 'h0y11', 'h0x12', 'h0y12', 'h0x13', 'h0y13', 'h0x14', 'h0y14', 'h0x15',
                        'h0y15', 'h0x16', 'h0y16', 'h0x17', 'h0y17', 'h0x18', 'h0y18', 'h0x19', 'h0y19', 'h0x20', 'h0y20', 'h1x0', 'h1y0',
                        'h1x1', 'h1y1', 'h1x2', 'h1y2', 'h1x3', 'h1y3', 'h1x4', 'h1y4', 'h1x5', 'h1y5', 'h1x6', 'h1y6', 'h1x7', 'h1y7', 'h1x8', 'h1y8', 
                        'h1x9', 'h1y9', 'h1x10', 'h1y10', 'h1x11', 'h1y11', 'h1x12', 'h1y12', 'h1x13', 'h1y13', 'h1x14', 'h1y14', 'h1x15', 'h1y15', 'h1x16', 
                        'h1y16', 'h1x17', 'h1y17', 'h1x18', 'h1y18', 'h1x19', 'h1y19', 'h1x20', 'h1y20','dist_x','dist_y','label', ]
                        

        counter = 0
        if path_to_datasource_file:  
            cap = cv2.VideoCapture(path_to_datasource_file)#using video path provided
        else:
            cap = cv2.VideoCapture(0)#using webcam 

        
        cap.set(1, 640)  # Width
        cap.set(4, 480)  # Height
        
        #create label.csv
        file = label + ".csv"

        if new_file:
            #create csv file, add headers
            with open(os.path.join(path_to_destination_folder, file), 'w', newline="") as f:
                writer = csv.writer(f)
                writer.writerow( headers)

        
        while True:
            #setup image capture from webcam
            success, img = cap.read()#reading the image from webcam
            try:
                if bothHands:
                    img = self.hand_detector.findHands(img,bothHands=True,draw=True)
                else:
                    img = self.hand_detector.findHands(img,bothHands=True,draw=True)
            except cv2.error:
                logging.info('End of video')
                break

            #only one hand is tracked for this program
            hand0_landmark_coordinates = self.hand_detector.findPosition(img, handNo=0, draw=False)

            if bothHands:
                hand1_landmark_coordinates = self.hand_detector.findPosition(img, handNo=1, draw=False)

        
            #2. every X frames, capture the landmarks
            if hand0_landmark_coordinates:
                if (counter % capture_rate) == 0:
                    hand0_norm = self.hand_detector.normalizeLandmarks(hand0_landmark_coordinates)
                    dataset_instance=hand0_norm
                    try:
                        hand1_norm = pd.DataFrame(np.zeros(hand0_norm.shape)) 
                        dataset_instance["dist_x"] = 0
                        dataset_instance["dist_y"] = 0
                        

                        if bothHands and hand1_landmark_coordinates:
                            hand1_norm = self.hand_detector.normalizeLandmarks(hand1_landmark_coordinates)

                            h0x0, h0y0 = hand0_landmark_coordinates[0]
                            h1x0, h1y0 = hand1_landmark_coordinates[0]
                            
        
                            #standard width and height
                            standard_hand_width =  abs(hand0_landmark_coordinates[5][0] - hand0_landmark_coordinates[17][0])
                            standard_hand_height = abs(hand0_landmark_coordinates[0][1] - hand0_landmark_coordinates[5][1])
        
                            #distance between palms, scaled by standard palm dimensions
                            dist_x = (h1x0 - h0x0)/standard_hand_width
                            dist_y = (h1y0 - h0y0)/standard_hand_height
        
                            
                            dataset_instance["dist_x"] = dist_x
                            dataset_instance["dist_y"] = dist_y

                        dataset_instance = pd.concat([hand0_norm, hand1_norm], axis=1)

                        #will return none if Zero error occurs   
                        if dataset_instance is not None:
                            #append label
                            dataset_instance["label"] = label
                            

                        
                        #4. add to training set
                        #file = 'peace.csv'
                        with open(os.path.join(path_to_destination_folder, file), 'a', newline="") as f:
                            writer = csv.writer(f)
                            writer.writerow(dataset_instance.iloc[0])
                    except AttributeError as e:
                        logging.info("hand0_norm is None. Skipping this frame.")

                #5. increase counter
                counter +=1
                

            img = cv2.flip(img,2)#flipped on x axis so finger movements, camera output, and cursor movement all align

            #7. save dataset to ML_pipeline/datasets
            cv2.putText(img, f"Collecting {label} data", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(img, f"Frames captured: {counter//capture_rate}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(img, f"Hit q to stop capturing", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Image", img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            elif cv2.waitKey(1) & 0xFF == ord('Q'):
                break
        logging.info('Task finished successfully')        

