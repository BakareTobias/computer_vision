import cv2
import modules.hand_tracking_module as htm
import os
import csv
import numpy as np
import pandas as pd

#for each clip
def main(label,video_path,csv_path,clip_id=0):
    file = label + '.csv'
    
    cap = cv2.VideoCapture(video_path)#using webcam no 0
    
    #handDetect
    hand_detector = htm.HandDetector()

    headers = [ 'h0x0', 'h0y0', 'h0x1', 'h0y1', 'h0x2', 'h0y2', 'h0x3', 'h0y3', 'h0x4', 'h0y4', 'h0x5', 'h0y5', 'h0x6', 'h0y6', 'h0x7', 'h0y7',
                'h0x8', 'h0y8', 'h0x9', 'h0y9', 'h0x10', 'h0y10', 'h0x11', 'h0y11', 'h0x12', 'h0y12', 'h0x13', 'h0y13', 'h0x14', 'h0y14', 'h0x15',
                'h0y15', 'h0x16', 'h0y16', 'h0x17', 'h0y17', 'h0x18', 'h0y18', 'h0x19', 'h0y19', 'h0x20', 'h0y20', 'h1x0', 'h1y0',
                'h1x1', 'h1y1', 'h1x2', 'h1y2', 'h1x3', 'h1y3', 'h1x4', 'h1y4', 'h1x5', 'h1y5', 'h1x6', 'h1y6', 'h1x7', 'h1y7', 'h1x8', 'h1y8', 
                'h1x9', 'h1y9', 'h1x10', 'h1y10', 'h1x11', 'h1y11', 'h1x12', 'h1y12', 'h1x13', 'h1y13', 'h1x14', 'h1y14', 'h1x15', 'h1y15', 'h1x16', 
                'h1y16', 'h1x17', 'h1y17', 'h1x18', 'h1y18', 'h1x19', 'h1y19', 'h1x20', 'h1y20','dist_x','dist_y','label', 'clip_id']

    path_exists = os.path.exists(csv_path)
    if not path_exists:
        #create csv file, add headers
        with open(os.path.join(csv_path), 'w', newline="") as f:
            writer = csv.writer(f)
            writer.writerow( headers)

#store landmark data(normalized)
#X = total no of frames//30 (30 frames per clip)
#if frame multiple of X, copy to new list
#store list
#append clip_id


    while True:
        success, img = cap.read()#reading the image from webcam
        
        img = hand_detector.findHands(img,bothHands=True,draw=True)

        hand0_landmark_coordinates = hand_detector.findPosition(img, handNo=0, draw=False)
        hand1_landmark_coordinates = hand_detector.findPosition(img, handNo=1, draw=False)

        #2. every X frames, capture the landmarks
        if hand0_landmark_coordinates:
            hand0_norm = hand_detector.normalizeLandmarks(hand0_landmark_coordinates)
            #init hand1 as  zeros
            dataset_instance = hand0_norm
            try:
                hand1_norm = pd.DataFrame(np.zeros(hand0_norm.shape)) 
            
                dataset_instance["dist_x"] = 0
                dataset_instance["dist_y"] = 0


                
                if hand1_landmark_coordinates:
                    hand1_norm = hand_detector.normalizeLandmarks(hand1_landmark_coordinates)
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
            except AttributeError:
                print("Error: hand0_norm is None. Skipping this frame.")
                #print("hand0_landmark_coordinates:", hand0_landmark_coordinates)
                pass
            dataset_instance = pd.concat([hand0_norm, hand1_norm], axis=1)


            #will return none if Zero error occurs   
            if dataset_instance is not None:
                #append label
                dataset_instance["label"] = label
                dataset_instance["clip_id"] = clip_id

            with open(os.path.join(csv_path), 'a', newline="") as f:
                                writer = csv.writer(f)
                                writer.writerow(dataset_instance.iloc[0])
        #resize fucking finally
        if success:
            img = cv2.resize(img,(640,480))
        cv2.imshow("Image", img)
        cv2.waitKey(1)#waiting for 1 millisecond before showing the next frame


video_path = 'ML_pipeline/ASL_words/cool/cool_13196.mp4'
label = 'cool'
csv_path = 'ML_pipeline/ASL_alphabet/Zz.csv'

#main(label,video_path,csv_path)

folder = os.listdir('ML_pipeline/ASL_alphabet/Z')
for i, clip in enumerate(folder):
    video_path = 'ML_pipeline/ASL_alphabet/Z/' + clip
    try:
        main('Z',video_path,csv_path,i)
    except cv2.error:
        pass