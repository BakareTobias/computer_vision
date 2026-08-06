import modules.hand_tracking_module as htm
import modules.hand_gesture_detection_module as hgd
import cv2
import math
from keras.models import load_model
import numpy as np

#1. start media pipe and hand detection with landmarks
def main():
    #setup webcam capture and hand detector
    cap = cv2.VideoCapture(0)#using webcam no 0
    cap.set(1, 640)  # Width
    cap.set(4, 480)  # Height

    # load model
    path_to_RF_model = 'ML_pipeline/models/random_forest_az_thumbsup_pinch.pkl'
    LSTM_model = load_model('ML_pipeline/models/LSTM_jz.keras')

    classes_static = ['A','B','C','D','E','F','G','H','I','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','thumbs_up','pinch']
    classes_dynamic = ['J','Z']

    #load modules
    hand_detector = htm.HandDetector()
    gesture_detector = hgd.GestureDetection(path_to_model=path_to_RF_model)

    text = ''
    last_predict = math.inf
    frame_rate = 7
    frame_counter = 0

    clip = []

    while True:
        #setup image capture from webcam
        success, img = cap.read()#reading the image from webcam
        img = hand_detector.findHands(img,bothHands=False,draw=True)

        #only one hand is tracked for this program
        hand0_landmark_coordinates = hand_detector.findPosition(img, handNo=0, draw=False)
        
        if hand0_landmark_coordinates:
            #normalize landmarks to generate dataset instance
            norm_landmarks = hand_detector.normalizeLandmarks(hand0_landmark_coordinates)
            #will return none if Zero error occurs   
            if norm_landmarks is not None:
                #add frames to rolling buffer 'clip' for dynamic gestures
                dynamic_landmarks = norm_landmarks.iloc[:, [8,9,20,21]]
                clip.append(dynamic_landmarks)

                #if clip has enough frames, run LSTM model to predict dynamic gesture
                #also remove the oldest frame from clip to maintain a rolling buffer of 27 frames
                if len(clip) == 27:
                    #run ml model on dataset instance to predict as well as confidence score
                    dynamic_input = np.array(clip).reshape(1,27,4)
                    gesture_detected_dynamic = LSTM_model.predict(dynamic_input)

                    predicted_class = np.argmax(gesture_detected_dynamic, axis=1)[0]
                    confidence_dynamic = np.max(gesture_detected_dynamic)
                    gesture_name = classes_dynamic[predicted_class]
                    #print(gesture_detected_dynamic)
                    if confidence_dynamic > 0.9:
                        text += str(gesture_name)
                        last_predict = gesture_name
                        clip = []  # Clear the clip after prediction
                    else:
                        clip.pop(0)  # Remove the oldest frame to maintain a rolling buffer of 27 frames
                #run ml model on dataset instance to predict as well as confidence score
                gesture_detected, confidence = gesture_detector.predict(dataset_instance=norm_landmarks,classes=classes_static)

                if confidence > 0.5:
                    #can only hit space/backspace once every X frames

                    if gesture_detected == 'pinch':
                        if frame_counter % frame_rate == 0:
                            text = text[:-1]
                        cv2.putText(img, "Backspace", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                    elif gesture_detected == 'thumbs_up':
                        if frame_counter % frame_rate == 0:
                            text += ' '
                        cv2.putText(img, "Space", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                    elif gesture_detected != last_predict:
                        if frame_counter % frame_rate == 0:
                            text += str(gesture_detected)
                            last_predict = gesture_detected 
                else:
                    cv2.putText(img, "Unrecognized", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.putText(img, f"{text}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Image", img)
        cv2.waitKey(1)#waiting for 1 millisecond before showing the next frame
        frame_counter += 1


if __name__ == "__main__":
    main()
