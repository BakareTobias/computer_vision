import modules.hand_tracking_module as htm
import modules.hand_gesture_detection_module as hgd
import cv2
import math

#1. start media pipe and hand detection with landmarks
def main():
    #setup webcam capture and hand detector
    cap = cv2.VideoCapture(0)#using webcam no 0
    cap.set(1, 640)  # Width
    cap.set(4, 480)  # Height

    # load model
    path_to_model = 'ML_pipeline/models/random_forest_as_thumbsup_pinch.pkl'
    classes = ['A','B','C','D','E','F','G','H','I','K','L','M','N','O','P','R','S','thumbs_up','pinch']

    #load modules
    hand_detector = htm.HandDetector()
    gesture_detector = hgd.GestureDetection(path_to_model=path_to_model)

    text = ''
    last_predict = math.inf
    frame_rate = 7
    frame_counter = 0

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
                #run ml model on dataset instance to predict as well as confidence score
                gesture_detected, confidence = gesture_detector.predict(dataset_instance=norm_landmarks,classes=classes)

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
