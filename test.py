import modules.hand_tracking_module as htm
import modules.face_detection_module as fdm
import cv2

def main():
    cap = cv2.VideoCapture(0)#using webcam no 0
    hand_detector = htm.HandDetector()
    face_detector = fdm.FaceDetector()

    while True:
        success, img = cap.read()#reading the image from webcam
        img = hand_detector.findHands(img)
        hand0_landmark_coordinates = hand_detector.findPosition(img, handNo=0, draw=True)
        hand1_landmark_coordinates = hand_detector.findPosition(img, handNo=1, draw=False)
        img = hand_detector.displayFPS(img)

        #print("Hand 0 Landmark Coordinates:", hand0_landmark_coordinates[4] if hand0_landmark_coordinates else "No hand detected")
        #print("Hand 1 Landmark Coordinates:", hand1_landmark_coordinates[4] if hand1_landmark_coordinates else "No hand detected")


        img = face_detector.findFaces(img)

        cv2.imshow("Image", img)
        cv2.waitKey(1)#waiting for 1 millisecond before showing the next frame

        
        
    


if __name__ == "__main__":
    main()

