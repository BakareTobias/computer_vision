import hand_tracking_module as htm
import cv2

def main():
    cap = cv2.VideoCapture(0)#using webcam no 0
    detector = htm.HandDetector()
    while True:
        success, img = cap.read()#reading the image from webcam
        img = detector.findHands(img)
        hand0_landmark_coordinates = detector.findPosition(img, handNo=0, draw=True)
        hand1_landmark_coordinates = detector.findPosition(img, handNo=1, draw=False)
        img = detector.displayFPS(img)

        print("Hand 0 Landmark Coordinates:", hand0_landmark_coordinates[4] if hand0_landmark_coordinates else "No hand detected")
        print("Hand 1 Landmark Coordinates:", hand1_landmark_coordinates[4] if hand1_landmark_coordinates else "No hand detected")

        cv2.imshow("Image", img)
        cv2.waitKey(1)#waiting for 1 millisecond before showing the next frame

        
        
    


if __name__ == "__main__":
    main()

