import modules.hand_tracking_module as htm
import cv2
from pycaw.pycaw import AudioUtilities



def main():
    cap = cv2.VideoCapture(0)#using webcam no 0
    hand_detector = htm.HandDetector()

    while True:
        success, img = cap.read()#reading the image from webcam
        img = hand_detector.findHands(img,bothHands=False)
        hand0_landmark_coordinates = hand_detector.findPosition(img, handNo=0, draw=False)
        #hand1_landmark_coordinates = hand_detector.findPosition(img, handNo=1, draw=False)
        img = hand_detector.displayFPS(img)

        #print("Hand 0 Landmark Coordinates:", hand0_landmark_coordinates[4] if hand0_landmark_coordinates else "No hand detected")
        #print("Hand 1 Landmark Coordinates:", hand1_landmark_coordinates[4] if hand1_landmark_coordinates else "No hand detected")

        if hand0_landmark_coordinates:
            #coordinates of landmarks 4 and 8
            #print(f"landmark 4 coordinates: {hand0_landmark_coordinates[4] if hand0_landmark_coordinates else 'No hand detected'}")
            #print(f"landmark 8 coordinates: {hand0_landmark_coordinates[8] if hand0_landmark_coordinates else 'No hand detected'}")

            cv2.line(img, hand0_landmark_coordinates[4], hand0_landmark_coordinates[8], (255, 0, 255), 3)
            #compute distance between landmarks 4 and 8 if both are detected
            if (hand0_landmark_coordinates[4][1] < hand0_landmark_coordinates[12][1]) and (hand0_landmark_coordinates[12][1] < hand0_landmark_coordinates[8][1]):
                pass
            else:
                if 4 in hand0_landmark_coordinates and 8 in hand0_landmark_coordinates:  
                    x4,y4 = hand0_landmark_coordinates[4]
                    x8,y8 = hand0_landmark_coordinates[8]
                    distance = ((x8 - x4) ** 2 + (y8 - y4) ** 2) ** 0.5
                    #print(f"Distance between landmark 4 and 8: {distance:.2f} pixels")
                    distance_threshold = 30 

                    #PYCAW: Adjust volume based on distance between landmarks 4 and 8
                    device = AudioUtilities.GetSpeakers()
                    volume = device.EndpointVolume
                    #print(f"Audio output: {device.FriendlyName}")
                    #print(f"- Muted: {bool(volume.GetMute())}")
                    #print(f"- Volume level: {volume.GetMasterVolumeLevel()} dB")
                    #print(f"- Volume range: {volume.GetVolumeRange()[0]} dB - {volume.GetVolumeRange()[1]} dB")
                    vol = volume.GetMasterVolumeLevel()
                    vol_step = 1
                    if distance > distance_threshold:
                        vol +=vol_step
                    elif vol < distance_threshold:
                        vol -=vol_step
                    clamped_vol = max(volume.GetVolumeRange()[0], min(vol, volume.GetVolumeRange()[1]))
                    volume.SetMasterVolumeLevel(clamped_vol, None)

                
                     
        cv2.imshow("Image", img)
        cv2.waitKey(1)#waiting for 1 millisecond before showing the next frame

        
        
    


if __name__ == "__main__":
    main()
