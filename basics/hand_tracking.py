import cv2
import mediapipe as mp
import time


cap = cv2.VideoCapture(0)#using webcam no 0

mpHands = mp.solutions.hands#using hands module from mediapipe
""" hands object has default params of 
        static_image_mode False     # if True, will continuously detect hands in the video stream, slowing function down
        max_num_hands 2             # maximum number of hands to detect
        detection confidence  0.5   # threshold for detection confidence
        tracking confidence  0.5    # threshold for tracking confidence
"""
hands = mpHands.Hands(
        static_image_mode = False,    # if True, will continuously detect hands in the video stream, slowing function down
        max_num_hands  =2,           # maximum number of hands to detect
        dec =  0.5,   # threshold for detection confidence
        tracking_confidence = 0.5    # threshold for tracking confidence
)
mpDraw = mp.solutions.drawing_utils#drawing utils to draw the landmarks and connections on the hand

pTime = 0
cTime = 0

while True:
    success, img = cap.read()#reading the image from webcam

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)#normalizing the landmark coordinates to pixel values
                #print(id, cx, cy)
                if id == 20 or id == 4:
                    cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)

            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)#
    
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.putText(img, f'FPS: {int(fps)}', (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)#waiting for 1 millisecond before showing the next frame

