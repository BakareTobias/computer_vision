import cv2 
import mediapipe as mp 
import time 



#initialize variables for FPS calculation
pTime = 0
cTime = 0

cap = cv2.VideoCapture(0)#using webcam no 0

#face detection module from mediapipe#
mpFaceDetection = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils#drawing utils to draw the landmarks and connections on

while True:
    #setup image capture from webcam
    success, img = cap.read()#reading the image from webcam
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    #process the image to detect faces
    faceDetection = mpFaceDetection.FaceDetection()
    results = faceDetection.process(imgRGB)

    if results.detections:
        for id, detection in enumerate(results.detections):
            #draw the detection on the image
            mpDraw.draw_detection(img, detection, mpDraw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2))
            #print the detection score and bounding box coordinates
            #print(f"Detection {id}: Score: {detection.score}, Bounding Box: {detection.location_data.relative_bounding_box}")
            BBox = detection.location_data.relative_bounding_box

    #calculate FPS
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    #display
    cv2.putText(img, f'FPS: {int(fps)}', (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
    cv2.putText(img, f'{detection.score[0]*100:.0f}%', (int(BBox.xmin*img.shape[1]), int(BBox.ymin*img.shape[0]-20)), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
    cv2.imshow("Image", img)
    cv2.waitKey(1)#waiting for 1 millisecond before showing the next frame