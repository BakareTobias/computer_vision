import cv2
import mediapipe as mp
import time

class FaceDetector():
    def __init__(self, min_detection_confidence=0.5, pTime=0, cTime=0):
        self.min_detection_confidence = min_detection_confidence
        self.pTime = pTime
        self.cTime = cTime

        self.mpFaceDetection = mp.solutions.face_detection
        self.faceDetection = self.mpFaceDetection.FaceDetection()
        self.mpDraw = mp.solutions.drawing_utils

    def findFaces(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.faceDetection.process(imgRGB)

        if results.detections:
            for id, detection in enumerate(results.detections):
                if draw:
                    self.mpDraw.draw_detection(img, detection, self.mpDraw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2))
                    self.BBox = detection.location_data.relative_bounding_box
                    cv2.putText(img, f'{detection.score[0]*100:.0f}%', (int(self.BBox.xmin*img.shape[1]), int(self.BBox.ymin*img.shape[0]-20)), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)


        return img

    def displayFPS(self, img):
        self.cTime = time.time()
        fps = 1/(self.cTime-self.pTime)
        self.pTime = self.cTime

        cv2.putText(img, f'FPS: {int(fps)}', (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

        return img