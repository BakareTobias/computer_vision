import modules.hand_tracking_module as htm
import cv2
import pickle
import pandas as pd

class GestureDetection():
    def __init__(self, path_to_model):
        self.path_to_model = path_to_model
        with open(self.path_to_model, 'rb') as f:
                self.model = pickle.load(f)

        pass

    def predict(self,dataset_instance,classes):
        gesture_detected = self.model.predict(dataset_instance)
        proba = self.model.predict_proba(dataset_instance)

        confidence = proba[0][gesture_detected[0]]
        gesture_detected = classes[gesture_detected[0]]

        return gesture_detected, confidence