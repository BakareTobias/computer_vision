import pickle
import numpy as np
from keras.models import load_model


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
    
class DynamicGestureDetection:
    def __init__(self, path_to_model):
        self.path_to_model = path_to_model
        self.model = load_model(self.path_to_model)

    def predict(self,dataset_instance, classes, sequence_length=27, no_features=42):
        if isinstance(dataset_instance, np.ndarray):
            pass
        else: 
            dataset_instance = np.array(dataset_instance).reshape(1, sequence_length, no_features)

        result_array = self.model.predict(dataset_instance)

        gesture_detected_dynamic = self.model.predict(dataset_instance)
        
        predicted_class = np.argmax(gesture_detected_dynamic, axis=1)[0]
        confidence = np.max(gesture_detected_dynamic)
        gesture_detected = classes[predicted_class]

        return gesture_detected, confidence