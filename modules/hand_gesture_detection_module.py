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
        #format dataset appropriately
        dataset_instance = np.array(dataset_instance)
        dataset_instance = dataset_instance.reshape(1,-1)

        gesture_detected = self.model.predict(dataset_instance)
        proba = self.model.predict_proba(dataset_instance)

        confidence = proba[0][gesture_detected[0]]
        gesture_detected = classes[gesture_detected[0]]

        return gesture_detected, confidence
    
class DynamicGestureDetection:
    def __init__(self, path_to_model):
        self.path_to_model = path_to_model
        self.model = load_model(self.path_to_model)

    def predict(self,dataset_instance, classes, sequence_length, no_features=42):
        #format dataset appropriately
        dataset_instance = np.array(dataset_instance)
        #if no of clips != sequence_length??
        if dataset_instance.shape[0] !=sequence_length:
            raise ValueError("sequence_length must match no. of clips provided")
        if dataset_instance.shape[-1] != no_features:
             raise ValueError("no_features must match no. of features provided")
             

        dataset_instance = dataset_instance.reshape(1, sequence_length, no_features)

        gesture_detected_dynamic = self.model.predict(dataset_instance)
        
        predicted_class = np.argmax(gesture_detected_dynamic, axis=1)[0]
        confidence = np.max(gesture_detected_dynamic)
        gesture_detected = classes[predicted_class]

        return gesture_detected, confidence