import unittest
from modules.hand_gesture_detection_module import GestureDetection, DynamicGestureDetection


class TestGestureDetection(unittest.TestCase):
    #test model is loaded properly
    path_to_model = 'ML_pipeline/models/random_forest_az_thumbsup_pinch.pkl'

    def test_path_to_model(self):
        self.assertIsInstance(self.path_to_model, str)
        self.assertTrue(self.path_to_model.endswith("pkl"))

    gdt = GestureDetection(path_to_model)


    #test that predict function(with all argument passed correctly)
    #produces a gesture detected(str?) and a confidence(float)
    def test_predict(self):
        dataset_instance1 = [0.0,0.0,0.474576,0.253521,0.762712,0.71831,0.779661,1.098592,0.898305,1.380282,0.338983,1.0,0.372881,1.253521,0.322034,0.915493,0.305085,0.619718,-0.016949,0.971831,0.0,1.225352,0.0,0.816901,0.033898,0.535211,-0.338983,0.915493,-0.322034,1.169014,-0.271186,0.774648,-0.220339,0.492958,-0.661017,0.816901,-0.644068,1.070423,-0.559322,0.802817,-0.508475,0.591549]#'A'
        dataset_instance2 = [0.0,0.0,-0.666667,0.6,-2.888889,1.244444,-4.666667,1.755556,-5.666667,2.155556,-6.555556,1.0,-7.888889,1.088889,-5.444444,0.911111,-4.777778,0.866667,-7.222222,0.6,-8.222222,0.644444,-5.222222,0.555556,-4.777778,0.555556,-7.444444,0.2,-8.555556,0.244444,-5.777778,0.2,-5.0,0.2,-7.555556,-0.2,-8.888889,-0.133333,-6.888889,-0.133333,-6.0,-0.133333]#'thumbs_up'

        classes_static = ['A','B','C','D','E','F','G','H','I','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','thumbs_up','pinch']

        gesturedetected1, confidence1 = self.gdt.predict(dataset_instance1, classes_static)
        self.assertIsInstance(gesturedetected1,str)
        self.assertIsInstance(confidence1, float)

        gesturedetected2, confidence2 = self.gdt.predict(dataset_instance2, classes_static)
        self.assertIsInstance(gesturedetected2, str)
        self.assertIsInstance(confidence2, float)


class TestDynamicGestureDetection(unittest.TestCase):
    #test model is loaded properly
    path_to_model = 'ML_pipeline/models/LSMT_jz_all_landmarks_other.keras'

    def test_path_to_model(self):
        self.assertIsInstance(self.path_to_model, str)
        self.assertTrue(self.path_to_model.endswith("keras"))

    dgdt = DynamicGestureDetection(path_to_model)


    #test that predict function(with all argument passed correctly)
    #produces a gesture detected(str?) and a confidence(float)
    def test_predict(self):
        #test wrong length dataset_instance should raise value error
        wrong_length_dataset_instance = [0.0,0.0,0.474576,0.253521,0.762712,0.71831,0.779661,1.098592,0.898305,1.380282,0.338983,1.0,0.372881,1.253521,0.322034,0.915493,0.305085,0.619718,-0.016949,0.971831,0.0,1.225352,0.0,0.816901,0.033898,0.535211,-0.338983,0.915493,-0.322034,1.169014,-0.271186,0.774648,-0.220339,0.492958,-0.661017,0.816901,-0.644068,1.070423,-0.559322,0.802817,-0.508475,0.591549]#'A'
        classes_dynamic = ['J','Z','other']

        
        with self.assertRaises(ValueError) as context:
            gesture_detected, confidence = self.dgdt.predict(wrong_length_dataset_instance,
                                                         classes=classes_dynamic,
                                                         sequence_length=27,
                                                         no_features=42)
            
        self.assertEqual(str(context.exception), "sequence_length must match no. of clips provided")



    



if __name__ == "__main__":
    unittest.main()