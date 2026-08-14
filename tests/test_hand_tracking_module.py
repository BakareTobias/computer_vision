import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from modules.hand_tracking_module import HandDetector

hand=HandDetector()

class TestHandDetector(unittest.TestCase):
    fake_img = np.zeros((480,640,3), dtype=np.uint8)
    def test_findHads(self, ):

        img = hand.findHands(
            img=self.fake_img
        )
        #should be equal as no hands in fake img
        self.assertEqual(self.fake_img.all(), img.all())

    @patch("mediapipe.solutions.hands.Hands.process")
    def test_findPosition(self, mock_process):
        #test if fake img return None as no hands in fake img
        mock_results = MagicMock()
        mock_results.multi_hand_landmarks = [MagicMock()]

        mock_process.return_value = mock_results
        
        landmark_coordinates = hand.findPosition(
            img=self.fake_img,
            handNo=0
        )

        self.assertIsNone(landmark_coordinates)


    def test_normalizeLandmarks(self):
        hand_coordinates = {0: (463, 464), 1: (425, 423), 2: (404, 362), 3: (383, 313), 4: (348, 285), 5: (462, 286), 6: (459, 223), 7: (458, 182), 8: (461, 147), 9: (504, 285), 10: (519, 212), 11: (530, 167), 12: (540, 126), 13: (540, 302), 14: (569, 234), 15: (592, 192), 16: (610, 152), 17: (571, 333), 18: (607, 289), 19: (632, 261), 20: (651, 234)}
        hand_coordinates_norm = hand.normalizeLandmarks(hand_coordinates)


        hand_coordinates_zero_error_case = {0: (463, 464), 1: (425, 423), 2: (404, 362), 3: (383, 313), 4: (348, 285), 5: (462, 464), 6: (459, 223), 7: (458, 182), 8: (461, 147), 9: (504, 285), 10: (519, 212), 11: (530, 167), 12: (540, 126), 13: (540, 302), 14: (569, 234), 15: (592, 192), 16: (610, 152), 17: (462, 333), 18: (607, 289), 19: (632, 261), 20: (651, 234)}
        hand_coordinates_zero_error_case_norm = hand.normalizeLandmarks(hand_coordinates_zero_error_case)


        self.assertEqual(len(hand_coordinates),len(hand_coordinates_norm.keys())/2)#each value tuple in hand coordinates norm becomes 2 separate items in the hand_coordinates_norm list

        self.assertIsNone(hand_coordinates_zero_error_case_norm)#if zero division error in calculation, function should return None




if __name__ == "__main__":
    unittest.main()