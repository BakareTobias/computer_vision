import unittest

from modules.hand_tracking_module import HandDetector

hand=HandDetector()

class TestHandDetector(unittest.TestCase):
    def test_findHads(self):
        pass

    def test_findPosition(self):
        pass

    def test_normalizeLandmarks(self):
        hand_coordinates = {0: (463, 464), 1: (425, 423), 2: (404, 362), 3: (383, 313), 4: (348, 285), 5: (462, 286), 6: (459, 223), 7: (458, 182), 8: (461, 147), 9: (504, 285), 10: (519, 212), 11: (530, 167), 12: (540, 126), 13: (540, 302), 14: (569, 234), 15: (592, 192), 16: (610, 152), 17: (571, 333), 18: (607, 289), 19: (632, 261), 20: (651, 234)}
        hand_coordinates_norm = hand.normalizeLandmarks(hand_coordinates)


        hand_coordinates_zero_error_case = {0: (463, 464), 1: (425, 423), 2: (404, 362), 3: (383, 313), 4: (348, 285), 5: (462, 464), 6: (459, 223), 7: (458, 182), 8: (461, 147), 9: (504, 285), 10: (519, 212), 11: (530, 167), 12: (540, 126), 13: (540, 302), 14: (569, 234), 15: (592, 192), 16: (610, 152), 17: (462, 333), 18: (607, 289), 19: (632, 261), 20: (651, 234)}
        hand_coordinates_zero_error_case_norm = hand.normalizeLandmarks(hand_coordinates_zero_error_case)


        self.assertEqual(len(hand_coordinates),len(hand_coordinates_norm.keys())/2)#each value tuple in hand coordinates norm becomes 2 separate items in the hand_coordinates_norm list

        self.assertIsNone(hand_coordinates_zero_error_case_norm)#if zero division error in calculation, function should return None

    def test_displayFPS(self):
        pass


if __name__ == "__main__":
    unittest.main()