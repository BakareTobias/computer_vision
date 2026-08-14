import os
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
from modules.data_collection_module import DataCollection

class TestDataCollection(unittest.TestCase):
    @patch("cv2.waitKey")
    @patch("cv2.VideoCapture")


    def test_capture_data(self,  mock_videoCapture, mock_waitKey):
        #by using mocks, i can simulate parts of the code involving cv2 and opening a webcam without actually doing so
        fake_cap = MagicMock()
        fake_cap.read.return_value = (
            True,
            np.zeros((480,640,3), dtype=np.uint8)
        )
        mock_videoCapture.return_value = fake_cap
        mock_waitKey.return_value = ord('q')

        label="A"
        path_to_destination_folder="."

        collector = DataCollection()
        collector.capture_data(
            label=label,
            path_to_destination_folder=path_to_destination_folder
        )
        
        #if new_file=True, new file created using label and path_to_destination_folder
        self.assertTrue(
            os.path.exists(f"{path_to_destination_folder}/{label}.csv")
        )

        os.remove(f"{path_to_destination_folder}/{label}.csv")

        #program end on q
        mock_waitKey.assert_called()
        self.assertTrue(mock_waitKey.called)

    

if __name__ == "__main__":
    unittest.main()