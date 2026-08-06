import modules.hand_tracking_module as htm
import modules.hand_gesture_detection_module as hgd
import cv2
import math
from keras.models import load_model
import numpy as np
import streamlit as st
from streamlit_webrtc import VideoProcessorBase, webrtc_streamer
import av

st.title('ASL Text Interface')
st.markdown(
        """
    This app types letters as you sign them in **American Sign Language (ASL)**.
    Show a hand sign to the webcam and hold it steady — the recognized letter is
    typed into the text box automatically.
    
    - 👍 **Thumbs up** → space
    - 🤏 **Pinch** (touch thumb + index finger tip) → backspace

    GitHub link: https://github.com/BakareTobias/computer_vision
    """
    )
col1, col2 = st.columns(2)


class MyProcessor(VideoProcessorBase):

    def __init__(self):
        self.latest_frame = None

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        self.latest_frame = img.copy()

        return av.VideoFrame.from_ndarray(
            img,
            format="bgr24"
        )

#1. start media pipe and hand detection with landmarks
def main():
    # load model
    path_to_RF_model = 'ML_pipeline/models/random_forest_az_thumbsup_pinch.pkl'
    LSTM_model = load_model('ML_pipeline/models/LSMT_jz_all_landmarks_other.keras',compile=False)

    classes_static = ['A','B','C','D','E','F','G','H','I','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','thumbs_up','pinch']
    classes_dynamic = ['J','Z','other']

    #load modules
    hand_detector = htm.HandDetector()
    gesture_detector = hgd.GestureDetection(path_to_model=path_to_RF_model)

    text = ''
    def clear_text():
        text = ''
        return text
    last_predict = math.inf
    frame_rate = 5
    frame_counter = 0

    clip = []
    with col2: 
        placeholder = st.empty()  # Placeholder for the video feed
        placeholder.text("Starting webcam...")

    st.button("Clear text", on_click=clear_text)

    #setup webcam capture and hand detector
    cap = webrtc_streamer(
    key="test",
    media_stream_constraints={
        "video": True,
        "audio": False
        
    },
    video_processor_factory=MyProcessor,
    desired_playing_state=True,
    )#using webcam no 0

    
    
     
    
    while cap.state.playing:
        #setup image capture from webcam
        #success, img = cap.read()#reading the image from webcam
        img = cap.video_processor.latest_frame
        img = hand_detector.findHands(img,bothHands=False,draw=True)
        img = cv2.flip(img,2)#flipped on x axis

        #only one hand is tracked for this program
        hand0_landmark_coordinates = hand_detector.findPosition(img, handNo=0, draw=False)
        
        if hand0_landmark_coordinates:
            #normalize landmarks to generate dataset instance
            norm_landmarks = hand_detector.normalizeLandmarks(hand0_landmark_coordinates)
            #will return none if Zero error occurs   
            if norm_landmarks is not None:
                #add frames to rolling buffer 'clip' for dynamic gestures
                #dynamic_landmarks = norm_landmarks.iloc[:, [8,9,20,21]]
                clip.append(norm_landmarks)
                

                #if clip has enough frames, run LSTM model to predict dynamic gesture
                #also remove the oldest frame from clip to maintain a rolling buffer of 27 frames
                """ if len(clip) == 27:
                    #run ml model on dataset instance to predict as well as confidence score
                    dynamic_input = np.array(clip).reshape(1,27,42)
                    gesture_detected_dynamic = LSTM_model.predict(dynamic_input)

                    predicted_class = np.argmax(gesture_detected_dynamic, axis=1)[0]
                    confidence_dynamic = np.max(gesture_detected_dynamic)
                    gesture_name = classes_dynamic[predicted_class]
                    #print(gesture_detected_dynamic)
                    if gesture_name == 'other':
                        clip.pop(0)  # Remove the oldest frame to maintain a rolling buffer of 27 frames
                    elif confidence_dynamic > 0.5:
                        text += str(gesture_name)
                        last_predict = gesture_name
                        clip = []  # Clear the clip after prediction """
                        
                #run ml model on dataset instance to predict as well as confidence score
                gesture_detected, confidence = gesture_detector.predict(dataset_instance=norm_landmarks,classes=classes_static)

                if confidence > 0.5:
                    #can only hit space/backspace once every X frames

                    if gesture_detected == 'pinch':
                        if frame_counter % frame_rate == 0:
                            text = text[:-1]
                        cv2.putText(img, "Backspace", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        #st.warning('Backspace')

                    elif gesture_detected == 'thumbs_up':
                        if frame_counter % frame_rate == 0:
                            text += ' '
                        cv2.putText(img, "Space", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        #st.info('Space')

                    elif gesture_detected != last_predict:
                        if frame_counter % frame_rate == 0:
                            text += str(gesture_detected)
                            last_predict = gesture_detected 
                else:
                    cv2.putText(img, "Unrecognized", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    #st.error('Unrecognized')

        
        cv2.putText(img, f"{text}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        placeholder.image(img, channels="BGR")
        #cv2.imshow("Image", img)
        cv2.waitKey(1)#waiting for 1 millisecond before showing the next frame
        frame_counter += 1 


if __name__ == "__main__":
    
    

    with col1:
        st.subheader("Here's a guide to the letters you can sign:")
        st.image("asl_chart.jpg")
    main()

