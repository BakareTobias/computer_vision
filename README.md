# Computer Vision Projects

A collection of computer vision projects built on **OpenCV** and **MediaPipe**.

## Key Results

- Recognizes all 26 ASL alphabet letters
- Supports dynamic gesture recognition for J and Z using LSTM
- Includes custom space and backspace gestures
- Real-time webcam inference using MediaPipe landmarks
- Supports touchless cursor, click, scroll, and volume control

## Repository Structure


```
computer_vision/
│
├── ML_pipeline/                 # datasets, trained models, ML artifacts
├── basics/                      # OpenCV/MediaPipe experiments
├── modules/                     # reusable CV modules
│   ├── face_detection_module.py
│   ├── hand_tracking_module.py
│   └── hand_gesture_detection_module.py
│
├── asl_speller.py               # ASL translator
├── asl_speller_st.py            # Streamlit deployment
├── data_collection.py           # Collecting one-handed data from video/webcam
├── data_collection_dynamic.py   # Collecting two-handed data from video/webcam
├── model_training.py            # Loading data, training, evaluating model
├── model_training_lstm.py       # Loading data, training, evaluating LSTM
│
├── finger_mouse.py              # cursor control
├── mouse_scroller.py            # gesture scrolling
├── volume_controller.py         # volume control
│
├── requirements.txt
└── README.md
```
## Features

- Real-time multi-hand detection
- Kalman-filter smoothing
- Modular MediaPipe wrappers
- ML based gesture classification

## Applications
1. Alphabet-Level ASL Translation Interface
2. Touchless Computer Navigation
    - cursor control
    - Left click
    - Right click
    - Click-and-drag
    - Volume control


### 1. Alphabet-Level ASL Translation Interface `asl_speller.py`

```
Webcam
   ↓
MediaPipe
   ↓
Landmark Normalization
   ↓
 ┌─────────────┬─────────────┐
 │ RandomForest│    LSTM     │
 │ Static ASL  │ Dynamic ASL │
 └─────────────┴─────────────┘
            ↓
     Text Translation
```
     
<img width="360" height="270" alt="cdf4e035-deca-45ac-807e-f5ecae65f9dc" src="https://github.com/user-attachments/assets/cdc57a23-06b4-42e3-9c83-b85b9ed95a55" />

  A real-time alphabet-level American Sign Language (ASL) translation interface built on a custom hand gesture recognition pipeline. The system recognizes all 26 letters of the ASL alphabet, plus custom 'space' and 'backspace' gestures for a coherent typing experience. Built on webcam input using MediaPipe hand landmarks, a Random Forest classifier for static letter signs, and a Long Short Term Memory RNN for dynamic letters.


#### Data Collection & Preprocessing `data_collection.py`, `data_collection_dynamic.py`, `data_collection_img.py`
Data is sourced from a combination of personal recordings and video/img datasets from Kaggle to create variation and prevent overfitting to one set of hand proportions. 5,00+ distinct instances are recorded and labelled across 28 classes.
Each data instance consists of 42 features representing the x, y coordinates of 21 hand landmarks, along with a label for the associated pose. `data_collection_dynamic.py` can process and store data for 86 landmarks(84 landmarks representing two hands, and two additional features representing the x and y distances between both hands)


Landmark coordinates are recalculated using `landmark_0` as the origin point, then scaled relative to:

- **Palm width** — distance between landmarks 5 and 17
- **Palm height** — distance between landmarks 0 and 5

This keeps values fairly consistent across different hand positions and rotations from the webcam. Data for each class is stored in its own CSV file. 

#### Model Selection & Training

- Data from all classes is combined and split 70/30 into training and validation sets.
- `stratify=True` ensures an equal ratio of all classes in both sets, avoiding class imbalance bias.
- Class labels are mapped to integers, since ML models can't process strings directly.

Initial model choices were *Logistic Regression* and *Random Forest Classifier*. Both performed very well with only two classes, but as the number of classes increased:

- Precision, accuracy, and F1 scores showed a slight drop-off
- Type I and Type II errors in the confusion matrix increased


**Random Forest maintained stronger performance as the number of classes increased.**

**Model Evaluation**

<img width="2385" height="2085" alt="image" src="https://github.com/user-attachments/assets/e32f473f-6177-4ae2-bc90-cdc1053738a6" />
<img width="3284" height="1481" alt="image" src="https://github.com/user-attachments/assets/960833f6-013d-4d46-a7c6-fa5c0b630f8a" />

At 28 classes, Random Forest outperforms Logistic Regression more often, and by higher margins.


#### Dynamic Gesture Recognition (LSTM)
Two letters(J, Z) in the ASL alphabet are formed by a motion, not a single pose. An RNN was selected because dynamic gestures depend on a sequence of hand poses rather than a single frame. An LSTM
allows the Neural Network to only remember the most recent poses relevant to the use case. 

Landmark data is collected from video clips sourced on Kaggle. These clips are then processed down to a uniform sized sequence of frames. A third class 'other', comprising various clips of hand motions that are not J or Z are included 
to avoid forced-choice error


**Real-World Testing**

The model performs well on live camera feed, static images, as well as recorded videos, detecting trained classes with high confidence. However, there is a tendency for misclassification of similar letters.

#### Libraries Used

`opencv`, `mediapipe`, `sklearn`, `pandas`, `pickle`, `tensorflow`

---
---

### 2. Touchless Computer Navigation `volume_control.py`, `finger_mouse.py`, `mouse_scroller.py`


A suite of scripts built using a combination of heuristics and Machine Learning for hands-free interaction with a computer. 


### A. Volume Control 
<img id="volume-control-using-fingers" width="426" height="240" alt="volume_gesture_control" src="https://github.com/user-attachments/assets/fb3f00e1-fb80-460a-bbea-65be5708e7d3" />

#### Setup

```bash
git clone https://github.com/BakareTobias/computer_vision.git
pip install -r requirements.txt
python3 volume_control.py
```

#### How Does It Work?

1. The webcam checks if your hand is in view. (So keep your hands where it can see 'em.)
2. It will do nothing if the base of your palm ([Landmark 0](#final-image)) is below a set threshold position (approximately anything below your nose). This prevents accidental volume changes while you're typing or have your hands idle.
3. Palm above your nose? Okay! Spread your thumb and index finger apart to raise the volume, and touch them together to reduce it.

---

### B. Mouse Control
<img width="360" height="202" alt="mouse_control_demo" src="https://github.com/user-attachments/assets/54a483db-f080-4670-b6ba-c729be263d3e" />


#### Setup

```bash
git clone https://github.com/BakareTobias/computer_vision.git
pip install -r requirements.txt
python3 finger_mouse.py
```

#### How Does It Work?

1. The webcam checks if your hand is in view. (So keep your hands where it can see 'em.)
2. A virtual trackpad is displayed in the webcam feed. Moving your index fingertip in this region controls the cursor*, allowing the entire screen to be reached without exaggerated hand movements.
3. Tap your index and middle finger together for a left click. Hold them together for a click-and-hold (highlighting/dragging operations).
4. Tap your index, middle, and ring fingers together for a right click. That's it!

*The cursor uses a moving average, as well as a Kalman filter, to reduce output jitter while preserving responsiveness.*

#### Libraries Used

`opencv`, `mediapipe`, `pynput` (smoother mouse experience than `pyautogui`)

---

### C. Mouse Scroller


#### Setup

```bash
git clone https://github.com/BakareTobias/computer_vision.git
pip install -r requirements.txt
python3 mouse_scroller.py
```

#### How Does It Work?

1. The webcam checks if your hand is in view. (So keep your hands where it can see 'em.)
2. A 'peace' sign scrolls the document down
3. An 'L' sign scrolls the document up

#### Libraries Used

`opencv`, `mediapipe`, `pynput` (smoother mouse experience than `pyautogui`)

---

### Relevance to Robotics

The techniques used in this project are directly applicable to:

- Human Robot Interaction (HRI)
- Vision-guided manipulation
- Gesture-based robot control
- Perception pipelines for autonomous systems

<a name="final-image"></a>

<img width="590" height="537" alt="Hand landmark reference" src="https://github.com/user-attachments/assets/d5bd8364-8a81-42eb-97e0-9a5e665775d0" />
