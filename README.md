# Computer Vision Projects

A collection of computer vision projects built on **OpenCV** and **MediaPipe**.

## Features

- Gesture recognition using a Random Forest Classifier
- Modular MediaPipe wrapper classes
- Real-time hand landmark detection
- Kalman-filter landmark smoothing
- Virtual trackpad
  - Gesture-based cursor control
  - Left click
  - Right click
  - Click-and-drag
- Gesture-based volume control

## Table of Contents

- [Modules](#modules)
- [Hand-Based Projects](#hand-based-projects)
  - [1. Volume Control Using Fingers](#1-volume-control-using-fingers)
  - [2. Mouse Control Using Fingers](#2-mouse-control-using-fingers)
  - [3. Gesture Recognition](#3-gesture-recognition)

## Modules

The modules are the foundation of the projects below. They are Python classes that make it easy to reuse basic MediaPipe functions, e.g. subject detection and landmark detection.

## Hand-Based Projects

### 1. Volume Control Using Fingers `volume_control.py`

#### Demo

<img width="426" height="240" alt="volume_gesture_control" src="https://github.com/user-attachments/assets/fb3f00e1-fb80-460a-bbea-65be5708e7d3" />

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

### 2. Mouse Control Using Fingers `finger_mouse.py`

#### Demo

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

### 3. Gesture Recognition `data_collection.py`, `model_training.py`, `gesture_recognition.py`

#### Demo
<img width="240" height="180" alt="gesture+recognition" src="https://github.com/user-attachments/assets/4b0383bd-ffd8-4f2d-bdac-591a53790876" />


#### How It Works

**Data Collection**

Landmark data is collected via webcam, with 100+ instances per class. Each data instance consists of 42 features representing the x, y coordinates of 21 hand landmarks, along with a label for the associated pose.

Landmark coordinates are recalculated using `landmark_0` as the origin point, then scaled relative to:

- **Palm width** — distance between landmarks 5 and 17
- **Palm height** — distance between landmarks 0 and 5

This keeps values fairly consistent across different hand positions and distances from the webcam. Data for each class is stored in its own CSV file.

**Data Preprocessing**

- Data from all classes is combined and split 70/30 into training and validation sets.
- `stratify=True` ensures an equal ratio of all classes in both sets, avoiding class imbalance bias.
- Class labels are mapped to integers, since ML models can't process strings directly.

**Model Selection**

Initial model choices were **Logistic Regression** and **Random Forest Classifier**. Both performed very well with only two classes, but as the number of classes increased:

- Precision, accuracy, and F1 scores showed a slight drop-off
- Type I and Type II errors in the confusion matrix increased

**Random Forest maintained stronger performance as the number of classes increased.**

**Model Evaluation**

<img width="3200" height="1200" alt="Model comparison: Logistic Regression vs Random Forest" src="https://github.com/user-attachments/assets/21c9a009-55b0-4846-8345-bf78c418bb65" />

At 6 classes, Random Forest outperforms Logistic Regression more often, and by higher margins.


**Real-World Testing**

The model performed very well on live camera feed and static images, detecting trained classes with high confidence. However, untrained poses may still be misclassified.

#### Libraries Used

`opencv`, `mediapipe`, `sklearn`, `pandas`, `pickle`


<a name="final-image"></a>

<img width="590" height="537" alt="Hand landmark reference" src="https://github.com/user-attachments/assets/d5bd8364-8a81-42eb-97e0-9a5e665775d0" />
