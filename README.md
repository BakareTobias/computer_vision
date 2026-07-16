# A collection of computer vision projects built on opencv and mediapipe 

## Features

- Modular MediaPipe wrapper classes 
- Real-time hand landmark detection
- Kalman-filter landmark smoothing
- Virtual trackpad
- Gesture-based cursor control
- Left click
- Right click
- Click-and-drag
- Gesture-based volume control

## Modules
The modules are the foundation of the following projects. They are python Classes that make it easy to reuse basic MediaPipe functions e.g. subject detection, landmark detection. 

## Hand based Projects

### 1. Volume Control using fingers

#### Demo
<img width="426" height="240" alt="volume_gesture_control" src="https://github.com/user-attachments/assets/fb3f00e1-fb80-460a-bbea-65be5708e7d3" />

#### Setup
    git clone https://github.com/BakareTobias/computer_vision.git
    pip install -r requirements.txt
    python3 volume_control.py

#### How Does it Work?
Step 1:  The webcam checks if your hand is in view. (So keep your hands where it can see 'em)

Step 2:  It will do nothing if the base of your palm ([Landmark 0](#final-image)) is below a set threshold position (approx. anything below your nose). This prevents accidental volume changes while you are typing or have your hands just idle. 

Step 3:  Palm above your nose? Okay! Spread your thumb and index finger apart to raise the volume, and touch them together to reduce it.

### 2. Mouse Control using fingers

#### Demo
<img width="360" height="202" alt="awrkbo" src="https://github.com/user-attachments/assets/54a483db-f080-4670-b6ba-c729be263d3e" />

#### Setup
    git clone https://github.com/BakareTobias/computer_vision.git
    pip install -r requirements.txt
    python3 finger_mouse.py

#### How Does it Work?
Step 1:  The webcam checks if your hand is in view. (So keep your hands where it can see 'em)

Step 2:  A virtual trackpad is displayed in the webcam feed. Moving your index fingertip in this region controls the cursor*, allowing the entire screen to be reached without exaggerated hand movements 

Step 3:  Tap your index and your middle finger together for a left click. Hold them together for a click and hold(higlighting/dragging operations)

Step 4: Tap your index, middle, and ring fingers for a right click. Thats it!

*the cursor uses a moving average, as well as a Kalman filter to reduce output jitter while preserving responsiveness

#### Libraries used: opencv, mediapipe, pynput(smoother mouth experience than pyautogui)

<a name="final-image"></a>
<img width="590" height="537" alt="image" src="https://github.com/user-attachments/assets/d5bd8364-8a81-42eb-97e0-9a5e665775d0" />
