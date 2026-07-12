# A collection of computer vision projects built on opencv and mediapipe 



## Modules
The modules are built on principles of Obect Oriented Programming. This makes it easy to reuse basic functions for as subject and landmark detection

## Hand based Projects

### 1. Volume Control using fingers

#### Demo
<img width="426" height="240" alt="volume_gesture_control" src="https://github.com/user-attachments/assets/fb3f00e1-fb80-460a-bbea-65be5708e7d3" />

#### Description
This project uses a webcam to monitor hands and adjust volume based on how far apart the tips of the fingers are 

It also uses the base of the palm (landmark 0) as a filter. The volume control gesture will only work when the landmark 0 is in the upper section of the webcam view.
This is to prevent accidentally changing the volume while the hands are resting on a table or in some other idle position. [See reference image below](#final-image).

#### Libraries used: opencv, mediapipe, pyautogui

<a name="final-image"></a>
<img width="590" height="537" alt="image" src="https://github.com/user-attachments/assets/d5bd8364-8a81-42eb-97e0-9a5e665775d0" />
