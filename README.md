# A collection of computer vision projects built on opencv and mediapipe 



## Modules
The modules are built on principles of Obect Oriented Programming. This makes it easy to reuse basic functions for as subject and landmark detection

## Hand based Projects
<img width="590" height="537" alt="image" src="https://github.com/user-attachments/assets/d5bd8364-8a81-42eb-97e0-9a5e665775d0" />
Mediapipe landmarks for a hand 


### 1. Volume Control using fingers
This project uses a webcam to monitor hands and then increases or decreases volume based on how far apart the tips of the fingers are 
It also muses the base of the palm as a trigger. The volume will only be adjusted when the landmark 0 is in the upper section of the webcam view.
This is to prevent accidentally changig the volume while the hands are resting on a table or in some other passive position
