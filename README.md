# Yellow-Box-Junction Violation Automatic Challan System

## Purpose
The Yellow-Box-Junction Violation Automatic Challan System is designed to automate the detection and reporting of violations of the yellow box junction rule. The rule mandates that vehicles should not stop inside a yellow box and should not enter the yellow box if the exit is not clear.
This model performs detections on virtual stimulation but it can be scaled to real world detection using powerful machine and high definition cameras that can clearly capture licence-plate numbers . 

### Demo
(Might take a few seconds to load.)
<img src="https://github.com/deva022/Yellow-Box-Junction-Violation-Automatic-Challan-System/assets/112040328/524c65c0-094d-41a4-824a-6d68075f22cc" style="width:60vw" alt="Demo Loading...">

The licence plate detector weights are provided in Licence_plate_detector_weights . Along with that some demo videos are provided to test the model .Clone the repository mentioned in requirements.txt for Sort algorith for tracking the vehicles . You can download YOLOv8x weights locally on your machine also and add its path in code line .
```bash
 coco_model = YOLO('Path_goes_here')
```


## Overview
This system leverages advanced computer vision and deep learning techniques to identify vehicles, track their movements, and detect violations within the yellow box junction. The core components of the system include:

- **Vehicle Detection:** Utilizing the YOLOv8x model from Ultralytics to detect vehicles in real-time.
- **License Plate Detection:** Fine-tuned YOLO model on custom data to accurately detect vehicle license plates.
- **Vehicle Tracking:** Using Deep SORT to track vehicles and assign them unique IDs.
- **Violation Detection:** Monitoring vehicle positions and durations within the yellow box to identify violations. The system logs vehicles that stop inside the yellow box for longer than a specified time limit or enter the yellow box when the exit is not clear.

## Key Concepts Used
- **YOLOv8x Model:** State-of-the-art object detection model used for real-time vehicle detection.
- **Fine-Tuning:** Customizing the YOLO model to accurately detect license plates using a custom dataset.
- **Deep SORT:** An algorithm for tracking multiple objects, ensuring each vehicle is uniquely identified and tracked over time.
- **Violation Detection Logic:** Implementing a rule-based system to monitor and detect violations based on vehicle positions and movements.


### Installation
1. Clone the repository:
```bash
    git clone https://github.com/yourusername/Yellow-Box-Junction-Violation-Automatic-Challan-System.git
```
2. Clone the Sort algorithm repository for vehicle tracking:
```bash
    git clone https://github.com/abewley/sort.git
```
3. Install the required packages using the following command:
```bash
    pip install -r requirements.txt
```
4. Run the Yellow_junction_model_run.py file:
   
   Run the python file and is there is some module error install that module in your system .
6. Change the demo video in the same video if you want :
```bash
   cap = cv2.VideoCapture(r"demo_videos\v1.mp4")
```
Change video name in this line to run model on different video.


## If there are any issues or suggestions fell free to contact me 😊

