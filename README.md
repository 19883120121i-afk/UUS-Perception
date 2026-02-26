# UUS-Perception
This repository contains a deep learning pipeline for the human-centric evaluation of urban micro-environments. By fusing street-level visual data (video frames) with continuous temporal sensor data, this model predicts human-centric environmental perception values (e.g., comfort, safety, or aesthetic scores). 

The architecture leverages a Vision-Temporal-Sensor fusion approach, making it highly suitable for spatial-temporal urban analysis and pedestrian viewpoint evaluations.


## 📊 Data Structure

The dataset requires synchronized video frames and sensor readings. By default, the `Config` class looks for the following structure:

```text
data_root/
├── sp/                 # Video frames directory
│   ├── video_001/      # Contains sequential images (e.g., frame_0.jpg, frame_1.jpg)
│   ├── video_002/
│   └── ...
├── bg/                 # Sensor data directory
│   ├── video_001.csv   # Synchronized sensor data for video_001
│   ├── video_002.csv
│   └── ...
└── models/             # Output directory for saved weights
