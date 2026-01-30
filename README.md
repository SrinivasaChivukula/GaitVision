# GaitVision

**2D Gait Analysis for Clinical Use**

An Android application for clinical gait assessment using computer vision and machine learning. The system processes 2D video on device to extract gait features for assessment.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Requirements](#requirements)
- [Technology Stack](#technology-stack)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset & Validation](#dataset--validation)
- [References](#References)
- [Acknowledgments](#acknowledgments)

---

## Overview

GaitVision implements a video-based gait analysis pipeline on Android (API 24+) requiring only a smartphone camera. The system extracts pose landmarks from video frames, computes gait signals, detects stride cycles, and calculates 16 clinical features across temporal, spatial, kinematic, and smoothness domains. Three machine learning models (Autoencoder, PCA, Ridge Regression) provide gait quality assessments, achieving >0.97 AUC for normal vs impaired classification.

---

## Features

### Capabilities
- Video input from camera(WIP) or device storage
- 16 gait features: cadence, stride time/length, knee ROM, movement smoothness (LDJ), trunk stability, asymmetry metrics
- Pose wireframe and angle overlay visualization
- Time-series charts for joint angles and gait signals
- Patient database with analysis history tracking
- CSV export of features and angle data

### Analysis
1. MediaPipe extracts 33 landmarks per frame from input video, currently only using 6 to extract signals and features
2. Computes joint angles, inter-ankle distance, velocities; applies EMA smoothing and interpolation
3. Three detection modes (inter-ankle distance, ankle velocity, knee angle) evaluated; best mode selected automatically
4. Calculates 16 features from detected stride cycles (temporal, spatial, kinematic, smoothness). Feature data is only gathered from chosen gait cycles
5. Three models (Autoencoder, PCA, Ridge) generate independent scores (0-100 scale)

---

## Requirements

### Runtime
- Android 7.0+ (API level 24)
- Camera, storage, and media permissions
- Sufficient storage for video and analysis data

### Development
- Android Studio
- JDK 8+
- Gradle (via wrapper)
- Android SDK API 24-34

---

## Technology Stack

- Android (Kotlin)
- MediaPipe Tasks Vision
- TensorFlow Lite
- Room (SQLite)

---

## Installation

### For End Users (APK Installation)

1. Transfer the `.apk` file to your Android-compatible device
2. Open the `.apk` file using a file manager
3. Click "Install" and allow installation from unknown sources if prompted
4. Grant all required permissions when prompted

### Preferred Method (Android Studio)

1. Clone or download the repository
2. Open the project in Android Studio
3. Wait for Gradle sync to complete
4. Connect an Android device via USB
5. Enable USB debugging on the device
6. Alternatively, Android Studio emulators work but may have degraded performance during video processing


### Build from Source (Outdated, Untested)

```bash
git clone https://github.com/SrinivasaChivukula/GaitVision
cd GaitVision
./gradlew build
./gradlew installDebug
```

   Or use the provided scripts:
   - Windows: `run_app.bat` or `run_app.ps1`
   - Linux/Mac: `./gradlew installDebug`

---

## Usage

1. Create or select a patient profile with demographic information

2. Select or record video:
   - Click "Record Video" to capture a new video, or
   - Click "Select Video" to choose an existing video from storage

3. Record walking pattern:
   - Have the participant walk normally
   - Ensure at least 2 complete gait cycles (approximately 5 seconds)
   - Record from a side view

4. Perform analysis:
   - Click "Perform Analysis" to process the video
   - Wait for analysis to complete

5. View and export results:
   - Review the annotated video showing angles at each timepoint
   - Click "View Analysis" to see detailed results and graphs
   - Export CSV file containing all angle measurements

---

## Dataset & Validation

### Training Dataset
Models trained on the [Gait Dataset for Knee Osteoarthritis and Parkinson's Disease Analysis With Severity Levels](https://data.mendeley.com/datasets/44pfnysy89/1)

- AUC: >0.97 (normal vs impaired classification)
- Validation: 5-fold patient-level cross-validation
- Clinical correlation: Spearman ρ=0.82

---

## References

- [MediaPipe](https://developers.google.com/mediapipe/solutions/vision/pose_landmarker) by Google
- [TensorFlow Lite](https://www.tensorflow.org/lite) by Google
- [MPAndroidChart](https://github.com/PhilJay/MPAndroidChart) by PhilJay (Apache 2.0 License)
- [Android Room & Kotlin](https://developer.android.com/training/data-storage/room)
- Kour, N., Gupta, S., & Arora, S. (2020). [Gait Dataset for Knee Osteoarthritis and Parkinson's Disease Analysis With Severity Levels. Mendeley Data, V1.](https://doi.org/10.17632/44pfnysy89.1)
---

## Acknowledgments

Special thanks to:

- Guna Sindhuja Siripurapu
- Dr. Rita Patterson
- Dr. Mark Albert
- University of North Texas

---

_Last Updated: January 2026_
