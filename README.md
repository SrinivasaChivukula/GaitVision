# GaitVision

[![Live Demo](https://img.shields.io/badge/Live%20Demo-GitHub%20Pages-38BDF8?style=for-the-badge&logo=googlechrome&logoColor=black)](https://srinivasachivukula.github.io/GaitVision/)
[![Android](https://img.shields.io/badge/Platform-Android%20API%2024+-3DDC84?style=for-the-badge&logo=android&logoColor=white)](https://developer.android.com)
[![Kotlin](https://img.shields.io/badge/Kotlin-KMP%20%2B%20Jetpack-7F52FF?style=for-the-badge&logo=kotlin&logoColor=white)](https://kotlinlang.org)
[![Machine Learning](https://img.shields.io/badge/ML%20AUC->0.97-orange?style=for-the-badge&logo=scikitlearn&logoColor=white)](https://github.com/SrinivasaChivukula/GaitVision)

**2D Gait Analysis for Clinical Use**

🔗 **Interactive Live Biomechanics & Gait Telemetry Studio**: [https://srinivasachivukula.github.io/GaitVision/](https://srinivasachivukula.github.io/GaitVision/)


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
- [References](#references)
- [Acknowledgments](#acknowledgments)

---

## Overview
GaitVision is an Android application focused on improving access to gait analysis in low-resource settings. It requires minimal hardware—only a smartphone with a camera or stored video.

The application implements a video-based gait analysis pipeline on Android (API 24+). It extracts pose landmarks from video frames, computes gait signals, detects stride cycles, and calculates 16 clinical features across temporal, spatial, kinematic, and smoothness domains.

Three machine learning models (Autoencoder, PCA, Ridge Regression) provide gait quality assessments, achieving >0.97 AUC for normal vs impaired classification.
<p align="center">
  <img src="https://github.com/user-attachments/assets/48f1214e-0ade-47ec-9052-ff6d5ccfbb57" width="30%" />
  <img src="https://github.com/user-attachments/assets/93b9cd79-4684-49b5-b0a9-438c131036f2" width="30%" />
  <img src="https://github.com/user-attachments/assets/b719c49b-11d2-4798-9e55-e3f73793aa6a" width="30%" />
</p>

---

## Team

- Bracken Conner
- Srinivasa Chivukula
- Aghomi Dickson
- Michael Hill
- Richard Wang

## Acknowledgment

Jacob Gallagher - Research testing and analysis collaborator.

---

## Features

### Capabilities
- Video input from camera (WIP) or device storage
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

---

## PC Development Environment Setup

### Requirements
- Git  
- Android Studio  
- JDK 17 or compatible version  

### Setup Steps
1. Clone the repository  
2. Open the project in Android Studio  
3. Allow Gradle to sync and install dependencies  
4. Install any required SDK components if prompted  
5. Select the Android app configuration  
6. Run the app on an emulator or connected Android device  

---

## Preferred Method (Android Studio)

1. Clone or download the repository  
2. Open the project in Android Studio  
3. Wait for Gradle sync to complete  
4. Connect an Android device via USB  
5. Enable USB debugging on the device  
6. Alternatively, Android Studio emulators may have reduced performance during video processing  

---

## Deployment Artifacts

The following artifacts should be included for deployment:
- Android APK file  
- Source code repository (GitHub)  
- Autoencoder (AE) model files  
- PCA model files  
- Linear regression scoring model  
- Sample test videos  
- Installation and setup instructions  

---

## Build from Source (Optional)

```bash
git clone https://github.com/SrinivasaChivukula/GaitVision/
cd GaitVision
./gradlew build
./gradlew installDebug
```

## Usage

1. Create or select a patient profile and input required information:
   - Unique Participant ID
   - Participant height (feet and inches)

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
   - Select graphs as needed
   - Export CSV file or return to the main menu
   
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
- Capstone Team of 2024-25
---

_Last Updated: April 2026_
