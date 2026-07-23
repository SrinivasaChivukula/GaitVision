# GaitLens

Smartphone-based, markerless gait quality assessment. Record a short video of
someone walking, and the app produces an interpretable **gait quality score
(0–100)** entirely on the phone — no motion-capture lab, no wearable sensors,
no markers.

This README covers what works today, how to install it on a phone, and how accurate it is so far.

---

## What it does (current state)

**Working, on-device:**
- Records or imports a **side-view** walking video.
- Runs MediaPipe pose estimation (33 body landmarks per frame) — no markers.
- Extracts **16 gait features** (cadence, stride timing, knee range of motion,
  trunk lean, smoothness, symmetry, etc.).
- Scores gait with an autoencoder and reports a **0–100 quality score** plus a
  per-model breakdown, a signals dashboard, and CSV export.

**Partially built (preliminary):**
- **Back-view (posterior) analysis** — 10 extra frontal-plane features
  (pelvic drop, hip sway, step-width asymmetry) that side view can't see.
  The Python pipeline works; the on-device port is not finished (see
  Limitations).
- **Multiview scoring** — combines side + back into a 26-feature score. Trained
  on 9 subjects and wired into the app as **"Multiview Analysis (Beta)"**, but
  not yet working end-to-end on the phone.

---

## Accuracy so far

The core **side-view screening** model was validated on 188 videos
(59 normal, 98 knee osteoarthritis, 31 Parkinson's) using **leave-one-subject-out
cross-validation** — every subject is scored by a model that never saw them in
training.

| Metric | Value |
|---|---|
| AUC (normal vs. impaired) | **0.898** |
| Sensitivity | **85%** |
| Specificity | **88%** |
| Validation | Leave-one-subject-out |

**What this means:** the model catches ~85% of impaired gait while
false-flagging ~12% of healthy walkers — on people it was never trained on.
Importantly, it flags knee osteoarthritis and Parkinson's **without ever having
been trained on either condition**, because it learns what *normal* walking looks
like and flags deviations. This is why it works as a general screening tool.

**Caveats — read before quoting these numbers:**
- Generalization is validated on **KOA and Parkinson's only**. Other gait
  patterns (ataxic, hemiplegic, etc.) were collected but not yet tested.
- The **multiview / back-view** results are preliminary (small N) and should not
  be presented as validated.

---

## Install it on a smartphone

There is no Play Store release — you build the app and install it directly
(sideload). One-time setup:

**1. On the phone:** enable Developer Options (Settings → About phone → tap
"Build number" 7 times), then turn on **USB debugging**.

**2. On your computer** (needs Android Studio + JDK 17):
```bash
git clone https://github.com/audreylena/GaitVision.git
cd GaitVision/GaitVisionAndroid
export JAVA_HOME=$(/usr/libexec/java_home -v 17)   # macOS; re-run each terminal session
```

**3. Plug the phone in via USB**, confirm the "Allow USB debugging?" prompt on
the phone, then build and install:
```bash
./gradlew installDebug
```

The app appears on the phone as **GaitVision**. (Alternatively: open the
`GaitVisionAndroid` folder in Android Studio, pick your phone as the target
device, and press Run.)

**Recording tips for good results:**
- Stationary camera (a tripod or propped phone). Moving-camera video is not
  supported.
- Side view, full lower body in frame (hips to ankles).
- At least 8–10 seconds of continuous walking.

---

## Repo layout

```
GaitVision/
├── GaitVisionAndroid/        Android app (Kotlin)
│   └── app/src/main/
│       ├── java/GaitVision/com/gait/    feature extraction + scoring
│       ├── java/GaitVision/com/mediapipe/  pose backend
│       ├── java/GaitVision/com/ui/      screens
│       └── assets/           .tflite models + .json configs
└── GaitFeatureExtraction/    Python pipeline (offline, for training)
    ├── extract_features.py           side-view features
    ├── extract_back_features.py      back-view features
    ├── train_autoencoder.py          trains the scoring model
    └── loso_auc.py                   leave-one-subject-out validation
```

---

## Retraining the model (Python side)

```bash
cd GaitFeatureExtraction
python3 -m venv venv && source venv/bin/activate
pip install mediapipe opencv-python numpy pandas scipy scikit-learn tensorflow matplotlib
python3 extract_features.py          # videos -> features.csv
python3 train_autoencoder.py         # trains on NORMAL gait only -> .tflite + .json
```
Copy the exported `.tflite` and `.json` into
`GaitVisionAndroid/app/src/main/assets/`, rebuild, reinstall. The Android feature
order must match `GaitFeatures.FEATURE_COLUMNS`.

The autoencoder is trained on **normal gait only** by design: it learns the
distribution of healthy walking so it can flag anything that deviates, including
conditions not in the dataset.

---

## Known limitations

- **JDK 17 is required.** Newer JDKs fail with a cryptic version error. On macOS:
  `export JAVA_HOME=$(/usr/libexec/java_home -v 17)` (must be re-run per terminal).
- **Multiview back-view step detection is broken on-device** — the Kotlin
  `BackFeatureExtractor.detectSteps` returns 0 steps where the Python version
  works. The Beta multiview screen runs but can't produce a score until this is
  fixed (it needs a faithful port of scipy's peak-prominence logic).
- **iPhone videos may not decode on the Android emulator** (Dolby Vision codec).
  Transcode to H.264 first: `ffmpeg -i in.mov -c:v libx264 out.mp4`.
- **Emulator stuck "offline" / "already running"** — delete the stale
  `~/.android/avd/<name>.avd/multiinstance.lock` and cold-boot with
  `-no-snapshot-load`.

---

## Next steps

1. Fix the Kotlin back-view step detection so multiview works on-device.
2. Collect more paired side+back subjects and retrain the multiview model
   (currently N=9).
3. Extend screening validation to gait patterns beyond KOA and Parkinson's.
4. Consider improving the smoothness (LDJ) features — currently RMS angular
   acceleration with poor dynamic range.

---

## Reference

The scoring approach follows the Shriners Gait Index (an autoencoder-based single
gait-quality metric):

> Wang, S.-J., et al. (2024). Creating an autoencoder single summary metric to
> assess gait quality to compare surgical outcomes in children with cerebral
> palsy: The Shriners Gait Index (SGI). *Journal of Biomechanics*, 168, 112092.
> https://doi.org/10.1016/j.jbiomech.2024.112092

## Acknowledgments

- [MediaPipe](https://github.com/google-ai-edge/mediapipe) by Google
- [TensorFlow Lite](https://www.tensorflow.org/lite) by Google
- [MPAndroidChart](https://github.com/PhilJay/MPAndroidChart) by PhilJay (Apache 2.0 License)
- [Android Room](https://developer.android.com/jetpack/androidx/releases/room) & [Kotlin](https://kotlinlang.org) by Google / JetBrains

**Dataset:** Kour, N., Gupta, S., & Arora, S. (2020). [KOA-PD-NM gait dataset](https://doi.org/10.17632/44pfnysy89.1). Mendeley Data.
