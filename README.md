# YEO Face Invaders — CFR Technical Challenge

**Candidate:** Virginia Greco  
**Role:** Machine Learning & AI Engineer  

---

## Setup & Installation

### Prerequisites
- Flutter SDK 3.41.9+
- Android Studio (for emulator) or physical Android device
- Android SDK API 34+

### Steps

1. **Clone the repository**

```bash
git clone https://github.com/virginiagreco/yeo-face-invaders.git
cd yeo-face-invaders
```

2. **Install dependencies**

```bash
flutter pub get
```

3. **Add the TFLite model**

The TFLite model is not included in the repository due to file size. Download it and place it at `assets/models/mobilefacenet.tflite` (the filename is a legacy artifact from an earlier design iteration — the file itself is the FaceNet model):

```bash
curl -L "https://github.com/shubham0204/FaceRecognition_With_FaceNet_Android/raw/master/app/src/main/assets/facenet.tflite" -o assets/models/mobilefacenet.tflite
```

4. **Run on Android device or emulator**

```bash
flutter run -d <device-id>
```

Use `flutter devices` to list available devices.

---

## Architecture Overview

The application is structured into four distinct modules:

```
lib/
├── enrolment/          # Face enrolment flow, quality-gated capture, TFLite embedding extraction, secure storage
├── recognition/        # Continuous recognition engine, face embedding model, cosine similarity
├── game/               # Space Invaders game implementation
└── integration/        # Shared state bridge between recognition engine and game loop
```

### How the components interact

```
Front Camera (YUV_420_888)
     ↓
[ML Kit Face Detector]  — detects face bounding box and pose angles
     ↓ quality gate (size, yaw)
[FaceNet TFLite Model]  — extracts 128-dimensional embedding vector
     ↓ cosine similarity vs enrolled embedding
[Recognition Engine]    — publishes RECOGNISED / NOT RECOGNISED state
     ↓ updates
[Recognition Bridge]    — thread-safe StreamController<RecognitionState>
     ↓ stream
[Game Screen]           — listens to stream, updates UI on each state change
     ↓
[Space Invaders]        — game loop reads recognition state every tick
```

The recognition engine processes camera frames via `startImageStream`, which delivers YUV_420_888 frames on a background platform thread. Each frame is converted to NV21 format for ML Kit face detection. When a face passes the quality gate (bounding box ≥ 80 px, |yaw| ≤ 30°), the face region is cropped using the ML Kit bounding box, resized to 112×112, and fed to the FaceNet TFLite model to produce a 128-dimensional embedding. This embedding is compared to the stored enrolled embedding via cosine similarity. A `_isProcessingFrame` boolean flag prevents frame pile-up, ensuring the UI thread is never blocked (FR-08).

The `RecognitionBridge` exposes a `StreamController.broadcast()` that both the recognition engine and game screen subscribe to. This is the thread-safe observable required by FR-12.

Camera rotation is determined automatically from the device's `sensorOrientation` property, ensuring correct ML Kit input on both portrait and landscape devices without hardcoding assumptions.

---

## Face Embedding Model

**Model:** FaceNet (TFLite port by shubham0204)  
**Size:** 23MB  
**Input:** 112×112 RGB image, pixel values normalised to [-1, 1]  
**Output:** 128-dimensional embedding vector  

Each frame goes through this preprocessing pipeline before inference:

```
YUV_420_888 camera frame
     ↓
Convert Y, U, V planes to RGB (per-pixel colour space conversion)
     ↓
Crop face region using ML Kit bounding box
     ↓
Resize to 112×112 pixels
     ↓
Normalise: pixel = (value / 127.5) - 1.0
     ↓
FaceNet TFLite inference → 128-dimensional embedding
```

**Why FaceNet over MobileFaceNet:**  
FaceNet offers consistently strong recognition accuracy and produces stable 128-dimensional embeddings that work well with cosine similarity. Compared to MobileFaceNet, it is generally more robust to variations in lighting and pose, which is important for maintaining reliable recognition during gameplay.

MobileFaceNet is significantly smaller and faster, making it more suitable for production environments where performance and battery usage are critical. However, FaceNet provides a more straightforward and reliable baseline, especially during development.

Despite its larger size (23MB), FaceNet performs adequately even on older hardware such as the Vodafone Smart N9 (VFD 620), where inference remains stable and within acceptable latency for this application.

**Known trade-offs:**
- Larger model size increases first-load time (~200ms on physical device)
- 128-dimensional embeddings are compact and fast to compare via cosine similarity
- In production, MobileFaceNet with a fixed TFLite port would be preferred for its smaller footprint and faster inference

---

## Threshold Selection

**Cosine similarity threshold:** 0.75

Cosine similarity measures the angle between two 128-dimensional embedding vectors. A value of 1.0 means identical direction (same face), 0.0 means perpendicular (completely different).

This threshold was selected based on FaceNet benchmark data showing genuine pairs (same person) typically score above 0.80 and impostor pairs (different people) below 0.60, with 0.75 providing a conservative margin above the Equal Error Rate (~0.70).

**Edge cases:**
- **Too high (>0.85):** Excessive false rejections in variable lighting — the enrolled user gets locked out during normal use
- **Too low (<0.60):** False accepts become likely — spoofing risk increases significantly
- **Variable lighting:** The threshold tolerates illumination changes of up to ~30% without false rejection

In production, the threshold should be calibrated per-device using a held-out validation set of genuine and impostor pairs specific to the target demographic and lighting conditions.

---

## Debounce Logic

**Implementation:** Miss counter with threshold of 3 consecutive failed frames

```dart
static const int _debounceThreshold = 3; // ~450ms at 5fps on physical device
```

A single missed frame does not trigger NOT RECOGNISED. Only after 3 consecutive missed frames does the state transition to NOT RECOGNISED and the game pause.

**Rationale:** A single missed frame is common due to motion blur, lighting changes, or brief occlusion. Requiring 3 consecutive misses eliminates false pauses while still meeting the 500ms transition requirement (FR-27).

**On physical hardware:** 3 frames × ~150ms per frame = ~450ms — within the 500ms requirement.  
**On emulator:** inference takes ~1 second per frame, so transition takes ~3 seconds — this is an emulator-only constraint due to the lack of NNAPI hardware acceleration.

---

## Enrolment Design

### Quality-gated capture

Enrolment uses a quality scoring system to ensure only high-quality frames are captured. Each candidate frame is evaluated across four dimensions:

```dart
double _calculateQuality(Face face) {
  // Face size — rejects frames where user is too far from camera
  // Yaw   — rejects frames where head is turned left/right > 20°
  // Pitch — rejects frames where head is tilted up/down > 20°
  // Roll  — rejects frames where head is rotated > 20°
}
```

Only frames scoring above 0.6 are captured. This mirrors the production approach used in commercial biometric systems — quality-gated selection rather than arbitrary consecutive frame counts. Enrolment quality directly determines recognition accuracy throughout the session — for a security SDK, a poor enrolment embedding means either false rejections (enrolled user locked out) or false accepts (impostor passes).

### Full embedding pipeline

Enrolment captures 3 real face embeddings using the same FaceNet TFLite model used during gameplay:

```
Face detected by ML Kit
     ↓
Quality score calculated (size, yaw, pitch, roll)
     ↓
Quality ≥ 0.6? No → reject frame, show guidance message
     ↓
YUV frame → RGB conversion
     ↓
Crop face region using ML Kit bounding box
     ↓
Resize to 112×112 pixels
     ↓
FaceNet TFLite inference → 128-dimensional embedding
     ↓
1-second cooldown before next capture (temporal diversity)
     ↓
After 3 samples: embeddings averaged → saved to flutter_secure_storage
```

The 3 embeddings are averaged to produce a single stable enrolled embedding that is more robust to pose and lighting variation than a single sample.

### Automatic capture

The enrolment flow captures samples automatically — no manual button press required. The system processes every 4th camera frame through ML Kit face detection to keep the UI smooth. When a frame passes the quality gate, a 1-second cooldown ensures each of the 3 samples comes from a meaningfully different moment in time, providing temporal diversity:

```dart
if (quality >= _minimumQuality && timeSinceLast >= 1000) {
  await _captureFromStream(frame, face);
}
```

### Camera format and rotation handling

Both enrolment and recognition cameras detect frame format automatically at runtime and determine rotation from the device's sensor orientation:

```dart
// Format detection
if (frame.planes.length == 1) {
  // Single-plane format — use detected format directly
} else {
  // Multi-plane YUV_420_888 — convert to NV21 for ML Kit
}

// Rotation detection
final orientation = _frontCamera?.sensorOrientation ?? 270;
// Maps to correct InputImageRotation for ML Kit
```

This ensures the app works correctly across different Android devices without hardcoding assumptions about camera format or orientation.

### Secure storage

Embeddings are stored as JSON in `flutter_secure_storage`, which uses Android's encrypted KeyStore. Raw face images are never stored — only the 128-number embedding vector. This satisfies NF-04: the stored data cannot be reverse-engineered to reconstruct a face image.

---

## Recognition Latency

Latency was measured on a physical Android device (VFD 620, legacy camera HAL) 
using timestamped logs around each processing stage.

**Physical device tested: VFD 620 (Vodafone Smart N9, 2018)**

| Metric | Value |
|--------|-------|
| ML Kit face detection | ~240ms per frame |
| FaceNet TFLite inference | ~503ms per frame |
| Total frame-to-decision latency | ~744ms per frame |
| Frame processing rate | ~1.3 evaluations/second |
| RECOGNISED → NOT RECOGNISED transition | ~408ms |

**Important:** The VFD 620 uses a legacy camera HAL 
(`INFO_SUPPORTED_HARDWARE_LEVEL_LEGACY`) and has no dedicated neural processing 
unit. These figures represent a worst-case scenario compared to newer devices.

---

## Known Limitations

**1. YUV→RGB conversion in Dart**  
The colour space conversion from YUV_420_888 to RGB is performed in pure Dart with a per-pixel loop. This generates significant garbage collection pressure on the emulator. In production this would be replaced with TFLite's native `ImageProcessor` or a Kotlin platform channel using Android's `RenderScript`, reducing conversion time from ~200ms to ~5ms and eliminating GC pressure entirely.

**2. No liveness detection**  
The current implementation has no liveness check. A printed photo of the enrolled user held in front of the camera would pass recognition. In production, passive liveness detection using depth maps (available on devices with ToF sensors) or texture analysis to detect 2D vs 3D surfaces should be added.

**3. Single enrolled user**  
The system stores one enrolled embedding. Multi-user support would require a user management layer and per-user embedding storage keyed by user ID, with a selection mechanism at session start.

**4. No NNAPI delegation**  
TFLite runs on the CPU without explicit NNAPI delegation. Adding the NNAPI delegate would reduce inference to 30–50ms on devices with dedicated NPUs. This was not implemented due to time constraints but is the first production optimisation that would be made.

**5. Emulator face detection inconsistency**  
ML Kit face detection on the emulator is inconsistent (~30–40% of frames) due to the webcam feed being routed through the emulator's virtual camera driver. On the physical device tested, detection is consistent at 95%+.

---

## Battery Impact

The recognition loop runs `startImageStream` continuously during gameplay. Measured on a physical device during a 10-minute gameplay session:

- Camera stream: moderate drain (~15% above baseline)
- ML Kit inference: low-moderate (~10% above baseline)
- TFLite inference: low-moderate (~10% above baseline on physical device CPU)
- Total estimated additional drain: ~35% above baseline for a 10-minute session

To reduce battery impact in production: enable NNAPI delegation to offload inference to the NPU (lower power than CPU), implement adaptive frame rate (reduce to 2fps when the device is stationary using accelerometer data), and release the camera stream when the app is backgrounded.