# HyperMotion — Full Motion Capture & Animation System for UE5 Football Games

## CRITICAL INSTRUCTIONS
You are building a complete, production-grade motion capture and animation system called HyperMotion. This is NOT a prototype, NOT a lite version, NOT simplified. Build every module fully with complete header and implementation files. All C++20, no Python anywhere.

When building this project, create ALL 8 modules with COMPLETE implementations. Do not skip modules, do not write stubs, do not say "TODO". Every class needs a full .h and .cpp file.

## Project Overview
- **Purpose**: Video-to-animation pipeline + real-time UE5 animation system for a football (soccer) game
- **Language**: C++20 exclusively. No Python.
- **Engine**: Unreal Engine 5.3+
- **ML Framework**: LibTorch (C++ PyTorch) for training, ONNX Runtime + TensorRT for inference
- **Vision**: OpenCV 4.8+ with DNN module
- **Math**: Eigen 3.4+
- **Serialisation**: nlohmann/json
- **Build**: CMake 3.20+
- **GPU**: CUDA 12.0+, TensorRT 8.6+

## Directory Structure
```
hypermotion/
├── CMakeLists.txt
├── CLAUDE.md
├── include/HyperMotion/
│   ├── core/          # Types, MathUtils, Logger
│   ├── pose/          # Module 1: Multi-Person Pose Estimation
│   ├── skeleton/      # Module 2: Skeleton Mapping
│   ├── signal/        # Module 3: Signal Processing
│   ├── segmenter/     # Module 4: Motion Segmentation
│   ├── ml/            # Module 5: ML Animation Generation
│   ├── style/         # Module 6: Player Style Fingerprinting
│   ├── export/        # Module 8: Export Pipeline
│   └── runtime/       # Module 7: UE5 Plugin (separate build)
├── src/
│   ├── core/
│   ├── pose/
│   ├── skeleton/
│   ├── signal/
│   ├── segmenter/
│   ├── ml/
│   ├── style/
│   ├── export/
│   └── Pipeline.cpp   # Top-level orchestrator
├── ue5_plugin/         # Module 7: UE5 Plugin
│   └── Source/HyperMotion/
│       ├── Public/
│       └── Private/
└── tools/              # CLI executables
    ├── hm_extract.cpp
    ├── hm_train.cpp
    ├── hm_style.cpp
    └── hm_batch.cpp
```

## Namespace Convention
- `hm::` — core types and utilities
- `hm::pose::` — Module 1
- `hm::skeleton::` — Module 2
- `hm::signal::` — Module 3
- `hm::segmenter::` — Module 4
- `hm::ml::` — Module 5
- `hm::style::` — Module 6
- `hm::xport::` — Module 8 (export is a reserved word)
- UE5 plugin uses standard Unreal macros (UCLASS, USTRUCT, etc.)

---

## CORE TYPES (include/HyperMotion/core/Types.h)

### Joint Enum — 22-joint UE5 Mannequin skeleton
```
Hips(0), Spine(1), Spine1(2), Spine2(3), Neck(4), Head(5),
LeftShoulder(6), LeftArm(7), LeftForeArm(8), LeftHand(9),
RightShoulder(10), RightArm(11), RightForeArm(12), RightHand(13),
LeftUpLeg(14), LeftLeg(15), LeftFoot(16), LeftToeBase(17),
RightUpLeg(18), RightLeg(19), RightFoot(20), RightToeBase(21)
JOINT_COUNT = 22
```

### Hierarchy (parent indices)
```
Hips(-1), Spine(0), Spine1(1), Spine2(2), Neck(3), Head(4),
LeftShoulder(3), LeftArm(6), LeftForeArm(7), LeftHand(8),
RightShoulder(3), RightArm(10), RightForeArm(11), RightHand(12),
LeftUpLeg(0), LeftLeg(14), LeftFoot(15), LeftToeBase(16),
RightUpLeg(0), RightLeg(18), RightFoot(19), RightToeBase(20)
```

### Constants
- ROTATION_DIM = 6 (6D rotation representation per joint)
- FRAME_DIM = 132 (22 joints x 6D)
- STYLE_DIM = 64
- MotionCondition::DIM = 78

### Motion Types (16 categories)
```
Idle(0), Walk(1), Jog(2), Sprint(3), TurnLeft(4), TurnRight(5),
Decelerate(6), Jump(7), Slide(8), Kick(9), Tackle(10), Shield(11),
Receive(12), Celebrate(13), Goalkeeper(14), Unknown(15)
MOTION_TYPE_COUNT = 16
```

### Key Structs
- **Keypoint2D**: Vec2 position (normalised), float confidence
- **Keypoint3D**: Vec3 position (cm, Y-up), float confidence
- **DetectedPerson**: id, keypoints2D[17], keypoints3D[17], bbox, classLabel, reidFeature[128]
- **PoseFrameResult**: timestamp, frameIndex, vector<DetectedPerson>, videoWidth/Height
- **JointTransform**: localRotation (Quat), localEulerDeg (Vec3), rotation6D (Vec6), worldPosition (Vec3), confidence
- **SkeletonFrame**: timestamp, frameIndex, trackingID, rootPosition, rootRotation, rootVelocity, rootAngularVel, joints[22]
- **MotionSegment**: type, startFrame, endFrame, avgVelocity, avgDirection, confidence, trackingID
- **AnimClip**: name, fps, trackingID, vector<SkeletonFrame>, vector<MotionSegment>
- **PlayerStyle**: playerID, playerName, embedding[64], manual overrides (strideLengthScale, armSwingIntensity, sprintLeanAngle, hipRotationScale, kneeLiftScale, cadenceScale, decelerationSharpness, turnLeadBody)
- **MotionCondition**: velocity(3), speed, direction, targetDirection, ballRelativePos(3), ballDistance, requestedAction, fatigue, archetypeID, styleEmbedding[64] -> flattened to 78D vector
- **GeneratedMotion**: vector<SkeletonFrame> (64 frames), quality, inferenceTimeMs

### MathUtils (include/HyperMotion/core/MathUtils.h)
Must implement:
- Quaternion <-> Euler <-> 6D rotation conversions (Zhou et al. 6D representation using Gram-Schmidt)
- RotationBetween(from, to), SafeSlerp, LookRotation
- ForwardKinematics: root + local rotations -> world positions
- InverseKinematics: world positions -> root + local rotations
- SkeletonToVector / VectorToSkeleton: frame <-> flat float vector (132D)
- ClipToMatrix / MatrixToClip: clip <-> Eigen matrix [frames x 132]

---

## MODULE 1: Multi-Person Pose Estimation (hm::pose)

### Files
- PlayerDetector.h/.cpp — YOLOv8-L player/referee/GK detection
- SinglePoseEstimator.h/.cpp — HRNet-W48 single-person 17-keypoint estimation
- PoseTracker.h/.cpp — Temporal tracking with Hungarian matching
- DepthLifter.h/.cpp — Learned 2D->3D lifting network (LibTorch)
- MultiPersonPoseEstimator.h/.cpp — Top-level orchestrator

### PlayerDetector
- Input: cv::Mat frame
- Output: vector<Detection> with bbox, confidence, classID (0=player, 1=referee, 2=gk)
- Backend: OpenCV DNN (primary), TensorRT (optional)
- YOLOv8 output parsing: handle [1, 4+nclass, ndet] shape, transpose if needed
- Per-class NMS with configurable IoU threshold
- Max 30 detections per frame

### SinglePoseEstimator
- Input: cv::Mat frame + Detection bbox
- Output: array<Keypoint2D, 17> with normalised positions and confidence
- Crop person with 20% padding, adjust aspect ratio to 256x192
- ImageNet normalisation (mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
- HRNet heatmap output: [1, 17, H/4, W/4]
- Sub-pixel refinement via quadratic peak fitting
- Optional flip test: average original + horizontally flipped predictions
- Batch processing support

### PoseTracker
- Maintains vector<Tracklet> with: id, classLabel, age, hitCount, framesSinceLast, lastDetection, lastPose, positionHistory, velocityHistory, reidFeature, predictedCenter/Bbox
- Update loop: predict -> build cost matrix -> Hungarian match -> update matched -> create new -> prune dead
- Cost = weighted sum of (1-IoU) + (1-OKS) + (1-ReID_sim) with configurable weights
- OKS uses per-keypoint COCO sigmas
- Linear motion prediction from position history
- Track confirmation after minHitsToConfirm detections
- Track deletion after lostTimeout frames without detection

### DepthLifter
- Learned network: Linear(34,1024) -> BN -> ReLU -> [Linear(1024,1024) -> BN -> ReLU + skip] x2 -> Linear(1024,51)
- Input: 17 keypoints x 2D = 34, normalised relative to bbox centre
- Output: 17 keypoints x 3D = 51, scaled by subject height
- Geometric fallback when no trained model: estimate from torso length ratio
- Batch support

### MultiPersonPoseEstimator
- Orchestrates: detect -> pose -> track -> lift
- ProcessVideo(): reads video, handles frame skipping for target FPS
- ProcessFrame(): single frame for streaming
- Debug visualisation: bboxes, skeletons, IDs, colour by class
- Progress callback support

---

## MODULE 2: Skeleton Mapping (hm::skeleton)

### Files
- SkeletonMapper.h/.cpp
- RotationSolver.h/.cpp
- SkeletonRetargeter.h/.cpp

### SkeletonMapper
- Input: DetectedPerson (17 COCO keypoints in 3D)
- Output: SkeletonFrame (22-joint skeleton with rotations)
- Maps COCO keypoints to game skeleton joints:
  - Hips = midpoint(LeftHip, RightHip)
  - Spine chain: interpolate between hips and shoulder midpoint (3 spine joints at 33%, 66%, 100%)
  - Head/Neck: from shoulder midpoint to nose
  - Arms: shoulder->elbow->wrist (map to Shoulder, Arm, ForeArm, Hand)
  - Legs: hip->knee->ankle (map to UpLeg, Leg, Foot, ToeBase)
- Root rotation from hip-shoulder orientation matrix

### RotationSolver
- Computes local joint rotations from world positions
- Direction-matching: rotation that takes rest-pose bone direction to current-pose direction
- Uses RotationBetween() from MathUtils
- Outputs both quaternion and 6D rotation representation
- Handles edge cases: zero-length bones, near-parallel vectors

### SkeletonRetargeter
- Maps between different skeleton definitions
- Joint name mapping table
- Scale compensation for different body proportions
- Useful for importing third-party mocap data

---

## MODULE 3: Signal Processing (hm::signal)

### Files
- OutlierFilter.h/.cpp
- SavitzkyGolay.h/.cpp
- ButterworthFilter.h/.cpp
- QuaternionSmoother.h/.cpp
- FootContactFilter.h/.cpp
- SignalPipeline.h/.cpp

### OutlierFilter
- Median Absolute Deviation (MAD) filter
- Sliding window, replace outliers > 3 MAD from local median
- Process each joint X, Y, Z independently
- Configurable window size and threshold

### SavitzkyGolay
- Polynomial smoothing filter
- Default: window=7, polynomial order=3
- Compute coefficients via Vandermonde matrix + least squares
- Apply as 1D convolution with mirror boundary conditions
- Process per-joint per-axis

### ButterworthFilter
- 4th-order IIR low-pass filter
- Configurable cutoff frequency: 12 Hz for body joints, 8 Hz for extremities (hands, feet)
- Forward-backward filtering (zero phase distortion)
- Bilinear transform from analog prototype

### QuaternionSmoother
- SLERP-based rotation smoothing
- Configurable smoothing factor (default=0.3)
- Handles quaternion double-cover (sign flipping)
- Ensures shortest path interpolation

### FootContactFilter
- Detect foot-ground contact from foot velocity and height
- Contact threshold: velocity < 2 cm/s AND height < 5 cm
- During contact: snap foot Y to ground, lock XZ to prevent sliding
- Smooth transitions in/out of contact

### SignalPipeline
- Chains all 5 filters in order
- Process entire vector<SkeletonFrame> in-place
- Configurable: enable/disable individual stages

---

## MODULE 4: Motion Segmentation (hm::segmenter)

### Files
- MotionFeatureExtractor.h/.cpp
- TemporalConvNet.h/.cpp
- MotionSegmenter.h/.cpp

### MotionFeatureExtractor
- Input: SkeletonFrame -> 70D feature vector
  - 22 joints x 3 (Euler XYZ) = 66D + root velocity 3D + angular velocity 1D

### TemporalConvNet (LibTorch)
- 6 dilated causal conv blocks, dilation [1,2,4,8,16,32], receptive field 190 frames
- Hidden: 128 channels, kernel: 3
- Each block: Conv1D -> BN -> ReLU -> Dropout(0.1) -> Conv1D -> BN -> ReLU + residual
- Output: Conv1D(128, 14, k=1) -> per-frame logits
- ~600K parameters

### MotionSegmenter
- Sliding window TCN classification
- Merge consecutive same-label frames into segments
- Minimum segment length: 10 frames

---

## MODULE 5: ML Animation Generation (hm::ml)

### Files
- ConditionEncoder.h/.cpp
- NoiseScheduler.h/.cpp
- MotionTransformer.h/.cpp
- MotionDiffusionModel.h/.cpp
- MotionGenerator.h/.cpp

### ConditionEncoder (LibTorch)
- Linear(78, 512) -> ReLU -> Linear(512, 512) -> ReLU -> Linear(512, 256)

### NoiseScheduler
- Linear beta: 0.0001 to 0.02, T=1000 steps
- DDIM sampling: 50 steps for inference

### MotionTransformer (LibTorch)
- Input: Linear(132, 512), timestep: sinusoidal PE(512) + MLP, condition: Linear(256, 512)
- 8-layer transformer encoder, 8 heads, 512D, FFN 2048, pre-norm, GELU, dropout 0.1
- Output: Linear(512, 132)
- ~17.6M parameters

### MotionDiffusionModel
- Training: x0 -> sample t -> add noise -> predict noise -> MSE loss
- Inference: noise -> 50 DDIM steps -> 64 frames of clean motion

### MotionGenerator
- High-level inference: MotionCondition -> GeneratedMotion (64 frames)
- Post-processing: joint limits, foot contact, plausibility check

---

## MODULE 6: Player Style Fingerprinting (hm::style)

### Files
- StyleEncoder.h/.cpp
- ContrastiveLoss.h/.cpp
- StyleTrainer.h/.cpp
- StyleLibrary.h/.cpp

### StyleEncoder (LibTorch)
- Input: variable-length, 201D per frame (132 rotations + 3 root vel + 66 angular vel)
- Conv1D(201,128) -> 4 ResBlocks (128->128->256->256->512) -> GAP -> Linear(512,256) -> ReLU -> Linear(256,64) -> L2 norm
- ~1.9M parameters

### ContrastiveLoss
- NT-Xent, tau=0.07, positive=same player, negative=different players

### StyleTrainer
- Adam lr=1e-4, cosine annealing, 200 epochs, batch 32 pairs
- Data augmentation: temporal crop, speed perturbation, noise

### StyleLibrary
- map<string, PlayerStyle>, save/load JSON, interpolate, nearest search

---

## MODULE 7: UE5 Plugin

### UHyperMotionComponent
- 14 movement states + 9 ball states
- Speed thresholds: Idle<10, Walk<120, Jog<250, Run<450, Sprint>450 cm/s
- Computed: Speed, NormalizedSpeed, Direction, LeanAngle, Fatigue, TransitionAlpha
- FPlayerAnimStyle per-player overrides

### UHyperMotionAnimInstance
- Reads from component, exposes to AnimBP
- Speed, Direction, LeanAngle, Fatigue, bIsMoving, bIsSprinting, bIsTurning, bHasBall

### UHyperMotionMLInference
- ONNX Runtime wrapper, async inference, GPU tensors

### UHyperMotionFootIK
- Per-foot traces, hip offset, ground normal rotation

### UHyperMotionStyleManager
- Style library loading, player lookup, interpolation

---

## MODULE 8: Export Pipeline (hm::xport)

### BVHExporter
- HIERARCHY with recursive joints, OFFSET, CHANNELS
- MOTION with per-frame rotations matching hierarchy order

### JSONExporter
- Full data: frames with all joints (rotation, position, quaternion), segments, metadata

### AnimClipUtils
- SubClip, SplitBySegments, Concatenate, Resample, Mirror

---

## CLI TOOLS

### hm_extract
```
hm_extract --detector yolov8l.onnx --pose hrnet_w48.onnx --input match.mp4 --output anims/ --fps 30 --visualize --split
```

### hm_train
```
hm_train --mode diffusion --data clips/ --epochs 500 --batch 64 --lr 1e-4
hm_train --mode classifier --data segments/ --epochs 100
```

### hm_style
```
hm_style --train --data player_clips/ --epochs 200 --output styles/
```

### hm_batch
```
hm_batch --detector yolov8l.onnx --pose hrnet_w48.onnx --input videos/ --output anims/
```

---

## IMPORTANT REMINDERS
- This is a FULL system, not lite, not simplified
- Build EVERY module completely
- Every .h needs a corresponding .cpp with full implementation
- No stubs, no TODOs, no placeholders
- Production-grade code with proper error handling
- Use LibTorch for all neural network definitions and training
- Use ONNX Runtime for optimised inference
- The UE5 plugin is a separate build target with proper .Build.cs and .uplugin files
