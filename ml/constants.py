"""Shared constants matching the C++ core (HyperMotion/core/Types.h).

Every numeric value here MUST stay in sync with the C++ header.
"""

# Skeleton
JOINT_COUNT = 22
ROTATION_DIM = 6          # 6D rotation per joint (Zhou et al.)
FRAME_DIM = 132            # JOINT_COUNT * ROTATION_DIM
COCO_KEYPOINTS = 17

# Embeddings
STYLE_DIM = 64
MOTION_EMBEDDING_DIM = 128
REID_DIM = 128

# Conditions
CONDITION_DIM = 78         # MotionCondition flattened

# Feature extraction
FEATURE_DIM_SEGMENTER = 70   # 66 Euler + 3 vel + 1 angular vel
STYLE_INPUT_DIM = 201         # 132 + 3 vel + 66 angular vel
MOTION_INPUT_DIM = 66         # JOINT_COUNT * 3 world positions

# Classification
MOTION_TYPE_COUNT = 16

MOTION_TYPES = [
    "Idle", "Walk", "Jog", "Sprint",
    "TurnLeft", "TurnRight", "Decelerate", "Jump",
    "Slide", "Kick", "Tackle", "Shield",
    "Receive", "Celebrate", "Goalkeeper", "Unknown",
]

# Joint names matching the C++ Joint enum order
JOINT_NAMES = [
    "Hips", "Spine", "Spine1", "Spine2", "Neck", "Head",
    "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand",
    "RightShoulder", "RightArm", "RightForeArm", "RightHand",
    "LeftUpLeg", "LeftLeg", "LeftFoot", "LeftToeBase",
    "RightUpLeg", "RightLeg", "RightFoot", "RightToeBase",
]

# COCO keypoint names
COCO_KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]

# COCO skeleton connections (for visualization)
COCO_SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),      # head
    (5, 6),                                  # shoulders
    (5, 7), (7, 9), (6, 8), (8, 10),       # arms
    (5, 11), (6, 12),                        # torso
    (11, 12),                                # hips
    (11, 13), (13, 15), (12, 14), (14, 16), # legs
]

# COCO keypoint flip pairs (for augmentation)
COCO_FLIP_PAIRS = [
    (1, 2), (3, 4), (5, 6), (7, 8),
    (9, 10), (11, 12), (13, 14), (15, 16),
]

# Default sequence length for training
DEFAULT_SEQ_LEN = 64

# ONNX export
ONNX_OPSET = 17

# ImageNet normalization (used by pose estimators)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
