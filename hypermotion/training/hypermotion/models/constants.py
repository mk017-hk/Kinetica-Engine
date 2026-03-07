"""Shared constants matching the C++ core/Types.h definitions."""

JOINT_COUNT = 22
ROTATION_DIM = 6          # 6D rotation per joint (Zhou et al.)
FRAME_DIM = 132           # 22 * 6
STYLE_DIM = 64
CONDITION_DIM = 78        # MotionCondition flattened
MOTION_TYPE_COUNT = 16

# Feature dimensions
FEATURE_DIM_SEGMENTER = 70   # 66 euler + 3 root vel + 1 angular vel
STYLE_INPUT_DIM = 201         # 132 rot + 3 root vel + 66 angular vel
MOTION_EMBEDDING_DIM = 128    # motion encoder output dimension
MOTION_INPUT_DIM = JOINT_COUNT * 3  # 22 * 3 = 66 (world positions)

# Skeleton hierarchy (parent indices, -1 = root)
PARENT_INDICES = [
    -1,  # 0  Hips
     0,  # 1  Spine
     1,  # 2  Spine1
     2,  # 3  Spine2
     3,  # 4  Neck
     4,  # 5  Head
     3,  # 6  LeftShoulder
     6,  # 7  LeftArm
     7,  # 8  LeftForeArm
     8,  # 9  LeftHand
     3,  # 10 RightShoulder
    10,  # 11 RightArm
    11,  # 12 RightForeArm
    12,  # 13 RightHand
     0,  # 14 LeftUpLeg
    14,  # 15 LeftLeg
    15,  # 16 LeftFoot
    16,  # 17 LeftToeBase
     0,  # 18 RightUpLeg
    18,  # 19 RightLeg
    19,  # 20 RightFoot
    20,  # 21 RightToeBase
]

JOINT_NAMES = [
    "Hips", "Spine", "Spine1", "Spine2", "Neck", "Head",
    "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand",
    "RightShoulder", "RightArm", "RightForeArm", "RightHand",
    "LeftUpLeg", "LeftLeg", "LeftFoot", "LeftToeBase",
    "RightUpLeg", "RightLeg", "RightFoot", "RightToeBase",
]

MOTION_TYPE_NAMES = [
    "Idle", "Walk", "Jog", "Sprint", "TurnLeft", "TurnRight",
    "Decelerate", "Jump", "Slide", "Kick", "Tackle", "Shield",
    "Receive", "Celebrate", "Goalkeeper", "Unknown",
]
