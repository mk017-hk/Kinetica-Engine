#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>
#include <cmath>

namespace hm {

// -------------------------------------------------------------------
// Linear Algebra Primitives
// -------------------------------------------------------------------

struct Vec2 {
    float x = 0.0f, y = 0.0f;

    Vec2() = default;
    Vec2(float x_, float y_) : x(x_), y(y_) {}

    Vec2 operator+(const Vec2& o) const { return {x + o.x, y + o.y}; }
    Vec2 operator-(const Vec2& o) const { return {x - o.x, y - o.y}; }
    Vec2 operator*(float s) const { return {x * s, y * s}; }
    Vec2 operator/(float s) const { return {x / s, y / s}; }
    float dot(const Vec2& o) const { return x * o.x + y * o.y; }
    float length() const { return std::sqrt(x * x + y * y); }
    Vec2 normalized() const { float l = length(); return l > 1e-8f ? Vec2{x / l, y / l} : Vec2{}; }
};

struct Vec3 {
    float x = 0.0f, y = 0.0f, z = 0.0f;

    Vec3() = default;
    Vec3(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}

    Vec3 operator+(const Vec3& o) const { return {x + o.x, y + o.y, z + o.z}; }
    Vec3 operator-(const Vec3& o) const { return {x - o.x, y - o.y, z - o.z}; }
    Vec3 operator*(float s) const { return {x * s, y * s, z * s}; }
    Vec3 operator/(float s) const { return {x / s, y / s, z / s}; }
    Vec3& operator+=(const Vec3& o) { x += o.x; y += o.y; z += o.z; return *this; }
    Vec3& operator-=(const Vec3& o) { x -= o.x; y -= o.y; z -= o.z; return *this; }
    Vec3& operator*=(float s) { x *= s; y *= s; z *= s; return *this; }
    float dot(const Vec3& o) const { return x * o.x + y * o.y + z * o.z; }
    Vec3 cross(const Vec3& o) const {
        return {y * o.z - z * o.y, z * o.x - x * o.z, x * o.y - y * o.x};
    }
    float length() const { return std::sqrt(x * x + y * y + z * z); }
    float lengthSq() const { return x * x + y * y + z * z; }
    Vec3 normalized() const { float l = length(); return l > 1e-8f ? Vec3{x / l, y / l, z / l} : Vec3{}; }
};

inline Vec3 operator*(float s, const Vec3& v) { return v * s; }

struct Vec4 {
    float x = 0.0f, y = 0.0f, z = 0.0f, w = 0.0f;

    Vec4() = default;
    Vec4(float x_, float y_, float z_, float w_) : x(x_), y(y_), z(z_), w(w_) {}
};

using Vec6 = std::array<float, 6>;

struct Quat {
    float w = 1.0f, x = 0.0f, y = 0.0f, z = 0.0f;

    Quat() = default;
    Quat(float w_, float x_, float y_, float z_) : w(w_), x(x_), y(y_), z(z_) {}

    static Quat identity() { return {1.0f, 0.0f, 0.0f, 0.0f}; }

    Quat operator*(const Quat& q) const {
        return {
            w * q.w - x * q.x - y * q.y - z * q.z,
            w * q.x + x * q.w + y * q.z - z * q.y,
            w * q.y - x * q.z + y * q.w + z * q.x,
            w * q.z + x * q.y - y * q.x + z * q.w
        };
    }

    Quat conjugate() const { return {w, -x, -y, -z}; }

    float norm() const { return std::sqrt(w * w + x * x + y * y + z * z); }

    Quat normalized() const {
        float n = norm();
        if (n < 1e-8f) return identity();
        return {w / n, x / n, y / n, z / n};
    }

    Vec3 rotate(const Vec3& v) const {
        Quat qv{0.0f, v.x, v.y, v.z};
        Quat result = (*this) * qv * conjugate();
        return {result.x, result.y, result.z};
    }

    float dot(const Quat& q) const { return w * q.w + x * q.x + y * q.y + z * q.z; }
};

struct Mat3 {
    float m[3][3] = {{1,0,0},{0,1,0},{0,0,1}};

    static Mat3 identity() { return {}; }

    Vec3 col(int c) const { return {m[0][c], m[1][c], m[2][c]}; }
    Vec3 row(int r) const { return {m[r][0], m[r][1], m[r][2]}; }

    void setCol(int c, const Vec3& v) { m[0][c] = v.x; m[1][c] = v.y; m[2][c] = v.z; }
    void setRow(int r, const Vec3& v) { m[r][0] = v.x; m[r][1] = v.y; m[r][2] = v.z; }

    Mat3 operator*(const Mat3& o) const {
        Mat3 result;
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j) {
                result.m[i][j] = 0;
                for (int k = 0; k < 3; ++k)
                    result.m[i][j] += m[i][k] * o.m[k][j];
            }
        return result;
    }

    Vec3 operator*(const Vec3& v) const {
        return {
            m[0][0] * v.x + m[0][1] * v.y + m[0][2] * v.z,
            m[1][0] * v.x + m[1][1] * v.y + m[1][2] * v.z,
            m[2][0] * v.x + m[2][1] * v.y + m[2][2] * v.z
        };
    }

    Mat3 transposed() const {
        Mat3 r;
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                r.m[i][j] = m[j][i];
        return r;
    }

    float determinant() const {
        return m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
             - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
             + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
    }
};

// -------------------------------------------------------------------
// Bounding Box
// -------------------------------------------------------------------

struct BBox {
    float x = 0.0f, y = 0.0f, width = 0.0f, height = 0.0f;
    float confidence = 0.0f;

    float centerX() const { return x + width * 0.5f; }
    float centerY() const { return y + height * 0.5f; }
    float area() const { return width * height; }

    float iou(const BBox& other) const {
        float x1 = std::max(x, other.x);
        float y1 = std::max(y, other.y);
        float x2 = std::min(x + width, other.x + other.width);
        float y2 = std::min(y + height, other.y + other.height);
        float inter = std::max(0.0f, x2 - x1) * std::max(0.0f, y2 - y1);
        float unionArea = area() + other.area() - inter;
        return unionArea > 0.0f ? inter / unionArea : 0.0f;
    }
};

// -------------------------------------------------------------------
// Joint Enum and Hierarchy
// -------------------------------------------------------------------

enum class Joint : int {
    Hips = 0, Spine = 1, Spine1 = 2, Spine2 = 3, Neck = 4, Head = 5,
    LeftShoulder = 6, LeftArm = 7, LeftForeArm = 8, LeftHand = 9,
    RightShoulder = 10, RightArm = 11, RightForeArm = 12, RightHand = 13,
    LeftUpLeg = 14, LeftLeg = 15, LeftFoot = 16, LeftToeBase = 17,
    RightUpLeg = 18, RightLeg = 19, RightFoot = 20, RightToeBase = 21
};

constexpr int JOINT_COUNT = 22;

// Parent indices for skeleton hierarchy (-1 = root)
constexpr int JOINT_PARENT[JOINT_COUNT] = {
    -1,  // Hips
     0,  // Spine -> Hips
     1,  // Spine1 -> Spine
     2,  // Spine2 -> Spine1
     3,  // Neck -> Spine2
     4,  // Head -> Neck
     3,  // LeftShoulder -> Spine2
     6,  // LeftArm -> LeftShoulder
     7,  // LeftForeArm -> LeftArm
     8,  // LeftHand -> LeftForeArm
     3,  // RightShoulder -> Spine2
    10,  // RightArm -> RightShoulder
    11,  // RightForeArm -> RightArm
    12,  // RightHand -> RightForeArm
     0,  // LeftUpLeg -> Hips
    14,  // LeftLeg -> LeftUpLeg
    15,  // LeftFoot -> LeftLeg
    16,  // LeftToeBase -> LeftFoot
     0,  // RightUpLeg -> Hips
    18,  // RightLeg -> RightUpLeg
    19,  // RightFoot -> RightLeg
    20   // RightToeBase -> RightFoot
};

constexpr const char* JOINT_NAMES[JOINT_COUNT] = {
    "Hips", "Spine", "Spine1", "Spine2", "Neck", "Head",
    "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand",
    "RightShoulder", "RightArm", "RightForeArm", "RightHand",
    "LeftUpLeg", "LeftLeg", "LeftFoot", "LeftToeBase",
    "RightUpLeg", "RightLeg", "RightFoot", "RightToeBase"
};

// -------------------------------------------------------------------
// Constants
// -------------------------------------------------------------------

constexpr int ROTATION_DIM = 6;      // 6D rotation per joint
constexpr int FRAME_DIM = 132;       // 22 joints x 6D
constexpr int STYLE_DIM = 64;        // Style embedding dimension
constexpr int COCO_KEYPOINTS = 17;   // COCO pose keypoints
constexpr int REID_DIM = 128;        // Re-identification feature dimension

// -------------------------------------------------------------------
// Motion Types
// -------------------------------------------------------------------

enum class MotionType : int {
    Idle = 0, Walk = 1, Jog = 2, Sprint = 3,
    TurnLeft = 4, TurnRight = 5, Decelerate = 6,
    Jump = 7, Slide = 8, Kick = 9, Tackle = 10,
    Shield = 11, Receive = 12, Celebrate = 13,
    Goalkeeper = 14, Unknown = 15
};

constexpr int MOTION_TYPE_COUNT = 16;

constexpr const char* MOTION_TYPE_NAMES[MOTION_TYPE_COUNT] = {
    "Idle", "Walk", "Jog", "Sprint", "TurnLeft", "TurnRight",
    "Decelerate", "Jump", "Slide", "Kick", "Tackle", "Shield",
    "Receive", "Celebrate", "Goalkeeper", "Unknown"
};

// -------------------------------------------------------------------
// Pose Estimation Structs
// -------------------------------------------------------------------

struct Keypoint2D {
    Vec2 position;   // Normalised [0,1]
    float confidence = 0.0f;
};

struct Keypoint3D {
    Vec3 position;   // cm, Y-up
    float confidence = 0.0f;
};

struct Detection {
    BBox bbox;
    float confidence = 0.0f;
    int classID = 0;       // 0=player, 1=referee, 2=goalkeeper
    std::string classLabel;
};

struct DetectedPerson {
    int id = -1;
    std::array<Keypoint2D, COCO_KEYPOINTS> keypoints2D;
    std::array<Keypoint3D, COCO_KEYPOINTS> keypoints3D;
    BBox bbox;
    std::string classLabel;
    std::array<float, REID_DIM> reidFeature{};
};

struct PoseFrameResult {
    double timestamp = 0.0;
    int frameIndex = 0;
    std::vector<DetectedPerson> persons;
    int videoWidth = 0;
    int videoHeight = 0;
};

// -------------------------------------------------------------------
// Skeleton Structs
// -------------------------------------------------------------------

struct JointTransform {
    Quat localRotation = Quat::identity();
    Vec3 localEulerDeg;
    Vec6 rotation6D{};
    Vec3 worldPosition;
    float confidence = 0.0f;
};

struct SkeletonFrame {
    double timestamp = 0.0;
    int frameIndex = 0;
    int trackingID = -1;
    Vec3 rootPosition;
    Quat rootRotation = Quat::identity();
    Vec3 rootVelocity;
    Vec3 rootAngularVel;
    std::array<JointTransform, JOINT_COUNT> joints;
};

// -------------------------------------------------------------------
// Motion Segmentation
// -------------------------------------------------------------------

struct MotionSegment {
    MotionType type = MotionType::Unknown;
    int startFrame = 0;
    int endFrame = 0;
    float avgVelocity = 0.0f;
    Vec3 avgDirection;
    float confidence = 0.0f;
    int trackingID = -1;
};

// -------------------------------------------------------------------
// Animation Clip
// -------------------------------------------------------------------

struct AnimClip {
    std::string name;
    float fps = 30.0f;
    int trackingID = -1;
    std::vector<SkeletonFrame> frames;
    std::vector<MotionSegment> segments;
};

// -------------------------------------------------------------------
// Player Style
// -------------------------------------------------------------------

struct PlayerStyle {
    std::string playerID;
    std::string playerName;
    std::array<float, STYLE_DIM> embedding{};

    // Manual overrides
    float strideLengthScale = 1.0f;
    float armSwingIntensity = 1.0f;
    float sprintLeanAngle = 0.0f;
    float hipRotationScale = 1.0f;
    float kneeLiftScale = 1.0f;
    float cadenceScale = 1.0f;
    float decelerationSharpness = 1.0f;
    float turnLeadBody = 0.0f;
};

// -------------------------------------------------------------------
// Motion Condition (Input to ML generation)
// -------------------------------------------------------------------

struct MotionCondition {
    static constexpr int DIM = 78;

    Vec3 velocity;
    float speed = 0.0f;
    float direction = 0.0f;
    float targetDirection = 0.0f;
    Vec3 ballRelativePos;
    float ballDistance = 0.0f;
    MotionType requestedAction = MotionType::Idle;
    float fatigue = 0.0f;
    int archetypeID = 0;
    std::array<float, STYLE_DIM> styleEmbedding{};

    // Flatten to 78D vector
    std::array<float, DIM> flatten() const {
        std::array<float, DIM> v{};
        int i = 0;
        v[i++] = velocity.x; v[i++] = velocity.y; v[i++] = velocity.z;
        v[i++] = speed;
        v[i++] = direction;
        v[i++] = targetDirection;
        v[i++] = ballRelativePos.x; v[i++] = ballRelativePos.y; v[i++] = ballRelativePos.z;
        v[i++] = ballDistance;
        v[i++] = static_cast<float>(static_cast<int>(requestedAction));
        v[i++] = fatigue;
        v[i++] = static_cast<float>(archetypeID);
        for (int s = 0; s < STYLE_DIM; ++s)
            v[i++] = styleEmbedding[s];
        // i should be 13 + 64 = 77; pad last
        v[i++] = 0.0f;
        return v;
    }
};

// -------------------------------------------------------------------
// Generated Motion (Output from ML)
// -------------------------------------------------------------------

struct GeneratedMotion {
    std::vector<SkeletonFrame> frames; // 64 frames
    float quality = 0.0f;
    float inferenceTimeMs = 0.0f;
};

// -------------------------------------------------------------------
// Rest-pose bone directions (Y-up, cm)
// Used by RotationSolver and ForwardKinematics
// -------------------------------------------------------------------

inline const std::array<Vec3, JOINT_COUNT>& getRestPoseBoneOffsets() {
    static const std::array<Vec3, JOINT_COUNT> offsets = {{
        {0.0f,   0.0f,   0.0f},   // Hips (root)
        {0.0f,  10.0f,   0.0f},   // Spine
        {0.0f,  10.0f,   0.0f},   // Spine1
        {0.0f,  10.0f,   0.0f},   // Spine2
        {0.0f,   8.0f,   0.0f},   // Neck
        {0.0f,   8.0f,   0.0f},   // Head
        {-5.0f,  0.0f,   0.0f},   // LeftShoulder
        {-8.0f,  0.0f,   0.0f},   // LeftArm
        {-25.0f, 0.0f,   0.0f},   // LeftForeArm
        {-23.0f, 0.0f,   0.0f},   // LeftHand
        {5.0f,   0.0f,   0.0f},   // RightShoulder
        {8.0f,   0.0f,   0.0f},   // RightArm
        {25.0f,  0.0f,   0.0f},   // RightForeArm
        {23.0f,  0.0f,   0.0f},   // RightHand
        {-9.0f, -3.0f,   0.0f},   // LeftUpLeg
        {0.0f, -40.0f,   0.0f},   // LeftLeg
        {0.0f, -38.0f,   0.0f},   // LeftFoot
        {0.0f,  -3.0f,   8.0f},   // LeftToeBase
        {9.0f,  -3.0f,   0.0f},   // RightUpLeg
        {0.0f, -40.0f,   0.0f},   // RightLeg
        {0.0f, -38.0f,   0.0f},   // RightFoot
        {0.0f,  -3.0f,   8.0f}    // RightToeBase
    }};
    return offsets;
}

} // namespace hm
