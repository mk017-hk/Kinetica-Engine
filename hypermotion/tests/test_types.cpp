#include <gtest/gtest.h>
#include "HyperMotion/core/Types.h"
#include "test_helpers.h"
#include <cmath>

using namespace hm;

// ---------------------------------------------------------------
// Vec2 Tests
// ---------------------------------------------------------------

TEST(Vec2Test, DefaultConstruction) {
    Vec2 v;
    EXPECT_FLOAT_EQ(v.x, 0.0f);
    EXPECT_FLOAT_EQ(v.y, 0.0f);
}

TEST(Vec2Test, Addition) {
    Vec2 a{1.0f, 2.0f}, b{3.0f, 4.0f};
    auto c = a + b;
    EXPECT_FLOAT_EQ(c.x, 4.0f);
    EXPECT_FLOAT_EQ(c.y, 6.0f);
}

TEST(Vec2Test, Subtraction) {
    Vec2 a{5.0f, 7.0f}, b{2.0f, 3.0f};
    auto c = a - b;
    EXPECT_FLOAT_EQ(c.x, 3.0f);
    EXPECT_FLOAT_EQ(c.y, 4.0f);
}

TEST(Vec2Test, ScalarMultiply) {
    Vec2 v{3.0f, 4.0f};
    auto r = v * 2.0f;
    EXPECT_FLOAT_EQ(r.x, 6.0f);
    EXPECT_FLOAT_EQ(r.y, 8.0f);
}

TEST(Vec2Test, Dot) {
    Vec2 a{1.0f, 0.0f}, b{0.0f, 1.0f};
    EXPECT_FLOAT_EQ(a.dot(b), 0.0f);

    Vec2 c{2.0f, 3.0f}, d{4.0f, 5.0f};
    EXPECT_FLOAT_EQ(c.dot(d), 23.0f);
}

TEST(Vec2Test, Length) {
    Vec2 v{3.0f, 4.0f};
    EXPECT_FLOAT_EQ(v.length(), 5.0f);
}

TEST(Vec2Test, Normalized) {
    Vec2 v{3.0f, 4.0f};
    auto n = v.normalized();
    EXPECT_NEAR(n.length(), 1.0f, test::kEps);
    EXPECT_NEAR(n.x, 0.6f, test::kEps);
    EXPECT_NEAR(n.y, 0.8f, test::kEps);
}

TEST(Vec2Test, NormalizedZeroVector) {
    Vec2 v{0.0f, 0.0f};
    auto n = v.normalized();
    EXPECT_FLOAT_EQ(n.x, 0.0f);
    EXPECT_FLOAT_EQ(n.y, 0.0f);
}

// ---------------------------------------------------------------
// Vec3 Tests
// ---------------------------------------------------------------

TEST(Vec3Test, Addition) {
    Vec3 a{1, 2, 3}, b{4, 5, 6};
    auto c = a + b;
    EXPECT_FLOAT_EQ(c.x, 5.0f);
    EXPECT_FLOAT_EQ(c.y, 7.0f);
    EXPECT_FLOAT_EQ(c.z, 9.0f);
}

TEST(Vec3Test, CrossProduct) {
    Vec3 x{1, 0, 0}, y{0, 1, 0};
    auto z = x.cross(y);
    EXPECT_NEAR(z.x, 0.0f, test::kEps);
    EXPECT_NEAR(z.y, 0.0f, test::kEps);
    EXPECT_NEAR(z.z, 1.0f, test::kEps);
}

TEST(Vec3Test, CrossProductAnticommutative) {
    Vec3 a{1, 2, 3}, b{4, 5, 6};
    auto ab = a.cross(b);
    auto ba = b.cross(a);
    EXPECT_NEAR(ab.x, -ba.x, test::kEps);
    EXPECT_NEAR(ab.y, -ba.y, test::kEps);
    EXPECT_NEAR(ab.z, -ba.z, test::kEps);
}

TEST(Vec3Test, DotOrthogonal) {
    Vec3 a{1, 0, 0}, b{0, 1, 0};
    EXPECT_FLOAT_EQ(a.dot(b), 0.0f);
}

TEST(Vec3Test, Length) {
    Vec3 v{1, 2, 2};
    EXPECT_FLOAT_EQ(v.length(), 3.0f);
}

TEST(Vec3Test, LengthSq) {
    Vec3 v{3, 4, 0};
    EXPECT_FLOAT_EQ(v.lengthSq(), 25.0f);
}

TEST(Vec3Test, CompoundAssignment) {
    Vec3 v{1, 2, 3};
    v += Vec3{10, 20, 30};
    EXPECT_FLOAT_EQ(v.x, 11.0f);
    v -= Vec3{1, 2, 3};
    EXPECT_FLOAT_EQ(v.x, 10.0f);
    v *= 2.0f;
    EXPECT_FLOAT_EQ(v.x, 20.0f);
}

TEST(Vec3Test, ScalarLeftMultiply) {
    Vec3 v{1, 2, 3};
    auto r = 2.0f * v;
    EXPECT_FLOAT_EQ(r.x, 2.0f);
    EXPECT_FLOAT_EQ(r.y, 4.0f);
    EXPECT_FLOAT_EQ(r.z, 6.0f);
}

// ---------------------------------------------------------------
// Quaternion Tests
// ---------------------------------------------------------------

TEST(QuatTest, IdentityMultiply) {
    Quat id = Quat::identity();
    Quat q{0.707f, 0.707f, 0.0f, 0.0f};
    auto r = id * q;
    EXPECT_NEAR(r.w, q.w, test::kEps);
    EXPECT_NEAR(r.x, q.x, test::kEps);
}

TEST(QuatTest, ConjugateMultiply) {
    Quat q{0.707f, 0.707f, 0.0f, 0.0f};
    q = q.normalized();
    auto product = q * q.conjugate();
    EXPECT_NEAR(product.w, 1.0f, test::kEps);
    EXPECT_NEAR(product.x, 0.0f, test::kEps);
    EXPECT_NEAR(product.y, 0.0f, test::kEps);
    EXPECT_NEAR(product.z, 0.0f, test::kEps);
}

TEST(QuatTest, Norm) {
    Quat q{1, 2, 3, 4};
    float expected = std::sqrt(1 + 4 + 9 + 16);
    EXPECT_NEAR(q.norm(), expected, test::kEps);
}

TEST(QuatTest, Normalized) {
    Quat q{1, 2, 3, 4};
    auto n = q.normalized();
    EXPECT_NEAR(n.norm(), 1.0f, test::kEps);
}

TEST(QuatTest, RotateVector) {
    // 90 degree rotation around Y axis should map X -> -Z
    float half = 3.14159265f / 4.0f;
    Quat q{std::cos(half), 0.0f, std::sin(half), 0.0f};
    Vec3 v{1, 0, 0};
    auto r = q.rotate(v);
    EXPECT_NEAR(r.x, 0.0f, test::kEps);
    EXPECT_NEAR(r.y, 0.0f, test::kEps);
    EXPECT_NEAR(r.z, -1.0f, test::kEps);
}

TEST(QuatTest, RotateIdentity) {
    Vec3 v{3.5f, -2.1f, 7.8f};
    auto r = Quat::identity().rotate(v);
    EXPECT_NEAR(r.x, v.x, test::kEps);
    EXPECT_NEAR(r.y, v.y, test::kEps);
    EXPECT_NEAR(r.z, v.z, test::kEps);
}

// ---------------------------------------------------------------
// Mat3 Tests
// ---------------------------------------------------------------

TEST(Mat3Test, IdentityMultiply) {
    Mat3 id = Mat3::identity();
    Vec3 v{1, 2, 3};
    auto r = id * v;
    EXPECT_FLOAT_EQ(r.x, 1.0f);
    EXPECT_FLOAT_EQ(r.y, 2.0f);
    EXPECT_FLOAT_EQ(r.z, 3.0f);
}

TEST(Mat3Test, Determinant) {
    Mat3 id = Mat3::identity();
    EXPECT_NEAR(id.determinant(), 1.0f, test::kEps);
}

TEST(Mat3Test, TransposeIdentity) {
    Mat3 id = Mat3::identity();
    auto t = id.transposed();
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            EXPECT_FLOAT_EQ(t.m[i][j], id.m[i][j]);
}

TEST(Mat3Test, TransposeInverts) {
    Mat3 m;
    m.m[0][1] = 5.0f;
    m.m[1][0] = 3.0f;
    auto t = m.transposed();
    EXPECT_FLOAT_EQ(t.m[0][1], 3.0f);
    EXPECT_FLOAT_EQ(t.m[1][0], 5.0f);
}

TEST(Mat3Test, ColumnRowAccess) {
    Mat3 m = Mat3::identity();
    m.setCol(1, {10, 20, 30});
    auto c = m.col(1);
    EXPECT_FLOAT_EQ(c.x, 10.0f);
    EXPECT_FLOAT_EQ(c.y, 20.0f);
    EXPECT_FLOAT_EQ(c.z, 30.0f);
}

// ---------------------------------------------------------------
// BBox Tests
// ---------------------------------------------------------------

TEST(BBoxTest, CenterAndArea) {
    BBox b{10, 20, 100, 200};
    EXPECT_FLOAT_EQ(b.centerX(), 60.0f);
    EXPECT_FLOAT_EQ(b.centerY(), 120.0f);
    EXPECT_FLOAT_EQ(b.area(), 20000.0f);
}

TEST(BBoxTest, IoUSameBox) {
    BBox b{0, 0, 100, 100};
    EXPECT_NEAR(b.iou(b), 1.0f, test::kEps);
}

TEST(BBoxTest, IoUNoOverlap) {
    BBox a{0, 0, 10, 10};
    BBox b{20, 20, 10, 10};
    EXPECT_FLOAT_EQ(a.iou(b), 0.0f);
}

TEST(BBoxTest, IoUPartialOverlap) {
    BBox a{0, 0, 10, 10};  // area=100
    BBox b{5, 5, 10, 10};  // area=100
    // intersection: 5x5=25, union: 100+100-25=175
    EXPECT_NEAR(a.iou(b), 25.0f / 175.0f, test::kEps);
}

// ---------------------------------------------------------------
// MotionCondition Tests
// ---------------------------------------------------------------

TEST(MotionConditionTest, FlattenDimension) {
    MotionCondition cond;
    cond.velocity = {1, 2, 3};
    cond.speed = 5.0f;
    auto flat = cond.flatten();
    EXPECT_EQ(flat.size(), static_cast<size_t>(MotionCondition::DIM));
    EXPECT_FLOAT_EQ(flat[0], 1.0f);
    EXPECT_FLOAT_EQ(flat[1], 2.0f);
    EXPECT_FLOAT_EQ(flat[2], 3.0f);
    EXPECT_FLOAT_EQ(flat[3], 5.0f);
}

TEST(MotionConditionTest, FlattenStyleEmbedding) {
    MotionCondition cond;
    cond.styleEmbedding[0] = 0.5f;
    cond.styleEmbedding[63] = 0.9f;
    auto flat = cond.flatten();
    EXPECT_FLOAT_EQ(flat[13], 0.5f);   // style starts at index 13
    EXPECT_FLOAT_EQ(flat[76], 0.9f);   // index 13+63 = 76
}
