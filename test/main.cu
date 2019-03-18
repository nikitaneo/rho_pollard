#include <gtest/gtest.h>
#include <iostream>
#include <int128_t.h>
#include <pollard.h>

TEST(INT256, Simple)
{
    int128_t a = -5;
    int128_t b = 10;
    ASSERT_EQ(a + b, 5);
    ASSERT_EQ(-a, 5);
    ASSERT_EQ(-2 * a, 10);
    ASSERT_EQ(a * a, 25);
    ASSERT_EQ(a < 0, true);
    ASSERT_EQ(a > 0, false);
    ASSERT_EQ(a == -5, true);
    ASSERT_EQ(a != -6, true);
    ASSERT_EQ(-2 * b, 4 * a);
    ASSERT_EQ(b - a, 15);
    ASSERT_EQ(b * a, -50);
    ASSERT_EQ(b >> 1, 5);
    ASSERT_EQ(b << 1, 20);
    ASSERT_EQ(b << 2, 40);
    ASSERT_EQ(a / 1, -5);
    ASSERT_EQ(a * 2, -10);
    ASSERT_EQ(a * (-1), 5);
    ASSERT_EQ(a < -2, false);
    ASSERT_EQ(b > a, true);
    ASSERT_EQ(++b, 11);
    ASSERT_EQ(++a, -4);
    ASSERT_EQ(--a, -5);
    ASSERT_EQ(a += 1, -4);
    ASSERT_EQ(a -= 1, -5);
    ASSERT_EQ(a - 1, -6);
    ASSERT_EQ(b - 1, 10);
    ASSERT_EQ(b / 2, 5);
    ASSERT_EQ(b / 2 * 2, 10);
    ASSERT_EQ(b % 2, 1);
    ASSERT_EQ(a % 3, -2);
}

TEST(Detail, EGCD)
{
    int128_t a = 612, b = 342, x = 0, y = 0;
    ASSERT_EQ(detail::EGCD(a, b, x, y), 18);
    ASSERT_EQ(x, -5);
    ASSERT_EQ(y, 9);
}

TEST(Detail, Inverse)
{
    int128_t a = 4, b = 101;
    ASSERT_EQ(detail::InvMod(a, b) * a % b, 1);
}

TEST(ELLIPTIC_CURVE, rho_pollard_CPU)
{
    int p = 751;
    int a = -1;
    int b = 1;

    EllipticCurve<int> ec(a, b, p);
    EllipticCurve<int>::Point P(0, 1, ec);

    for(int i = 1; i < 31; i++)
    {
        ASSERT_EQ(cpu::rho_pollard<int>(i * P, P, 31), i);
        std::cout << "success #" << i << std::endl;
    }
}

TEST(ELLIPTIC_CURVE, rho_pollard_GPU)
{
    int p = 751;
    int a = -1;
    int b = 1;

    EllipticCurve<int> ec(a, b, p);
    EllipticCurve<int>::Point P(0, 1, ec);

    for(int i = 1; i < 31; i++)
    {
        ASSERT_EQ(gpu::rho_pollard<int>(i * P, P, 31), i);
        std::cout << "success #" << i << std::endl;
    }
}

TEST(ELLIPTIC_CURVE, rho_pollard_GPU)
{
    int128_t p = 751;
    int128_t a = -1;
    int128_t b = 1;

    EllipticCurve<int128_t> ec(a, b, p);
    EllipticCurve<int128_t>::Point P(0, 1, ec, 31);

    for(int i = 1; i < 31; i++)
    {
        ASSERT_EQ(gpu::rho_pollard<int128_t>(i * P, P), i);
        std::cout << "success #" << i << std::endl;
    }
}

TEST(ELLIPTIC_CURVE, sec112r1)
{
    using namespace boost::multiprecision;

    checked_int128_t p("0xdb7c2abf62e35e668076bead208b");

    checked_int128_t a("0xdb7c2abf62e35e668076bead2088");
    checked_int128_t b("0x659ef8ba043916eede8911702b22");

    EllipticCurve<checked_int128_t> ec(a, b, p);
    EllipticCurve<checked_int128_t>::Point g(checked_int128_t("0x9487239995a5ee76b55f9c2f098"), checked_int128_t("0xa89ce5af8724c0a23e0e0ff77500"),
        ec, checked_int128_t("0xDB7C2ABF62E35E7628DFAC6561C5"));
    EllipticCurve<checked_int128_t>::Point h(checked_int128_t("0x45cf81634b4ca4c6aac505843b94"), checked_int128_t("0xbda8eea7a5004255fa03c48d4ae8"), ec);

    checked_int128_t m("0xf6893de509504e9be7e85b7ae3b");
    ASSERT_EQ(m * g, h);
}

TEST(ELLIPTIC_CURVE, secp256k1)
{
    using namespace boost::multiprecision;

    checked_int512_t p("0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F");

    checked_int512_t a("0x0000000000000000000000000000000000000000000000000000000000000000");
    checked_int512_t b("0x0000000000000000000000000000000000000000000000000000000000000007");

    EllipticCurve<checked_int512_t> ec(a, b, p);
    EllipticCurve<checked_int512_t>::Point g(checked_int512_t("0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798"), 
        checked_int512_t("0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8"), ec,
        checked_int512_t("0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141"));
    EllipticCurve<checked_int512_t>::Point h(checked_int512_t("0x34F9460F0E4F08393D192B3C5133A6BA099AA0AD9FD54EBCCFACDFA239FF49C6"), 
        checked_int512_t("0x0B71EA9BD730FD8923F6D25A7A91E7DD7728A960686CB5A901BB419E0F2CA232"), ec);

    checked_int512_t m("0xAA5E28D6A97A2479A65527F7290311A3624D4CC0FA1578598EE3C2613BF99522");
    ASSERT_EQ(m * g, h);
}

TEST(ELLIPTIC_CURVE, P256)
{
    using namespace boost::multiprecision;

    checked_int512_t p("0xffffffff00000001000000000000000000000000ffffffffffffffffffffffff");

    checked_int512_t a = -3;
    checked_int512_t b("0x5ac635d8aa3a93e7b3ebbd55769886bc651d06b0cc53b0f63bce3c3e27d2604b");

    EllipticCurve<checked_int512_t> ec(a, b, p);
    EllipticCurve<checked_int512_t>::Point S(checked_int512_t("0xde2444bebc8d36e682edd27e0f271508617519b3221a8fa0b77cab3989da97c9"),
        checked_int512_t("0xc093ae7ff36e5380fc01a5aad1e66659702de80f53cec576b6350b243042a256"), ec);

    checked_int512_t d("0xc51e4753afdec1e6b6c6a5b992f43f8dd0c7a8933072708b6522468b2ffb06fd");

    ASSERT_EQ(d * S, EllipticCurve<checked_int512_t>::Point(checked_int512_t("0x51d08d5f2d4278882946d88d83c97d11e62becc3cfc18bedacc89ba34eeca03f"),
        checked_int512_t("0x75ee68eb8bf626aa5b673ab51f6e744e06f8fcf8a6c0cf3035beca956a7b41d5"), ec));
}

int main(int argc, char ** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
