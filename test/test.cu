#include <gtest/gtest.h>
#include <iostream>
#include <int128_t.h>
#include <pollard.h>

TEST(INT128, Simple)
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
    //ASSERT_EQ(a % 3, -2);

    int128_t c("0xdb7c2abf62e35e668076bead2088");
    int128_t d("0x659ef8ba043916eede8911702b22");
    ASSERT_EQ(c + d, int128_t("0x1411b2379671c75555effd01d4baa"));
    ASSERT_EQ(c - d, int128_t("0x75dd32055eaa4777a1edad3cf566"));
    ASSERT_EQ(c/2, int128_t("0x6dbe155fb171af33403b5f569044"));
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
    ASSERT_EQ(detail::invmod(a, b) * a % b, 1);
}

TEST(ELLIPTIC_CURVE, teske)
{
    int128_t p = 967;
    int128_t a = 0;
    int128_t b = 7;

    EllipticCurve<int128_t> ec(a, b, p);
    Point<int128_t> P(47, 19);
    int128_t order = 907;

    Point<int128_t> O(0, 0);
    assert(ec.mul(order, P) == O);

    Point<int128_t> Q = ec.mul(3, P);
    assert(ec.mul(order, Q) == O);

    int128_t c; c.random(order);
    int128_t d = 0;
    Point<int128_t> T =  ec.add(ec.mul(c, P), ec.mul(d, Q));

    for(unsigned i = 0; i < 100; i++)
    {  
        int128_t a; a = a.random(order);
        int128_t b; b = b.random(order);
        Point<int128_t> R = ec.add(ec.mul(a, P), ec.mul(b, Q));
    
        auto T1 = ec.add(T, R);
        int128_t c1 = c + a; // if(c1 > order) c1 -= order;
        int128_t d1 = d + b; // if(d1 > order) d1 -= order;
        auto P1 = ec.mul(c1, P); auto P2 = ec.mul(d1, Q);
        auto tmp = ec.add(P1, P2);
        assert( tmp == T1 );
        assert( ec.check(tmp) );
    }
}

TEST(ELLIPTIC_CURVE, rho_pollard_cpu)
{
    int128_t p = 967;
    int128_t a = 0;
    int128_t b = 7;

    EllipticCurve<int128_t> ec(a, b, p);
    Point<int128_t> P(47, 19);
    int128_t order = 907;

    for(int i = 1; i < order; i++)
    {
        ASSERT_EQ(cpu::rho_pollard<int128_t>(ec.mul(i, P), P, order, ec), i);
        // std::cout << "CPU success #" << i << std::endl;
    }
}

TEST(ELLIPTIC_CURVE, rho_pollard_gpu)
{
    int128_t p = 967;
    int128_t a = 0;
    int128_t b = 7;

    EllipticCurve<int128_t> ec(a, b, p);
    Point<int128_t> P(47, 19);
    int128_t order = 907;

    for(int i = 1; i < order; i++)
    {
        ASSERT_EQ(gpu::rho_pollard<int128_t>(ec.mul(i, P), P, order, ec), i);
        // std::cout << "GPU success #" << i << std::endl;
    }
}

/*
TEST(ELLIPTIC_CURVE, sec112r1)
{
    int128_t p("0xdb7c2abf62e35e668076bead208b");

    int128_t a("0xdb7c2abf62e35e668076bead2088");
    int128_t b("0x659ef8ba043916eede8911702b22");

    EllipticCurve<int128_t> ec(a, b, p);
    Point<int128_t> g(int128_t("0x9487239995a5ee76b55f9c2f098"), int128_t("0xa89ce5af8724c0a23e0e0ff77500"));
    int128_t g_order("0xDB7C2ABF62E35E7628DFAC6561C5");
    Point<int128_t> h(int128_t("0x45cf81634b4ca4c6aac505843b94"), int128_t("0xbda8eea7a5004255fa03c48d4ae8"));

    int128_t m("0xf6893de509504e9be7e85b7ae3b");
    ASSERT_EQ(cpu::rho_pollard<int128_t>(h, g, g_order, ec), m);
}
*/
int main(int argc, char ** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}