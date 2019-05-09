#include <gtest/gtest.h>
#include <iostream>
#include <int128_t.h>
#include <pollard.h>

class bits10_curve : public ::testing::Test
{
public:
    EllipticCurve<int128_t> ec{ 0, 7, 967 };
    Point<int128_t> g{ 47, 19 };
    int128_t order{ 907 };
    int128_t m{ 505 };
};

class bits35_curve : public ::testing::Test
{
public:
    EllipticCurve<int128_t> ec{ int128_t{"0x1fdc776b2"}, int128_t{"0x1b92df928"}, int128_t{"0x54f174ca1"} };
    Point<int128_t> g{ int128_t{"0x348eb7221"}, int128_t{"0x49e2e6396"} };
    int128_t order{"0x54f1be6b9"};
    int128_t m{"0x2a78df35c"};
};

class bits45_curve : public ::testing::Test
{
public:
    EllipticCurve<int128_t> ec{ int128_t{"0x4d074790451"}, int128_t{"0xd854b3cef28"}, int128_t{"0x1b480a4ae579"} };
    Point<int128_t> g{ int128_t{"0xf2c89c6cbd5"}, int128_t{"0x34cb9515c4e"} };
    int128_t order{"0x1b480a938913"};
    int128_t m{"0xda40549c489"};
};

class bits55_curve : public ::testing::Test
{
public:
    EllipticCurve<int128_t> ec{ 0, int128_t{"0x19785be46bfd52"}, int128_t{"0x6151f5f993b467"} };
    Point<int128_t> g{ int128_t{"0x1e4d05e577096e"}, int128_t{"0xfd69cab66a1c"} };
    int128_t order{"0x6151f60b0f7219"};
    int128_t m{"0x30a8fb0587b90c"};
};

class bits65_curve : public ::testing::Test
{
public:
    EllipticCurve<int128_t> ec{ int128_t{"0x3aa9d64204542061"}, int128_t{"0x1e1f2ab759811962"}, int128_t{"0x97c940edc00408ed"} };
    Point<int128_t> g{ int128_t{"0x1b0c41e43e0cf2e9"}, int128_t{"0x782454f18de8203d"} };
    int128_t order{"0x97c940ef3f44ee07"};
    int128_t m{"0x4be4a0779fa27800"};
};

TEST(int128_t, arithmetics)
{
    int128_t a = -5;
    int128_t b = 10;
    ASSERT_EQ(b >> 1, 5);
    ASSERT_EQ(b << 1, 20);
    ASSERT_EQ(b << 2, 40);
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
    ASSERT_EQ(int128_t(1) << 33, 8589934592);
    ASSERT_EQ(int128_t(8589934592) >> 33, 1);

    int128_t c("0xdb7c2abf62e35e668076bead2088");
    int128_t d("0x659ef8ba043916eede8911702b22");
    ASSERT_EQ(c + d, int128_t("0x1411b2379671c75555effd01d4baa"));
    ASSERT_EQ(c - d, int128_t("0x75dd32055eaa4777a1edad3cf566"));
    ASSERT_EQ(c/2, int128_t("0x6dbe155fb171af33403b5f569044"));

    c = int128_t("0x37276cf767b9e78402ecbaed");
    d = int128_t("0x176aeb86b56fcbe861482259");
    ASSERT_EQ(c / d, 2);
}

TEST(detail, inverse)
{
    int128_t a = 4, b = 101;
    ASSERT_EQ(detail::invmod(a, b) * a % b, 1);
}

TEST(detail, point_addition)
{
    int128_t p = 967;
    int128_t a = 0;
    int128_t b = 7;

    EllipticCurve<int128_t> ec(a, b, p);
    Point<int128_t> P(47, 19);
    int128_t order = 907;

    Point<int128_t> O(0, 0);
    ASSERT_EQ(ec.add(P, O), P);

    Point<int128_t> P2 = ec.add(P, P);
    ASSERT_EQ(P2, Point<int128_t>(895, 656));

    ASSERT_EQ(ec.add(P, P2), Point<int128_t>(57, 774));
}

TEST(detail, teske)
{
    int128_t p = 967;
    int128_t a = 0;
    int128_t b = 7;

    EllipticCurve<int128_t> ec(a, b, p);
    Point<int128_t> P(47, 19);
    int128_t order = 907;

    Point<int128_t> O(0, 0);
    ASSERT_EQ(ec.mul(order, P), O);

    Point<int128_t> Q = ec.mul(3, P);
    ASSERT_EQ(ec.mul(order, Q), O);

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
        ASSERT_EQ(tmp, T1);
        ASSERT_EQ(ec.check(tmp), true);
    }
}

TEST_F(bits10_curve, cpu)
{
    auto res = cpu::rho_pollard<int128_t>(ec.mul(m, g), g, order, ec);
    ASSERT_EQ(std::get<0>( res ), m);
}

TEST_F(bits10_curve, gpu)
{
    auto res = gpu::rho_pollard<int128_t>(ec.mul(m, g), g, order, ec);
    ASSERT_EQ(std::get<0>( res ), m);
}

TEST_F(bits35_curve, cpu)
{
    auto result = cpu::rho_pollard<int128_t>(ec.mul(m, g), g, order, ec);
    ASSERT_EQ(std::get<0>( result ), m);

    std::cout << "[bits35_cpu] Average number of rho-pollard iterations per second: " << std::get<1>( result )
        << ". Calculation time: " << std::get<2>( result ) << " sec." << std::endl;
}

TEST_F(bits35_curve, gpu)
{
    auto result = gpu::rho_pollard<int128_t>(ec.mul(m, g), g, order, ec);
    ASSERT_EQ(std::get<0>( result ), m);

    std::cout << "[bits35_gpu] Average number of rho-pollard iterations per second: " << std::get<1>( result ) 
        << ". Calculation time: " << std::get<2>( result ) << " sec." << std::endl;
}

TEST_F(bits45_curve, cpu)
{
    auto result = cpu::rho_pollard<int128_t>(ec.mul(m, g), g, order, ec);
    ASSERT_EQ(std::get<0>( result ), m);

    std::cout << "[bits45_cpu] Average number of rho-pollard iterations per second: " << std::get<1>( result ) 
        << ". Calculation time: " << std::get<2>( result ) << " sec." << std::endl;
}

TEST_F(bits45_curve, gpu)
{
    auto result = gpu::rho_pollard<int128_t>(ec.mul(m, g), g, order, ec);
    ASSERT_EQ(std::get<0>( result ), m);

    std::cout << "[bits45_gpu] Average number of rho-pollard iterations per second: " << std::get<1>( result ) 
        << ". Calculation time: " << std::get<2>( result ) << " sec." << std::endl;
}

TEST_F(bits55_curve, cpu)
{
    auto result = cpu::rho_pollard<int128_t>(ec.mul(m, g), g, order, ec);
    ASSERT_EQ(std::get<0>( result ), m);

    std::cout << "[bits55_cpu] Average number of rho-pollard iterations per second: " << std::get<1>( result ) 
        << ". Calculation time: " << std::get<2>( result ) << " sec." << std::endl;
}

TEST_F(bits55_curve, gpu)
{
    auto result = gpu::rho_pollard<int128_t>(ec.mul(m, g), g, order, ec);
    ASSERT_EQ(std::get<0>( result ), m);

    std::cout << "[bits55_gpu] Average number of rho-pollard iterations per second: " << std::get<1>( result ) 
        << ". Calculation time: " << std::get<2>( result ) << " sec." << std::endl;
}

TEST_F(bits65_curve, cpu)
{
    auto result = cpu::rho_pollard<int128_t>(ec.mul(m, g), g, order, ec);
    ASSERT_EQ(std::get<0>( result ), m);

    std::cout << "[bits65_cpu] Average number of rho-pollard iterations per second: " << std::get<1>( result ) 
        << ". Calculation time: " << std::get<2>( result ) << " sec." << std::endl;
}

TEST_F(bits65_curve, gpu)
{
    auto result = gpu::rho_pollard<int128_t>(ec.mul(m, g), g, order, ec);
    ASSERT_EQ(std::get<0>( result ), m);

    std::cout << "[bits65_gpu] Average number of rho-pollard iterations per second: " << std::get<1>( result ) 
        << ". Calculation time: " << std::get<2>( result ) << " sec." << std::endl;
}

int main(int argc, char ** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}