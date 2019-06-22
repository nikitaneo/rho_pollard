#include <gtest/gtest.h>
#include <iostream>
#include <pollard.h>
#include <arith.h>

class bits10_curve : public ::testing::Test
{
public:
    EllipticCurve<base_uint<2>> ec{ 0, 7, 967 };
    Point<base_uint<2>> g{ 47, 19 };
    base_uint<2> order{ 907 };
    base_uint<2> m{ 505 };
};

class bits35_curve : public ::testing::Test
{
public:
    EllipticCurve<base_uint<3>> ec{ base_uint<3>{"0x1fdc776b2"}, base_uint<3>{"0x1b92df928"}, base_uint<3>{"0x54f174ca1"} };
    Point<base_uint<3>> g{ base_uint<3>{"0x348eb7221"}, base_uint<3>{"0x49e2e6396"} };
    base_uint<3> order{"0x54f1be6b9"};
    base_uint<3> m{"0x2a78df35c"};
};

class bits45_curve : public ::testing::Test
{
public:
    EllipticCurve<base_uint<4>> ec{ base_uint<4>{"0x4d074790451"}, base_uint<4>{"0xd854b3cef28"}, base_uint<4>{"0x1b480a4ae579"} };
    Point<base_uint<4>> g{ base_uint<4>{"0xf2c89c6cbd5"}, base_uint<4>{"0x34cb9515c4e"} };
    base_uint<4> order{"0x1b480a938913"};
    base_uint<4> m{"0xda40549c489"};
};

class bits55_curve : public ::testing::Test
{
public:
    EllipticCurve<base_uint<4>> ec{ 0, base_uint<4>{"0x19785be46bfd52"}, base_uint<4>{"0x6151f5f993b467"} };
    Point<base_uint<4>> g{ base_uint<4>{"0x1e4d05e577096e"}, base_uint<4>{"0xfd69cab66a1c"} };
    base_uint<4> order{"0x6151f60b0f7219"};
    base_uint<4> m{"0x30a8fb0587b90c"};
};

class bits64_curve : public ::testing::Test
{
public:
    EllipticCurve<base_uint<4>> ec{ base_uint<4>{"0x3aa9d64204542061"}, base_uint<4>{"0x1e1f2ab759811962"}, base_uint<4>{"0x97c940edc00408ed"} };
    Point<base_uint<4>> g{ base_uint<4>{"0x1b0c41e43e0cf2e9"}, base_uint<4>{"0x782454f18de8203d"} };
    base_uint<4> order{"0x97c940ef3f44ee07"};
    base_uint<4> m{"0x4be4a0779fa27800"};
};

class bits70_curve : public ::testing::Test
{
public:
    EllipticCurve<base_uint<5>> ec{ base_uint<5>{"0x3f985150c872f59d6"}, base_uint<5>{"0xe7d05995e25f482ef"}, base_uint<5>{"0x258d5eb731dd705cf3"} };
    Point<base_uint<5>> g{ base_uint<5>{"0x1567b532e6008e4fd2"}, base_uint<5>{"0x4077305eb22f45144"} };
    base_uint<5> order{"0x258d5eb732fbdffddd"};
    base_uint<5> m{"0x12c6af5b997deffeee"};
};

class bits79_curve : public ::testing::Test
{
public:
    EllipticCurve<base_uint<5>> ec{ base_uint<5>{"0x4a2e38a8f66d7f4c385f"}, base_uint<5>{"0x2c0bb31c6becc03d68a7"}, base_uint<5>{"0x80000000000000000201"} };
    Point<base_uint<5>> g{ base_uint<5>{"0x30cb127b63e42792f10f"}, base_uint<5>{"0x547b2c88266bb04f713b"} };
    base_uint<5> order{"0x40000000004531a2562b"};
    base_uint<5> m{"0x3aa068a09f1ed21e2582"};
};

__global__ void test_arithmetics()
{
    base_uint<4> a = -5;
    base_uint<4> b = 10;
    assert(b >> 1 == 5);
    assert(b << 1 == 20);
    assert(b << 2 == 40);
    assert(a + b == 5);
    assert(-a == 5);
    assert(-2 * a == 10);
    assert(a * a == 25);
    assert(a < 0 == true);
    assert(a > 0 == false);
    assert(a == -5);
    assert(a != -6);
    assert(-2 * b == 4 * a);
    assert(b - a == 15);
    assert(b * a == -50);
    assert(a * 2 == -10);
    assert(a * (-1) == 5);
    assert(a < -2 == false);
    assert(b > a == true);
    assert(++b == 11);
    assert(++a == -4);
    assert(--a == -5);
    a += 1;
    assert(a == -4);
    a -= 1;
    assert(a == -5);
    assert(a - 1 == -6);
    assert(b - 1 == 10);
    assert(b / 2 == 5);
    assert(b / 2 * 2 == 10);
    assert(b % 2 == 1);
    assert(base_uint<4>(1) << 33 == 8589934592);
    assert(base_uint<4>(8589934592) >> 33 == 1);
}

TEST(base_uint, gpu_arithmetics)
{
    test_arithmetics<<<1, 1>>>();
}

TEST(base_uint, arithmetics)
{
    base_uint<4> a = -5;
    base_uint<4> b = 10;
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
    ASSERT_EQ(base_uint<4>(1) << 33, 8589934592);
    ASSERT_EQ(base_uint<4>(8589934592) >> 33, 1);

    base_uint<4> c("0xdb7c2abf62e35e668076bead2088");
    base_uint<4> d("0x659ef8ba043916eede8911702b22");
    ASSERT_EQ(c + d, base_uint<4>("0x1411b2379671c75555effd01d4baa"));
    ASSERT_EQ(c - d, base_uint<4>("0x75dd32055eaa4777a1edad3cf566"));
    ASSERT_EQ(c/2, base_uint<4>("0x6dbe155fb171af33403b5f569044"));

    c = base_uint<4>("0x37276cf767b9e78402ecbaed");
    d = base_uint<4>("0x176aeb86b56fcbe861482259");
    ASSERT_EQ(c / d, 2);
}

TEST(detail, inverse)
{
    base_uint<4> a = 4, b = 101;
    ASSERT_EQ(a.inv_modp(b) * a % b, 1);
}

TEST(detail, point_addition)
{
    base_uint<2> p = 967;
    base_uint<2> a = 0;
    base_uint<2> b = 7;

    EllipticCurve<base_uint<2>> ec(a, b, p);
    Point<base_uint<2>> P(47, 19);
    base_uint<2> order = 907;

    Point<base_uint<2>> O(0, 0);
    ASSERT_EQ(ec.plus(P, O), P);

    Point<base_uint<2>> P2 = ec.plus(P, P);
    ASSERT_EQ(P2, Point<base_uint<2>>(895, 656));

    ASSERT_EQ(ec.plus(P, P2), Point<base_uint<2>>(57, 774));
}

TEST(detail, teske)
{
    base_uint<4> p = 967;
    base_uint<4> a = 0;
    base_uint<4> b = 7;

    EllipticCurve<base_uint<4>> ec(a, b, p);
    Point<base_uint<4>> P(47, 19);
    base_uint<4> order = 907;

    Point<base_uint<4>> O(0, 0);
    ASSERT_EQ(ec.mul(order, P), O);

    Point<base_uint<4>> Q = ec.mul(3, P);
    ASSERT_EQ(ec.mul(order, Q), O);

    base_uint<4> c; c.random(order);
    base_uint<4> d = 0;
    Point<base_uint<4>> T =  ec.plus(ec.mul(c, P), ec.mul(d, Q));

    for(unsigned i = 0; i < 100; i++)
    {  
        base_uint<4> a; a = a.random(order);
        base_uint<4> b; b = b.random(order);
        Point<base_uint<4>> R = ec.plus(ec.mul(a, P), ec.mul(b, Q));
    
        auto T1 = ec.plus(T, R);
        base_uint<4> c1 = c + a; // if(c1 > order) c1 -= order;
        base_uint<4> d1 = d + b; // if(d1 > order) d1 -= order;
        auto P1 = ec.mul(c1, P); auto P2 = ec.mul(d1, Q);
        auto tmp = ec.plus(P1, P2);
        ASSERT_EQ(tmp, T1);
        ASSERT_EQ(ec.check(tmp), true);
    }
}

TEST_F(bits10_curve, gpu)
{
    auto res = gpu::rho_pollard(ec.mul(m, g), g, order, ec);
    ASSERT_EQ(std::get<0>( res ), m);
}

TEST_F(bits10_curve, cpu)
{
    auto res = cpu::rho_pollard(ec.mul(m, g), g, order, ec);
    ASSERT_EQ(std::get<0>( res ), m);
}

TEST_F(bits35_curve, gpu)
{
    auto result = gpu::rho_pollard(ec.mul(m, g), g, order, ec);
    ASSERT_EQ(std::get<0>( result ), m);

    std::cout << "[bits35_gpu] Average number of rho-pollard iterations per second: " << std::get<1>( result ) << "." << std::endl;
    std::cout << "[bits35_gpu] Preparation time: " << std::get<2>( result ) << " ms." << std::endl;
    std::cout << "[bits35_gpu] Calculation time: " << std::get<3>( result ) << " ms." << std::endl;
    std::cout << "[bits35_gpu] Total time: " << std::get<2>( result ) + std::get<3>( result ) << " ms." << std::endl;
}

TEST_F(bits45_curve, gpu)
{
    auto result = gpu::rho_pollard(ec.mul(m, g), g, order, ec);
    ASSERT_EQ(std::get<0>( result ), m);

    std::cout << "[bits45_gpu] Average number of rho-pollard iterations per second: " << std::get<1>( result ) << "." << std::endl;
    std::cout << "[bits45_gpu] Preparation time: " << std::get<2>( result ) << " ms." << std::endl;
    std::cout << "[bits45_gpu] Calculation time: " << std::get<3>( result ) << " ms." << std::endl;
    std::cout << "[bits45_gpu] Total time: " << std::get<2>( result ) + std::get<3>( result ) << " ms." << std::endl;
}

TEST_F(bits55_curve, gpu)
{
    auto result = gpu::rho_pollard(ec.mul(m, g), g, order, ec);
    ASSERT_EQ(std::get<0>( result ), m);

    std::cout << "[bits55_gpu] Average number of rho-pollard iterations per second: " << std::get<1>( result ) << "." << std::endl;
    std::cout << "[bits55_gpu] Preparation time: " << std::get<2>( result ) << " ms." << std::endl;
    std::cout << "[bits55_gpu] Calculation time: " << std::get<3>( result ) << " ms." << std::endl;
    std::cout << "[bits55_gpu] Total time: " << std::get<2>( result ) + std::get<3>( result ) << " ms." << std::endl;
}

TEST_F(bits35_curve, cpu)
{
    auto result = cpu::rho_pollard(ec.mul(m, g), g, order, ec);
    ASSERT_EQ(std::get<0>( result ), m);

    std::cout << "[bits35_cpu] Average number of rho-pollard iterations per second: " << std::get<1>( result ) << "." << std::endl;
    std::cout << "[bits35_cpu] Preparation time: " << std::get<2>( result ) << " ms." << std::endl;
    std::cout << "[bits35_cpu] Calculation time: " << std::get<3>( result ) << " ms." << std::endl;
    std::cout << "[bits35_cpu] Total time: " << std::get<2>( result ) + std::get<3>( result ) << " ms." << std::endl;
}

TEST_F(bits64_curve, gpu)
{
    auto result = gpu::rho_pollard(ec.mul(m, g), g, order, ec);
    ASSERT_EQ(std::get<0>( result ), m);

    std::cout << "[bits64_gpu] Average number of rho-pollard iterations per second: " << std::get<1>( result ) << "." << std::endl;
    std::cout << "[bits64_gpu] Preparation time: " << std::get<2>( result ) << " ms." << std::endl;
    std::cout << "[bits64_gpu] Calculation time: " << std::get<3>( result ) << " ms." << std::endl;
    std::cout << "[bits64_gpu] Total time: " << std::get<2>( result ) + std::get<3>( result ) << " ms." << std::endl;
}

TEST_F(bits70_curve, gpu)
{
    auto result = gpu::rho_pollard(ec.mul(m, g), g, order, ec);
    ASSERT_EQ(std::get<0>( result ), m);

    std::cout << "[bits70_gpu] Average number of rho-pollard iterations per second: " << std::get<1>( result ) << "." << std::endl;
    std::cout << "[bits70_gpu] Preparation time: " << std::get<2>( result ) << " ms." << std::endl;
    std::cout << "[bits70_gpu] Calculation time: " << std::get<3>( result ) << " ms." << std::endl;
    std::cout << "[bits70_gpu] Total time: " << std::get<2>( result ) + std::get<3>( result ) << " ms." << std::endl;
}

TEST_F(bits45_curve, cpu)
{
    auto result = cpu::rho_pollard(ec.mul(m, g), g, order, ec);
    ASSERT_EQ(std::get<0>( result ), m);

    std::cout << "[bits45_cpu] Average number of rho-pollard iterations per second: " << std::get<1>( result ) << "." << std::endl;
    std::cout << "[bits45_cpu] Preparation time: " << std::get<2>( result ) << " ms." << std::endl;
    std::cout << "[bits45_cpu] Calculation time: " << std::get<3>( result ) << " ms." << std::endl;
    std::cout << "[bits45_cpu] Total time: " << std::get<2>( result ) + std::get<3>( result ) << " ms." << std::endl;
}

TEST_F(bits55_curve, cpu)
{
    auto result = cpu::rho_pollard(ec.mul(m, g), g, order, ec);
    ASSERT_EQ(std::get<0>( result ), m);

    std::cout << "[bits55_cpu] Average number of rho-pollard iterations per second: " << std::get<1>( result ) << "." << std::endl;
    std::cout << "[bits55_cpu] Preparation time: " << std::get<2>( result ) << " ms." << std::endl;
    std::cout << "[bits55_cpu] Calculation time: " << std::get<3>( result ) << " ms." << std::endl;
    std::cout << "[bits55_cpu] Total time: " << std::get<2>( result ) + std::get<3>( result ) << " ms." << std::endl;
}

TEST_F(bits64_curve, cpu)
{
    auto result = cpu::rho_pollard(ec.mul(m, g), g, order, ec);
    ASSERT_EQ(std::get<0>( result ), m);

    std::cout << "[bits64_cpu] Average number of rho-pollard iterations per second: " << std::get<1>( result ) << "." << std::endl;
    std::cout << "[bits64_cpu] Preparation time: " << std::get<2>( result ) << " ms." << std::endl;
    std::cout << "[bits64_cpu] Calculation time: " << std::get<3>( result ) << " ms." << std::endl;
    std::cout << "[bits64_cpu] Total time: " << std::get<2>( result ) + std::get<3>( result ) << " ms." << std::endl;
}

TEST_F(bits70_curve, cpu)
{
    auto result = cpu::rho_pollard(ec.mul(m, g), g, order, ec);
    ASSERT_EQ(std::get<0>( result ), m);

    std::cout << "[bits70_cpu] Average number of rho-pollard iterations per second: " << std::get<1>( result ) << "." << std::endl;
    std::cout << "[bits70_cpu] Preparation time: " << std::get<2>( result ) << " ms." << std::endl;
    std::cout << "[bits70_cpu] Calculation time: " << std::get<3>( result ) << " ms." << std::endl;
    std::cout << "[bits70_cpu] Total time: " << std::get<2>( result ) + std::get<3>( result ) << " ms." << std::endl;
}

TEST_F(bits79_curve, cpu)
{
    auto result = cpu::rho_pollard(ec.mul(m, g), g, order, ec);
    ASSERT_EQ(std::get<0>( result ), m);

    std::cout << "[bits64_cpu] Average number of rho-pollard iterations per second: " << std::get<1>( result ) << "." << std::endl;
    std::cout << "[bits64_cpu] Preparation time: " << std::get<2>( result ) << " ms." << std::endl;
    std::cout << "[bits64_cpu] Calculation time: " << std::get<3>( result ) << " ms." << std::endl;
    std::cout << "[bits64_cpu] Total time: " << std::get<2>( result ) + std::get<3>( result ) << " ms." << std::endl;
}

TEST_F(bits79_curve, gpu)
{
    auto result = cpu::rho_pollard(ec.mul(m, g), g, order, ec);
    ASSERT_EQ(std::get<0>( result ), m);

    std::cout << "[bits64_cpu] Average number of rho-pollard iterations per second: " << std::get<1>( result ) << "." << std::endl;
    std::cout << "[bits64_cpu] Preparation time: " << std::get<2>( result ) << " ms." << std::endl;
    std::cout << "[bits64_cpu] Calculation time: " << std::get<3>( result ) << " ms." << std::endl;
    std::cout << "[bits64_cpu] Total time: " << std::get<2>( result ) + std::get<3>( result ) << " ms." << std::endl;
}

int main(int argc, char ** argv)
{
    checkCudaErrors(cudaSetDeviceFlags(cudaDeviceMapHost));

    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
