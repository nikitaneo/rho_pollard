#include <stdio.h>
#include <gtest/gtest.h>
#include <iostream>
#include <cmath>
#include <uint256.h>
#include <elliptic.h>
#include <boost/multiprecision/cpp_int.hpp>
 
void new_xab( int& x, int& a, int& b, int alpha, int beta, int p )
{
    switch( x % 3 )
    {
        case 0: x = x*x     % p;  a =  a*2  % (p - 1);  b =  b*2  % (p - 1);  break;
        case 1: x = x*alpha % p;  a = (a+1) % (p - 1);                        break;
        case 2: x = x*beta  % p;                        b = (b+1) % (p - 1);  break;
    }
}

template<typename T>
T rho_pollard( const T &alpha,  const T &beta, const T &p )
{
    T x=1, a=0, b=0;
    T X=x, A=a, B=b;
    for( T i = 1; i < p - 1; ++i )
    {
        new_xab( x, a, b, alpha, beta, p );
        new_xab( X, A, B, alpha, beta, p );
        new_xab( X, A, B, alpha, beta, p );
        if( x == X )
            return std::fabs(A - a)/std::fabs(B - b);
    }
}

TEST(RHO_POLLARD_CPU, Simple)
{
    int p = 1019;  /* N = 1019 -- prime     */
    int alpha = 2; /* generator             */
    int beta = 5;  /* 2^{10} = 1024 = 5 (N) */
    ASSERT_EQ(rho_pollard( alpha, beta, p ), 10);
}

TEST(ELLIPTIC_CURVE, P256_arithm)
{
    using namespace boost::multiprecision;

    int1024_t p("0xffffffff00000001000000000000000000000000ffffffffffffffffffffffff");

    int1024_t a = -3;
    int1024_t b("0x5ac635d8aa3a93e7b3ebbd55769886bc651d06b0cc53b0f63bce3c3e27d2604b");

    EllipticCurve<int1024_t> ec(a, b, p);
    EllipticCurve<int1024_t>::Point S(int1024_t("0xde2444bebc8d36e682edd27e0f271508617519b3221a8fa0b77cab3989da97c9"),
        int1024_t("0xc093ae7ff36e5380fc01a5aad1e66659702de80f53cec576b6350b243042a256"), ec);

    int1024_t d("0xc51e4753afdec1e6b6c6a5b992f43f8dd0c7a8933072708b6522468b2ffb06fd");

    ASSERT_EQ(d * S, EllipticCurve<int1024_t>::Point(int1024_t("0x51d08d5f2d4278882946d88d83c97d11e62becc3cfc18bedacc89ba34eeca03f"),
        int1024_t("0x75ee68eb8bf626aa5b673ab51f6e744e06f8fcf8a6c0cf3035beca956a7b41d5"), ec));
}

TEST(ELLIPTIC_CURVE, sec112r1)
{
    using namespace boost::multiprecision;

    int256_t p("0xdb7c2abf62e35e668076bead208b");

    int256_t a("0xdb7c2abf62e35e668076bead2088");
    int256_t b("0x659ef8ba043916eede8911702b22");

    EllipticCurve<int256_t> ec(a, b, p);
    EllipticCurve<int256_t>::Point g(int256_t("0x9487239995a5ee76b55f9c2f098"), int256_t("0xa89ce5af8724c0a23e0e0ff77500"), ec);
    EllipticCurve<int256_t>::Point h(int256_t("0x45cf81634b4ca4c6aac505843b94"), int256_t("0xbda8eea7a5004255fa03c48d4ae8"), ec);

    int256_t m("0xf6893de509504e9be7e85b7ae3b");
    ASSERT_EQ(m * g, h);
}

TEST(ELLIPTIC_CURVE, secp256k1)
{
    using namespace boost::multiprecision;

    int1024_t p("0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F");

    int1024_t a("0x0000000000000000000000000000000000000000000000000000000000000000");
    int1024_t b("0x0000000000000000000000000000000000000000000000000000000000000007");

    EllipticCurve<int1024_t> ec(a, b, p);
    EllipticCurve<int1024_t>::Point g(int1024_t("0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798"), 
        int1024_t("0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8"), ec);
    EllipticCurve<int1024_t>::Point h(int1024_t("0x34F9460F0E4F08393D192B3C5133A6BA099AA0AD9FD54EBCCFACDFA239FF49C6"), 
        int1024_t("0x0B71EA9BD730FD8923F6D25A7A91E7DD7728A960686CB5A901BB419E0F2CA232"), ec);

    int1024_t m("0xAA5E28D6A97A2479A65527F7290311A3624D4CC0FA1578598EE3C2613BF99522");
    ASSERT_EQ(m * g, h);
}

int main(int argc, char ** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}