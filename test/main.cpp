#include <stdio.h>
#include <gtest/gtest.h>
#include <iostream>
#include <cmath>
#include <uint256.h>
#include <elliptic.h>
#include <random>
#include <boost/multiprecision/cpp_int.hpp>
#include <boost/random.hpp>

#define POLLARD_SET_COUNT 16

// TODO: Proper partition should be implemented
template<typename T>
void iteration( typename EllipticCurve<T>::Point &r,
                T &c,
                T &d,
                const std::vector<T> &a,
                const std::vector<T> &b,
                const std::vector<typename EllipticCurve<T>::Point> &R,
                const T& order )
{
    T index = r.x().i() & T(POLLARD_SET_COUNT - 1);
    r += R[index];
    c = (c + a[index]) % order;
    d = (d + b[index]) % order;
}

// Solve Q = xP
template<typename T>
T rho_pollard( const typename EllipticCurve<T>::Point &Q, const typename EllipticCurve<T>::Point &P )
{
    T order = P.Order();

    boost::random::mt19937 gen;
    boost::random::uniform_int_distribution<T> distr(1, order);

    T c1, c2, d1, d2;

    while( true )
    {
        std::vector<T> a(POLLARD_SET_COUNT);
        std::vector<T> b(POLLARD_SET_COUNT);
        std::vector<typename EllipticCurve<T>::Point> R(POLLARD_SET_COUNT, P);
        for(unsigned i = 0; i < POLLARD_SET_COUNT; i++)
        {
            a[i] = distr(gen);
            b[i] = distr(gen);
            R[i] = a[i] * P + b[i] * Q;
        }

        c1 = c2 = distr(gen);
        d1 = d2 = distr(gen);
        typename EllipticCurve<T>::Point X1 = c1 * P + d1 * Q;
        typename EllipticCurve<T>::Point X2 = X1;
        
        do
        {
            iteration(X1, c1, d1, a, b, R, order);
            iteration(X2, c2, d2, a, b, R, order);
            iteration(X2, c2, d2, a, b, R, order);
        }
        while(X1 != X2 && X1.check() && X2.check());

        if(!X1.check() || !X2.check())
            continue;

        if((c1 * P + d1 * Q == X1) && (c2 * P + d2 * Q == X2))
        {
            T c = c1 - c2; if(c < 0) c += order;
            T d = d2 - d1; if(d < 0) d += order;
            if(d == 0)
                continue;
            return (c * detail::InvMod(d, order)) % order;
        }
    }
}

TEST(Detail, EGCD)
{
    int a = 612, b = 342, x = 0, y = 0;
    ASSERT_EQ(detail::EGCD(a, b, x, y), 18);
    ASSERT_EQ(x, -5);
    ASSERT_EQ(y, 9);
}

TEST(Detail, Inverse)
{
    ASSERT_EQ(detail::InvMod(4, 101) * 4 % 101, 1);
}

TEST(ELLIPTIC_CURVE, Simple)
{
    int p = 751;

    int a = -1;
    int b = 1;

    EllipticCurve<int> ec(a, b, p);
    EllipticCurve<int>::Point P(0, 1, ec, 31);
    EllipticCurve<int>::Point P2(188, 93, ec);
    EllipticCurve<int>::Point P3(56, 419, ec);

    ASSERT_EQ(1 * P, P);
    ASSERT_EQ(2 * P, P2);
    ASSERT_EQ(3 * P, P3);
    ASSERT_EQ(rho_pollard<int>(P, P), 1);
    ASSERT_EQ(rho_pollard<int>(P2, P), 2);
    ASSERT_EQ(rho_pollard<int>(P3, P), 3);
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

TEST(ELLIPTIC_CURVE, sec112r1)
{
    using namespace boost::multiprecision;

    checked_int128_t p("0xdb7c2abf62e35e668076bead208b");

    checked_int128_t a("0xdb7c2abf62e35e668076bead2088");
    checked_int128_t b("0x659ef8ba043916eede8911702b22");

    EllipticCurve<checked_int128_t> ec(a, b, p);
    EllipticCurve<checked_int128_t>::Point g(checked_int128_t("0x9487239995a5ee76b55f9c2f098"), checked_int128_t("0xa89ce5af8724c0a23e0e0ff77500"),
        ec, /* order */checked_int128_t("0xDB7C2ABF62E35E7628DFAC6561C5"));
    EllipticCurve<checked_int128_t>::Point h(checked_int128_t("0x45cf81634b4ca4c6aac505843b94"), checked_int128_t("0xbda8eea7a5004255fa03c48d4ae8"), ec);

    checked_int128_t m("0xf6893de509504e9be7e85b7ae3b");
    ASSERT_EQ(m * g, h);
    //ASSERT_EQ(rho_pollard<checked_int128_t>(m * g, h), m);
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

int main(int argc, char ** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}