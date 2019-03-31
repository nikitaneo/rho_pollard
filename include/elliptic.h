#include <iostream>

#ifndef ELLIPTIC_H
#define ELLIPTIC_H

#define CUDA_CALLABLE __host__ __device__

template<typename T>
class Point;

template<typename T>
class FiniteField;

// helper functions
namespace detail
{

template<typename T>
CUDA_CALLABLE T mulmod(const T &A, const T &B, const T &mod) 
{ 
    T res{ 0 };
    T a = A;
    T b = B; 
    while( b != 0 ) 
    { 
        if( (b & 1) == 1 ) 
            res += a; if(res > mod) res -= mod; 
  
        a <<= 1; if(a > mod) a -= mod; 
        b >>= 1; 
    } 
  
    // Return result 
    return res; 
}

template<typename T>
CUDA_CALLABLE T ToMont(const T &A, const T &mod)
{
    // for module 907, B = 2, n = 10
    return (A << 10) % mod;
} 

template<typename T>
CUDA_CALLABLE T FromMont(const T &A, const T &mod)
{
    // for module 907, B = 2, n = 10
    return (A >> 10) % mod;
} 

// A and B should be in Montgomery representation
template<typename T>
CUDA_CALLABLE T MontProd(const T &A, const T &B, const T &mod)
{
    return 0;
} 

template<typename T>
CUDA_CALLABLE T EGCD(const T &a, const T &b, T &u, T &v)
{
    u = 1;
    v = 0;
    T g = a;
    T u1 = 0;
    T v1 = 1;
    T g1 = b;
    while( g1 != 0 )
    {
        T q = g / g1; // Integer divide
        T t1 = u - q * u1;
        T t2 = v - q * v1;
        T t3 = g - q * g1;
        u = u1;
        v = v1;
        g = g1;
        u1 = t1;
        v1 = t2;
        g1 = t3;
    }

    return g;
}

// Solve linear congruence equation x * z == 1 (mod n) for z
template<typename T>
CUDA_CALLABLE T invmod(const T &x, const T &n)
{
    T X = x;
    T u, v, g, z;
    g = EGCD(X, n, u, v);
    if(g != 1)
    {
        z = 0;
    }
    else
    {
        z = u % n;
    }
    return z.isLessZero() ? z + n : z;
}
} // namespace detail

template <typename T>
class FiniteFieldElement
{
    T i_;
    T P;

    CUDA_CALLABLE void assign(const T &i)
    {
        if(i >= P)
            i_ = i % P;
        else
            i_ = i;

        if (i.isLessZero())
        {
            i_ += P;
        }
    }

  public:
    // ctor
    CUDA_CALLABLE FiniteFieldElement()
        : i_(0), P(0)
    {
    }
    // ctor
    CUDA_CALLABLE explicit FiniteFieldElement(const T &i, const T &p) : P(p)
    {
        assign(i);
    }
    // copy ctor
    CUDA_CALLABLE FiniteFieldElement(const FiniteFieldElement<T> &rhs)
        : i_(rhs.i_), P(rhs.P)
    {
    }

    // access "raw" integer
    CUDA_CALLABLE T i() const { return i_; }

    CUDA_CALLABLE T p() const { return P; }

    CUDA_CALLABLE operator T() { return i_; }

    // negate
    CUDA_CALLABLE FiniteFieldElement operator-() const
    {
        return FiniteFieldElement(-i_, P);
    }

    CUDA_CALLABLE FiniteFieldElement<T> &operator=(const FiniteFieldElement<T> &rhs)
    {
        i_ = rhs.i_;
        P = rhs.P;
        return *this;
    }

    CUDA_CALLABLE FiniteFieldElement<T> &operator*=(const FiniteFieldElement<T> &rhs)
    {
        i_ = detail::mulmod(i_, rhs.i_, P);
        return *this;
    }

    CUDA_CALLABLE friend bool operator==(const FiniteFieldElement<T> &lhs, const FiniteFieldElement<T> &rhs)
    {
        return (lhs.i_ == rhs.i_);
    }

    CUDA_CALLABLE friend bool operator!=(const FiniteFieldElement<T> &lhs, const FiniteFieldElement<T> &rhs)
    {
        return !(lhs.i_ == rhs.i_);
    }

    CUDA_CALLABLE friend bool operator==(const FiniteFieldElement<T> &lhs, const T &rhs)
    {
        return (lhs.i_ == rhs);
    }

    CUDA_CALLABLE friend bool operator!=(const FiniteFieldElement<T> &lhs, const T &rhs)
    {
        return (lhs.i_ != rhs);
    }

    CUDA_CALLABLE friend FiniteFieldElement<T> operator/(const FiniteFieldElement<T> &lhs, const FiniteFieldElement<T> &rhs)
    {
        return FiniteFieldElement<T>(detail::mulmod(lhs.i_, detail::invmod(rhs.i_, rhs.P), rhs.P), rhs.P);
    }
 
    CUDA_CALLABLE friend FiniteFieldElement<T> operator+(const FiniteFieldElement<T> &lhs, const FiniteFieldElement<T> &rhs)
    {
        return FiniteFieldElement<T>(lhs.i_ + rhs.i_, rhs.P);
    }

    CUDA_CALLABLE friend FiniteFieldElement<T> operator-(const FiniteFieldElement<T> &lhs, const FiniteFieldElement<T> &rhs)
    {
        return FiniteFieldElement<T>(lhs.i_ - rhs.i_, rhs.P);
    }

    CUDA_CALLABLE friend FiniteFieldElement<T> operator+(const FiniteFieldElement<T> &lhs, const T &i)
    {
        return FiniteFieldElement<T>(lhs.i_ + i, lhs.P);
    }

    CUDA_CALLABLE friend FiniteFieldElement<T> operator+(const T &i, const FiniteFieldElement<T> &rhs)
    {
        return FiniteFieldElement<T>(rhs.i_ + i, rhs.P);
    }

    CUDA_CALLABLE friend FiniteFieldElement<T> operator*(const T &n, const FiniteFieldElement<T> &rhs)
    {
        return FiniteFieldElement<T>(detail::mulmod(n, rhs.i_, rhs.P), rhs.P);
    }

    CUDA_CALLABLE friend FiniteFieldElement<T> operator*(const FiniteFieldElement<T> &lhs, const FiniteFieldElement<T> &rhs)
    {
        return FiniteFieldElement<T>(detail::mulmod(lhs.i_, rhs.i_, rhs.P), rhs.P);
    }
};

/*
    Elliptic Curve over a finite field of order P:
    y^2 mod P = x^3 + ax + b mod P
            
    NOTE: this implementation is simple and uses normal machine integers for its calculations.
    No special "big integer" manipulation is supported so anything _big_ won't work. 
    However, it is a complete finite field EC implementation in that it can be used 
    to learn and understand the behaviour of these curves and in particular to experiment with them 
    for their use in cryptography
                  
    Template parameter P is the order of the finite field Fp over which this curve is defined
*/

template <typename T>
class EllipticCurve
{
  public:
    // this curve is defined over the finite field (Galois field) Fp, this is the
    // typedef of elements in it
    typedef FiniteFieldElement<T> ffe_t;

    // Initialize EC as y^2 = x^3 + ax + b
    CUDA_CALLABLE EllipticCurve(const T &a, const T &b, const T &p)
        : a(a), b(b), P(p)
    {
    }

    // core of the doubling multiplier algorithm (see below)
    // multiplies acc by m as a series of "2*acc's"
    CUDA_CALLABLE void addDouble(const T &m, Point<T> &p) const
    {
        Point<T> r = p;
        for (T n = 0; n < m; ++n)
        {
            r = add(r, r); // doubling step
        }
        p = r;
    }

    CUDA_CALLABLE Point<T> add(const Point<T> &lhs, const Point<T> &rhs) const
    {
        if(lhs.x == 0 && lhs.y == 0)
        {
            return rhs;
        }
        else if(rhs.x == 0 && rhs.y == 0)
        {
            return lhs;
        }
        else if(lhs.y == (-rhs.y + P) && lhs.x == rhs.x)
        {
            return Point<T>(0, 0);
        }
        
        T xR, yR;
        ffe_t x1(lhs.x, P), y1(lhs.y, P), x2(rhs.x, P), y2(rhs.y, P), s(0, P); 
        if (lhs.x == rhs.x && lhs.y == rhs.y && lhs.y != 0 ) // P == Q, doubling
        {
            s = (3 * x1 * x1 + a) / ffe_t(y1.i() << 1, P);
            xR = s * s - ffe_t(x1.i() << 1, P);
        }
        else
        {
            s = (y1 - y2) * ffe_t(detail::invmod((x1 - x2).i(), P), P);
            xR = s * s - x1 - x2;
        }
        yR = -y1 + s * (x1 - ffe_t(xR, P));

        return Point<T>(xR, yR);
    }

    CUDA_CALLABLE Point<T> mul(const T &k, const Point<T> &rhs) const
    {
        Point<T> acc = rhs;
        Point<T> res(0, 0);
        T i = 0, j = 0;
        T b = k;

        while(b != 0)
        {
            if((b & 1) != 0)
            {
                // bit is set; acc = 2^(i-j)*acc
                addDouble(i - j, acc);
                res = add(res, acc);
                j = i; // last bit set
            }
            b >>= 1;
            ++i;
        }
        return res;
    }

    CUDA_CALLABLE bool check( const Point<T> &point ) const
    {
        ffe_t y(point.y, P), x(point.x, P);
        if((point.x == 0 && point.y == 0) || (y * y == x * x * x + a * x + b))
            return true;
        return false;
    }

  private:
    T a, b;
    T P;
};

template<typename T>
class Point
{
    friend class EllipticCurve<T>;
    T x{ 0 };
    T y{ 0 };

public:

    CUDA_CALLABLE Point()
    {
    }

    CUDA_CALLABLE Point(const T &x, const T &y)
        : x(x), y(y)
    {
    }

    CUDA_CALLABLE Point(const Point &rhs)
    {
        x = rhs.x;
        y = rhs.y;
    }

    CUDA_CALLABLE Point &operator=(const Point &rhs)
    {
        x = rhs.x;
        y = rhs.y;
        return *this;
    }

    CUDA_CALLABLE T getX() const { return x; }

    CUDA_CALLABLE T getY() const { return y; }

    CUDA_CALLABLE friend bool operator==(const Point &lhs, const Point &rhs)
    {
        return (lhs.x == rhs.x) && (lhs.y == rhs.y);
    }

    CUDA_CALLABLE friend bool operator!=(const Point &lhs, const Point &rhs)
    {
        return !(lhs == rhs);
    }

    __host__ friend std::ostream &operator<<(std::ostream &os, const Point &p)
    {
        return (os << "(" << p.x << ", " << p.y << ")");
    }
};

#endif