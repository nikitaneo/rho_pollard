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
    T res = 0; // Initialize result 
    T a = A % mod;
    T b = B % mod; 
    while (b > 0) 
    { 
        if ((b & 1) == 1) 
            res = (res + a) % mod; 
  
        // Multiply 'a' with 2 
        a = (a << 1) % mod; 
  
        // Divide b by 2 
        b >>= 1; 
    } 
  
    // Return result 
    return res % mod; 
}

//From Knuth; Extended GCD gives g = a*u + b*v
template<typename T>
CUDA_CALLABLE T EGCD(const T &a, const T &b, T &u, T &v)
{
    u = 1;
    v = 0;
    T g = a;
    T u1 = 0;
    T v1 = 1;
    T g1 = b;
    while (g1 != 0)
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

template<typename T>
CUDA_CALLABLE T InvMod(const T &x, const T &n) // Solve linear congruence equation x * z == 1 (mod n) for z
{
    //n = Abs(n);
    T X = x % n;
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
    return z < 0 ? z + n : z;
}
} // namespace detail

/*
    An element in a Galois field FP
    Adapted for the specific behaviour of the "mod" function where (-n) mod m returns a negative number
    Allows basic arithmetic operations between elements:
    +,-,/,scalar multiply                
*/
template <typename T>
class FiniteFieldElement
{
    T i_;
    T P;

    CUDA_CALLABLE void assign(const T &i)
    {
        i_ = i % P;
        if (i < 0)
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
        return FiniteFieldElement<T>(detail::mulmod(lhs.i_, detail::InvMod(rhs.i_, rhs.P), rhs.P), rhs.P);
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
        return FiniteFieldElement<T>(n * rhs.i_, rhs.P);
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
    typedef EllipticCurve<T> this_t;

    // Initialize EC as y^2 = x^3 + ax + b
    CUDA_CALLABLE EllipticCurve(const T &a, const T &b, const T &p)
        : a(a), b(b), P(p)
    {
    }

    // core of the doubling multiplier algorithm (see below)
    // multiplies acc by m as a series of "2*acc's"
    CUDA_CALLABLE void addDouble(const T &m, Point<T> &p) const
    {
        if (m > 0)
        {
            Point<T> r = p;
            for (T n = 0; n < m; ++n)
            {
                r = add(r, r); // doubling step
            }
            p = r;
        }
    }

    // adding two points on the curve
    CUDA_CALLABLE void add_(const ffe_t &x1, const ffe_t &y1, const ffe_t &x2, const ffe_t &y2, ffe_t &xR, ffe_t &yR) const
    {
        // special cases involving the additive identity
        if (x1 == 0 && y1 == 0)
        {
            xR = x2;
            yR = y2;
            return;
        }

        if (x2 == 0 && y2 == 0)
        {
            xR = x1;
            yR = y1;
            return;
        }

        if(y1 == -y2 && x1 == x2)
        {
            xR = yR = ffe_t(0, P);
            return;
        }

        ffe_t s(0, P);
        if (x1 == x2 && y1 == y2 && y1 != 0 ) // P == Q, doubling
        {
            s = (3 * detail::mulmod(x1.i(), x1.i(), P) + ffe_t(a, P)) / (2 * y1);
            xR = s * s - 2 * x1;
            yR = -y1 + s * (x1 - xR);
        }
        else
        {
            s = (y1 - y2) / (x1 - x2);
            xR = s * s - x1 - x2;
            yR = -y1 + s * (x1 - xR);
        }
    }

    CUDA_CALLABLE Point<T> add(const Point<T> &lhs, const Point<T> &rhs) const
    {
        ffe_t xR(0, P), yR(0, P), x1(lhs.x_, P), y1(lhs.y_, P), x2(rhs.x_, P), y2(rhs.y_, P); 
        add_(x1, y1, x2, y2, xR, yR);
        return Point<T>(xR.i(), yR.i());
    }

    CUDA_CALLABLE Point<T> mul(const T &k, const Point<T> &rhs) const
    {
        Point<T> acc = rhs;
        Point<T> res(0, 0, P);
        T i = 0, j = 0;
        T b = k;

        while (b != 0)
        {
            if ((b & 1) != 0)
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
        ffe_t y(point.y_, P), x(point.x_, P);
        if((point.x_ == 0 && point.y_ == 0) || (y * y == x * x * x + a * x + b))
            return true;
        return false;
    }

  private:
    T a, b;
    T P;
};

/*
    A point, or group element, on the EC, consisting of two elements of the field FP
    Points can only created by the EC instance itself as they have to be 
    elements of the group generated by the EC
*/
template<typename T>
class Point
{
    friend class EllipticCurve<T>;
    T x_;
    T y_;

    CUDA_CALLABLE Point(const T &x, const T &y)
        : x_(x), y_(y)
    {
    }

public:

    CUDA_CALLABLE Point() : x_(0), y_(0)
    {
    }

    CUDA_CALLABLE Point(const T &x, const T &y, const EllipticCurve<T> &ec)
        : x_(x), y_(y)
    {
        // do nothing
    }

    CUDA_CALLABLE Point(const T &x, const T &y, const T &p)
        : x_(x), y_(y)
    {
        // do nothing
    }

    CUDA_CALLABLE Point(const Point &rhs)
    {
        x_ = rhs.x_;
        y_ = rhs.y_;
    }

    CUDA_CALLABLE Point &operator=(const Point &rhs)
    {
        x_ = rhs.x_;
        y_ = rhs.y_;
        return *this;
    }

    CUDA_CALLABLE T x() const { return x_; }

    CUDA_CALLABLE T y() const { return y_; }

    CUDA_CALLABLE Point operator-()
    {
        return Point(x_, -y_);
    }

    CUDA_CALLABLE friend bool operator==(const Point &lhs, const Point &rhs)
    {
        return (lhs.x_ == rhs.x_) && (lhs.y_ == rhs.y_);
    }

    CUDA_CALLABLE friend bool operator!=(const Point &lhs, const Point &rhs)
    {
        return !(lhs == rhs);
    }

    __host__ friend std::ostream &operator<<(std::ostream &os, const Point &p)
    {
        return (os << "(" << p.x_ << ", " << p.y_ << ")");
    }
};

#endif