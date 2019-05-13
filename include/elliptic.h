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

// Not used
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

// Solve linear congruence equation x * z == 1 (mod n) for z. buf is size of 8 * sizeof(int128_t)
CUDA_CALLABLE int128_t invmod(const int128_t &x, const int128_t &n)
{
    int128_t u = 1, g = x;
    int128_t u1 = 0, g1 = n;
    int128_t t1, t2, q;
    while( g1 != 0 )
    {
        q = g / g1;
        t1 = u - q * u1;
        t2 = g - q * g1;
        u = u1;
        g = g1;
        u1 = t1;
        g1 = t2;
    }

    q = u % n;

    return q.isLessZero() ? q + n : q;
}

} // namespace detail

template <typename T>
class FiniteFieldElement
{
    T i_;

    CUDA_CALLABLE void assign(const T &i, const T &p)
    {
        if(i >= p)
            i_ = i % p;
        else if(i.isLessZero())
            i_ = i + p;
        else
            i_ = i;
    }

  public:

    CUDA_CALLABLE FiniteFieldElement() : i_(0)
    {
    }

    CUDA_CALLABLE explicit FiniteFieldElement(const T &i, const T &p)
    {
        assign(i, p);
    }

    CUDA_CALLABLE FiniteFieldElement(const FiniteFieldElement<T> &rhs)
        : i_(rhs.i_)
    {
    }

    // access "raw" integer
    CUDA_CALLABLE const T& i() const { return i_; }

    CUDA_CALLABLE operator T() { return i_; }

    CUDA_CALLABLE FiniteFieldElement<T> &operator=(const FiniteFieldElement<T> &rhs)
    {
        i_ = rhs.i_;
        return *this;
    }

    CUDA_CALLABLE friend bool operator==(const FiniteFieldElement<T> &lhs, const FiniteFieldElement<T> &rhs)
    {
        return (lhs.i_ == rhs.i_);
    }

    CUDA_CALLABLE friend bool operator!=(const FiniteFieldElement<T> &lhs, const FiniteFieldElement<T> &rhs)
    {
        return (lhs.i_ != rhs.i_);
    }

    CUDA_CALLABLE friend bool operator==(const FiniteFieldElement<T> &lhs, const T &rhs)
    {
        return (lhs.i_ == rhs);
    }

    CUDA_CALLABLE friend bool operator!=(const FiniteFieldElement<T> &lhs, const T &rhs)
    {
        return (lhs.i_ != rhs);
    }

    CUDA_CALLABLE FiniteFieldElement<T> div(const FiniteFieldElement<T> &rhs, const T &p) const
    {
        return FiniteFieldElement<T>(i_ * detail::invmod(rhs.i_, p), p);
    }
 
    CUDA_CALLABLE FiniteFieldElement<T> add(const FiniteFieldElement<T> &rhs, const T &p) const
    {
        return FiniteFieldElement<T>(i_ + rhs.i_, p);
    }

    CUDA_CALLABLE FiniteFieldElement<T> add(const T &n, const T &p) const
    {
        return FiniteFieldElement<T>(i_ + n, p);
    }

    CUDA_CALLABLE FiniteFieldElement<T> sub(const FiniteFieldElement<T> &rhs, const T &p) const 
    {
        return FiniteFieldElement<T>(i_ - rhs.i_, p);
    }

    CUDA_CALLABLE FiniteFieldElement<T> sub(const T &n, const T &p) const 
    {
        return FiniteFieldElement<T>(i_ - n, p);
    }

    CUDA_CALLABLE FiniteFieldElement<T> mul(const T &n, const T &p) const 
    {
        return FiniteFieldElement<T>(i_ * n, p);
    }

    CUDA_CALLABLE FiniteFieldElement<T> mul(const FiniteFieldElement<T> &rhs, const T &p) const 
    {
        return FiniteFieldElement<T>(i_ * rhs.i_, p);
    }

    CUDA_CALLABLE FiniteFieldElement<T> lshift(unsigned shift, const T &p) const
    {
        return FiniteFieldElement<T>( i_ << shift, p );
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

    CUDA_CALLABLE EllipticCurve()
    {
    }

    // Initialize EC as y^2 = x^3 + ax + b
    CUDA_CALLABLE EllipticCurve(const T &a, const T &b, const T &p)
        : a(a), b(b), P(p)
    {
    }

    CUDA_CALLABLE const T& getA() const
    {
        return a;
    }

    CUDA_CALLABLE const T& getB() const
    {
        return b;
    }

    CUDA_CALLABLE const T& getP() const
    {
        return P;
    }

    // core of the doubling multiplier algorithm (see below)
    // multiplies acc by m as a series of "2*acc's"
    CUDA_CALLABLE void addDouble(const T &m, Point<T> &p) const
    {
        Point<T> r = p;
        for(T n = 0; n < m; ++n)
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
        
        ffe_t xR(0, P), yR(0, P);
        ffe_t x1(lhs.x, P), y1(lhs.y, P), x2(rhs.x, P), y2(rhs.y, P), s(0, P); 
        if(x1 == x2 && y1 == y2 && y1 != 0 ) // P == Q, doubling
        {
            s = x1.mul(3, P).mul(x1, P).add(a, P).div(y1.lshift(1, P), P);
            xR = s.mul(s, P).sub(x1.lshift(1, P), P);
        }
        else
        {
            s = y1.sub(y2, P).div(x1.sub(x2, P), P);
            xR = s.mul(s, P).sub(x1, P).sub(x2, P);
        }
        yR = s.mul(x1.sub(xR, P), P).sub(y1, P);

        return Point<T>(xR, yR);
    }

    __device__ void addition(Point<T> &lhs, const Point<T> &rhs) const
    {
        const ffe_t x1(lhs.x, P), y1(lhs.y, P), x2(rhs.x, P), y2(rhs.y, P);
        const ffe_t s = y1.sub(y2, P).div(x1.sub(x2, P), P);
        lhs.x = s.mul(s, P).sub(x1, P).sub(x2, P);
        lhs.y = s.mul(x1.sub(lhs.x, P), P).sub(y1, P);
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
        if((point.x == 0 && point.y == 0) || (y.mul(y, P) == x.mul(x, P).mul(x, P).add(x.mul(a, P).add(b, P), P)))
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
    T x;
    T y;

public:

    CUDA_CALLABLE Point() { }

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

    CUDA_CALLABLE inline uint32_t hash() const
    {
        uint32_t hash = 0;
        hash = (hash + (324723947 + x.hash())) ^ 93485734985;
        hash = (hash + (324723947 + y.hash())) ^ 93485734985;
        return hash;
    }

    CUDA_CALLABLE const T& getX() const { return x; }

    CUDA_CALLABLE const T& getY() const { return y; }

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

namespace detail
{

class HashFunc
{
public:
    __device__ __host__ std::size_t operator() (const Point<int128_t> &arg) const noexcept
    {
        return arg.hash();
    }
};

}

#endif
