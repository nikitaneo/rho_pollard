#include <iostream>

#ifndef ELLIPTIC_H
#define ELLIPTIC_H

#define CUDA_CALLABLE __host__ __device__

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

    /*
        A point, or group element, on the EC, consisting of two elements of the field FP
        Points can only created by the EC instance itself as they have to be 
        elements of the group generated by the EC
    */
    class Point
    {
        friend class EllipticCurve<T>;
        typedef FiniteFieldElement<T> ffe_t;
        ffe_t x_;
        ffe_t y_;
        ffe_t a_;
        ffe_t b_;
        T p_;

        // core of the doubling multiplier algorithm (see below)
        // multiplies acc by m as a series of "2*acc's"
        CUDA_CALLABLE void addDouble(const T &m)
        {
            if (m > 0)
            {
                Point r = *this;
                for (T n = 0; n < m; ++n)
                {
                    r += r; // doubling step
                }
                *this = r;
            }
        }
        // doubling multiplier algorithm
        // multiplies a by k by expanding in multiplies by 2
        // a is also an accumulator that stores the intermediate results
        // between the "1s" of the binary form of the input scalar k
        CUDA_CALLABLE Point scalarMultiply(const T &k) const
        {
            Point acc = *this;
            Point res = Point(0, 0, a_.i(), b_.i(), p_);
            T i = 0, j = 0;
            T b = k;

            while (b != 0)
            {
                if ((b & 1) != 0)
                {
                    // bit is set; acc = 2^(i-j)*acc
                    acc.addDouble(i - j);
                    res += acc;
                    j = i; // last bit set
                }
                b >>= 1;
                ++i;
            }
            return res;
        }
        // adding two points on the curve
        CUDA_CALLABLE void add(ffe_t x1, ffe_t y1, ffe_t x2, ffe_t y2, ffe_t &xR, ffe_t &yR) const
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
            if (y1 == -y2)
            {
                xR = yR = ffe_t(0, p_);
                return;
            }

            // the additions
            ffe_t s(0, p_);
            if (x1 == x2 && y1 == y2)
            {
                //2P
                s = (3 * detail::mulmod(x1.i(), x1.i(), p_) + a_) / (2 * y1);
                xR = s * s - 2 * x1;
            }
            else
            {
                //P+Q
                s = (y1 - y2) / (x1 - x2);
                xR = ((s * s) - x1 - x2);
            }

            if (s != 0)
            {
                yR = (-y1 + s * (x1 - xR));
            }
            else
            {
                xR = yR = ffe_t(0, p_);
            }
        }

        CUDA_CALLABLE Point(const ffe_t &x, const ffe_t &y, const EllipticCurve<T> &ec)
            : x_(x),
              y_(y),
              a_(ec.a()),
              b_(ec.b()),
              p_(ec.p())
        {
        }

      public:
        static Point ONE;

        CUDA_CALLABLE Point() : x_(0, 1), y_(0, 1), a_(0, 1), b_(0, 1), p_(1)
        {
        }

        CUDA_CALLABLE Point(const T &x, const T &y, const EllipticCurve<T> &ec)
            : x_(x, ec.p()),
              y_(y, ec.p()),
              a_(ec.a()),
              b_(ec.b()),
              p_(ec.p())
        {
            // do nothing
        }

        CUDA_CALLABLE Point(const T &x, const T &y, const T &a, const T &b, const T &p)
            : x_(x, p),
              y_(y, p),
              a_(a, p),
              b_(b, p),
              p_(p)

        {
            // do nothing
        }

        CUDA_CALLABLE Point(const Point &rhs)
        {
            x_ = rhs.x_;
            y_ = rhs.y_;
            a_ = rhs.a_;
            b_ = rhs.b_;
            p_ = rhs.p_;
        }

        CUDA_CALLABLE Point &operator=(const Point &rhs)
        {
            x_ = rhs.x_;
            y_ = rhs.y_;
            a_ = rhs.a_;
            b_ = rhs.b_;
            p_ = rhs.p_;
            return *this;
        }

        CUDA_CALLABLE ffe_t x() const { return x_; }

        CUDA_CALLABLE ffe_t y() const { return y_; }

        CUDA_CALLABLE Point operator-()
        {
            return Point(x_, -y_, a_, b_, p_);
        }

        CUDA_CALLABLE friend bool operator==(const Point &lhs, const Point &rhs)
        {
            return (lhs.a_ == rhs.a_) && (lhs.b_ == rhs.b_) && (lhs.p_ == rhs.p_) && (lhs.x_ == rhs.x_) && (lhs.y_ == rhs.y_);
        }

        CUDA_CALLABLE friend bool operator!=(const Point &lhs, const Point &rhs)
        {
            return !(lhs == rhs);
        }

        CUDA_CALLABLE friend Point operator+(const Point &lhs, const Point &rhs)
        {
            ffe_t xR(0, lhs.p_), yR(0, lhs.p_);
            lhs.add(lhs.x_, lhs.y_, rhs.x_, rhs.y_, xR, yR);
            return Point(xR.i(), yR.i(), lhs.a_.i(), lhs.b_.i(), lhs.p_);
        }

        CUDA_CALLABLE friend Point operator*(const T &k, const Point &rhs)
        {
            return Point(rhs).operator*=(k);
        }

        CUDA_CALLABLE Point &operator+=(const Point &rhs)
        {
            add(x_, y_, rhs.x_, rhs.y_, x_, y_);
            return *this;
        }

        CUDA_CALLABLE Point &operator*=(const T &k)
        {
            return (*this = scalarMultiply(k));
        }

        friend std::ostream &operator<<(std::ostream &os, const Point &p)
        {
            return (os << "(" << p.x_.i() << ", " << p.y_.i() << ")");
        }

        CUDA_CALLABLE bool check()
        {
            if(y_*y_ == x_*x_*x_ + a_ * x_ + b_)
                return true;
            return false;
        }
    };

    typedef EllipticCurve<T> this_t;
    typedef class EllipticCurve<T>::Point point_t;

    // Initialize EC as y^2 = x^3 + ax + b
    CUDA_CALLABLE EllipticCurve(const T &a, const T &b, const T &p)
        : a_(a, p),
          b_(b, p),
          P(p)
    {
    }

    CUDA_CALLABLE T p() const { return P; }

    CUDA_CALLABLE FiniteFieldElement<T> a() const { return a_; }

    CUDA_CALLABLE FiniteFieldElement<T> b() const { return b_; }


  private:
    FiniteFieldElement<T> a_;
    FiniteFieldElement<T> b_;
    T P; // group order
};

template <typename T>
typename EllipticCurve<T>::Point EllipticCurve<T>::Point::ONE(0, 0);

#endif