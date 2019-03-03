#include <iostream>
#include <uint256.h>

#ifndef ELLIPTIC_H
#define ELLIPTIC_H

// helper functions
namespace detail
{
//From Knuth; Extended GCD gives g = a*u + b*v
template<typename T>
T EGCD(const T &a, const T &b, T &u, T &v)
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
T InvMod(T x, T n) // Solve linear congruence equation x * z == 1 (mod n) for z
{
    //n = Abs(n);
    x = x % n; // % is the remainder function, 0 <= x % n < |n|
    T u, v, g, z;
    g = EGCD(x, n, u, v);
    if (g != 1)
    {
        // x and n have to be relative prime for there to exist an x^-1 mod n
        z = 0;
    }
    else
    {
        z = u % n;
    }
    return z;
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

    void assign(const T &i)
    {
        i_ = i;
        if (i < 0)
        {
            // ensure (-i) mod p correct behaviour
            // the (i%P) term is to ensure that i is in the correct range before normalizing
            i_ = (i % P) + 2 * P;
        }

        i_ %= P;
    }

  public:
    // ctor
    FiniteFieldElement()
        : i_(0), P(0)
    {
    }
    // ctor
    explicit FiniteFieldElement(const T &i, const T &p) : P(p)
    {
        assign(i);
    }
    // copy ctor
    FiniteFieldElement(const FiniteFieldElement<T> &rhs)
        : i_(rhs.i_), P(rhs.P)
    {
    }

    // access "raw" integer
    T i() const { return i_; }
    // negate
    FiniteFieldElement operator-() const
    {
        return FiniteFieldElement(-i_, P);
    }

    // assign from field element
    FiniteFieldElement<T> &operator=(const FiniteFieldElement<T> &rhs)
    {
        i_ = rhs.i_;
        P = rhs.P;
        return *this;
    }
    // *=
    FiniteFieldElement<T> &operator*=(const FiniteFieldElement<T> &rhs)
    {
        i_ = (i_ * rhs.i_) % P;
        return *this;
    }
    // ==
    friend bool operator==(const FiniteFieldElement<T> &lhs, const FiniteFieldElement<T> &rhs)
    {
        return (lhs.i_ == rhs.i_);
    }
    // == int
    friend bool operator==(const FiniteFieldElement<T> &lhs, const T &rhs)
    {
        return (lhs.i_ == rhs);
    }
    // !=
    friend bool operator!=(const FiniteFieldElement<T> &lhs, const T &rhs)
    {
        return (lhs.i_ != rhs);
    }
    // a / b
    friend FiniteFieldElement<T> operator/(const FiniteFieldElement<T> &lhs, const FiniteFieldElement<T> &rhs)
    {
        return FiniteFieldElement<T>(lhs.i_ * detail::InvMod(rhs.i_, rhs.P), rhs.P);
    }
    // a + b
    friend FiniteFieldElement<T> operator+(const FiniteFieldElement<T> &lhs, const FiniteFieldElement<T> &rhs)
    {
        return FiniteFieldElement<T>(lhs.i_ + rhs.i_, rhs.P);
    }
    // a - b
    friend FiniteFieldElement<T> operator-(const FiniteFieldElement<T> &lhs, const FiniteFieldElement<T> &rhs)
    {
        return FiniteFieldElement<T>(lhs.i_ - rhs.i_, rhs.P);
    }
    // a + int
    friend FiniteFieldElement<T> operator+(const FiniteFieldElement<T> &lhs, const T &i)
    {
        return FiniteFieldElement<T>(lhs.i_ + i, lhs.P);
    }
    // int + a
    friend FiniteFieldElement<T> operator+(const T &i, const FiniteFieldElement<T> &rhs)
    {
        return FiniteFieldElement<T>(rhs.i_ + i, rhs.P);
    }
    // int * a
    friend FiniteFieldElement<T> operator*(const T &n, const FiniteFieldElement<T> &rhs)
    {
        return FiniteFieldElement<T>(n * rhs.i_, rhs.P);
    }
    // a * b
    friend FiniteFieldElement<T> operator*(const FiniteFieldElement<T> &lhs, const FiniteFieldElement<T> &rhs)
    {
        return FiniteFieldElement<T>(lhs.i_ * rhs.i_, rhs.P);
    }

    // ostream handler
    friend std::ostream &operator<<(std::ostream &os, const FiniteFieldElement<T> &g)
    {
        return os << g.i_;
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
        const EllipticCurve *ec_;

        // core of the doubling multiplier algorithm (see below)
        // multiplies acc by m as a series of "2*acc's"
        void addDouble(const T &m, Point &acc)
        {
            if (m > 0)
            {
                Point r = acc;
                for (T n = 0; n < m; ++n)
                {
                    r += r; // doubling step
                }
                acc = r;
            }
        }
        // doubling multiplier algorithm
        // multiplies a by k by expanding in multiplies by 2
        // a is also an accumulator that stores the intermediate results
        // between the "1s" of the binary form of the input scalar k
        Point scalarMultiply(const T &k, const Point &a)
        {
            Point acc = a;
            Point res = Point(0, 0, *ec_);
            T i = 0, j = 0;
            T b = k;

            while (b != 0)
            {
                if ((b & 1) != 0)
                {
                    // bit is set; acc = 2^(i-j)*acc
                    addDouble(i - j, acc);
                    res += acc;
                    j = i; // last bit set
                }
                b >>= 1;
                ++i;
            }
            return res;
        }
        // adding two points on the curve
        void add(ffe_t x1, ffe_t y1, ffe_t x2, ffe_t y2, ffe_t &xR, ffe_t &yR) const
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
                xR = yR = ffe_t(0, ec_->p());
                return;
            }

            // the additions
            ffe_t s(0, ec_->P);
            if (x1 == x2 && y1 == y2)
            {
                //2P
                s = (3 * (x1.i() * x1.i()) + ec_->a()) / (2 * y1);
                xR = ((s * s) - 2 * x1);
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
                xR = yR = ffe_t(0, ec_->p());
            }
        }

        Point(const ffe_t &x, const ffe_t &y, const EllipticCurve<T> &EC)
            : x_(x),
              y_(y),
              ec_(&EC)
        {
        }

      public:
        static Point ONE;

        Point(const T &x, const T &y, const EllipticCurve<T> &EC)
            : x_(x, EC.P),
              y_(y, EC.P),
              ec_(&EC)
        {
        }

        // copy ctor
        Point(const Point &rhs)
        {
            x_ = rhs.x_;
            y_ = rhs.y_;
            ec_ = rhs.ec_;
        }
        // assignment
        Point &operator=(const Point &rhs)
        {
            x_ = rhs.x_;
            y_ = rhs.y_;
            ec_ = rhs.ec_;
            return *this;
        }
        // access x component as element of Fp
        ffe_t x() const { return x_; }
        // access y component as element of Fp
        ffe_t y() const { return y_; }

        // calculate the order of this point by brute-force additions
        // WARNING: this can be VERY slow if the period is long and might not even converge
        // so maxPeriod should probably be set to something sensible...
        unsigned int Order(unsigned int maxPeriod = ~0) const
        {
            Point r = *this;
            unsigned int n = 0;
            while (r.x_ != 0 && r.y_ != 0)
            {
                ++n;
                r += *this;
                if (n > maxPeriod)
                    break;
            }
            return n;
        }

        Point operator-()
        {
            return Point(x_, -y_, *ec_);
        }

        friend bool operator==(const Point &lhs, const Point &rhs)
        {
            return (lhs.ec_ == rhs.ec_) && (lhs.x_ == rhs.x_) && (lhs.y_ == rhs.y_);
        }

        friend bool operator!=(const Point &lhs, const Point &rhs)
        {
            return (lhs.ec_ != rhs.ec_) || (lhs.x_ != rhs.x_) || (lhs.y_ != rhs.y_);
        }

        friend Point operator+(const Point &lhs, const Point &rhs)
        {
            if(lhs == rhs)
                return lhs.scalarMultiply(2, lhs);

            ffe_t xR(0, lhs.ec_->p()), yR(0, lhs.ec_->p());
            lhs.add(lhs.x_, lhs.y_, rhs.x_, rhs.y_, xR, yR);
            return Point(xR, yR, *lhs.ec_);
        }

        friend Point operator*(const T &k, const Point &rhs)
        {
            return Point(rhs).operator*=(k);
        }

        Point &operator+=(const Point &rhs)
        {
            add(x_, y_, rhs.x_, rhs.y_, x_, y_);
            return *this;
        }

        Point &operator*=(const T &k)
        {
            return (*this = scalarMultiply(k, *this));
        }

        friend std::ostream &operator<<(std::ostream &os, const Point &p)
        {
            return (os << "(" << p.x_ << ", " << p.y_ << ")");
        }
    };

    // ==================================================== EllipticCurve impl

    typedef EllipticCurve<T> this_t;
    typedef class EllipticCurve<T>::Point point_t;

    // ctor
    // Initialize EC as y^2 = x^3 + ax + b
    EllipticCurve(const T &a, const T &b, const T &p)
        : a_(a, p),
          b_(b, p),
          P(p),
          m_table_(),
          table_filled_(false)
    {
    }

    // Calculate *all* the points (group elements) for this EC
    //NOTE: if the order of this curve is large this could take some time...
    void CalculatePoints()
    {
        uint256 x_val[P];
        uint256 y_val[P];
        for (int n = 0; n < P; ++n)
        {
            int nsq = n * n;
            x_val[n] = ((n * nsq) + a_.i() * n + b_.i()) % P;
            y_val[n] = nsq % P;
        }

        for (int n = 0; n < P; ++n)
        {
            for (int m = 0; m < P; ++m)
            {
                if (x_val[n] == y_val[m])
                {
                    m_table_.push_back(Point(n, m, *this));
                }
            }
        }

        table_filled_ = true;
    }

    // get a point (group element) on the curve
    Point operator[](int n)
    {
        if (!table_filled_)
        {
            CalculatePoints();
        }

        return m_table_[n];
    }

    // number of elements in this group
    size_t Size() const { return m_table_.size(); }
    // the degree P of this EC
    T Degree() const { return P; }
    // the parameter a (as an element of Fp)
    FiniteFieldElement<T> a() const { return a_; }
    // the paramter b (as an element of Fp)
    FiniteFieldElement<T> b() const { return b_; }

    T p() const { return P; }

    // ostream handler: print this curve in human readable form
    template<typename S>
    friend std::ostream &operator<<(std::ostream &os, const EllipticCurve<S> &EllipticCurve);
    // print all the elements of the EC group
    std::ostream &PrintTable(std::ostream &os, int columns = 4)
    {
        if (table_filled_)
        {
            int col = 0;
            typename EllipticCurve<T>::m_table_t::iterator iter = m_table_.begin();
            for (; iter != m_table_.end(); ++iter)
            {
                os << "(" << (*iter).x_.i() << ", " << (*iter).y_.i() << ") ";
                if (++col > columns)
                {
                    os << "\n";
                    col = 0;
                }
            }
        }
        else
        {
            os << "EllipticCurve, F_" << P;
        }
        return os;
    }

  private:
    typedef std::vector<Point> m_table_t;

    m_table_t m_table_;       // table of points
    FiniteFieldElement<T> a_; // paramter a of the EC equation
    FiniteFieldElement<T> b_; // parameter b of the EC equation
    T P; // parameter p of the EC equation
    bool table_filled_;       // true if the table has been calculated
};

template <typename T>
typename EllipticCurve<T>::Point EllipticCurve<T>::Point::ONE(0, 0);

template <typename T>
std::ostream &operator<<(std::ostream &os, const EllipticCurve<T> &EllipticCurve)
{
    os << "y^2 mod " << EllipticCurve.P << " = (x^3" << std::showpos;
    if (EllipticCurve.a_ != 0)
    { 
        os << EllipticCurve.a_ << "x";
    }

    if (EllipticCurve.b_.i() != 0)
    {
        os << EllipticCurve.b_;
    }

    os << std::noshowpos << ") mod " << EllipticCurve.P;
    return os;
}

#endif