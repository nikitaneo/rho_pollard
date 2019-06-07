#ifndef ELLIPTIC_H
#define ELLIPTIC_H

#include <iostream>

#define CUDA_CALLABLE __host__ __device__

template<typename T>
class Point;

/*
    Elliptic Curve over a finite field of order P:
    y^2 mod P = x^3 + ax + b mod P
*/

template <typename T>
class EllipticCurve
{
  public:

    CUDA_CALLABLE EllipticCurve()
    {
    }

    // Initialize EC as y^2 = x^3 + ax + b
    CUDA_CALLABLE EllipticCurve(const T &a, const T &b, const T &p)
        : a(a), b(b), P(p)
    {
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
            r = plus(r, r); // doubling step
        }
        p = r;
    }

    CUDA_CALLABLE Point<T> plus(const Point<T> &lhs, const Point<T> &rhs) const
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
        
        T s = 0, xR, yR;
        if(lhs.x == rhs.x && lhs.y == rhs.y && lhs.y != 0 ) // P == Q, doubling
        {
            s = lhs.x.mul_modp(3, P).mul_modp(lhs.x, P).add_modp(a, P).div_modp(lhs.y.mul_modp(2, P), P);
            xR = s.mul_modp(s, P).sub_modp(lhs.x.mul_modp(2, P), P);
        }
        else
        {
            s = lhs.y.sub_modp(rhs.y, P).div_modp(lhs.x.sub_modp(rhs.x, P), P);
            xR = s.mul_modp(s, P).sub_modp(lhs.x, P).sub_modp(rhs.x, P);
        }
        yR = s.mul_modp(lhs.x.sub_modp(xR, P), P).sub_modp(lhs.y, P);

        return Point<T>(xR, yR);
    }

    CUDA_CALLABLE void addition(Point<T> &lhs, const Point<T> &rhs) const
    {
        T s = lhs.y.sub_modp(rhs.y, P).div_modp(lhs.x.sub_modp(rhs.x, P), P);
        T xR = s.mul_modp(s, P).sub_modp(lhs.x, P).sub_modp(rhs.x, P);
        lhs.y = s.mul_modp(lhs.x.sub_modp(xR, P), P).sub_modp(lhs.y, P);
        lhs.x = xR;
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
                res = plus(res, acc);
                j = i; // last bit set
            }
            b >>= 1;
            ++i;
        }
        return res;
    }

    CUDA_CALLABLE bool check( const Point<T> &p ) const
    {
        if((p.x == 0 && p.y == 0) || (p.y.mul_modp(p.y, P) == p.x.mul_modp(p.x, P).mul_modp(p.x, P).add_modp(p.x.mul_modp(a, P).add_modp(b, P), P)))
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

#endif
