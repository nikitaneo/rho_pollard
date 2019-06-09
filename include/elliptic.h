#ifndef ELLIPTIC_H
#define ELLIPTIC_H

#include <iostream>
#include <unordered_map>

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
    __host__ EllipticCurve(const T &a, const T &b, const T &p)
        : a(a), b(b), P(p)
    {
    }

    CUDA_CALLABLE const T& getP() const
    {
        return P;
    }

    // core of the doubling multiplier algorithm (see below)
    // multiplies acc by m as a series of "2*acc's"
    __host__ void addDouble(const T &m, Point<T> &p) const
    {
        Point<T> r = p;
        for(T n = 0; n < m; ++n)
        {
            r = plus(r, r); // doubling step
        }
        p = r;
    }

    __host__ Point<T> plus(const Point<T> &lhs, const Point<T> &rhs) const
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

    __host__ void addition(Point<T> &lhs, const Point<T> &rhs) const
    {
        T s = lhs.y.sub_modp(rhs.y, P).div_modp(lhs.x.sub_modp(rhs.x, P), P);
        T xR = s.mul_modp(s, P).sub_modp(lhs.x, P).sub_modp(rhs.x, P);
        lhs.y = s.mul_modp(lhs.x.sub_modp(xR, P), P).sub_modp(lhs.y, P);
        lhs.x = xR;
    }

    __device__ void addition(Point<T> &lhs, const Point<T> &rhs, const T &inv) const
    {
        T s = lhs.y.sub_modp(rhs.y, P).mul_modp(inv, P);
        T xR = s.mul_modp(s, P).sub_modp(lhs.x, P).sub_modp(rhs.x, P);
        lhs.y = s.mul_modp(lhs.x.sub_modp(xR, P), P).sub_modp(lhs.y, P);
        lhs.x = xR;
    }

    __host__ Point<T> mul(T k, const Point<T> &rhs) const
    {
        Point<T> acc = rhs;
        Point<T> res(0, 0);

        while(k != 0)
        {
            if((k & 1) != 0)
            {
                res = plus(res, acc);
            }
            
            k >>= 1;
            acc = plus(acc, acc);
        }
        return res;
    }

    __host__ Point<T> mul(T k, const Point<T> &rhs, std::unordered_map<unsigned, Point<T>> &cash) const
    {
        unsigned pow2 = 0;
        Point<T> acc = rhs;
        Point<T> res(0, 0);

        while(k != 0)
        {
            if((k & 1) != 0)
            {
                res = plus(res, acc);
            }
            
            k >>= 1; pow2++;
            if(cash.find(pow2) != cash.end())
            {
                acc = cash[pow2];
            }
            else
            {
                acc = plus(acc, acc);
                cash[pow2] = acc;
            }
        }
        return res;
    }

    __host__ bool check( const Point<T> &p ) const
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

    CUDA_CALLABLE T& getX() { return x; }

    CUDA_CALLABLE T& getY() { return y; }

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