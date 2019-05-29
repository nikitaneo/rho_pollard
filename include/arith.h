#ifndef _ARITH_H
#define _ARITH_H

#include <assert.h>
#include <limits>
#include <stdint.h>
#include <string>
#include <vector>
#include <random>
#include <functional>
#include <algorithm>
#include <cuda_runtime.h>

#define CUDA_CALLABLE __host__ __device__

__host__ char ToLower(char c)
{
    return (c >= 'A' && c <= 'Z' ? (c - 'A') + 'a' : c);
}

inline __host__ bool IsSpace(char c) noexcept 
{
    return c == ' ' || c == '\f' || c == '\n' || c == '\r' || c == '\t' || c == '\v';
}

__host__ signed char HexDigit(char c)
{
    static const signed char p_util_hexdigit[256] =
    { -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
    0,1,2,3,4,5,6,7,8,9,-1,-1,-1,-1,-1,-1,
    -1,0xa,0xb,0xc,0xd,0xe,0xf,-1,-1,-1,-1,-1,-1,-1,-1,-1,
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
    -1,0xa,0xb,0xc,0xd,0xe,0xf,-1,-1,-1,-1,-1,-1,-1,-1,-1,
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1, };

    return p_util_hexdigit[(unsigned char)c];
}

template<typename T>
__host__ std::string HexStr(const T itbegin, const T itend, bool fSpaces=false)
{
    std::string rv;
    static const char hexmap[16] = { '0', '1', '2', '3', '4', '5', '6', '7',
                                     '8', '9', 'a', 'b', 'c', 'd', 'e', 'f' };
    rv.reserve((itend-itbegin)*3);
    for(T it = itbegin; it < itend; ++it)
    {
        unsigned char val = (unsigned char)(*it);
        if(fSpaces && (it - itbegin) % 4 == 0 && it != itbegin)
            rv.push_back(' ');
        rv.push_back(hexmap[val>>4]);
        rv.push_back(hexmap[val&15]);
    }

    return rv;
}

/** Template base class for unsigned big integers. */
template<unsigned int WIDTH>
class base_uint
{
protected:
    uint32_t pn[WIDTH];
public:

    CUDA_CALLABLE base_uint()
    {
    }

    CUDA_CALLABLE base_uint(const base_uint& b)
    {
        for (int i = 0; i < WIDTH; i++)
            pn[i] = b.pn[i];
    }

    CUDA_CALLABLE base_uint& operator=(const base_uint& b)
    {
        for (int i = 0; i < WIDTH; i++)
            pn[i] = b.pn[i];
        return *this;
    }

    CUDA_CALLABLE base_uint(int64_t b)
    {
        memset(pn, 0, sizeof(pn));
        
        if(b >= 0)
        {
            pn[0] = (unsigned int)b;
            pn[1] = (unsigned int)(b >> 32);
        }
        else
        {
            b = -b;
            pn[0] = (unsigned int)b;
            pn[1] = (unsigned int)(b >> 32);
            *this = -(*this);
        }
    }

    __host__ explicit base_uint(const std::string& str)
    {
        SetHex(str);
    }

    CUDA_CALLABLE const base_uint operator~() const
    {
        base_uint ret;
        for (int i = 0; i < WIDTH; i++)
            ret.pn[i] = ~pn[i];
        return ret;
    }

    CUDA_CALLABLE const base_uint operator-() const
    {
        base_uint ret;
        for (int i = 0; i < WIDTH; i++)
            ret.pn[i] = ~pn[i];
        ++ret;
        return ret;
    }

    CUDA_CALLABLE base_uint& operator=(uint64_t b)
    {
        pn[0] = (unsigned int)b;
        pn[1] = (unsigned int)(b >> 32);
        for (int i = 2; i < WIDTH; i++)
            pn[i] = 0;
        return *this;
    }

    CUDA_CALLABLE base_uint& operator^=(const base_uint& b)
    {
        for (int i = 0; i < WIDTH; i++)
            pn[i] ^= b.pn[i];
        return *this;
    }

    CUDA_CALLABLE base_uint& operator&=(const base_uint& b)
    {
        for (int i = 0; i < WIDTH; i++)
            pn[i] &= b.pn[i];
        return *this;
    }

    CUDA_CALLABLE base_uint& operator|=(const base_uint& b)
    {
        for (int i = 0; i < WIDTH; i++)
            pn[i] |= b.pn[i];
        return *this;
    }

    CUDA_CALLABLE base_uint& operator^=(uint64_t b)
    {
        pn[0] ^= (unsigned int)b;
        pn[1] ^= (unsigned int)(b >> 32);
        return *this;
    }

    CUDA_CALLABLE base_uint& operator|=(uint64_t b)
    {
        pn[0] |= (unsigned int)b;
        pn[1] |= (unsigned int)(b >> 32);
        return *this;
    }

    CUDA_CALLABLE base_uint& operator<<=(unsigned int shift)
    {
        base_uint<WIDTH> a(*this);
        for (int i = 0; i < WIDTH; i++)
            pn[i] = 0;
        int k = shift / 32;
        shift = shift % 32;
        for (int i = 0; i < WIDTH; i++) {
            if (i + k + 1 < WIDTH && shift != 0)
                pn[i + k + 1] |= (a.pn[i] >> (32 - shift));
            if (i + k < WIDTH)
                pn[i + k] |= (a.pn[i] << shift);
        }
        return *this;
    }

    CUDA_CALLABLE base_uint& operator>>=(unsigned int shift)
    {
        base_uint<WIDTH> a(*this);
        for (int i = 0; i < WIDTH; i++)
            pn[i] = 0;
        int k = shift / 32;
        shift = shift % 32;
        for (int i = 0; i < WIDTH; i++) {
            if (i - k - 1 >= 0 && shift != 0)
                pn[i - k - 1] |= (a.pn[i] << (32 - shift));
            if (i - k >= 0)
                pn[i - k] |= (a.pn[i] >> shift);
        }
        return *this;
    }

    CUDA_CALLABLE base_uint& operator+=(const base_uint& b)
    {
#ifdef __CUDA_ARCH__
        unsigned int *a = pn; // hack!
        // No carry in
        asm volatile( "add.cc.u32 %0, %1, %2;\n\t" : "=r"(a[ 0 ]) : "r"(b.pn[ 0 ]), "r"(pn[ 0 ]) );

        // Carry in and carry out    
        #pragma unroll
        for(int i = 1; i < WIDTH; i++)
        {
            asm volatile( "addc.cc.u32 %0, %1, %2;\n\t" : "=r"(a[ i ]) : "r"(b.pn[ i ]), "r"(pn[ i ]) );
        }
#else
        uint64_t carry = 0;
        for (int i = 0; i < WIDTH; i++)
        {
            uint64_t n = carry + pn[i] + b.pn[i];
            pn[i] = n & 0xffffffff;
            carry = n >> 32;
        }
#endif
        return *this;
    }

    CUDA_CALLABLE base_uint& operator-=(const base_uint& b)
    {
#ifdef __CUDA_ARCH__
        unsigned int *a = pn; // hack!
        asm volatile( "sub.cc.u32 %0, %1, %2;\n\t" : "=r"(a[ 0 ]) : "r"(pn[ 0 ]), "r"(b.pn[ 0 ]) );
        
        #pragma unroll
        for(int i = 1; i < WIDTH; ++i)
        {
            asm volatile( "subc.cc.u32 %0, %1, %2;\n\t" : "=r"(a[ i ]) : "r"(pn[ i ]), "r"(b.pn[ i ]) );
        }
#else
        *this += -b;
#endif
        return *this;
    }

    CUDA_CALLABLE base_uint& operator+=(uint64_t b64)
    {
        base_uint b;
        b = b64;
        *this += b;
        return *this;
    }

    CUDA_CALLABLE base_uint& operator-=(uint64_t b64)
    {
        base_uint b;
        b = b64;
        *this += -b;
        return *this;
    }

    CUDA_CALLABLE base_uint& operator*=(const base_uint& b)
    {
#ifdef __CUDA_ARCH__
        base_uint<WIDTH> a = 0;
        uint64_t n;
        for( unsigned j = 0; j < WIDTH; j++ )
        {
            n = 0;
            for( unsigned i = 0; i + j < WIDTH; i++)
            {
                asm volatile( "mad.wide.u32 %0, %1, %2, %3;\n\t" : "=l"(n) : "r"(pn[j]), "r"(b.pn[i]), "l"((n >> 32) + a.pn[i + j]) );
                a.pn[i + j] = n;
            }
        }
#else
        base_uint<WIDTH> a = 0;
        for (int j = 0; j < WIDTH; j++)
        {
            uint64_t carry = 0;
            for (int i = 0; i + j < WIDTH; i++)
            {
                uint64_t n = carry + a.pn[i + j] + (uint64_t)pn[j] * b.pn[i];
                a.pn[i + j] = n & 0xffffffff;
                carry = n >> 32;
            }
        }
#endif
        *this = a;
        return *this;
    }

    CUDA_CALLABLE base_uint& operator/=(const base_uint& b)
    {
        base_uint div = b;
        base_uint num = (*this);

        memset(pn, 0, sizeof(pn));

        int shift = num.bitlength() - div.bitlength();
        if(shift < 0)
            return (*this);

        div <<= shift; // shift so that div and num align
        
        for(; shift >= 0; --shift)
        {
            if(num >= div)
            {
                num -= div;
                pn[shift >> 5] |= (1 << (shift & 31)); // set a bit of the result
            }
            div >>= 1; // shift back
        }

        return (*this);
    }

    CUDA_CALLABLE base_uint& operator++()
    {
        // prefix operator
        int i = 0;
        while (i < WIDTH && ++pn[i] == 0)
            i++;
        return *this;
    }

    CUDA_CALLABLE const base_uint operator++(int)
    {
        // postfix operator
        const base_uint ret = *this;
        ++(*this);
        return ret;
    }

    CUDA_CALLABLE base_uint& operator--()
    {
        // prefix operator
        int i = 0;
        while (i < WIDTH && --pn[i] == 0xffffffff)
            i++;
        return *this;
    }

    CUDA_CALLABLE const base_uint operator--(int)
    {
        // postfix operator
        const base_uint ret = *this;
        --(*this);
        return ret;
    }

    CUDA_CALLABLE base_uint& operator%=(const base_uint& b)
    {
        *this = *this % b;
        return *this;
    }

    CUDA_CALLABLE inline uint32_t hash() const
    {
        uint32_t hash = 0;
        for(unsigned i = 0; i < WIDTH; ++i)
        {
            hash = (hash + (324723947 + pn[i])) ^ 93485734985;
        }
        return hash;
    }

    CUDA_CALLABLE friend inline const base_uint operator+(const base_uint& a, const base_uint& b) { return base_uint(a) += b; }
    CUDA_CALLABLE friend inline const base_uint operator-(const base_uint& a, const base_uint& b) { return base_uint(a) -= b; }
    CUDA_CALLABLE friend inline const base_uint operator*(const base_uint& a, const base_uint& b) { return base_uint(a) *= b; }
    CUDA_CALLABLE friend inline const base_uint operator/(const base_uint& a, const base_uint& b) { return base_uint(a) /= b; }
    CUDA_CALLABLE friend inline const base_uint operator|(const base_uint& a, const base_uint& b) { return base_uint(a) |= b; }
    CUDA_CALLABLE friend inline const base_uint operator&(const base_uint& a, const base_uint& b) { return base_uint(a) &= b; }
    CUDA_CALLABLE friend inline uint32_t operator&(const base_uint& a, const uint32_t b) { return a.pn[0] & b; }
    CUDA_CALLABLE friend inline const base_uint operator^(const base_uint& a, const base_uint& b) { return base_uint(a) ^= b; }
    CUDA_CALLABLE friend inline const base_uint operator>>(const base_uint& a, int shift) { return base_uint(a) >>= shift; }
    CUDA_CALLABLE friend inline const base_uint operator<<(const base_uint& a, int shift) { return base_uint(a) <<= shift; }
    CUDA_CALLABLE friend inline bool operator==(const base_uint& a, const base_uint& b)
    {
        for(unsigned i = 0; i < WIDTH; ++i)
        {
            if(a.pn[i] != b.pn[i])
                return false;
        }
        return true;
    }
    CUDA_CALLABLE friend inline bool operator!=(const base_uint& a, const base_uint& b) { return !(a == b); }
    
    CUDA_CALLABLE friend inline bool operator>(const base_uint& a, const base_uint& b)
    {
        const bool lhsSign = a.isLessZero();
        const bool rhsSign = b.isLessZero();

        if( lhsSign != rhsSign )
            return lhsSign < rhsSign;

        int i = WIDTH - 1;
        for(; a.pn[i] == b.pn[i] && i > 0; --i)
        {
            // do nothing;
        }

        if(a.pn[i] > b.pn[i])
            return !lhsSign;
        else
            return lhsSign;
    }
    
    CUDA_CALLABLE friend inline bool operator<(const base_uint& a, const base_uint& b)
    {
        const bool lhsSign = a.isLessZero();
        const bool rhsSign = b.isLessZero();

        if( lhsSign != rhsSign )
            return lhsSign > rhsSign;
    
        int i = WIDTH - 1;
        for(; a.pn[i] == b.pn[i] && i > 0; --i)
        {
            // do nothing;
        }

        if(a.pn[i] < b.pn[i])
            return !lhsSign;
        else
            return lhsSign;
    }
    
    CUDA_CALLABLE friend inline bool operator>=(const base_uint& a, const base_uint& b) { return !(a < b); }
    CUDA_CALLABLE friend inline bool operator<=(const base_uint& a, const base_uint& b) { return !(a > b); }

    __host__ friend inline std::ostream& operator<<(std::ostream& out, const base_uint &a) { return out << a.GetHex(); }

    CUDA_CALLABLE friend inline const base_uint operator%(const base_uint& a, const base_uint& b)
    {
        return a - b * ( a / b );
    }

    __host__ std::string GetHex() const
    {
        return HexStr(std::reverse_iterator<const uint8_t*>((const uint8_t*)pn + sizeof(pn)), std::reverse_iterator<const uint8_t*>((const uint8_t*)pn), true);
    }

    void SetHex(const char* psz)
    {
        memset(pn, 0, sizeof(pn));

        // skip leading spaces
        while (IsSpace(*psz))
            psz++;

        // skip 0x
        if (psz[0] == '0' && ToLower(psz[1]) == 'x')
            psz += 2;

        // hex string to uint
        const char* pbegin = psz;
        while (::HexDigit(*psz) != -1)
            psz++;
        psz--;
        unsigned char* p1 = (unsigned char*)pn;
        unsigned char* pend = p1 + 16;
        while (psz >= pbegin && p1 < pend)
        {
            *p1 = ::HexDigit(*psz--);
            if (psz >= pbegin)
            {
                *p1 |= ((unsigned char)::HexDigit(*psz--) << 4);
                p1++;
            }
        }
    }

    void SetHex(const std::string& str)
    {
        SetHex(str.c_str());
    }

    CUDA_CALLABLE unsigned int size() const
    {
        return sizeof(pn);
    }

    /**
     * Returns the position of the highest bit set plus one, or zero if the
     * value is zero.
     */
    CUDA_CALLABLE unsigned int bitlength() const
    {
        for(int pos = WIDTH - 1; pos >= 0; pos--)
        {
            if (pn[pos])
            {
#ifdef __CUDA_ARCH__
                return 32 * (pos + 1) + 1 - __clz(pn[pos]);
#else
                return 32 * (pos + 1) + 1 - __builtin_clz(pn[pos]);
#endif
            }
        }
        return 0;
    }

    const base_uint& random( const base_uint &mod )
    {
        static std::mt19937 gen( time(NULL) );
        std::generate(pn, pn + WIDTH, std::ref(gen));
        pn[WIDTH - 1] &= ~(1U << 31);
        if( *this >= mod )
            *this %= mod;
        return *this;
    }

    CUDA_CALLABLE bool isLessZero() const
    {
        return pn[WIDTH - 1] & 0x80000000;
    }
};

#endif // _ARITH_H