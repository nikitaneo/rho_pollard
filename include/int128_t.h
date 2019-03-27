// Copyright (c) 2009-2010 Satoshi Nakamoto
// Copyright (c) 2009-2018 The Bitcoin Core developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#ifndef BITCOIN_uint256_H
#define BITCOIN_uint256_H

#include <assert.h>
#include <limits>
#include <stdint.h>
#include <string>
#include <vector>
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

class int128_t
{
protected:
    static constexpr int WIDTH = 128 / 32;
    uint32_t pn[WIDTH];
public:

    CUDA_CALLABLE int128_t()
    {
        memset(pn, 0, sizeof(pn));
    }

    CUDA_CALLABLE int128_t(const int128_t& b)
    {
        memcpy(pn, b.pn, sizeof(pn));
    }

    CUDA_CALLABLE int128_t& operator=(const int128_t& b)
    {
        memcpy(pn, b.pn, sizeof(pn));
        return *this;
    }

    __host__ int128_t& operator=(const std::string &str)
    {
        SetHex(str.c_str());

        return *this;
    }

    CUDA_CALLABLE int128_t(int64_t b)
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

    __host__ explicit int128_t(const std::string& str)
    {
        SetHex(str.c_str());
    }

    CUDA_CALLABLE const int128_t operator~() const
    {
        int128_t ret;
        ret.pn[0] = ~pn[0];
        ret.pn[1] = ~pn[1];
        ret.pn[2] = ~pn[2];
        ret.pn[3] = ~pn[3];
        return ret;
    }

    CUDA_CALLABLE const int128_t operator-() const
    {
        int128_t ret = ~(*this);
        ++ret;
        return ret;
    }

    CUDA_CALLABLE int128_t& operator=(int64_t b)
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
        return *this;
    }

    CUDA_CALLABLE int128_t& operator^=(const int128_t& b)
    {
        pn[0] ^= b.pn[0];
        pn[1] ^= b.pn[1];
        pn[2] ^= b.pn[2];
        pn[3] ^= b.pn[3];
        return *this;
    }

    CUDA_CALLABLE int128_t& operator&=(const int128_t& b)
    {
        pn[0] &= b.pn[0];
        pn[1] &= b.pn[1];
        pn[2] &= b.pn[2];
        pn[3] &= b.pn[3];

        return *this;
    }

    CUDA_CALLABLE int128_t& operator|=(const int128_t& b)
    {
        pn[0] |= b.pn[0];
        pn[1] |= b.pn[1];
        pn[2] |= b.pn[2];
        pn[3] |= b.pn[3];
        return *this;
    }

    CUDA_CALLABLE int128_t& operator^=(uint64_t b)
    {
        pn[0] ^= (unsigned int)b;
        pn[1] ^= (unsigned int)(b >> 32);
        return *this;
    }

    CUDA_CALLABLE int128_t& operator|=(uint64_t b)
    {
        pn[0] |= (unsigned int)b;
        pn[1] |= (unsigned int)(b >> 32);
        return *this;
    }

    CUDA_CALLABLE int128_t& operator<<=(unsigned int shift);
    CUDA_CALLABLE int128_t& operator>>=(unsigned int shift);

    CUDA_CALLABLE int128_t& operator%=(const int128_t& b)
    {
        *this = *this % b;
        return *this;
    }

    CUDA_CALLABLE int128_t& operator+=(const int128_t& b)
    {
        uint64_t n = 0;

        n = uint64_t(pn[0]) + b.pn[0];
        pn[0] = n & 0xffffffff;

        n = (n >> 32) + pn[1] + b.pn[1];
        pn[1] = n & 0xffffffff;

        n = (n >> 32) + pn[2] + b.pn[2];
        pn[2] = n & 0xffffffff;
        
        n = (n >> 32) + pn[3] + b.pn[3];
        pn[3] = n & 0xffffffff;
    
        return *this;
    }

    CUDA_CALLABLE int128_t& operator-=(const int128_t& b)
    {
        *this += -b;
        return *this;
    }

    CUDA_CALLABLE int128_t& operator*=(const int128_t& b);
    CUDA_CALLABLE int128_t& operator/=(const int128_t& b);

    CUDA_CALLABLE int128_t& operator++()
    {
        // prefix operator
        int i = 0;
        while (i < WIDTH && ++pn[i] == 0)
            ++i;
        return *this;
    }

    CUDA_CALLABLE const int128_t operator++(int)
    {
        // postfix operator
        const int128_t ret = *this;
        ++(*this);
        return ret;
    }

    CUDA_CALLABLE int128_t& operator--()
    {
        // prefix operator
        int i = 0;
        while (i < WIDTH && --pn[i] == 0xffffffff)
            ++i;
        return *this;
    }

    CUDA_CALLABLE const int128_t operator--(int)
    {
        // postfix operator
        const int128_t ret = *this;
        --(*this);
        return ret;
    }

    CUDA_CALLABLE int CompareTo(const int128_t& b) const;

    CUDA_CALLABLE friend inline const int128_t operator%(const int128_t& a, const int128_t& b) { return a - b * ( a / b ); }
    CUDA_CALLABLE friend inline const int128_t operator+(const int128_t& a, const int128_t& b) { return int128_t(a) += b; }
    CUDA_CALLABLE friend inline const int128_t operator-(const int128_t& a, const int128_t& b) { return int128_t(a) -= b; }
    CUDA_CALLABLE friend inline const int128_t operator*(const int128_t& a, const int128_t& b) { return int128_t(a) *= b; }
    CUDA_CALLABLE friend inline const int128_t operator/(const int128_t& a, const int128_t& b) { return int128_t(a) /= b; }
    CUDA_CALLABLE friend inline const int128_t operator|(const int128_t& a, const int128_t& b) { return int128_t(a) |= b; }
    CUDA_CALLABLE friend inline const int128_t operator&(const int128_t& a, const int128_t& b) { return int128_t(a) &= b; }
    CUDA_CALLABLE friend inline uint32_t operator&(const int128_t& a, const uint32_t b) { return a.pn[0] & b; }
    CUDA_CALLABLE friend inline const int128_t operator^(const int128_t& a, const int128_t& b) { return int128_t(a) ^= b; }
    CUDA_CALLABLE friend inline const int128_t operator>>(const int128_t& a, int shift) { return int128_t(a) >>= shift; }
    CUDA_CALLABLE friend inline const int128_t operator<<(const int128_t& a, int shift) { return int128_t(a) <<= shift; }

    CUDA_CALLABLE friend inline bool operator==(const int128_t& a, const int128_t& b)
    {
        for(unsigned i = 0; i < 3; ++i)
        {
            if(a.pn[i] != b.pn[i])
                return false;
        }
        return true;
    }

    CUDA_CALLABLE friend inline bool operator!=(const int128_t& a, const int128_t& b) { return !(a == b); }
    CUDA_CALLABLE friend inline bool operator>(const int128_t& a, const int128_t& b) { return a.CompareTo(b) > 0; }
    CUDA_CALLABLE friend inline bool operator<(const int128_t& a, const int128_t& b) { return a.CompareTo(b) < 0; }
    CUDA_CALLABLE friend inline bool operator>=(const int128_t& a, const int128_t& b) { return a.CompareTo(b) >= 0; }
    CUDA_CALLABLE friend inline bool operator<=(const int128_t& a, const int128_t& b) { return a.CompareTo(b) <= 0; }
    __host__ friend inline std::ostream& operator<<(std::ostream& out, const int128_t &a) { return out << a.GetHex(); }

    __host__ std::string GetHex() const;
    void SetHex(const char* psz);

    /**
     * Returns the position of the highest bit set plus one, or zero if the
     * value is zero.
     */
    CUDA_CALLABLE unsigned int bits() const
    {
        for (int pos = 3; pos >= 0; --pos)
        {
            if (pn[pos])
            {
                unsigned p = 0;
                uint32_t y = pn[pos];
                while (0 != y)
                    ++p, y >>= 1;
                return (pos << 5) + p;
            }
        }
        return 0;
    }

    __host__ const int128_t& random( const int128_t &mod );
};

CUDA_CALLABLE int128_t& int128_t::operator<<=(unsigned int shift)
{
    int128_t a(*this);
    memset(pn, 0, sizeof(pn));
    int k = shift / 32;
    shift = shift % 32;
    for (int i = 0; i < 4; ++i)
    {
        if (i + k + 1 < WIDTH && shift != 0)
            pn[i + k + 1] |= (a.pn[i] >> (32 - shift));
        if (i + k < WIDTH)
            pn[i + k] |= (a.pn[i] << shift);
    }
    return *this;
}

CUDA_CALLABLE int128_t& int128_t::operator>>=(unsigned int shift)
{
    int128_t a(*this);
    memset(pn, 0, sizeof(pn));
    int k = shift / 32;
    shift = shift % 32;

    for(unsigned i = 0; i + k < 4; ++i)
        pn[i] = a.pn[i + k];

    uint32_t mask_least = (1 << shift) - 1;
    uint32_t carry1 = 0, carry2 = 0;
    for(int i = 3 - k; i >= 0; --i)
    {
        carry1 = pn[i] & mask_least;
        pn[i] >>= shift;
        pn[i] |= (carry2 << (32 - shift));
        carry2 = carry1;
    }

    return *this;
}

int128_t& int128_t::operator*=(const int128_t& b)
{
    int128_t a;

    uint64_t n = 0;
    
    n = a.pn[0] + (uint64_t)pn[0] * b.pn[0];
    a.pn[0] = n & 0xffffffff;

    n = (n >> 32) + a.pn[1] + (uint64_t)pn[0] * b.pn[1];
    a.pn[1] = n & 0xffffffff;

    n = (n >> 32) + a.pn[2] + (uint64_t)pn[0] * b.pn[2];
    a.pn[2] = n & 0xffffffff;

    n = (n >> 32) + a.pn[3] + (uint64_t)pn[0] * b.pn[3];
    a.pn[3] = n & 0xffffffff;


    n = a.pn[1] + (uint64_t)pn[1] * b.pn[0];
    a.pn[1] = n & 0xffffffff;

    n = (n >> 32) + a.pn[2] + (uint64_t)pn[1] * b.pn[1];
    a.pn[2] = n & 0xffffffff;

    n = (n >> 32) + a.pn[3] + (uint64_t)pn[1] * b.pn[2];
    a.pn[3] = n & 0xffffffff;


    n = a.pn[2] + (uint64_t)pn[2] * b.pn[0];
    a.pn[2] = n & 0xffffffff;

    n = (n >> 32) + a.pn[3] + (uint64_t)pn[2] * b.pn[1];
    a.pn[3] = n & 0xffffffff;


    n = a.pn[3] + (uint64_t)pn[3] * b.pn[0];
    a.pn[3] = n & 0xffffffff;

    *this = a;
    return *this;
}

CUDA_CALLABLE int128_t& int128_t::operator/=(const int128_t& b)
{
    int sign = b > 0 && (*this) > 0 ? 1 : -1;
    int128_t div = b > 0 ? b : -b;     // make a copy, so we can shift.
    int128_t num = *this > 0 ? *this : -(*this); // make a copy, so we can subtract.
    memset(pn, 0, sizeof(pn));
    int num_bits = num.bits();
    int div_bits = div.bits();
    assert(div_bits != 0 && "Division by zero");
    if (div_bits > num_bits) // the result is certainly 0.
        return *this;
    int shift = num_bits - div_bits;
    div <<= shift; // shift so that div and num align.
    while (shift >= 0)
    {
        if (num >= div)
        {
            num -= div;
            pn[shift / 32] |= (1 << (shift & 31)); // set a bit of the result.
        }
        div >>= 1; // shift back.
        --shift;
    }

    if( sign < 0 )
        (*this) = -(*this);
    return (*this);
}

CUDA_CALLABLE int int128_t::CompareTo(const int128_t& b) const
{
    bool lhsSign = pn[3] >> 31;
    bool rhsSign = b.pn[3] >> 31;

    if( lhsSign && !rhsSign )
    {
        return -1;
    }
    else if( !lhsSign && rhsSign )
    {
        return 1;
    }
    else if( lhsSign && rhsSign )
    {
        int i = 3;
        while(pn[i] == b.pn[i] && i > 0)
            --i;
        if(pn[i] > b.pn[i])
            return -1;
        else if(pn[i] == b.pn[i])
            return 0;
        else
            return 1;
    }
    else
    {
        int i = 3;
        while(pn[i] == b.pn[i] && i > 0)
            --i;
        if(pn[i] > b.pn[i])
            return 1;
        else if(pn[i] == b.pn[i])
            return 0;
        else
            return -1;
    }
}

__host__ std::string int128_t::GetHex() const
{
    return HexStr(std::reverse_iterator<const uint8_t*>((const uint8_t*)pn + sizeof(pn)), std::reverse_iterator<const uint8_t*>((const uint8_t*)pn), true);
}

void int128_t::SetHex(const char* psz)
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
    unsigned char* pend = p1 + WIDTH * 4;
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

const int128_t& int128_t::random( const int128_t &mod )
{
    static std::mt19937 gen( time(NULL) );
    std::generate(pn, pn + WIDTH, std::ref(gen));
    pn[WIDTH - 1] &= ~(1U << 31);
    if( *this >= mod )
        *this %= mod;
    return *this;
}

#endif // BITCOIN_uint256_H