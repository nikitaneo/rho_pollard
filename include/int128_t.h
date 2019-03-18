// Copyright (c) 2009-2010 Satoshi Nakamoto
// Copyright (c) 2009-2018 The Bitcoin Core developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#ifndef BITCOIN_uint256_H
#define BITCOIN_uint256_H

#include <assert.h>
#include <limits>
#include <stdexcept>
#include <stdint.h>
#include <string>
#include <vector>

#define CUDA_CALLABLE __host__ __device__

CUDA_CALLABLE char ToLower(char c)
{
    return (c >= 'A' && c <= 'Z' ? (c - 'A') + 'a' : c);
}

inline CUDA_CALLABLE bool IsSpace(char c) noexcept 
{
    return c == ' ' || c == '\f' || c == '\n' || c == '\r' || c == '\t' || c == '\v';
}

CUDA_CALLABLE signed char HexDigit(char c)
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

/*
template<typename T>
CUDA_CALLABLE std::string HexStr(const T itbegin, const T itend, bool fSpaces=false)
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
*/

/** Template base class for unsigned big integers. */
template<unsigned int BITS>
class base_uint
{
protected:
    static constexpr int WIDTH = BITS / 32;
    uint32_t pn[WIDTH];
public:

    CUDA_CALLABLE base_uint()
    {
        static_assert(BITS/32 > 0 && BITS%32 == 0, "Template parameter BITS must be a positive multiple of 32.");
        memset(pn, 0, sizeof(pn));
    }

    CUDA_CALLABLE base_uint(const base_uint& b)
    {
        static_assert(BITS/32 > 0 && BITS%32 == 0, "Template parameter BITS must be a positive multiple of 32.");

        memcpy(pn, b.pn, sizeof(pn));
    }

    CUDA_CALLABLE base_uint& operator=(const base_uint& b)
    {
        memcpy(pn, b.pn, sizeof(pn));
        return *this;
    }

    CUDA_CALLABLE base_uint& operator=(const char *str)
    {
        static_assert(BITS/32 > 0 && BITS%32 == 0, "Template parameter BITS must be a positive multiple of 32.");

        SetHex(str);
    }

    CUDA_CALLABLE base_uint(int64_t b)
    {
        static_assert(BITS/32 > 0 && BITS%32 == 0, "Template parameter BITS must be a positive multiple of 32.");
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

    CUDA_CALLABLE explicit base_uint(const std::string& str);

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

    CUDA_CALLABLE base_uint& operator=(int64_t b)
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

    CUDA_CALLABLE base_uint& operator<<=(unsigned int shift);
    CUDA_CALLABLE base_uint& operator>>=(unsigned int shift);

    CUDA_CALLABLE base_uint& operator%=(const base_uint& b)
    {
        *this = *this % b;
        return *this;
    }

    CUDA_CALLABLE base_uint& operator+=(const base_uint& b)
    {
        uint64_t carry = 0;
        for (int i = 0; i < WIDTH; i++)
        {
            uint64_t n = carry + pn[i] + b.pn[i];
            pn[i] = n & 0xffffffff;
            carry = n >> 32;
        }
        return *this;
    }

    CUDA_CALLABLE base_uint& operator-=(const base_uint& b)
    {
        *this += -b;
        return *this;
    }

    CUDA_CALLABLE base_uint& operator*=(const base_uint& b);
    CUDA_CALLABLE base_uint& operator/=(const base_uint& b);

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
        while (i < WIDTH && --pn[i] == uint32_t(-1))
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

    CUDA_CALLABLE int CompareTo(const base_uint& b) const;

    CUDA_CALLABLE friend inline const base_uint operator%(const base_uint& a, const base_uint& b) { return a - ( a / b ) * b; }
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
    CUDA_CALLABLE friend inline bool operator==(const base_uint& a, const base_uint& b) { return memcmp(a.pn, b.pn, sizeof(a.pn)) == 0; }
    CUDA_CALLABLE friend inline bool operator!=(const base_uint& a, const base_uint& b) { return memcmp(a.pn, b.pn, sizeof(a.pn)) != 0; }
    CUDA_CALLABLE friend inline bool operator>(const base_uint& a, const base_uint& b) { return a.CompareTo(b) > 0; }
    CUDA_CALLABLE friend inline bool operator<(const base_uint& a, const base_uint& b) { return a.CompareTo(b) < 0; }
    CUDA_CALLABLE friend inline bool operator>=(const base_uint& a, const base_uint& b) { return a.CompareTo(b) >= 0; }
    CUDA_CALLABLE friend inline bool operator<=(const base_uint& a, const base_uint& b) { return a.CompareTo(b) <= 0; }
    // CUDA_CALLABLE friend inline std::ostream& operator<<(std::ostream& out, const base_uint &a) { return out << a.GetHex(); }

    // CUDA_CALLABLE std::string GetHex() const;
    CUDA_CALLABLE void SetHex(const char* psz);

    /**
     * Returns the position of the highest bit set plus one, or zero if the
     * value is zero.
     */
    CUDA_CALLABLE unsigned int bits() const;
};

template <unsigned int BITS>
CUDA_CALLABLE base_uint<BITS>& base_uint<BITS>::operator<<=(unsigned int shift)
{
    base_uint<BITS> a(*this);
    for (int i = 0; i < WIDTH; i++)
        pn[i] = 0;
    int k = shift / 32;
    shift = shift % 32;
    for (int i = 0; i < WIDTH; i++)
    {
        if (i + k + 1 < WIDTH && shift != 0)
            pn[i + k + 1] |= (a.pn[i] >> (32 - shift));
        if (i + k < WIDTH)
            pn[i + k] |= (a.pn[i] << shift);
    }
    return *this;
}

template <unsigned int BITS>
CUDA_CALLABLE base_uint<BITS>& base_uint<BITS>::operator>>=(unsigned int shift)
{
    base_uint<BITS> a(*this);
    for (int i = 0; i < WIDTH; i++)
        pn[i] = 0;
    int k = shift / 32;
    shift = shift % 32;
    for (int i = 0; i < WIDTH; i++)
    {
        if (i - k - 1 >= 0 && shift != 0)
            pn[i - k - 1] |= (a.pn[i] << (32 - shift));
        if (i - k >= 0)
            pn[i - k] |= (a.pn[i] >> shift);
    }
    return *this;
}

template <unsigned int BITS>
CUDA_CALLABLE base_uint<BITS>& base_uint<BITS>::operator*=(const base_uint& b)
{
    base_uint<BITS> a;
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
    *this = a;
    return *this;
}

template <unsigned int BITS>
CUDA_CALLABLE base_uint<BITS>& base_uint<BITS>::operator/=(const base_uint& b)
{
    int sign = b * (*this) > 0 ? 1 : -1;
    base_uint<BITS> div = b < 0 ? -b : b;     // make a copy, so we can shift.
    base_uint<BITS> num = *this < 0 ? -(*this) : *this; // make a copy, so we can subtract.
    memset(pn, 0, sizeof(pn));
    int num_bits = num.bits();
    int div_bits = div.bits();
    assert(div_bits != 0);
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
        shift--;
    }
    // num now contains the remainder of the division.
    if( sign < 0 )
        (*this) = -(*this);
    return (*this);
}

template <unsigned int BITS>
CUDA_CALLABLE int base_uint<BITS>::CompareTo(const base_uint<BITS>& b) const
{
    // lhs negative, rhs positive
    bool lhsSign = this->pn[WIDTH-1] >> 31;
    bool rhsSign = b.pn[WIDTH-1] >> 31;

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
        for (int i = WIDTH - 1; i >= 0; --i)
        {
            if (pn[i] > b.pn[i])
                return -1;
            else if (pn[i] < b.pn[i])
                return 1;
        }
    }
    else
    {
        for (int i = WIDTH - 1; i >= 0; --i)
        {
            if (pn[i] < b.pn[i])
                return -1;
            else if (pn[i] > b.pn[i])
                return 1;
        }
    }
    return 0;
}

/*
template <unsigned int BITS>
CUDA_CALLABLE std::string base_uint<BITS>::GetHex() const
{
    return HexStr(std::reverse_iterator<const uint8_t*>((const uint8_t*)pn + sizeof(pn)), std::reverse_iterator<const uint8_t*>((const uint8_t*)pn), true);
}
*/

template <unsigned int BITS>
CUDA_CALLABLE void base_uint<BITS>::SetHex(const char* psz)
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

template <unsigned int BITS>
CUDA_CALLABLE unsigned int base_uint<BITS>::bits() const
{
    for (int pos = WIDTH - 1; pos >= 0; pos--)
    {
        if (pn[pos])
        {
            unsigned p = 0;
            uint32_t y = pn[pos];
            while (0 != y)
                p++, y >>= 1;
            return (pos << 5) + p;
        }
    }
    return 0;
}

typedef base_uint<128> int128_t;

#endif // BITCOIN_uint256_H