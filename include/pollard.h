#include <elliptic.h>
#include <stdlib.h>
#include <fstream>
#include <cuda_runtime.h>

#define POLLARD_SET_COUNT 16
#define THREADS 64
#define DEBUG

namespace cpu
{
// Solve Q = xP
template<typename T>
T rho_pollard( const Point<T> &Q, const Point<T> &P, const T &order, const EllipticCurve<T> &ec )
{
    T c1, c2, d1, d2;
    std::vector<T> a(POLLARD_SET_COUNT);
    std::vector<T> b(POLLARD_SET_COUNT);
    std::vector<Point<T>> R(POLLARD_SET_COUNT, P);
    
    while( true )
    {
        for(unsigned i = 0; i < POLLARD_SET_COUNT; i++)
        {
            a[i].random(order);
            b[i].random(order) ;
            R[i] = ec.add(ec.mul(a[i], P), ec.mul(b[i], Q));
        }

        c1 = c2.random(order);
        d1 = d2.random(order);
        Point<T> X1 = ec.add(ec.mul(c1, P), ec.mul(d1, Q));
        Point<T> X2 = X1;
        
        unsigned index = 0;
        do
        {
            index = X1.getX() & 0xF;
            X1 = ec.add(X1, R[index]);
            c1 += a[index]; if(c1 >= order) c1 -= order;
            d1 += b[index]; if(d1 >= order) d1 -= order;

            index = X2.getX() & 0xF;
            X2 = ec.add(X2, R[index]);
            c2 += a[index]; if(c2 >= order) c2 -= order;
            d2 += b[index]; if(d2 >= order) d2 -= order;
        
            index = X2.getX() & 0xF;
            X2 = ec.add(X2, R[index]);
            c2 += a[index]; if(c2 >= order) c2 -= order;
            d2 += b[index]; if(d2 >= order) d2 -= order;
        }
        while(X1 != X2);

        c1 = c1 % order;
        d1 = d1 % order;
        c2 = c2 % order;
        d2 = d2 % order;

        if(ec.add(ec.mul(c2, P), ec.mul(d2, Q)) != X2 || ec.add(ec.mul(c1, P), ec.mul(d1, Q)) != X1 || !ec.check(X1))
        {
            std::cerr << "[INFO] c1 * P + d1 * Q != X1 or c2 * P + d2 * Q != X2 or X1 is not on curve." << std::endl;
            continue;
        }

        T c = c2 - c1; if(c.isLessZero()) c += order;
        T d = d1 - d2; if(d.isLessZero()) d += order;
        if(d == 0)
        {
            std::cerr << "[INFO] d1 == d2" << std::endl;
            continue;
        }

        T d_inv = detail::invmod(d, order); if(d_inv.isLessZero()) d_inv += order;
        assert((c * d_inv) % order != 0);
        return (c * d_inv) % order;
    }
}
}   // namespace cpu

namespace gpu
{
__device__ bool found = false;

template<typename T>
__device__ inline void iteration(   Point<T> &r,
                                    uint32_t &c, uint32_t &d,
                                    const uint32_t *a,
                                    const uint32_t *b,
                                    const Point<T> *R,
                                    const EllipticCurve<T> &ec,
                                    unsigned rem,
                                    uint32_t *pool )
{
    unsigned index = r.getX() & 0xF;
    r = ec.add(r, R[index]);

    uint64_t n1, n2;
    ///////////////////////////////////////

    n1 = uint64_t(c) + a[index*4 + rem];
    pool[rem] = n1 & 0xffffffff;

    n2 = uint64_t(d) + b[index*4 + rem];
    pool[4 + rem] = n2 & 0xffffffff;

    __syncthreads();

    if(rem != 3)
    {
        pool[rem + 1] += (n1 >> 32);
        pool[4 + rem + 1] += (n2 >> 32);
    }

    __syncthreads();

    c = pool[rem];
    d = pool[4 + rem];
}

template<typename T>
__global__ void rho_pollard_kernel( const uint32_t *a,
                                    const uint32_t *b,
                                    const Point<T> *R,
                                    Point<T> *X1,
                                    Point<T> *X2,
                                    uint32_t *c1, uint32_t *c2, uint32_t *d1, uint32_t *d2,
                                    bool *indicators,
                                    const EllipticCurve<T> ec )
{
    __shared__ uint32_t pool[8];

    unsigned idx = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned r = threadIdx.x;

    Point<T> X1i = X1[idx];
    Point<T> X2i = X2[idx];
    uint32_t c1i = c1[idx], c2i = c2[idx], d1i = d1[idx], d2i = d2[idx];

    do
    {
        iteration(X1i, c1i, d1i, a, b, R, ec, r, pool);

        iteration(X2i, c2i, d2i, a, b, R, ec, r, pool);
        iteration(X2i, c2i, d2i, a, b, R, ec, r, pool);
    }
    while(X1i != X2i && !found);

    if(found)
        return;

    X1[idx] = X1i;
    X2[idx] = X2i;

    c1[idx] = c1i;
    c2[idx] = c2i;
    d1[idx] = d1i;
    d2[idx] = d2i;

    if(r == 0)
    {
        indicators[idx] = true;
        found = true;
    }
}

// Solve Q = xP
template<typename T>
T rho_pollard(  const Point<T> &Q,
                const Point<T> &P,
                const T &order,
                const EllipticCurve<T> &ec )
{
    // Host memory
    uint32_t c1_host[THREADS], c2_host[THREADS], d1_host[THREADS], d2_host[THREADS];
    uint32_t a_host[POLLARD_SET_COUNT * 4], b_host[POLLARD_SET_COUNT * 4];
    Point<T> R_host[THREADS];
    Point<T> X1_host[THREADS];
    Point<T> X2_host[THREADS];
    bool indicator_host[THREADS];

    // Device memory
    uint32_t *c1_device = nullptr, *c2_device = nullptr, *d1_device = nullptr, *d2_device = nullptr;
    uint32_t *a_device = nullptr, *b_device = nullptr;
    Point<T> *X1_device = nullptr, *X2_device = nullptr, *R_device = nullptr;
    bool *indicator_device = nullptr;

    cudaMalloc((void **)&c1_device, sizeof(c1_host));
    cudaMalloc((void **)&c2_device, sizeof(c2_host));
    cudaMalloc((void **)&d1_device, sizeof(d1_host));
    cudaMalloc((void **)&d2_device, sizeof(d2_host));

    cudaMalloc((void **)&a_device, sizeof(a_host));
 	cudaMalloc((void **)&b_device, sizeof(b_host));

    cudaMalloc((void **)&R_device, sizeof(R_host));

 	cudaMalloc((void **)&X1_device, sizeof(X1_host));
    cudaMalloc((void **)&X2_device, sizeof(X2_host));

    cudaMalloc((void **)&indicator_device, sizeof(indicator_host));

    T result = 0;
    while( true )
    {
        for(unsigned i = 0; i < POLLARD_SET_COUNT; i++)
        {
            T a; a.random(order);
            T b; b.random(order);
            R_host[i] = ec.add(ec.mul(a, P), ec.mul(b, Q));
            for(unsigned j = 0; j < 4; j++)
            {
                a_host[i*4+j] = a[j];
                b_host[i*4+j] = b[j];
            }
        }

        for(unsigned i = 0; i < THREADS / 4; i++)
        {
            T c; c.random(order);
            T d; d.random(order);

            Point<T> point = ec.add(ec.mul(c, P), ec.mul(d, Q));
            for(unsigned j = 0; j < 4; j++)
            {
                c1_host[i*4+j] = c2_host[i*4+j] = c[j];
                d1_host[i*4+j] = d2_host[i*4+j] = d[j];
                X1_host[i*4+j] = X2_host[i*4+j] = point;
            }
        }
        memset(indicator_host, false, sizeof(indicator_host));
        
        cudaMemcpy((void *)c1_device, (const void*)c1_host, sizeof(c1_host), cudaMemcpyHostToDevice);
        cudaMemcpy((void *)c2_device, (const void*)c2_host, sizeof(c2_host), cudaMemcpyHostToDevice);
        cudaMemcpy((void *)d1_device, (const void*)d1_host, sizeof(d1_host), cudaMemcpyHostToDevice);
        cudaMemcpy((void *)d2_device, (const void*)d2_host, sizeof(d2_host), cudaMemcpyHostToDevice);

        cudaMemcpy((void *)a_device, (const void*)a_host, sizeof(a_host), cudaMemcpyHostToDevice);
        cudaMemcpy((void *)b_device, (const void*)b_host, sizeof(b_host), cudaMemcpyHostToDevice);
        cudaMemcpy((void *)R_device, (const void*)R_host, sizeof(R_host), cudaMemcpyHostToDevice);

        cudaMemcpy((void *)X1_device, (const void*)X1_host, sizeof(X1_host), cudaMemcpyHostToDevice);
        cudaMemcpy((void *)X2_device, (const void*)X2_host, sizeof(X2_host), cudaMemcpyHostToDevice);

        cudaMemcpy((void *)indicator_device, (const void*)indicator_host, sizeof(indicator_host), cudaMemcpyHostToDevice);

        bool tmp = false;
        cudaMemcpyToSymbol(found, &tmp, sizeof(bool));
        // kerner invocation here
        rho_pollard_kernel<<<THREADS / 4, 4>>>(a_device, b_device, R_device, X1_device, X2_device, c1_device, c2_device, 
            d1_device, d2_device, indicator_device, ec);
        cudaDeviceSynchronize();

#ifdef DEBUG
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess)
        {
            printf("%s\n", cudaGetErrorString(error));
            throw std::exception();
        }
#endif
        
        cudaMemcpy((void *)c1_host, (const void*)c1_device, sizeof(c1_host), cudaMemcpyDeviceToHost);
        cudaMemcpy((void *)c2_host, (const void*)c2_device, sizeof(c2_host), cudaMemcpyDeviceToHost);
        cudaMemcpy((void *)d1_host, (const void*)d1_device, sizeof(d1_host), cudaMemcpyDeviceToHost);
        cudaMemcpy((void *)d2_host, (const void*)d2_device, sizeof(d2_host), cudaMemcpyDeviceToHost);

        cudaMemcpy((void *)X1_host, (const void*)X1_device, sizeof(X1_host), cudaMemcpyDeviceToHost);
        cudaMemcpy((void *)X2_host, (const void*)X2_device, sizeof(X1_host), cudaMemcpyDeviceToHost);

        cudaMemcpy((void *)indicator_host, (const void*)indicator_device, sizeof(indicator_host), cudaMemcpyDeviceToHost);

        auto iter = std::find(indicator_host, indicator_host + THREADS, true);
        unsigned idx = iter - std::begin(indicator_host);

        Point<T> X1 = X1_host[idx];
        Point<T> X2 = X2_host[idx];

        T c1, c2, d1, d2;
        for(unsigned i = 0; i < 4; i++)
        {
            c1[i] = c1_host[idx + i];
            c2[i] = c2_host[idx + i];
            d1[i] = d1_host[idx + i];
            d2[i] = d2_host[idx + i];
        }

        c1 = c1 % order;
        d1 = d1 % order;
        c2 = c2 % order;
        d2 = d2 % order;

        if(ec.add(ec.mul(c2, P), ec.mul(d2, Q)) != X2)
        {
            std::cerr << "[INFO] c2 * P + d2 * Q != X2" << std::endl;
            std::cerr << "c2 = " << c2 << "; d2 = " << d2 << "; X2 = " << X2 << std::endl;
            assert(false);
            continue;
        }

        if(ec.add(ec.mul(c1, P), ec.mul(d1, Q)) != X1)
        {
            std::cerr << "[INFO] c1 * P + d1 * Q != X1" << std::endl;
            assert(false);
            continue;
        }

        if(!ec.check(X1))
        {
            std::cerr << "X1 is not on curve." << std::endl;
            assert(false);
            continue;
        }

        T c = c1 - c2; if(c.isLessZero()) c += order;
        T d = d2 - d1; if(d.isLessZero()) d += order;
        if(d == 0)
        {
            std::cerr << "[INFO] d1 == d2" << std::endl;
            continue;
        }

        T d_inv = detail::invmod(d, order); if(d_inv.isLessZero()) d_inv += order;

        result = (c * d_inv) % order;
        break;
    }
    
    cudaFree(c1_device);
	cudaFree(c2_device);
	cudaFree(d1_device);
    cudaFree(d2_device);

	cudaFree(X1_device);
	cudaFree(X2_device);

	cudaFree(a_device);
	cudaFree(b_device);

    return result;
}
}   // namespace gpu
