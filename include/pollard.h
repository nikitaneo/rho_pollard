#include <elliptic.h>
#include <stdlib.h>
#include <fstream>
#include <cuda_runtime.h>

#define POLLARD_SET_COUNT 16
#define THREAD_PER_BLOCK 64
#define DEBUG

namespace cpu
{
template<typename T>
inline void iteration(  Point<T> &r,
                        T &c, T &d,
                        const std::vector<T> &a,
                        const std::vector<T> &b,
                        const std::vector<Point<T>> &R,
                        const EllipticCurve<T> &ec )
{
    unsigned index = r.getX() & 0xF;
    r = ec.add(r, R[index]);
    c += a[index];
    d += b[index];
}

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
            
        do
        {
            iteration(X1, c1, d1, a, b, R, ec);

            iteration(X2, c2, d2, a, b, R, ec);
            iteration(X2, c2, d2, a, b, R, ec);
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

        T c = c2 - c1; if(c < 0) c += order;
        T d = d1 - d2; if(d < 0) d += order;
        if(d == 0)
        {
            std::cerr << "[INFO] d1 == d2" << std::endl;
            continue;
        }

        T d_inv = detail::InvMod(d, order); if(d_inv < 0) d_inv += order;

        return (c * d_inv) % order;
    }
}
}   // namespace cpu

namespace gpu
{
__device__ int found_idx_d = -1;

template<typename T>
__device__ inline void iteration(   Point<T> &r,
                                    T &c, T &d,
                                    const T *a,
                                    const T *b,
                                    const Point<T> *R,
                                    const EllipticCurve<T> &ec )
{
    unsigned index = r.getX() & 0xF;
    r = ec.add(r, R[index]);
    c += a[index];
    d += b[index];
}

template<typename T>
__global__ void rho_pollard_kernel( const T *a,
                                    const T *b,
                                    const Point<T> *R,
                                    Point<T> *X1,
                                    Point<T> *X2,
                                    T *c1, T *c2, T *d1, T *d2,
                                    const EllipticCurve<T> ec )
{
    unsigned idx = threadIdx.x + blockDim.x * blockIdx.x;

    /*
    __shared__ T a_shared[POLLARD_SET_COUNT];
    __shared__ T b_shared[POLLARD_SET_COUNT];

    if(idx < POLLARD_SET_COUNT)
    {
        a_shared[idx] = a[idx];
        b_shared[idx] = b[idx];
    }

    __syncthreads();
    */

    Point<T> X1i = X1[idx];
    Point<T> X2i = X2[idx];
    T c1i = c1[idx], c2i = c2[idx], d1i = d1[idx], d2i = d2[idx];

    do
    {
        iteration(X1i, c1i, d1i, a, b, R, ec);

        iteration(X2i, c2i, d2i, a, b, R, ec);
        iteration(X2i, c2i, d2i, a, b, R, ec);
    }
    while(X1i != X2i && found_idx_d == -1);

    if(found_idx_d != -1)
        return;

    X1[idx] = X1i;
    X2[idx] = X2i;
    c1[idx] = c1i;
    c2[idx] = c2i;
    d1[idx] = d1i;
    d2[idx] = d2i;

    found_idx_d = idx;
}

// Solve Q = xP
template<typename T>
T rho_pollard(  const Point<T> &Q,
                const Point<T> &P,
                const T &order,
                const EllipticCurve<T> &ec )
{
    // Host memory
    T c1_host[THREAD_PER_BLOCK], c2_host[THREAD_PER_BLOCK], d1_host[THREAD_PER_BLOCK], d2_host[THREAD_PER_BLOCK];
    T a_host[POLLARD_SET_COUNT], b_host[POLLARD_SET_COUNT];
    Point<T> R_host[THREAD_PER_BLOCK];
    Point<T> X1_host[THREAD_PER_BLOCK];
    Point<T> X2_host[THREAD_PER_BLOCK];

    // Device memory
    T *c1_device = nullptr, *c2_device = nullptr, *d1_device = nullptr, *d2_device = nullptr;
    T *a_device = nullptr, *b_device = nullptr;
    Point<T> *X1_device = nullptr, *X2_device = nullptr, *R_device = nullptr;

    cudaMalloc((void **)&c1_device, sizeof(T) * THREAD_PER_BLOCK);
    cudaMalloc((void **)&c2_device, sizeof(T) * THREAD_PER_BLOCK);
    cudaMalloc((void **)&d1_device, sizeof(T) * THREAD_PER_BLOCK);
    cudaMalloc((void **)&d2_device, sizeof(T) * THREAD_PER_BLOCK);

    cudaMalloc((void **)&a_device, sizeof(T) * POLLARD_SET_COUNT);
 	cudaMalloc((void **)&b_device, sizeof(T) * POLLARD_SET_COUNT);

    cudaMalloc((void **)&R_device, sizeof(Point<T>) * POLLARD_SET_COUNT);

 	cudaMalloc((void **)&X1_device, sizeof(Point<T>) * THREAD_PER_BLOCK);
    cudaMalloc((void **)&X2_device, sizeof(Point<T>) * THREAD_PER_BLOCK);

    T result = 0;
    while( true )
    {
        for(unsigned i = 0; i < POLLARD_SET_COUNT; i++)
        {
            a_host[i].random(order);
            b_host[i].random(order);
            R_host[i] = ec.add(ec.mul(a_host[i], P), ec.mul(b_host[i], Q));
        }

        for(unsigned i = 0; i < THREAD_PER_BLOCK; i++)
        {
            c1_host[i] = c2_host[i].random(order);
            d1_host[i] = d2_host[i].random(order);  
            X2_host[i] = X1_host[i] = ec.add(ec.mul(c1_host[i], P), ec.mul(d1_host[i], Q));
        }
        
        cudaMemcpy((void *)c1_device, (const void*)c1_host, sizeof(T) * THREAD_PER_BLOCK, cudaMemcpyHostToDevice);
        cudaMemcpy((void *)c2_device, (const void*)c2_host, sizeof(T) * THREAD_PER_BLOCK, cudaMemcpyHostToDevice);
        cudaMemcpy((void *)d1_device, (const void*)d1_host, sizeof(T) * THREAD_PER_BLOCK, cudaMemcpyHostToDevice);
        cudaMemcpy((void *)d2_device, (const void*)d2_host, sizeof(T) * THREAD_PER_BLOCK, cudaMemcpyHostToDevice);

        cudaMemcpy((void *)a_device, (const void*)a_host, sizeof(T) * POLLARD_SET_COUNT, cudaMemcpyHostToDevice);
        cudaMemcpy((void *)b_device, (const void*)b_host, sizeof(T) * POLLARD_SET_COUNT, cudaMemcpyHostToDevice);
        cudaMemcpy((void *)R_device, (const void*)R_host, sizeof(Point<T>) * POLLARD_SET_COUNT, cudaMemcpyHostToDevice);

        cudaMemcpy((void *)X1_device, (const void*)X1_host, sizeof(Point<T>) * THREAD_PER_BLOCK, cudaMemcpyHostToDevice);
        cudaMemcpy((void *)X2_device, (const void*)X2_host, sizeof(Point<T>) * THREAD_PER_BLOCK, cudaMemcpyHostToDevice);

        int found_idx = -1;
        cudaMemcpyToSymbol(found_idx_d, &found_idx, sizeof(int));
        // kerner invocation here
        rho_pollard_kernel<<<1, THREAD_PER_BLOCK>>>(a_device, b_device, R_device, X1_device, X2_device, c1_device, c2_device, 
            d1_device, d2_device, ec);
        cudaDeviceSynchronize();
        cudaMemcpyFromSymbol(&found_idx, found_idx_d, sizeof(found_idx), 0, cudaMemcpyDeviceToHost);

#ifdef DEBUG
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess)
        {
            printf("%s\n", cudaGetErrorString(error));
            throw std::exception();
        }
#endif
        
        cudaMemcpy((void *)c1_host, (const void*)c1_device, sizeof(T) * THREAD_PER_BLOCK, cudaMemcpyDeviceToHost);
        cudaMemcpy((void *)c2_host, (const void*)c2_device, sizeof(T) * THREAD_PER_BLOCK, cudaMemcpyDeviceToHost);
        cudaMemcpy((void *)d1_host, (const void*)d1_device, sizeof(T) * THREAD_PER_BLOCK, cudaMemcpyDeviceToHost);
        cudaMemcpy((void *)d2_host, (const void*)d2_device, sizeof(T) * THREAD_PER_BLOCK, cudaMemcpyDeviceToHost);

        cudaMemcpy((void *)X1_host, (const void*)X1_device, sizeof(Point<T>) * THREAD_PER_BLOCK, cudaMemcpyDeviceToHost);
        cudaMemcpy((void *)X2_host, (const void*)X2_device, sizeof(Point<T>) * THREAD_PER_BLOCK, cudaMemcpyDeviceToHost);

        Point<T> X1 = X1_host[found_idx];
        Point<T> X2 = X2_host[found_idx];
        T c1 = c1_host[found_idx], c2 = c2_host[found_idx], d1 = d1_host[found_idx], d2 = d2_host[found_idx];

        c1 = c1 % order;
        d1 = d1 % order;
        c2 = c2 % order;
        d2 = d2 % order;

        if(ec.add(ec.mul(c2, P), ec.mul(d2, Q)) != X2 || ec.add(ec.mul(c1, P), ec.mul(d1, Q)) != X1 || !ec.check(X1))
        {
            std::cerr << "[INFO] c1 * P + d1 * Q != X1 or c2 * P + d2 * Q != X2 or X1 is not on curve." << std::endl;
            continue;
        }

        T c = c1 - c2; if(c < 0) c += order;
        T d = d2 - d1; if(d < 0) d += order;
        if(d == 0)
        {
            std::cerr << "[INFO] d1 == d2" << std::endl;
            continue;
        }

        T d_inv = detail::InvMod(d, order); if(d_inv < 0) d_inv += order;

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
