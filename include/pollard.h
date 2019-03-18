#include <elliptic.h>
#include <stdlib.h>
#include <fstream>
#include <cuda_runtime.h>

#define POLLARD_SET_COUNT 16
#define THREAD_PER_BLOCK 128
#define DEBUG

namespace cpu
{
template<typename T>
inline void iteration(  typename EllipticCurve<T>::Point &r,
                        T &c, T &d,
                        const std::vector<T> &a,
                        const std::vector<T> &b,
                        // const std::vector<typename EllipticCurve<T>::Point> &R,
                        const T& order,
                        const typename EllipticCurve<T>::Point &P,
                        const typename EllipticCurve<T>::Point &Q )
{
    unsigned index = r.x().i() & 0xF;
    //r += R[index];
    c = (c + a[index]) % order;
    d = (d + b[index]) % order;
    r = c * P + d * Q;
}

// Solve Q = xP
template<typename T>
T rho_pollard( const typename EllipticCurve<T>::Point &Q, const typename EllipticCurve<T>::Point &P, const T &order )
{
    // std::independent_bits_engine<std::mt19937, sizeof(T)*4, T> gen;
    // gen.seed(time(NULL));

    T c1, c2, d1, d2;
    std::vector<T> a(POLLARD_SET_COUNT);
    std::vector<T> b(POLLARD_SET_COUNT);
    // std::vector<typename EllipticCurve<T>::Point> R(POLLARD_SET_COUNT, P);
    
    srand(time(NULL));
    while( true )
    {
        for(unsigned i = 0; i < POLLARD_SET_COUNT; i++)
        {
            a[i] = rand() % order;
            b[i] = rand() % order;
            // R[i] = a[i] * P + b[i] * Q;
        }
    
        c1 = c2 = rand() % order;
        d1 = d2 = rand() % order;
        typename EllipticCurve<T>::Point X1 = c1 * P + d1 * Q;
        typename EllipticCurve<T>::Point X2 = X1;
            
        do
        {
            iteration(X1, c1, d1, a, b, order, P, Q);

            iteration(X2, c2, d2, a, b, order, P, Q);
            iteration(X2, c2, d2, a, b, order, P, Q);
        }
        while(X1 != X2);

        if(c1 * P + d1 * Q != X1 || c2 * P + d2 * Q != X2 || !X1.check())
        {
            continue;
        }

        T c = c1 - c2; if(c < 0) c += order;
        T d = d2 - d1; if(d < 0) d += order;
        if(d == 0)
            continue;
        T d_inv = detail::InvMod(d, order); if(d_inv < 0) d_inv += order;

        T result = (c * d_inv) % order;
        if(result * P == Q)
            return result;
    }
}
}   // namespace cpu

namespace gpu
{
__device__ int found_idx_d = -1;

template<typename T>
__device__ inline void iteration(   typename EllipticCurve<T>::Point &r,
                                    T &c, T &d,
                                    const T *a,
                                    const T *b,
                                    // const typename EllipticCurve<T>::Point *R,
                                    const T& order,
                                    const typename EllipticCurve<T>::Point P,
                                    const typename EllipticCurve<T>::Point Q  )
{
    unsigned index = r.x().i() & 0xF;
    // r += R[index];
    c = (c + a[index]) % order;
    d = (d + b[index]) % order;
    r = c * P + d * Q;
}

template<typename T>
__global__ void rho_pollard_kernel( const T *a,
                                    const T *b,
                                    typename EllipticCurve<T>::Point *X1,
                                    typename EllipticCurve<T>::Point *X2,
                                    T *c1, T *c2, T *d1, T *d2, T order,
                                    const typename EllipticCurve<T>::Point P,
                                    const typename EllipticCurve<T>::Point Q )
{
    __shared__ T a_shared[POLLARD_SET_COUNT];
    __shared__ T b_shared[POLLARD_SET_COUNT];

    unsigned idx = threadIdx.x + blockDim.x * blockIdx.x;

    if(idx < POLLARD_SET_COUNT)
    {
        a_shared[idx] = a[idx];
        b_shared[idx] = b[idx];
    }

    __syncthreads();

    typename EllipticCurve<T>::Point X1i = X1[idx];
    typename EllipticCurve<T>::Point X2i = X2[idx];
    T c1i = c1[idx], c2i = c2[idx], d1i = d1[idx], d2i = d2[idx];

    do
    {
        iteration(X1i, c1i, d1i, a_shared, b_shared, order, P, Q);

        iteration(X2i, c2i, d2i, a_shared, b_shared, order, P, Q);
        iteration(X2i, c2i, d2i, a_shared, b_shared, order, P, Q);
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
T rho_pollard(const typename EllipticCurve<T>::Point &Q, const typename EllipticCurve<T>::Point &P, const T &order)
{
    // std::independent_bits_engine<std::mt19937, sizeof(T)*4, T> gen;
    // gen.seed(time(NULL));

    // Host memory
    T c1_host[THREAD_PER_BLOCK], c2_host[THREAD_PER_BLOCK], d1_host[THREAD_PER_BLOCK], d2_host[THREAD_PER_BLOCK];
    T a_host[POLLARD_SET_COUNT], b_host[POLLARD_SET_COUNT];
    typename EllipticCurve<T>::Point X1_host[THREAD_PER_BLOCK];
    typename EllipticCurve<T>::Point X2_host[THREAD_PER_BLOCK];

    // Device memory
    T *c1_device = nullptr, *c2_device = nullptr, *d1_device = nullptr, *d2_device = nullptr;
    T *a_device = nullptr, *b_device = nullptr;
    typename EllipticCurve<T>::Point *X1_device = nullptr, *X2_device = nullptr;

    cudaMalloc((void **)&c1_device, sizeof(T) * THREAD_PER_BLOCK);
    cudaMalloc((void **)&c2_device, sizeof(T) * THREAD_PER_BLOCK);
    cudaMalloc((void **)&d1_device, sizeof(T) * THREAD_PER_BLOCK);
    cudaMalloc((void **)&d2_device, sizeof(T) * THREAD_PER_BLOCK);

    cudaMalloc((void **)&a_device, sizeof(T) * POLLARD_SET_COUNT);
 	cudaMalloc((void **)&b_device, sizeof(T) * POLLARD_SET_COUNT);

 	cudaMalloc((void **)&X1_device, sizeof(typename EllipticCurve<T>::Point) * THREAD_PER_BLOCK);
    cudaMalloc((void **)&X2_device, sizeof(typename EllipticCurve<T>::Point) * THREAD_PER_BLOCK);

    srand(time(NULL));
    
    T result = 0;
    while( !result )
    {
        for(unsigned i = 0; i < POLLARD_SET_COUNT; i++)
        {
            a_host[i] = rand() % order;
            b_host[i] = rand() % order;
        }

        for(unsigned i = 0; i < THREAD_PER_BLOCK; i++)
        {
            c1_host[i] = c2_host[i] = rand() % order;
            d1_host[i] = d2_host[i] = rand() % order;
            X1_host[i] = c1_host[i] * P + d1_host[i] * Q;
            X2_host[i] = c2_host[i] * P + d2_host[i] * Q;
        }
        
        cudaMemcpy((void *)c1_device, (const void*)c1_host, sizeof(T) * THREAD_PER_BLOCK, cudaMemcpyHostToDevice);
        cudaMemcpy((void *)c2_device, (const void*)c2_host, sizeof(T) * THREAD_PER_BLOCK, cudaMemcpyHostToDevice);
        cudaMemcpy((void *)d1_device, (const void*)d1_host, sizeof(T) * THREAD_PER_BLOCK, cudaMemcpyHostToDevice);
        cudaMemcpy((void *)d2_device, (const void*)d2_host, sizeof(T) * THREAD_PER_BLOCK, cudaMemcpyHostToDevice);
        
        cudaMemcpy((void *)a_device, (const void*)a_host, sizeof(T) * POLLARD_SET_COUNT, cudaMemcpyHostToDevice);
        cudaMemcpy((void *)b_device, (const void*)b_host, sizeof(T) * POLLARD_SET_COUNT, cudaMemcpyHostToDevice);

        cudaMemcpy((void *)X1_device, (const void*)X1_host, sizeof(typename EllipticCurve<T>::Point) * THREAD_PER_BLOCK, cudaMemcpyHostToDevice);
        cudaMemcpy((void *)X2_device, (const void*)X2_host, sizeof(typename EllipticCurve<T>::Point) * THREAD_PER_BLOCK, cudaMemcpyHostToDevice);

        int found_idx = -1;
        cudaMemcpyToSymbol(found_idx_d, &found_idx, sizeof(int));
        // kerner invocation here
        rho_pollard_kernel<<<1, THREAD_PER_BLOCK>>>(a_device, b_device, X1_device, X2_device, c1_device, c2_device, d1_device, d2_device, order, P, Q);
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

        cudaMemcpy((void *)X1_host, (const void*)X1_device, sizeof(typename EllipticCurve<T>::Point) * POLLARD_SET_COUNT, cudaMemcpyDeviceToHost);
        cudaMemcpy((void *)X2_host, (const void*)X2_device, sizeof(typename EllipticCurve<T>::Point) * POLLARD_SET_COUNT, cudaMemcpyDeviceToHost);

        typename EllipticCurve<T>::Point X1 = X1_host[found_idx];
        typename EllipticCurve<T>::Point X2 = X2_host[found_idx];
        T c1 = c1_host[found_idx], c2 = c2_host[found_idx], d1 = d1_host[found_idx], d2 = d2_host[found_idx];

        if(c1 * P + d1 * Q != X1 || c2 * P + d2 * Q != X2 || !X1.check())
            continue;

        T c = c1 - c2; if(c < 0) c += order;
        T d = d2 - d1; if(d < 0) d += order;
        if(d == 0)
            continue;

        T d_inv = detail::InvMod(d, order); if(d_inv < 0) d_inv += order;

        T tmp = (c * d_inv) % order;
        if(tmp * P == Q)
            result = tmp;
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
