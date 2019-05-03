#include <elliptic.h>
#include <stdlib.h>
#include <fstream>
#include <chrono>
#include <cuda_runtime.h>

#define POLLARD_SET_COUNT 16
#define THREADS_PER_BLOCK 48
#define THREADS 288
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
std::tuple<T, double, unsigned long long>
rho_pollard( const Point<T> &Q, const Point<T> &P, const T &order, const EllipticCurve<T> &ec )
{
    T c1, c2, d1, d2;
    std::vector<T> a(POLLARD_SET_COUNT);
    std::vector<T> b(POLLARD_SET_COUNT);
    std::vector<Point<T>> R(POLLARD_SET_COUNT, P);
    
    double iters_per_sec = 0;
    unsigned long long time = 0;
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
        
        int iters = 0;
        auto before = std::chrono::system_clock::now();  
        do
        {
            iteration(X1, c1, d1, a, b, R, ec);

            iteration(X2, c2, d2, a, b, R, ec);
            iteration(X2, c2, d2, a, b, R, ec);
            iters++;
        }
        while(X1 != X2);
        auto after = std::chrono::system_clock::now();

        time = std::chrono::duration_cast<std::chrono::milliseconds>(after - before).count();
        iters_per_sec = (iters * 3.0) / time * 1000;

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

        T d_inv = detail::invmod(d, order); if(d_inv < 0) d_inv += order;

        return std::make_tuple((c * d_inv) % order, iters_per_sec, time);
    }
}
}   // namespace cpu

namespace gpu
{
__constant__ int128_t a_const[POLLARD_SET_COUNT];
__constant__ int128_t b_const[POLLARD_SET_COUNT];

__constant__ Point<int128_t> R_const[POLLARD_SET_COUNT];

__constant__ EllipticCurve<int128_t> ec_const;

__device__ volatile int found_idx_d = -1;

#ifdef DEBUG
__device__ int num_iter_d = 0;
#endif

template<typename T>
__device__ inline void iteration(   Point<T> &r,
                                    T &c, T &d,
                                    const EllipticCurve<T> &ec )
{
    unsigned index = r.getX() & 0xF;
    ec.addition(r, R_const[index]);
    c += a_const[index];
    d += b_const[index];
}

template<typename T>
__global__ void rho_pollard_kernel( Point<T> *X1,
                                    Point<T> *X2,
                                    T *c1, T *c2, T *d1, T *d2 )
{
    __shared__ T c1_sh[THREADS_PER_BLOCK];
    __shared__ T c2_sh[THREADS_PER_BLOCK];
    __shared__ T d1_sh[THREADS_PER_BLOCK];
    __shared__ T d2_sh[THREADS_PER_BLOCK];

    __shared__ Point<T> X1_sh[THREADS_PER_BLOCK];
    __shared__ Point<T> X2_sh[THREADS_PER_BLOCK];

    volatile __shared__ bool someoneFoundIt;

    // initialize shared status
    if( threadIdx.x == 0 )
        someoneFoundIt = !(found_idx_d == -1);

    __syncthreads();

    unsigned idx = threadIdx.x + blockDim.x * blockIdx.x;

    c1_sh[threadIdx.x] = c1[idx];
    c2_sh[threadIdx.x] = c2[idx];
    d1_sh[threadIdx.x] = d1[idx];
    d2_sh[threadIdx.x] = d2[idx];

    X1_sh[threadIdx.x] = X1[idx];
    X2_sh[threadIdx.x] = X2[idx];

    unsigned num_iter = 0;
    do
    {
        iteration(X1_sh[threadIdx.x], c1_sh[threadIdx.x], d1_sh[threadIdx.x], ec_const);

        iteration(X2_sh[threadIdx.x], c2_sh[threadIdx.x], d2_sh[threadIdx.x], ec_const);
        iteration(X2_sh[threadIdx.x], c2_sh[threadIdx.x], d2_sh[threadIdx.x], ec_const);

#ifdef DEBUG
        num_iter++;
#endif

        if(X1_sh[threadIdx.x] == X2_sh[threadIdx.x])
        {
            someoneFoundIt = true;
            found_idx_d = idx;
        }

        if(threadIdx.x == 0 && found_idx_d != -1)
        {
            someoneFoundIt = true;
        }
    }
    while( !someoneFoundIt );

#ifdef DEBUG
    atomicAdd(&num_iter_d, num_iter);
#endif

    if(found_idx_d == idx)
    {
        X1[idx] = X1_sh[threadIdx.x];
        X2[idx] = X2_sh[threadIdx.x];
        c1[idx] = c1_sh[threadIdx.x];
        c2[idx] = c2_sh[threadIdx.x];
        d1[idx] = d1_sh[threadIdx.x];
        d2[idx] = d2_sh[threadIdx.x];
    }
}

// Solve Q = xP
template<typename T>
std::tuple<T, double, unsigned long long>
rho_pollard(const Point<T> &Q,
            const Point<T> &P,
            const T &order,
            const EllipticCurve<T> &ec )
{
    // Host memory
    T c1_host[THREADS], c2_host[THREADS], d1_host[THREADS], d2_host[THREADS];
    T a_host[POLLARD_SET_COUNT], b_host[POLLARD_SET_COUNT];
    Point<T> R_host[POLLARD_SET_COUNT];
    Point<T> X1_host[THREADS];
    Point<T> X2_host[THREADS];

    // Device memory
    T *c1_device = nullptr, *c2_device = nullptr, *d1_device = nullptr, *d2_device = nullptr;
    Point<T> *X1_device = nullptr, *X2_device = nullptr;

    cudaMalloc((void **)&c1_device, sizeof(c1_host));
    cudaMalloc((void **)&c2_device, sizeof(c2_host));
    cudaMalloc((void **)&d1_device, sizeof(d1_host));
    cudaMalloc((void **)&d2_device, sizeof(d2_host));

 	cudaMalloc((void **)&X1_device, sizeof(X1_host));
    cudaMalloc((void **)&X2_device, sizeof(X2_host));

    T result = 0;
    double iters_per_sec = 0;
    unsigned long long time = 0;
    while( true )
    {
        for(unsigned i = 0; i < POLLARD_SET_COUNT; i++)
        {
            a_host[i].random(order);
            b_host[i].random(order);
            R_host[i] = ec.add(ec.mul(a_host[i], P), ec.mul(b_host[i], Q));
        }

        for(unsigned i = 0; i < THREADS; i++)
        {
            c1_host[i] = c2_host[i].random(order);
            d1_host[i] = d2_host[i].random(order);  
            X2_host[i] = X1_host[i] = ec.add(ec.mul(c1_host[i], P), ec.mul(d1_host[i], Q));
        }
        
        cudaMemcpy((void *)c1_device, (const void*)c1_host, sizeof(c1_host), cudaMemcpyHostToDevice);
        cudaMemcpy((void *)c2_device, (const void*)c2_host, sizeof(c2_host), cudaMemcpyHostToDevice);
        cudaMemcpy((void *)d1_device, (const void*)d1_host, sizeof(d1_host), cudaMemcpyHostToDevice);
        cudaMemcpy((void *)d2_device, (const void*)d2_host, sizeof(d2_host), cudaMemcpyHostToDevice);

        cudaMemcpyToSymbol(a_const, a_host, sizeof(a_host));
        cudaMemcpyToSymbol(b_const, b_host, sizeof(b_host));

        cudaMemcpyToSymbol(R_const, R_host, sizeof(R_host));

        cudaMemcpyToSymbol(ec_const, &ec, sizeof(EllipticCurve<T>));

        cudaMemcpy((void *)X1_device, (const void*)X1_host, sizeof(X1_host), cudaMemcpyHostToDevice);
        cudaMemcpy((void *)X2_device, (const void*)X2_host, sizeof(X2_host), cudaMemcpyHostToDevice);

        int found_idx = -1;
        cudaMemcpyToSymbol(found_idx_d, &found_idx, sizeof(int));

#ifdef DEBUG
        int num_iter = 0;
        cudaMemcpyToSymbol(num_iter_d, &num_iter, sizeof(int));
#endif
        // kerner invocation here
        auto before = std::chrono::system_clock::now();
        rho_pollard_kernel<<<THREADS / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(X1_device, X2_device,
            c1_device, c2_device, d1_device, d2_device);
        cudaDeviceSynchronize();
        auto after = std::chrono::system_clock::now();
        cudaMemcpyFromSymbol(&found_idx, found_idx_d, sizeof(found_idx), 0, cudaMemcpyDeviceToHost);

#ifdef DEBUG
        cudaMemcpyFromSymbol(&num_iter, num_iter_d, sizeof(num_iter), 0, cudaMemcpyDeviceToHost);
#endif

        time = std::chrono::duration_cast<std::chrono::milliseconds>(after - before).count();
        iters_per_sec = (num_iter + 0.0) / time * 1000;

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

        T d_inv = detail::invmod(d, order); if(d_inv < 0) d_inv += order;

        result = (c * d_inv) % order;
        break;
    }
    
    cudaFree(c1_device);
    cudaFree(c2_device);
    cudaFree(d1_device);
    cudaFree(d2_device);

    cudaFree(X1_device);
    cudaFree(X2_device);

    return std::make_tuple(result, iters_per_sec, time);
}
}   // namespace gpu
