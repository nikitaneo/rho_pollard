#include <elliptic.h>
#include <stdlib.h>
#include <fstream>
#include <chrono>
#include <thread>
#include <unordered_map>
#include <helper_cuda.h>
#include <cuda_runtime.h>

#define POLLARD_SET_COUNT 0x20
#define THREADS_PER_BLOCK 48
#define THREADS 384
#define BUFFER_SIZE 8388608
#define DEBUG

template<class T>
struct PointCD
{
    Point<T> point;
    T c, d;

    __host__ __device__ PointCD( const Point<T> &p, const T &c, const T &d)
        : point( p ), c( c ), d( d )
    {
        // do nothing
    }

    __host__ __device__ PointCD()
    {
        // do nothing
    }
};

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
    unsigned index = r.getX() & (POLLARD_SET_COUNT - 1);
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
        
        unsigned long long iters = 0;
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

        time = std::chrono::duration_cast<std::chrono::seconds>(after - before).count();
        iters_per_sec = (iters * 3.0) / time;

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

#ifdef DEBUG
__device__ unsigned long long num_iter_d = 0;
#endif

template<typename T>
__global__ void rho_pollard_kernel( Point<T> *X_arr, T *c_arr, T *d_arr, PointCD<T> *buffer, int *buffer_tail_d, bool *found_d, unsigned pattern )
{
    volatile __shared__ bool someoneFoundIt;

    // initialize shared status
    if( threadIdx.x == 0 )
        someoneFoundIt = *found_d;

    __syncthreads();

    unsigned idx = threadIdx.x + blockDim.x * blockIdx.x;

    T c = c_arr[idx], d = d_arr[idx];
    Point<T> X = X_arr[idx];

    unsigned long long num_iter = 0;
    unsigned index = 0;
    while( !someoneFoundIt )
    {
        index = X.getX() & (POLLARD_SET_COUNT - 1);
        ec_const.addition(X, R_const[index]);
        c += a_const[index];
        d += b_const[index];

#ifdef DEBUG
        num_iter++;
#endif
        if((X.getX() & pattern) == 0)
        {
            int tail = *buffer_tail_d;
            *buffer_tail_d = (*buffer_tail_d + 1) % BUFFER_SIZE;
            buffer[tail] = PointCD<T>(X, c, d);
        }

        if(threadIdx.x == 0 && *found_d)
            someoneFoundIt = true;
    }

#ifdef DEBUG
    atomicAdd(&num_iter_d, num_iter);
#endif
}

// Solve Q = xP
template<typename T>
std::tuple<T, double, unsigned long long>
rho_pollard(const Point<T> &Q,
            const Point<T> &P,
            const T &order,
            const EllipticCurve<T> &ec )
{
    checkCudaErrors(cudaSetDeviceFlags(cudaDeviceMapHost));
    checkCudaErrors(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1));

    // Host memory
    T c1_host[THREADS], d1_host[THREADS];
    T a_host[POLLARD_SET_COUNT], b_host[POLLARD_SET_COUNT];
    Point<T> R_host[POLLARD_SET_COUNT];
    Point<T> X1_host[THREADS];

    // Device memory
    T *c1_device = nullptr, *d1_device = nullptr;
    Point<T> *X1_device = nullptr;
    PointCD<T> *device_buffer = nullptr;

    unsigned pattern = (1 << ec.getP().bitlength()) - 1;

    checkCudaErrors(cudaMalloc((void **)&c1_device, sizeof(c1_host)));
    checkCudaErrors(cudaMalloc((void **)&d1_device, sizeof(d1_host)));

    checkCudaErrors(cudaMalloc((void **)&X1_device, sizeof(X1_host)));

    checkCudaErrors(cudaMalloc((void **)&device_buffer, BUFFER_SIZE * sizeof(PointCD<T>)));

    volatile int *buffer_tail = nullptr;
    int *buffer_tail_d = nullptr;
    checkCudaErrors(cudaHostAlloc((void **)&buffer_tail, sizeof(int), cudaHostAllocMapped));
    checkCudaErrors(cudaHostGetDevicePointer((void **)&buffer_tail_d, (void *)buffer_tail, 0));

    volatile bool *found = nullptr;
    bool *found_d = nullptr;
    checkCudaErrors(cudaHostAlloc((void **)&found, sizeof(bool), cudaHostAllocMapped));
    checkCudaErrors(cudaHostGetDevicePointer((void **)&found_d, (void *)found, 0));
    *found = false;

    cudaStream_t kernel_stream, memory_stream;
    checkCudaErrors(cudaStreamCreate(&kernel_stream));
    checkCudaErrors(cudaStreamCreate(&memory_stream));

    T result = 0;
    double iters_per_sec = 0;
    unsigned long long time = 0;
    while( !(*found) )
    {
        *buffer_tail = 0;

        for(unsigned i = 0; i < POLLARD_SET_COUNT; i++)
        {
            a_host[i].random(order);
            b_host[i].random(order);
            R_host[i] = ec.add(ec.mul(a_host[i], P), ec.mul(b_host[i], Q));
        }

        for(unsigned i = 0; i < THREADS; i++)
        {
            c1_host[i].random(order);
            d1_host[i].random(order);  
            X1_host[i] = ec.add(ec.mul(c1_host[i], P), ec.mul(d1_host[i], Q));
        }      
        
        checkCudaErrors(cudaMemcpy((void *)c1_device, (const void*)c1_host, sizeof(c1_host), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy((void *)d1_device, (const void*)d1_host, sizeof(d1_host), cudaMemcpyHostToDevice));

        checkCudaErrors(cudaMemcpyToSymbol(a_const, a_host, sizeof(a_host)));
        checkCudaErrors(cudaMemcpyToSymbol(b_const, b_host, sizeof(b_host)));

        checkCudaErrors(cudaMemcpyToSymbol(R_const, R_host, sizeof(R_host)));

        checkCudaErrors(cudaMemcpyToSymbol(ec_const, &ec, sizeof(EllipticCurve<T>)));

        checkCudaErrors(cudaMemcpy((void *)X1_device, (const void*)X1_host, sizeof(X1_host), cudaMemcpyHostToDevice));

#ifdef DEBUG
        unsigned long long num_iter = 0;
        checkCudaErrors(cudaMemcpyToSymbol(num_iter_d, &num_iter, sizeof(int)));
#endif
        // kerner invocation here
        auto before = std::chrono::system_clock::now();
        rho_pollard_kernel<<<THREADS / THREADS_PER_BLOCK, THREADS_PER_BLOCK, 0, kernel_stream>>>(X1_device, c1_device, d1_device,
            device_buffer, buffer_tail_d, found_d, pattern);

        std::thread worker([&](){
            int tail = 0;
            std::unordered_map<Point<T>, std::pair<T, T>, detail::HashFunc> host_storage;
            PointCD<T> dist_point;
            while( !(*found) )
            {
                int tmp_tail = *buffer_tail;
                for(int i = tail; i < tmp_tail; i++)
                {
                    checkCudaErrors(cudaMemcpyAsync((void *)&dist_point, (const void*)(device_buffer + i), sizeof(dist_point), cudaMemcpyDeviceToHost, memory_stream));
                    checkCudaErrors(cudaStreamSynchronize(memory_stream));
                    auto iter = host_storage.find( dist_point.point );
                    if(iter != host_storage.end())
                    {
                        *found = true;

                        Point<T> X1 = dist_point.point, X2 = iter->first;
                        T c1 = dist_point.c, c2 = iter->second.first, d1 = dist_point.d, d2 = iter->second.second;

                        c1 = c1 % order;
                        d1 = d1 % order;
                        c2 = c2 % order;
                        d2 = d2 % order;

                        if(ec.add(ec.mul(c1, P), ec.mul(d1, Q)) != X1 || ec.add(ec.mul(c2, P), ec.mul(d2, Q)) != X2 || !ec.check(X1))
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
                    else
                    {
                        host_storage[dist_point.point] = std::make_pair(dist_point.c, dist_point.d);
                    }
                }
                tail = tmp_tail;
                std::this_thread::sleep_for(std::chrono::seconds(1));
            }
        });

        checkCudaErrors(cudaStreamSynchronize(kernel_stream));
        getLastCudaError("rho_pollard_kernel execution failed");
        worker.join();
        auto after = std::chrono::system_clock::now();

#ifdef DEBUG
        checkCudaErrors(cudaMemcpyFromSymbol(&num_iter, num_iter_d, sizeof(num_iter), 0, cudaMemcpyDeviceToHost));
#endif

        time = std::chrono::duration_cast<std::chrono::seconds>(after - before).count();
        iters_per_sec = (num_iter + 0.0) / time;
    }
    
    checkCudaErrors(cudaStreamDestroy(kernel_stream));
    checkCudaErrors(cudaStreamDestroy(memory_stream));

    checkCudaErrors(cudaFreeHost((void *)buffer_tail));

    checkCudaErrors(cudaFree(c1_device));
    checkCudaErrors(cudaFree(d1_device));

    checkCudaErrors(cudaFree(X1_device));

    checkCudaErrors(cudaFree(device_buffer));

    return std::make_tuple(result, iters_per_sec, time);
}
}   // namespace gpu
