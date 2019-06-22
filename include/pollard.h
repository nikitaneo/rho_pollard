#ifndef _POLLARD_H
#define _POLLARD_H

#include <elliptic.h>
#include <stdlib.h>
#include <fstream>
#include <chrono>
#include <thread>
#include <unordered_map>
#include <helper_cuda.h>
#include <cuda_runtime.h>

#include <arith.h>

#define POLLARD_SET_COUNT 0x20
#define POINTS_IN_PARALLEL 200

// For 610m
#define THREADS_PER_BLOCK 64
#define THREADS 512

// For K40c
// #define THREADS_PER_BLOCK 256
// #define THREADS 19200

#define BUFFER_SIZE 10000
#define DEBUG

namespace cpu
{
template<typename T>
inline void iteration(  Point<T> &R,
                        T &uR, T &vR,
                        const std::vector<T> &ug,
                        const std::vector<T> &vg,
                        const std::vector<Point<T>> &g,
                        const EllipticCurve<T> &EC )
{
    unsigned index = R.getX() & (POLLARD_SET_COUNT - 1);
    EC.addition(R, g[index]);
    uR += ug[index];
    vR += vg[index];
}

// Solve Q = xP
template<typename T>
std::tuple<T, double, unsigned long long, unsigned long long>
rho_pollard( const Point<T> &Q, const Point<T> &P, const T &order, const EllipticCurve<T> &EC )
{
    Point<T> R1, R2;
    T u1, u2, v1, v2;

    std::vector<Point<T>> g(POLLARD_SET_COUNT, P);
    std::vector<T> ug(POLLARD_SET_COUNT);
    std::vector<T> vg(POLLARD_SET_COUNT);
    
    double iters_per_sec = 0;
    unsigned long long prep_time = 0, iters_time = 0;
    while( true )
    {
        auto before_prep = std::chrono::system_clock::now();
        for(unsigned i = 0; i < POLLARD_SET_COUNT; i++)
        {
            ug[i].random(order);
            vg[i].random(order) ;
            g[i] = EC.plus(EC.mul(ug[i], P), EC.mul(vg[i], Q));
        }

        u1 = u2.random(order);
        v1 = v2.random(order);
        R2 = R1 = EC.plus(EC.mul(u1, P), EC.mul(v1, Q));
        auto after_prep = std::chrono::system_clock::now();
        prep_time = std::chrono::duration_cast<std::chrono::milliseconds>(after_prep - before_prep).count();
        
        unsigned long long iters = 0;

        bool stop_count_iters = false;

        auto before_iters = std::chrono::system_clock::now();  
        do
        {
            iteration(R1, u1, v1, ug, vg, g, EC);

            iteration(R2, u2, v2, ug, vg, g, EC);
            iteration(R2, u2, v2, ug, vg, g, EC);
            iters += 3;

            if( !stop_count_iters && iters >= 100000 )
            {
                stop_count_iters = true;
                auto t_now = std::chrono::system_clock::now();
                unsigned long long t = std::chrono::duration_cast<std::chrono::milliseconds>(t_now - before_iters).count();
                std::cout << "[INFO] Performance: " << (iters + 0.0) / t * 1000 << " p.a./s" << std::endl;
                return std::make_tuple(T(0), 0, 0, 0);
            }

        }
        while(R1 != R2);

        auto after_iters = std::chrono::system_clock::now();
        iters_time = std::chrono::duration_cast<std::chrono::milliseconds>(after_iters - before_iters).count();
        iters_per_sec = (iters + 0.0) / iters_time * 1000;

        u1 %= order; v1 %= order;
        u2 %= order; v2 %= order;

        if(EC.plus(EC.mul(u2, P), EC.mul(v2, Q)) != R2 || EC.plus(EC.mul(u1, P), EC.mul(v1, Q)) != R1 || !EC.check(R1))
        {
            std::cerr << "[INFO] u1 * P + v1 * Q != R1 or u2 * P + v2 * Q != R2 or R1 is not on curve." << std::endl;
            continue;
        }

        T u = u2.sub_modp(u1, order);
        T v = v1.sub_modp(v2, order);
        if(v == 0)
        {
            std::cerr << "[INFO] v1 == v2" << std::endl;
            continue;
        }

        T v_inv = v.inv_modp(order); if(v_inv < 0) v_inv += order;

        return std::make_tuple(u.mul_modp(v_inv, order), iters_per_sec, prep_time, iters_time);
    }
}
}   // namespace cpu

namespace gpu
{
template<class T>
struct PointWithCoeffs
{
    Point<T> point;
    T u, v;

    __host__ __device__ PointWithCoeffs( const Point<T> &p, const T &u, const T &v)
        : point( p ), u( u ), v( v )
    {
        // do nothing
    }

    __host__ __device__ PointWithCoeffs()
    {
        // do nothing
    }
};

// Support up to 256 bit arithmetics
__constant__ uint32_t ug_const[POLLARD_SET_COUNT * 8];
__constant__ uint32_t vg_const[POLLARD_SET_COUNT * 8];

__constant__ uint32_t g_const[POLLARD_SET_COUNT * 16];

#ifdef DEBUG
__device__ unsigned long long num_iter_d = 0;
#endif

template<typename T>
__global__ void rho_pollard_kernel( Point<T> *R_arr,
                                    T *u_arr,
                                    T *v_arr,
                                    PointWithCoeffs<T> *buffer,
                                    int *buffer_tail_d,
                                    bool *found_d,
                                    unsigned pattern,
                                    const EllipticCurve<T> EC)
{
    volatile __shared__ bool someoneFoundIt;

    // initialize shared status
    if( threadIdx.x == 0 )
        someoneFoundIt = *found_d;

    __syncthreads();

    unsigned idx = threadIdx.x + blockDim.x * blockIdx.x;

    // unsigned long long num_iter = 0;
    unsigned index = 0;
    T diff_buff[POINTS_IN_PARALLEL];
    T chain_buff[POINTS_IN_PARALLEL];
    T inv_diff, inverse, product;
    Point<T> R;
    while( !someoneFoundIt )
    {
        // Prepare for Montgomery inversion trick
        // Multiply differences together
        product = 1;
        for(int i = 0; i < POINTS_IN_PARALLEL; ++i)
        {
            R = R_arr[idx * POINTS_IN_PARALLEL + i];
            index = R.getX() & (POLLARD_SET_COUNT - 1);
            T diff = R.getX().sub_modp(reinterpret_cast<Point<T>*>(g_const)[index].getX(), EC.getP());
            diff_buff[i] = diff;
            product = product.mul_modp(diff, EC.getP());
            chain_buff[i] = product;
        }

        // Compute inverse
        inverse = product.inv_modp(EC.getP());

        // Extract inverse of the differences
        for(int i = POINTS_IN_PARALLEL - 1; i >= 0; i--)
        {
            // Get the inverse of the last difference by multiplying the inverse of the product of all the differences
            // with the product of all but the last difference
            if(i >= 1)
            {
                T tmp = chain_buff[i - 1];
                inv_diff = inverse.mul_modp(tmp, EC.getP());

                // Cancel out the last difference
                tmp = diff_buff[i];
                inverse = inverse.mul_modp(tmp, EC.getP());
            }
            else
            {
                inv_diff = inverse;
            }
            
            R = R_arr[idx * POINTS_IN_PARALLEL + i];
            index = R.getX() & (POLLARD_SET_COUNT - 1);
            EC.addition(R, reinterpret_cast<Point<T>*>(g_const)[index], inv_diff);

            R_arr[idx * POINTS_IN_PARALLEL + i] = R;
            u_arr[idx * POINTS_IN_PARALLEL + i] += reinterpret_cast<T *>(ug_const)[index];
            v_arr[idx * POINTS_IN_PARALLEL + i] += reinterpret_cast<T *>(vg_const)[index];

#ifdef DEBUG
            atomicAdd(&num_iter_d, 1);
#endif

            if((R.getX() & pattern) == 0)
            {
                int tail = *buffer_tail_d;
                *buffer_tail_d = (*buffer_tail_d + 1) % BUFFER_SIZE;
                buffer[tail] = PointWithCoeffs<T>(R, u_arr[idx * POINTS_IN_PARALLEL + i], v_arr[idx * POINTS_IN_PARALLEL + i]);
            }
        }
        
        if(threadIdx.x == 0 && *found_d)
            someoneFoundIt = true;
    }

#ifdef DEBUG
    // atomicAdd(&num_iter_d, num_iter);
#endif
}

// Solve Q = xP
template<typename T>
std::tuple<T, double, unsigned long long, unsigned long long>
rho_pollard(const Point<T> &Q,
            const Point<T> &P,
            const T &order,
            const EllipticCurve<T> &ec )
{
    checkCudaErrors(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1));

    // Host memory
    Point<T> R_host[THREADS*POINTS_IN_PARALLEL];
    T u_host[THREADS*POINTS_IN_PARALLEL], v_host[THREADS*POINTS_IN_PARALLEL];

    Point<T> g_host[POLLARD_SET_COUNT];
    T ug_host[POLLARD_SET_COUNT], vg_host[POLLARD_SET_COUNT];

    // Device memory
    T *u_device = nullptr, *v_device = nullptr;
    Point<T> *R_device = nullptr;
    PointWithCoeffs<T> *device_buffer = nullptr;

    unsigned pattern = (1 << (ec.getP().bitlength() / 4)) - 1;

    checkCudaErrors(cudaMalloc((void **)&u_device, sizeof(u_host)));
    checkCudaErrors(cudaMalloc((void **)&v_device, sizeof(v_host)));

    checkCudaErrors(cudaMalloc((void **)&R_device, sizeof(R_host)));

    checkCudaErrors(cudaMalloc((void **)&device_buffer, BUFFER_SIZE * sizeof(PointWithCoeffs<T>)));

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
    unsigned long long iters_time = 0, prep_time = 0;

    std::unordered_map<unsigned, Point<T>> cashP;   // cash to store k -> (2^k)*P
    std::unordered_map<unsigned, Point<T>> cashQ;   // cash to store k -> (2^k)*P
    while( !(*found) )
    {
        *buffer_tail = 0;

        // Preparation step
        auto before_prep = std::chrono::system_clock::now();
        for(unsigned i = 0; i < POLLARD_SET_COUNT; i++)
        {
            ug_host[i].random(order);
            vg_host[i].random(order);
            g_host[i] = ec.plus(ec.mul(ug_host[i], P, cashP), ec.mul(vg_host[i], Q, cashQ));
        }

        for(unsigned i = 0; i < THREADS*POINTS_IN_PARALLEL; i++)
        {
            u_host[i].random(order);
            v_host[i].random(order);  
            R_host[i] = ec.plus(ec.mul(u_host[i], P, cashP), ec.mul(v_host[i], Q, cashQ));
        }      
        
        checkCudaErrors(cudaMemcpy((void *)u_device, (const void*)u_host, sizeof(u_host), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy((void *)v_device, (const void*)v_host, sizeof(v_host), cudaMemcpyHostToDevice));

        checkCudaErrors(cudaMemcpyToSymbol(ug_const, ug_host, sizeof(ug_host)));
        checkCudaErrors(cudaMemcpyToSymbol(vg_const, vg_host, sizeof(vg_host)));

        checkCudaErrors(cudaMemcpyToSymbol(g_const, g_host, sizeof(g_host)));

        checkCudaErrors(cudaMemcpy((void *)R_device, (const void*)R_host, sizeof(R_host), cudaMemcpyHostToDevice));

#ifdef DEBUG
        unsigned long long num_iter = 0;
        checkCudaErrors(cudaMemcpyToSymbol(num_iter_d, &num_iter, sizeof(int)));
#endif
        auto after_prep = std::chrono::system_clock::now();
        prep_time = std::chrono::duration_cast<std::chrono::milliseconds>(after_prep - before_prep).count();


        // Kerner invocation here
        auto before_iters = std::chrono::system_clock::now();
        rho_pollard_kernel<<<THREADS / THREADS_PER_BLOCK, THREADS_PER_BLOCK, 0, kernel_stream>>>(   R_device, 
                                                                                                    u_device,
                                                                                                    v_device,
                                                                                                    device_buffer,
                                                                                                    buffer_tail_d,
                                                                                                    found_d,
                                                                                                    pattern,
                                                                                                    ec );


        // Collision search
        int tail = 0;
        std::unordered_map<Point<T>, std::pair<T, T>, detail::HashFunc> host_storage;
        PointWithCoeffs<T> dist_point;
        bool stop_count_iters = false;
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
#ifdef DEBUG
                    checkCudaErrors(cudaMemcpyFromSymbolAsync(&num_iter, num_iter_d, sizeof(num_iter), 0, cudaMemcpyDeviceToHost, memory_stream));
                    if( !stop_count_iters )
                    {
                        stop_count_iters = true;
                        auto t_now = std::chrono::system_clock::now();
                        unsigned long long t = std::chrono::duration_cast<std::chrono::milliseconds>(t_now - before_iters).count();
                        std::cout << "[INFO] Performance: " << (num_iter + 0.0) / t * 1000 << " p.a./s" << std::endl;
                        *found = true;
                        return std::make_tuple(T(0), 0, 0, 0);
                    }
#endif
                    Point<T> R1 = dist_point.point, R2 = iter->first;
                    T u1 = dist_point.u, u2 = iter->second.first, v1 = dist_point.v, v2 = iter->second.second;

                    u1 %= order; v1 %= order;
                    u2 %= order; v2 %= order;

                    if(ec.plus(ec.mul(u1, P), ec.mul(v1, Q)) != R1 || ec.plus(ec.mul(u2, P), ec.mul(v2, Q)) != R2 || !ec.check(R1))
                    {
                        std::cerr << "[INFO] u1 * P + v1 * Q != R1 or u2 * P + v2 * Q != R2 or R1 is not on curve." << std::endl;
                        continue;
                    }
                    
                    T u = u1.sub_modp(u2, order);
                    T v = v2.sub_modp(v1, order);
                    if(v == 0)
                    {
                        std::cerr << "[INFO] v1 == v2" << std::endl;
                        continue;
                    }

                    T v_inv = v.inv_modp(order); if(v_inv < 0) v_inv += order;
                    result = u.mul_modp(v_inv, order);

                    *found = true;

                    break;
                }
                else
                {
                    host_storage[dist_point.point] = std::make_pair(dist_point.u, dist_point.v);
                }
            }
            tail = tmp_tail;
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }

        checkCudaErrors(cudaStreamSynchronize(kernel_stream));
        getLastCudaError("rho_pollard_kernel execution failed");
        auto after_iters = std::chrono::system_clock::now();

#ifdef DEBUG
        checkCudaErrors(cudaMemcpyFromSymbol(&num_iter, num_iter_d, sizeof(num_iter), 0, cudaMemcpyDeviceToHost));
#endif

        iters_time = std::chrono::duration_cast<std::chrono::milliseconds>(after_iters - before_iters).count();
        iters_per_sec = (num_iter + 0.0) / iters_time * 1000;
    }
    
    checkCudaErrors(cudaStreamDestroy(kernel_stream));
    checkCudaErrors(cudaStreamDestroy(memory_stream));

    checkCudaErrors(cudaFreeHost((void *)buffer_tail));

    checkCudaErrors(cudaFree(u_device));
    checkCudaErrors(cudaFree(v_device));

    checkCudaErrors(cudaFree(R_device));

    checkCudaErrors(cudaFree(device_buffer));

    return std::make_tuple(result, iters_per_sec, prep_time, iters_time);
}
}   // namespace gpu

#endif // _POLLARD_H
