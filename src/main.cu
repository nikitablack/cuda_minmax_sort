#include <opencv2/cudaarithm.hpp>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <stdexcept>
#include <string>

#define CHECK_ERROR(ans)                       \
    {                                          \
        checkError((ans), __FILE__, __LINE__); \
    }

inline void checkError(cudaError_t code, const char *file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        printf("GPU Kernel Error: %s %s %d\n", cudaGetErrorString(code), file, line);

        if (abort)
        {
            throw std::runtime_error{"Cuda error " + std::to_string(code)};
        }
    }
}

int main()
{
    cv::cuda::GpuMat m(1000, 1000, CV_32FC1);

    for (uint32_t i{0}; i < 10; ++i)
    {
        double maxValue;
        cv::cuda::minMax(m, nullptr, &maxValue);
    }

    constexpr size_t n{1'000'000};

    int *id;
    float *a;
    float *b;
    float *c;
    CHECK_ERROR(cudaMalloc(&id, n * sizeof(int)));
    CHECK_ERROR(cudaMalloc(&a, n * sizeof(float)));
    CHECK_ERROR(cudaMalloc(&b, n * sizeof(float)));
    CHECK_ERROR(cudaMalloc(&c, n * sizeof(float)));

    thrust::device_ptr<int> id_ptr(id);
    thrust::device_ptr<float> a_ptr(a);
    thrust::device_ptr<float> b_ptr(b);
    thrust::device_ptr<float> c_ptr(c);

    auto it = thrust::make_zip_iterator(thrust::make_tuple(a_ptr, b_ptr, c_ptr));

    for (uint32_t i{0}; i < 10; ++i)
    {
        thrust::sort_by_key(id_ptr, id_ptr + n, it);
    }

    CHECK_ERROR(cudaFree(id));
    CHECK_ERROR(cudaFree(a));
    CHECK_ERROR(cudaFree(b));
    CHECK_ERROR(cudaFree(c));

    return 0;
}