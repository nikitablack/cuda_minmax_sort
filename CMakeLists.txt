cmake_minimum_required(VERSION 3.23 FATAL_ERROR)

# list(APPEND CMAKE_PREFIX_PATH "$ENV{HOME}/opencv;/usr/local/cuda-11.4")

project(test LANGUAGES CXX CUDA)

find_package(OpenCV 4.6 REQUIRED COMPONENTS cudaarithm)

add_executable(test
    src/main.cu
)

set_target_properties(test PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED YES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "87"
)

target_link_libraries(test PRIVATE
    ${OpenCV_LIBS}
)

set(CUDA_FLAGS)
target_compile_options(test PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:${CUDA_FLAGS};>")
