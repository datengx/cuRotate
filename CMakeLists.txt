cmake_minimum_required(VERSION 2.8)

## Manually specifying the location of the C and C++ compiler (NOTE the following might not work in CMAKE
## try working with commandLine argument such as -DCMAKE_CXX_COMPILER )

## If building on mac
if(APPLE)
	set(CMAKE_CXX_COMPILER "/Users/dateng/Developer/apple_legacy/Xcode_7.2.xctoolchain/usr/bin/c++")
	set(CMAKE_C_COMPILER "/Users/dateng/Developer/apple_legacy/Xcode_7.2.xctoolchain/usr/bin/cc")
endif()



project(test_cufft)


add_definitions( -DCUDA_API_PER_THREAD_DEFAULT_STREAM )
find_package(CUDA QUIET REQUIRED)	# Find CUDA library


set( CMAKE_EXE_LINKER_FLAGS  "${CMAKE_EXE_LINKER_FLAGS}  -O3")


if(APPLE)
include_directories(
    "/Users/dateng/Developer/eigen-eigen-07105f7124f9"
    "/Users/dateng/Developer/itk_sample/cuRotate"
    "./" )
include_directories(
			"/usr/local/cuda/include"
	)
endif()

set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -arch=sm_21")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-write-strings")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Ofast")

cuda_add_executable(cuRotate 
				"./cuRotate.cu"
				)

# Link to CUDA and ITK libraries
# target_link_libraries( cuRotate  )

CUDA_ADD_CUFFT_TO_TARGET( cuRotate )

## Find ITK.
find_package(ITK REQUIRED)
# FIND_PACKAGE( FFTW ) 
include(${ITK_USE_FILE})

add_library(itk_io SHARED "./itk_io.cpp" )
target_link_libraries(itk_io ${ITK_LIBRARIES})

target_link_libraries(cuRotate itk_io)