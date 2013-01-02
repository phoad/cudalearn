CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
PROJECT(CudaLearn)

SET(target hello)
SET(CMAKE_BUILD_TYPE "DEBUG")

FIND_PACKAGE(CUDA REQUIRED)
ADD_DEFINITIONS(-DUSE_NVIDIA_OPENCL)
ADD_DEFINITIONS(-DCL_PLATFORM_NVIDIA)

message(STATUS "CUDA TOOLKIT " ${CUDA_TOOLKIT_ROOT_DIR})
message(STATUS "CUBLAS " ${CUDA_CUBLAS_LIBRARIES})

if (NOT CUDA_SDK_ROOT_DIR)
  set(CUDA_SDK_ROOT_DIR "/home/phoad/NVIDIA_GPU_Computing_SDK")
endif()

message(STATUS "SDK ROOT " ${CUDA_SDK_ROOT_DIR})

#C:\phoad\development\libraries\cppunit\lib
#C:\phoad\development\libraries\cppunit\include
#FIND_PACKAGE(CppUnit)

INCLUDE_DIRECTORIES(
  ${CUDA_TOOLKIT_ROOT_DIR}/include
  ${CUDA_SDK_ROOT_DIR}/common/inc	
  ${CUDA_TOOLKIT_ROOT_DIR}/samples/common/inc
  
  "C:/ProgramData/NVIDIA Corporation/CUDA Samples/v5.0/common/inc"
#	"C:/phoad/development/libraries/cppunit/include"
)

LINK_DIRECTORIES(
  ${CUDA_SDK_ROOT_DIR}/common/lib/Win32
  ${CUDA_SDK_ROOT_DIR}/common/lib/x64
  ${CUDA_SDK_ROOT_DIR}/common/lib
  "C:/ProgramData/NVIDIA Corporation/CUDA Samples/v5.0/common/lib/x64"
#	"C:/phoad/development/libraries/cppunit/lib"
)

FILE(GLOB CPP_FILES *.cpp *.cu)

#Main executable
CUDA_ADD_EXECUTABLE(${target} ${CPP_FILES})

TARGET_LINK_LIBRARIES(${target}
	${CUDA_LIBRARIES}
	#	cppunit
	)
