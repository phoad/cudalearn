CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
PROJECT(CudaLearn)

SET(target hello)
SET(CMAKE_BUILD_TYPE "DEBUG")


#C:\phoad\development\libraries\cppunit\lib
#C:\phoad\development\libraries\cppunit\include
#FIND_PACKAGE(CppUnit)

INCLUDE_DIRECTORIES(
	"C:/phoad/development/libraries/cppunit/include"
)

LINK_DIRECTORIES(
	"C:/phoad/development/libraries/cppunit/lib"
)

FILE(GLOB CPP_FILES *.cpp)

#Main executable
ADD_EXECUTABLE(${target} ${CPP_FILES})

TARGET_LINK_LIBRARIES(${target} 
	cppunit
	)
