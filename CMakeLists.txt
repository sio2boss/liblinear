cmake_minimum_required(VERSION 3.7)
project(liblinear)


FIND_PACKAGE(SWIG REQUIRED)
find_package(Java REQUIRED COMPONENTS Runtime Development)
find_package(JNI REQUIRED)

include(UseJava)
include(UseSWIG)

INCLUDE_DIRECTORIES(${JAVA_INCLUDE_PATH2})
INCLUDE_DIRECTORIES(${JAVA_INCLUDE_PATH})
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/src/liblinear)

set(CMAKE_CXX_STANDARD 11)

set(CMAKE_SWIG_FLAGS -package org.liblinear)
set(SWIG_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/src/liblinear)
set(CMAKE_SWIG_OUTDIR ${CMAKE_CURRENT_SOURCE_DIR}/src/bindings/java/src/main/java/org/liblinear)


# Build shared library with core code
add_library(linear SHARED
        src/liblinear/tron.cpp
        src/liblinear/linear.cpp
        src/liblinear/io.cpp
        src/liblinear/blas/daxpy.c
        src/liblinear/blas/ddot.c
        src/liblinear/blas/dnrm2.c
        src/liblinear/blas/dscal.c)

# Executable using liblinear for training
add_executable(train src/train.cpp)
target_link_libraries (train linear)

# Executable for using liblinear for predicting
add_executable(predict src/predict.cpp)
target_link_libraries(predict linear)

# Java bindings
SET_SOURCE_FILES_PROPERTIES(src/bindings/java/liblinear.i PROPERTIES CPLUSPLUS ON)
SWIG_ADD_MODULE(linear-java java src/bindings/java/liblinear.i)
SWIG_LINK_LIBRARIES(linear-java linear)

# Installation targets
install(TARGETS train predict DESTINATION bin)
install(TARGETS linear DESTINATION lib)
