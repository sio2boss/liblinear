cmake_minimum_required(VERSION 3.7)
project(liblinear)

set(CMAKE_CXX_STANDARD 11)

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
target_link_libraries (predict linear)

# Installation targets
install(TARGETS train predict DESTINATION bin)
install(TARGETS linear DESTINATION lib)
