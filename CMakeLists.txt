cmake_minimum_required(VERSION 3.10)
project(MnistNeuralNetwork CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(SOURCES
    main.cpp
    conv2d/conv2d.cpp
    FC_Layer/fc_layer.cpp
    MaxPool/max_pool.cpp
    ReLu/relu.cpp
)

add_executable(ml_project ${SOURCES})
target_compile_definitions(ml_project PRIVATE GLIBCXX_ASSERTIONS)
target_compile_options(ml_project PRIVATE -g)

target_include_directories(ml_project PRIVATE
    .
    conv2d
    FC_Layer
    MaxPool
    ReLu
)