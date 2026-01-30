set("C:\\msys64\\ucrt64\\bin\\gdb.exe")

cmake_minimum_required(VERSION 3.10)
project(MachineLearning)

add_library(functions 
            Components/ActivationFunctions.cpp
            Components/Architecture.cpp
            Components/Layers.cpp
            Components/Basic.cpp
            Components/Loss.cpp
            Components/Optimizer.cpp
            )

target_include_directories(functions
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/Headers
)

add_executable(ml_test TestingFile.cpp)

target_link_libraries(ml_test PRIVATE functions)