cmake_minimum_required(VERSION 3.11)

project(ml_lib)

add_library(ml_lib 
    src/model_layer_initializer.cpp
    src/model_layer_type.cpp
    src/model_lossfunction.cpp
    src/model_optimizer.cpp
    src/tensor_element_autodiff_node.cpp
    src/tensor_element.cpp
    src/tensor.cpp)

target_include_directories(${PROJECT_NAME}
    PUBLIC ${PROJECT_SOURCE_DIR}/include)

target_compile_features(ml_lib PUBLIC cxx_std_20)