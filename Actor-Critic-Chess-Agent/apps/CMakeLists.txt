cmake_minimum_required(VERSION 3.11)

add_executable(main 
    main.cpp
    ../src/chess_agent_environment.cpp
    ../src/chess_agent_replay.cpp
    ../src/chess_agent_train.cpp
    ../src/chess_agent_test.cpp)

target_link_libraries(main PRIVATE chess_lib)
target_link_libraries(main PRIVATE ml_lib)

target_include_directories(main
    PUBLIC ${PROJECT_SOURCE_DIR}/include/)

target_compile_features(main PUBLIC cxx_std_20)