cmake_minimum_required(VERSION 3.11)

project(chess_lib)

add_library(chess_lib
    src/chess_board.cpp
    src/chess_game.cpp
    src/chess_move.cpp
    src/chess_piece.cpp)


target_include_directories(${PROJECT_NAME}
    PUBLIC ${PROJECT_SOURCE_DIR}/include)

target_compile_features(chess_lib PUBLIC cxx_std_20)