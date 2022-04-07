cmake_minimum_required(VERSION 3.11)

project(
    ActorCriticChessAgent
    VERSION 1.0
    DESCRIPTION "Simple Actor-Critic chess agent from scratch!"
)

add_subdirectory(apps)
add_subdirectory(libs/chess_lib)
add_subdirectory(libs/ml_lib)