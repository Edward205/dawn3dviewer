cmake_minimum_required(VERSION 3.13)
project(
    dawnviewer
    VERSION 0.0.1
    LANGUAGES CXX C
)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_executable(dawnviewer 
    src/main.cpp
)

set_target_properties(dawnviewer PROPERTIES COMPILE_WARNING_AS_ERROR ON)
if (MSVC)
target_compile_options(dawnviewer PRIVATE /W4)
else()
target_compile_options(dawnviewer PRIVATE -Wall -Wextra -pedantic)
endif()

set(DAWN_FETCH_DEPENDENCIES ON)
add_subdirectory("dawn" EXCLUDE_FROM_ALL)

target_include_directories(
    dawnviewer PRIVATE "${CMAKE_BINARY_DIR}/dawn/include"
    dawnviewer PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(dawnviewer PRIVATE dawn::webgpu_dawn glfw webgpu_glfw)
