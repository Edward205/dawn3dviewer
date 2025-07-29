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
    
    src/components/mesh.cpp

    src/core/app.cpp
    src/core/window.cpp

    src/scene.cpp
)

set_target_properties(dawnviewer PROPERTIES COMPILE_WARNING_AS_ERROR ON)
if (MSVC)
target_compile_options(dawnviewer PRIVATE /W4)
else()
target_compile_options(dawnviewer PRIVATE -Wall -Wextra -pedantic)
endif()
set(CMAKE_BUILD_TYPE Debug)
set(DAWN_FETCH_DEPENDENCIES ON)
add_subdirectory("external/dawn" EXCLUDE_FROM_ALL)
add_subdirectory("external/glm" EXCLUDE_FROM_ALL)
target_include_directories(
    dawnviewer PRIVATE ${CMAKE_BINARY_DIR}/external/dawn/include
    dawnviewer PRIVATE ${CMAKE_BINARY_DIR}/external/glm/
    dawnviewer PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/external/tinyobjloader/include
    dawnviewer PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/external/entt/include
    dawnviewer PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(dawnviewer PRIVATE dawn::webgpu_dawn glfw webgpu_glfw glm)

