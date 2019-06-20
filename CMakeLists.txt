cmake_minimum_required(VERSION 3.5.1)
project(vkcompute)

set(CMAKE_CXX_STANDARD 17)
set(THIRD_PARTY ${CMAKE_SOURCE_DIR}/third_party)

add_subdirectory(${THIRD_PARTY}/glfw EXCLUDE_FROM_ALL)
add_subdirectory(${THIRD_PARTY}/glm EXCLUDE_FROM_ALL)
add_subdirectory(${THIRD_PARTY}/spdlog EXCLUDE_FROM_ALL)

# Add volk as a library
add_library(volk STATIC ${THIRD_PARTY}/volk/volk.c ${THIRD_PARTY}/volk/volk.h)
target_include_directories(volk PUBLIC ${THIRD_PARTY}/volk ${THIRD_PARTY}/khronos)


set(SOURCES 
	src/main.cpp
)

find_package(Vulkan REQUIRED)
find_package(glfw3 REQUIRED)

add_executable(vkcompute ${SOURCES})
target_include_directories(vkrenderer PRIVATE ${THIRD_PARTY}/stb spdlog::spdlog)
target_link_libraries(vkrenderer PRIVATE glfw glm volk spdlog)
