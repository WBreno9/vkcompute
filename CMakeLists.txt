cmake_minimum_required(VERSION 3.7.3)
project(vkcompute)

set(CMAKE_CXX_STANDARD 17)

set(THIRD_PARTY ${CMAKE_SOURCE_DIR}/third_party)

add_subdirectory(${THIRD_PARTY}/spdlog EXCLUDE_FROM_ALL)

# Add volk as a library
add_library(volk STATIC ${THIRD_PARTY}/volk/volk.c ${THIRD_PARTY}/volk/volk.h)
target_include_directories(volk PUBLIC ${THIRD_PARTY}/volk ${THIRD_PARTY}/khronos)
target_link_libraries(volk PRIVATE dl)

find_package(Vulkan REQUIRED)

set(SOURCES 
	src/vkrpch.h
	src/VmaImpl.cpp
	src/main.cpp
)

add_executable(vkcompute ${SOURCES})

add_subdirectory(${CMAKE_SOURCE_DIR}/src/shaders)
add_dependencies(vkcompute shaders)

target_include_directories(vkcompute PRIVATE 
	${THIRD_PARTY}/stb 
	${THIRD_PARTY}/VulkanMemoryAllocator/src spdlog::spdlog 
)
target_link_libraries(vkcompute PRIVATE volk spdlog)
