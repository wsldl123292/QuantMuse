cmake_minimum_required(VERSION 3.14)
project(trading_system VERSION 1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(CURL REQUIRED)
find_package(nlohmann_json REQUIRED)
find_package(spdlog REQUIRED)
find_package(Boost REQUIRED COMPONENTS system)
find_package(OpenSSL REQUIRED)

# Add include directories
include_directories(${CMAKE_SOURCE_DIR}/include)

# Add subdirectories for different components
add_subdirectory(src)
add_subdirectory(tests)

# Set compiler flags
if(MSVC)
    add_compile_options(/W4)
else()
    add_compile_options(-Wall -Wextra -Wpedantic)
endif() 