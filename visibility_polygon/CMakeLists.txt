cmake_minimum_required(VERSION 3.24)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_TOOLCHAIN_FILE "C:/src/vcpkg/scripts/buildsystems/vcpkg.cmake" CACHE STRING "Vcpkg toolchain file")
project(visbility_polygon)
set(CMAKE_FIND_DEBUG_MODE ON)
set(CMAKE_CXX_STANDARD 17)
#set(CMAKE_POLICY_DEFAULT_CMP0094 NEW)
add_executable(visbility_polygon general_polygon_example.cpp)

find_package(CGAL REQUIRED)
target_link_libraries(visbility_polygon PRIVATE CGAL::CGAL)



#set(CMAKE_FIND_USE_CMAKE_PATH FALSE)
#set(Python_FIND_STRATEGY "LOCATION")
#set(Python_FIND_REGISTRY "NEVER")
#set(Python_FIND_FRAMEWORK "NEVER")
#set(Python_EXECUTABLE "C:/Users/user/anaconda3/envs/obstacleMobilityModel/python.exe")
#set(Python_ROOT_DIR "C:/Users/user/anaconda3/envs/obstacleMobilityModel/")
#set(Python_ROOT "C:/Users/user/anaconda3/envs/obstacleMobilityModel/")
#set(Python_INCLUDE_DIR "C:/Users/user/anaconda3/envs/obstacleMobilityModel/Include")
##find_package(Python COMPONENTS Interpreter Development REQUIRED)
#find_package(Python COMPONENTS Development REQUIRED)
#
#set(pybind11_DIR "C:/Users/user/PycharmProjects/obstacleMobilityModel/extern/pybind11/build/mock_install/share/cmake/pybind11")
#find_package(pybind11 CONFIG REQUIRED)
#
#pybind11_add_module(example MODULE example.cpp)

