cmake_minimum_required(VERSION 3.10.0)
project(main)

#configure_file(${CMAKE_CURRENT_SOURCE_DIR}/std.conf ${CMAKE_CURRENT_BINARY_DIR}/std.conf)

#find_package(aphros)

#set(CXX mpic++)
find_package(MPI REQUIRED)
include_directories(SYSTEM ${MPI_INCLUDE_PATH})

SET(KORALI_F "-D_KORALI_NO_MPI4PY -std=c++17 -I/home/pooria/korali/python/korali/include -I/home/pooria/korali/python/korali/../../../../include/ -I/home/pooria/korali/python/korali/../../source/ -I/home/pooria/korali/python/korali/../../build/source/ -I/opt/python/3.8.3/include/python3.8 -I/opt/python/3.8.3/include/python3.8 -I/opt/python/3.8.3/lib/python3.8/site-packages/pybind11/include -I/opt/python/3.8.3/include/python3.8 -I/opt/python/3.8.3/include/python3.8")
SET(KORALI_L "/home/pooria/korali/build/source/libkorali.cpython-38-x86_64-linux-gnu.so -Wl,-rpath,/home/pooria/korali/python/korali  -L/opt/python/3.8.3/lib -lpython3.8 -lcrypt -lpthread -ldl  -lutil -lm -lm  -Wl,-rpath,/opt/python/3.8.3/lib")

SET(CMAKE_CXX_FLAGS "${KORALI_F}")
SET(CMAKE_EXE_LINKER_FLAGS "${KORALI_L}")

SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -lX11 -lstdc++fs")

#add_subdirectory("_model")

set(EXE main)
add_executable(${EXE} main.cpp)
#target_link_libraries(${EXE} env)
#target_link_libraries(${EXE} aphros)
target_link_libraries(${EXE} ${MPI_C_LIBRARIES})

