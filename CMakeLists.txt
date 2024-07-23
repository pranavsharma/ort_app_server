cmake_minimum_required(VERSION 3.26)
project(ORTAppServer CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(TARGET ort_app_server)
set(TARGET_SRC_DIR src)
set(TARGET_SRCS ${TARGET_SRC_DIR}/ort_app_server.cc ${TARGET_SRC_DIR}/utils.h)

find_library(ORT_GENAI_LIB NAMES onnxruntime-genai PATHS ${ORT_GENAI_DIR}/lib)

include_directories(${ORT_GENAI_DIR}/include ${TARGET_SRC_DIR}/spdlog ${TARGET_SRC_DIR})
add_executable(${TARGET} ${TARGET_SRCS})
target_link_libraries(${TARGET} PRIVATE ${ORT_GENAI_LIB} pthread)