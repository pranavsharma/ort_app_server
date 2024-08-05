# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

cmake_minimum_required(VERSION 3.26)
project(ORTAppServer CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(TARGET ort_app_server)
set(TARGET_SRC_DIR src)
set(TARGET_SRCS
    ${TARGET_SRC_DIR}/ort_app_server.cc
    ${TARGET_SRC_DIR}/utils.h
    ${TARGET_SRC_DIR}/model_manager.h
    ${TARGET_SRC_DIR}/model_manager.cc
    ${TARGET_SRC_DIR}/model_downloader.h
    ${TARGET_SRC_DIR}/model_downloader.cc)

set_property(GLOBAL PROPERTY FIND_LIBRARY_USE_LIB64_PATHS TRUE )
find_library(ORT_LIB NAMES onnxruntime PATHS ${ORT_GENAI_DIR}/lib)
find_library(ORT_GENAI_LIB NAMES onnxruntime-genai PATHS ${ORT_GENAI_DIR}/lib)
find_library(LIB_SSL NAMES ssl PATHS /usr/local/lib64)
find_library(LIB_CRYPTO NAMES crypto PATHS /usr/local/lib64)

include_directories(${ORT_GENAI_DIR}/include ${TARGET_SRC_DIR}/spdlog ${TARGET_SRC_DIR})
add_executable(${TARGET} ${TARGET_SRCS})

target_link_libraries(${TARGET} PRIVATE ${ORT_GENAI_LIB} ${ORT_LIB} pthread stdc++fs ${LIB_SSL} ${LIB_CRYPTO})
