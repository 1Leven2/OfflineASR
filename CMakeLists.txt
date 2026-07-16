cmake_minimum_required(VERSION 3.18)
project(offline_asr VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ---- Build options ----
option(BUILD_TESTS "Build unit tests" OFF)
option(BUILD_BENCHMARKS "Build benchmarks" OFF)
option(BUILD_CLI "Build CLI tool" ON)

# ---- CMake modules ----
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
message(STATUS "CMAKE_MODULE_PATH: ${CMAKE_MODULE_PATH}")

# ---- Dependencies ----
include(FetchContent)

# spdlog (header-only logging)
FetchContent_Declare(spdlog
    GIT_REPOSITORY https://github.com/gabime/spdlog.git
    GIT_TAG v1.13.0
)
FetchContent_MakeAvailable(spdlog)

# yaml-cpp (YAML config parsing)
FetchContent_Declare(yaml-cpp
    GIT_REPOSITORY https://github.com/jbeder/yaml-cpp.git
    GIT_TAG yaml-cpp-0.7.0
)
set(YAML_CPP_BUILD_TESTS OFF CACHE BOOL "")
set(YAML_CPP_BUILD_TOOLS OFF CACHE BOOL "")
FetchContent_MakeAvailable(yaml-cpp)

# libsndfile (audio I/O)
find_library(SNDFILE_LIBRARY sndfile REQUIRED)
find_path(SNDFILE_INCLUDE_DIR sndfile.h REQUIRED)
if(NOT TARGET sndfile::sndfile)
    add_library(sndfile::sndfile UNKNOWN IMPORTED)
    set_target_properties(sndfile::sndfile PROPERTIES
        IMPORTED_LOCATION "${SNDFILE_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${SNDFILE_INCLUDE_DIR}"
    )
endif()
message(STATUS "libsndfile: ${SNDFILE_LIBRARY}")

# OpenMP (required by FBank)
find_package(OpenMP REQUIRED)

# ONNX Runtime
find_package(ONNXRUNTIME REQUIRED)
message(STATUS "ONNX Runtime include: ${ONNXRUNTIME_INCLUDE_DIRS}")
message(STATUS "ONNX Runtime library: ${ONNXRUNTIME_LIBRARIES}")

# FBank (existing library)
add_library(fbank STATIC IMPORTED)
set_target_properties(fbank PROPERTIES
    IMPORTED_LOCATION "${CMAKE_CURRENT_SOURCE_DIR}/../FBank/build/libfbank.a"
    INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_CURRENT_SOURCE_DIR}/../FBank/include"
)

# CTC decoder (existing library)
add_library(ctc_core STATIC IMPORTED)
set_target_properties(ctc_core PROPERTIES
    IMPORTED_LOCATION "${CMAKE_CURRENT_SOURCE_DIR}/../CTC/build/libctc_core.a"
    INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_CURRENT_SOURCE_DIR}/../CTC/include"
)

# ---- Core Library ----
add_library(offline_asr_core STATIC
    src/audio/wav_reader.cpp
    src/feature/fbank_extractor.cpp
    src/feature/streaming_fbank_extractor.cpp
    src/runtime/onnx_runtime.cpp
    src/runtime/runtime_factory.cpp
    src/decoder/ctc_wrapper.cpp
    src/decoder/token_table.cpp
    src/recognizer/recognizer.cpp
    src/recognizer/recognizer_config.cpp
    src/utils/logger.cpp
)

target_include_directories(offline_asr_core
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(offline_asr_core
    PUBLIC
        fbank
        ctc_core
        onnxruntime::onnxruntime
        sndfile::sndfile
        spdlog::spdlog
        yaml-cpp
        OpenMP::OpenMP_CXX
)

# ---- CLI ----
if(BUILD_CLI)
    add_subdirectory(cli)
endif()

# ---- Tests ----
if(BUILD_TESTS)
    enable_testing()
    FetchContent_Declare(googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v1.14.0
    )
    FetchContent_MakeAvailable(googletest)
    include(GoogleTest)
    add_subdirectory(tests)
endif()

# ---- Benchmarks ----
if(BUILD_BENCHMARKS)
    FetchContent_Declare(benchmark
        GIT_REPOSITORY https://github.com/google/benchmark.git
        GIT_TAG v1.8.3
    )
    FetchContent_MakeAvailable(benchmark)
    add_subdirectory(benchmark)
endif()
