# =============================================================================
# FindONNXRuntime.cmake — 查找 ONNX Runtime 安装
# =============================================================================
#
# 搜索顺序:
#   1. cmake 变量 ONNXRUNTIME_ROOT
#   2. 环境变量 ONNXRUNTIME_ROOT
#   3. 项目内置 third_party/onnxruntime/
#
# 导出变量:
#   ONNXRUNTIME_FOUND
#   ONNXRUNTIME_INCLUDE_DIRS
#   ONNXRUNTIME_LIBRARIES
# =============================================================================

set(_ONNXRUNTIME_SEARCH_PATHS
    ${ONNXRUNTIME_ROOT}
    $ENV{ONNXRUNTIME_ROOT}
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/onnxruntime
    /usr
    /usr/local
)

find_path(ONNXRUNTIME_INCLUDE_DIRS
    NAMES onnxruntime_cxx_api.h
    PATHS ${_ONNXRUNTIME_SEARCH_PATHS}
    PATH_SUFFIXES include include/onnxruntime
    NO_DEFAULT_PATH
)

if(NOT ONNXRUNTIME_INCLUDE_DIRS)
    find_path(ONNXRUNTIME_INCLUDE_DIRS
        NAMES onnxruntime_cxx_api.h
        PATH_SUFFIXES include include/onnxruntime
    )
endif()

find_library(ONNXRUNTIME_LIBRARIES
    NAMES onnxruntime libonnxruntime
    PATHS ${_ONNXRUNTIME_SEARCH_PATHS}
    PATH_SUFFIXES lib lib64
    NO_DEFAULT_PATH
)

if(NOT ONNXRUNTIME_LIBRARIES)
    find_library(ONNXRUNTIME_LIBRARIES
        NAMES onnxruntime libonnxruntime
        PATH_SUFFIXES lib lib64
    )
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(ONNXRUNTIME
    REQUIRED_VARS ONNXRUNTIME_INCLUDE_DIRS ONNXRUNTIME_LIBRARIES
)

if(ONNXRUNTIME_FOUND AND NOT TARGET onnxruntime::onnxruntime)
    add_library(onnxruntime::onnxruntime SHARED IMPORTED)
    set_target_properties(onnxruntime::onnxruntime PROPERTIES
        IMPORTED_LOCATION "${ONNXRUNTIME_LIBRARIES}"
        INTERFACE_INCLUDE_DIRECTORIES "${ONNXRUNTIME_INCLUDE_DIRS}"
    )
endif()

mark_as_advanced(ONNXRUNTIME_INCLUDE_DIRS ONNXRUNTIME_LIBRARIES)
