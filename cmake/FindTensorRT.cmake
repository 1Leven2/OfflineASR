# =============================================================================
# FindTensorRT.cmake — locate NVIDIA TensorRT installation
# =============================================================================
#
# Search order:
#   1. cmake variable TENSORRT_ROOT
#   2. environment variable TENSORRT_ROOT
#   3. system paths (/usr, /usr/local)
#
# Exported variables:
#   TENSORRT_FOUND
#   TENSORRT_INCLUDE_DIRS
#   TENSORRT_LIBRARIES        (nvinfer + nvonnxparser)
#   TENSORRT_NVINFER_LIBRARY
#   TENSORRT_NVONNXPARSER_LIBRARY
# =============================================================================

set(_TENSORRT_SEARCH_PATHS
    ${TENSORRT_ROOT}
    $ENV{TENSORRT_ROOT}
    /usr
    /usr/local
)

# Headers — look for NvInfer.h in typical Debian multiarch layout first
find_path(TENSORRT_INCLUDE_DIRS
    NAMES NvInfer.h
    PATHS ${_TENSORRT_SEARCH_PATHS}
    PATH_SUFFIXES include/x86_64-linux-gnu include include/tensorrt
    NO_DEFAULT_PATH
)

if(NOT TENSORRT_INCLUDE_DIRS)
    find_path(TENSORRT_INCLUDE_DIRS
        NAMES NvInfer.h
        PATH_SUFFIXES include/x86_64-linux-gnu include include/tensorrt
    )
endif()

# Verify NvOnnxParser.h is in the same location
if(TENSORRT_INCLUDE_DIRS)
    if(NOT EXISTS "${TENSORRT_INCLUDE_DIRS}/NvOnnxParser.h")
        message(WARNING "NvInfer.h found but NvOnnxParser.h not in ${TENSORRT_INCLUDE_DIRS}")
    endif()
endif()

# Libraries
find_library(TENSORRT_NVINFER_LIBRARY
    NAMES nvinfer libnvinfer
    PATHS ${_TENSORRT_SEARCH_PATHS}
    PATH_SUFFIXES lib/x86_64-linux-gnu lib lib64
    NO_DEFAULT_PATH
)

if(NOT TENSORRT_NVINFER_LIBRARY)
    find_library(TENSORRT_NVINFER_LIBRARY
        NAMES nvinfer libnvinfer
        PATH_SUFFIXES lib/x86_64-linux-gnu lib lib64
    )
endif()

find_library(TENSORRT_NVONNXPARSER_LIBRARY
    NAMES nvonnxparser libnvonnxparser
    PATHS ${_TENSORRT_SEARCH_PATHS}
    PATH_SUFFIXES lib/x86_64-linux-gnu lib lib64
    NO_DEFAULT_PATH
)

if(NOT TENSORRT_NVONNXPARSER_LIBRARY)
    find_library(TENSORRT_NVONNXPARSER_LIBRARY
        NAMES nvonnxparser libnvonnxparser
        PATH_SUFFIXES lib/x86_64-linux-gnu lib lib64
    )
endif()

set(TENSORRT_LIBRARIES ${TENSORRT_NVINFER_LIBRARY} ${TENSORRT_NVONNXPARSER_LIBRARY})

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(TensorRT
    REQUIRED_VARS TENSORRT_INCLUDE_DIRS TENSORRT_NVINFER_LIBRARY TENSORRT_NVONNXPARSER_LIBRARY
)

if(TENSORRT_FOUND AND NOT TARGET TensorRT::nvinfer)
    add_library(TensorRT::nvinfer SHARED IMPORTED)
    set_target_properties(TensorRT::nvinfer PROPERTIES
        IMPORTED_LOCATION "${TENSORRT_NVINFER_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${TENSORRT_INCLUDE_DIRS}"
    )
endif()

if(TENSORRT_FOUND AND NOT TARGET TensorRT::nvonnxparser)
    add_library(TensorRT::nvonnxparser SHARED IMPORTED)
    set_target_properties(TensorRT::nvonnxparser PROPERTIES
        IMPORTED_LOCATION "${TENSORRT_NVONNXPARSER_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${TENSORRT_INCLUDE_DIRS}"
    )
endif()

mark_as_advanced(
    TENSORRT_INCLUDE_DIRS
    TENSORRT_NVINFER_LIBRARY
    TENSORRT_NVONNXPARSER_LIBRARY
)
