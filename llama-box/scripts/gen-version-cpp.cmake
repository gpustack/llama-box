find_package(Git)
if (NOT Git_FOUND)
    find_program(GIT_EXECUTABLE NAMES git git.exe)
    if (GIT_EXECUTABLE)
        set(Git_FOUND TRUE)
    endif ()
endif ()

#
# LLaMA Box
#

set(LLAMA_BOX_COMMIT "unknown")
set(LLAMA_BOX_BUILD_NUMBER 0)
set(LLAMA_BOX_BUILD_VERSION "unknown")
set(LLAMA_BOX_BUILD_COMPILER "unknown")
set(LLAMA_BOX_BUILD_TARGET "unknown")

# Get the commit and count
if (Git_FOUND)
    # Get the commit
    execute_process(
            COMMAND ${GIT_EXECUTABLE} rev-parse --short HEAD
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
            OUTPUT_VARIABLE GIT_HEAD
            OUTPUT_STRIP_TRAILING_WHITESPACE
            RESULT_VARIABLE RES
    )
    if (RES EQUAL 0)
        set(LLAMA_BOX_COMMIT ${GIT_HEAD})
    endif ()
    # Get the commit count
    execute_process(
            COMMAND ${GIT_EXECUTABLE} rev-list --count HEAD
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
            OUTPUT_VARIABLE GIT_COUNT
            OUTPUT_STRIP_TRAILING_WHITESPACE
            RESULT_VARIABLE RES
    )
    if (RES EQUAL 0)
        set(LLAMA_BOX_BUILD_NUMBER ${GIT_COUNT})
    endif ()
endif ()

# Get the build version
if (DEFINED ENV{LLAMA_BOX_BUILD_VERSION})
    set(LLAMA_BOX_BUILD_VERSION $ENV{LLAMA_BOX_BUILD_VERSION})
else ()
    execute_process(
            COMMAND ${GIT_EXECUTABLE} status --porcelain
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
            OUTPUT_VARIABLE GIT_STATUS
            OUTPUT_STRIP_TRAILING_WHITESPACE
            RESULT_VARIABLE RES
    )
    if (RES EQUAL 0)
        if (GIT_STATUS)
            set(GIT_TREE_STATE "dirty")
        else ()
            set(GIT_TREE_STATE "clean")
        endif ()
    endif ()
    execute_process(
            COMMAND ${GIT_EXECUTABLE} rev-parse --abbrev-ref HEAD
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
            OUTPUT_VARIABLE GIT_VERSION
            OUTPUT_STRIP_TRAILING_WHITESPACE
            RESULT_VARIABLE RES
    )
    if (RES EQUAL 0)
        if (GIT_TREE_STATE STREQUAL "dirty")
            set(LLAMA_BOX_BUILD_VERSION "dev")
        else ()
            set(LLAMA_BOX_BUILD_VERSION ${GIT_VERSION})
        endif ()
    endif ()
endif ()

# Get the build compiler and target
if (MSVC)
    set(BUILD_COMPILER "${CMAKE_C_COMPILER_ID} ${CMAKE_C_COMPILER_VERSION}")
    set(BUILD_TARGET ${CMAKE_VS_PLATFORM_NAME})
else ()
    execute_process(
            COMMAND sh -c "$@ --version | head -1" _ ${CMAKE_C_COMPILER}
            OUTPUT_VARIABLE C_COMPILER
            OUTPUT_STRIP_TRAILING_WHITESPACE
            RESULT_VARIABLE RES
    )
    if (RES EQUAL 0)
        set(LLAMA_BOX_BUILD_COMPILER ${C_COMPILER})
    else ()
        message(WARNING "Failed to get the build compiler")
    endif ()
    execute_process(
            COMMAND ${CMAKE_C_COMPILER} -dumpmachine
            OUTPUT_VARIABLE C_TARGET
            OUTPUT_STRIP_TRAILING_WHITESPACE
            RESULT_VARIABLE RES
    )
    if (RES EQUAL 0)
        set(LLAMA_BOX_BUILD_TARGET ${C_TARGET})
    else ()
        message(WARNING "Failed to get the build target")
    endif ()
endif ()

# Print the version info
message(STATUS "LLaMA Box commit:                    ${LLAMA_BOX_COMMIT}")
message(STATUS "LLaMA Box build number:              ${LLAMA_BOX_BUILD_NUMBER}")
message(STATUS "LLaMA Box build version:             ${LLAMA_BOX_BUILD_VERSION}")
message(STATUS "LLaMA Box build compiler:            ${LLAMA_BOX_BUILD_COMPILER}")
message(STATUS "LLaMA Box build target:              ${LLAMA_BOX_BUILD_TARGET}")

#
# LLaMA CPP
#

set(LLAMA_CPP_COMMIT "unknown")
set(LLAMA_CPP_BUILD_NUMBER 0)

# Get the commit and count
if (Git_FOUND)
    # Get the commit
    execute_process(
            COMMAND ${GIT_EXECUTABLE} rev-parse --short HEAD
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../llama.cpp
            OUTPUT_VARIABLE GIT_HEAD
            OUTPUT_STRIP_TRAILING_WHITESPACE
            RESULT_VARIABLE RES
    )
    if (RES EQUAL 0)
        set(LLAMA_CPP_COMMIT ${GIT_HEAD})
    else ()
        message(WARNING "Failed to get the commit of llama.cpp")
    endif ()
    # Get the commit count
    execute_process(
            COMMAND ${GIT_EXECUTABLE} rev-list --count HEAD
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../llama.cpp
            OUTPUT_VARIABLE GIT_COUNT
            OUTPUT_STRIP_TRAILING_WHITESPACE
            RESULT_VARIABLE RES
    )
    if (RES EQUAL 0)
        set(LLAMA_CPP_BUILD_NUMBER ${GIT_COUNT})
    else ()
        message(WARNING "Failed to get the commit count of llama.cpp")
    endif ()
endif ()

# Print the version info
message(STATUS "LLaMA CPP commit:                    ${LLAMA_CPP_COMMIT}")
message(STATUS "LLaMA CPP build number:              ${LLAMA_CPP_BUILD_NUMBER}")

#
# Stable Diffusion CPP
#

set(STABLE_DIFFUSION_CPP_COMMIT "unknown")
set(STABLE_DIFFUSION_CPP_BUILD_NUMBER 0)

# Get the commit and count
if (Git_FOUND)
    # Get the commit
    execute_process(
            COMMAND ${GIT_EXECUTABLE} rev-parse --short HEAD
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../stable-diffusion.cpp
            OUTPUT_VARIABLE GIT_HEAD
            OUTPUT_STRIP_TRAILING_WHITESPACE
            RESULT_VARIABLE RES
    )
    if (RES EQUAL 0)
        set(STABLE_DIFFUSION_CPP_COMMIT ${GIT_HEAD})
    else ()
        message(WARNING "Failed to get the commit of stable-diffusion.cpp")
    endif ()
    # Get the commit count
    execute_process(
            COMMAND ${GIT_EXECUTABLE} rev-list --count HEAD
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../stable-diffusion.cpp
            OUTPUT_VARIABLE GIT_COUNT
            OUTPUT_STRIP_TRAILING_WHITESPACE
            RESULT_VARIABLE RES
    )
    if (RES EQUAL 0)
        set(STABLE_DIFFUSION_CPP_BUILD_NUMBER ${GIT_COUNT})
    else ()
        message(WARNING "Failed to get the commit count of stable-diffusion.cpp")
    endif ()
endif ()

# Print the version info
message(STATUS "Stable Diffusion CPP commit:         ${STABLE_DIFFUSION_CPP_COMMIT}")
message(STATUS "Stable Diffusion CPP build number:   ${STABLE_DIFFUSION_CPP_BUILD_NUMBER}")

#
# Write version info
#

set(TEMPLATE_FILE "${CMAKE_CURRENT_SOURCE_DIR}/version.cpp.in")
set(OUTPUT_FILE "${CMAKE_CURRENT_SOURCE_DIR}/version.cpp")
configure_file(${TEMPLATE_FILE} ${OUTPUT_FILE})
