set(GIT_TREE_STATE "unknown")
set(GIT_COMMIT "unknown")
set(GIT_VERSION "unknown")

# Look for git
find_package(Git)
if (NOT Git_FOUND)
    find_program(GIT_EXECUTABLE NAMES git git.exe)
    if (GIT_EXECUTABLE)
        set(Git_FOUND TRUE)
    endif ()
endif ()

# Get the commit count and hash
if (Git_FOUND)
    execute_process(
            COMMAND ${GIT_EXECUTABLE} status --porcelain
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
            OUTPUT_VARIABLE TREE_STATE
            OUTPUT_STRIP_TRAILING_WHITESPACE
            RESULT_VARIABLE RES
    )
    if (RES EQUAL 0)
        if (TREE_STATE)
            set(GIT_TREE_STATE "dirty")
        else ()
            set(GIT_TREE_STATE "clean")
        endif ()
    endif ()
    execute_process(
            COMMAND ${GIT_EXECUTABLE} rev-parse --short HEAD
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
            OUTPUT_VARIABLE HEAD
            OUTPUT_STRIP_TRAILING_WHITESPACE
            RESULT_VARIABLE RES
    )
    if (RES EQUAL 0)
        set(GIT_COMMIT ${HEAD})
    endif ()
    if (DEFINED ENV{LLAMA_BOX_BUILD_VERSION})
        set(GIT_VERSION $ENV{LLAMA_BOX_BUILD_VERSION})
    else ()
        execute_process(
                COMMAND ${GIT_EXECUTABLE} rev-parse --abbrev-ref HEAD
                WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                OUTPUT_VARIABLE VERSION
                OUTPUT_STRIP_TRAILING_WHITESPACE
                RESULT_VARIABLE RES
        )
        if (RES EQUAL 0)
            if (GIT_TREE_STATE STREQUAL "dirty")
                set(GIT_VERSION "dev")
            else()
                set(GIT_VERSION ${VERSION})
            endif ()
        endif ()
    endif ()
endif ()

message(STATUS "Git tree state: ${GIT_TREE_STATE}")
message(STATUS "Git commit: ${GIT_COMMIT}")
message(STATUS "Git version: ${GIT_VERSION}")

set(TEMPLATE_FILE "${CMAKE_CURRENT_SOURCE_DIR}/version.cpp.in")
set(OUTPUT_FILE "${CMAKE_CURRENT_SOURCE_DIR}/version.cpp")

# Only write the version if it changed
if(EXISTS ${OUTPUT_FILE})
    file(READ ${OUTPUT_FILE} CONTENTS)
    string(REGEX MATCH "LLAMA_BOX_GIT_TREE_STATE = \"([^\"]*)\";" _ ${CONTENTS})
    set(OLD_GIT_TREE_STATE ${CMAKE_MATCH_1})
    string(REGEX MATCH "LLAMA_BOX_GIT_COMMIT = \"([^\"]*)\";" _ ${CONTENTS})
    set(OLD_GIT_COMMIT ${CMAKE_MATCH_1})
    string(REGEX MATCH "LLAMA_BOX_GIT_VERSION = \"([^\"]*)\";" _ ${CONTENTS})
    set(OLD_GIT_VERSION ${CMAKE_MATCH_1})
    if (
        NOT OLD_GIT_TREE_STATE   STREQUAL GIT_TREE_STATE   OR
        NOT OLD_GIT_COMMIT STREQUAL GIT_COMMIT OR
        NOT OLD_GIT_VERSION   STREQUAL GIT_VERSION
    )
        configure_file(${TEMPLATE_FILE} ${OUTPUT_FILE})
    endif()
else()
    configure_file(${TEMPLATE_FILE} ${OUTPUT_FILE})
endif()
