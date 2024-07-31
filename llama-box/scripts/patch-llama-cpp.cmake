set(LLAMA_CPP_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../llama.cpp)

# Look for git
find_package(Git)
if (NOT Git_FOUND)
    find_program(GIT_EXECUTABLE NAMES git git.exe)
    if (GIT_EXECUTABLE)
        set(Git_FOUND TRUE)
    endif ()
endif ()

# Apply patch
if (Git_FOUND)
    execute_process(
            COMMAND ${GIT_EXECUTABLE} -C ${LLAMA_CPP_DIR} status --porcelain
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
            OUTPUT_VARIABLE TREE_STATE
            OUTPUT_STRIP_TRAILING_WHITESPACE
            RESULT_VARIABLE RES
    )
    if (NOT RES EQUAL 0)
        message(FATAL_ERROR "Failed to apply patches: Cannot get tree state")
    endif ()
    if (TREE_STATE)
        message(WARNING "Skipped to apply patches: Working tree is dirty")
        return()
    endif ()
    file(GLOB_RECURSE PATCHES "${CMAKE_CURRENT_SOURCE_DIR}/patches/*.patch")
    foreach (PATCH_FILE ${PATCHES})
        execute_process(
                COMMAND ${GIT_EXECUTABLE} -C ${LLAMA_CPP_DIR} apply --check ${PATCH_FILE}
                WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                RESULT_VARIABLE PATCH_RESULT
        )
        if (PATCH_RESULT EQUAL 0)
            execute_process(
                    COMMAND ${GIT_EXECUTABLE} -C ${LLAMA_CPP_DIR} apply --whitespace=nowarn ${PATCH_FILE}
                    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
            )
            message(STATUS "Applied patch file ${PATCH_FILE}")
        else ()
            message(FATAL_ERROR "Failed to apply patches: Cannot apply patch file ${PATCH_FILE}")
        endif ()
    endforeach ()
else ()
    message(FATAL_ERROR "Failed to apply patches: Git not found")
endif ()
