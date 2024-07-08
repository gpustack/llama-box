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
            COMMAND ${GIT_EXECUTABLE} -C ${LLAMA_CPP_DIR} reset --hard
            COMMAND ${GIT_EXECUTABLE} -C ${LLAMA_CPP_DIR} clean -df
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
            RESULT_VARIABLE PATCH_RESULT
    )
    if (PATCH_RESULT EQUAL 0)
        message(STATUS "Cleaned patches")
    else ()
        message(FATAL_ERROR "Failed to clean patches")
    endif ()
else ()
    message(FATAL_ERROR "Failed to clean patches: Git not found")
endif ()
