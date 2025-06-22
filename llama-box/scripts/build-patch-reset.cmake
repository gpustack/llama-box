set(VENDOR_PATHS ${CMAKE_CURRENT_SOURCE_DIR}/../llama.cpp ${CMAKE_CURRENT_SOURCE_DIR}/../stable-diffusion.cpp)

# Look for git
find_package(Git)
if (NOT Git_FOUND)
    find_program(GIT_EXECUTABLE NAMES git git.exe)
    if (GIT_EXECUTABLE)
        set(Git_FOUND TRUE)
    endif ()
endif ()
if (NOT Git_FOUND)
    message(FATAL_ERROR "Failed to clean patches: Git not found")
endif ()

# Revert patch
foreach (VENDOR_PATH ${VENDOR_PATHS})
    get_filename_component(VENDOR_NAME ${VENDOR_PATH} NAME)
    execute_process(
            COMMAND ${GIT_EXECUTABLE} -C ${VENDOR_PATH} reset --hard
            COMMAND ${GIT_EXECUTABLE} -C ${VENDOR_PATH} clean -df
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
            RESULT_VARIABLE PATCH_RESULT
    )
    if (NOT PATCH_RESULT EQUAL 0)
        message(FATAL_ERROR "Failed to clean vendor ${VENDOR_NAME}")
    endif ()
    message(STATUS "Cleaned vendor ${VENDOR_NAME}")
endforeach ()
