set(VENDOR_PATHS ${CMAKE_CURRENT_SOURCE_DIR}/../llama.cpp)

# Look for git
find_package(Git)
if (NOT Git_FOUND)
    find_program(GIT_EXECUTABLE NAMES git git.exe)
    if (GIT_EXECUTABLE)
        set(Git_FOUND TRUE)
    endif ()
endif ()
if (NOT Git_FOUND)
    message(FATAL_ERROR "Failed to apply patches: Git not found")
endif ()

# Apply patch
foreach (VENDOR_PATH ${VENDOR_PATHS})
    get_filename_component(VENDOR_NAME ${VENDOR_PATH} NAME)
    message(STATUS "Patching vendor ${VENDOR_NAME}")
    execute_process(
            COMMAND ${GIT_EXECUTABLE} -C ${VENDOR_PATH} status --porcelain
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
            OUTPUT_VARIABLE TREE_STATE
            OUTPUT_STRIP_TRAILING_WHITESPACE
            RESULT_VARIABLE RES
    )
    if (NOT RES EQUAL 0)
        message(FATAL_ERROR "Failed to patch vendor ${VENDOR_NAME}: Cannot get tree state")
    endif ()
    if (TREE_STATE)
        message(WARNING "Skipped to patch vendor ${VENDOR_NAME}: Working tree is dirty")
        continue()
    endif ()
    file(GLOB_RECURSE PATCHES "${CMAKE_CURRENT_SOURCE_DIR}/patches/${VENDOR_NAME}/*.patch")
    foreach (PATCH_FILE ${PATCHES})
        execute_process(
                COMMAND ${GIT_EXECUTABLE} -C ${VENDOR_PATH} apply --check ${PATCH_FILE}
                WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                RESULT_VARIABLE PATCH_RESULT
        )
        if (PATCH_RESULT EQUAL 0)
            execute_process(
                    COMMAND ${GIT_EXECUTABLE} -C ${VENDOR_PATH} apply --whitespace=nowarn ${PATCH_FILE}
                    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
            )
            message(STATUS "  Applied ${PATCH_FILE}")
        else ()
            message(FATAL_ERROR "  Failed to apply ${PATCH_FILE}")
        endif ()
    endforeach ()
    message(STATUS "Patched vendor ${VENDOR_NAME}")
endforeach ()
