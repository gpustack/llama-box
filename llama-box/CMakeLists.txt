# llama-box

#
# version
#
if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../.git")
    set(GIT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../.git")

    # Is git submodule
    if (NOT IS_DIRECTORY "${GIT_DIR}")
        file(READ ${GIT_DIR} REAL_GIT_DIR_LINK)
        string(REGEX REPLACE "gitdir: (.*)\n$" "\\1" REAL_GIT_DIR ${REAL_GIT_DIR_LINK})
        string(FIND "${REAL_GIT_DIR}" "." SLASH_POS)
        if (SLASH_POS EQUAL 0)
            set(GIT_DIR "${REAL_GIT_DIR}")
        else ()
            set(GIT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../${REAL_GIT_DIR}")
        endif ()
    endif ()

    if (EXISTS "${GIT_DIR}/index")
        set(GIT_INDEX "${GIT_DIR}/index")
    else ()
        set(GIT_INDEX "")
    endif ()
else ()
    set(GIT_INDEX "")
endif ()
add_custom_command(
        OUTPUT "${CMAKE_CURRENT_SOURCE_DIR}/version.cpp"
        COMMENT "Generating build details from Git"
        COMMAND ${CMAKE_COMMAND} -DMSVC=${MSVC} -DCMAKE_C_COMPILER_VERSION=${CMAKE_C_COMPILER_VERSION}
        -DCMAKE_C_COMPILER_ID=${CMAKE_C_COMPILER_ID} -DCMAKE_VS_PLATFORM_NAME=${CMAKE_VS_PLATFORM_NAME}
        -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER} -P "${CMAKE_CURRENT_SOURCE_DIR}/scripts/gen-version-cpp.cmake"
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/version.cpp.in" ${GIT_INDEX}
        VERBATIM
)
set(TARGET version)
add_library(${TARGET} OBJECT version.cpp)

#
# llama-box
#
set(TARGET llama-box)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
add_executable(${TARGET} main.cpp param.hpp ratelimiter.hpp utils.hpp)
target_link_libraries(${TARGET} PRIVATE version common ${CMAKE_THREAD_LIBS_INIT})
target_include_directories(${TARGET} PUBLIC ${CMAKE_SOURCE_DIR})
if (WIN32)
    TARGET_LINK_LIBRARIES(${TARGET} PRIVATE ws2_32)
endif()
target_compile_features(${TARGET} PUBLIC cxx_std_11)
