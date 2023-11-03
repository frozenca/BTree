set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH};${CMAKE_SOURCE_DIR}/cmake/modules")
cmake_minimum_required(VERSION 3.16)

set(MAJOR "0")
set(MINOR "0")
set(PATCH "1")
cmake_policy(SET CMP0048 NEW)

project(btree VERSION ${MAJOR}.${MINOR}.${PATCH})

set(CMAKE_CXX_STANDARD 20)

find_package(VersionHeader)
set(PROJECT_VERSION "${MAJOR}.${MINOR}.${PATCH}.${COMMITTER_SHORTSHA}")
file(WRITE ${CMAKE_SOURCE_DIR}/VERSION.txt "${MAJOR}.${MINOR}.${PATCH}")

find_package(Catch2 3 QUIET)
set(Catch_VER 3)
if (NOT Catch2_FOUND)
    find_package(Catch2 REQUIRED)
    set(Catch_VER 2)
endif()
set(CATCH_LIBS_ALIASES Catch2::Catch2 Catch2::Catch2WithMain)

function(get_linux_lsb_release_information)
    find_program(LSB_RELEASE_EXEC lsb_release)
    if(NOT LSB_RELEASE_EXEC)
        message(FATAL_ERROR "Could not detect lsb_release executable, can not gather required information")
    endif()

    execute_process(COMMAND "${LSB_RELEASE_EXEC}" --short --id OUTPUT_VARIABLE LSB_RELEASE_ID_SHORT OUTPUT_STRIP_TRAILING_WHITESPACE)
    execute_process(COMMAND "${LSB_RELEASE_EXEC}" --short --release OUTPUT_VARIABLE LSB_RELEASE_VERSION_SHORT OUTPUT_STRIP_TRAILING_WHITESPACE)
    execute_process(COMMAND "${LSB_RELEASE_EXEC}" --short --codename OUTPUT_VARIABLE LSB_RELEASE_CODENAME_SHORT OUTPUT_STRIP_TRAILING_WHITESPACE)

    set(LSB_RELEASE_ID_SHORT "${LSB_RELEASE_ID_SHORT}" PARENT_SCOPE)
    set(LSB_RELEASE_VERSION_SHORT "${LSB_RELEASE_VERSION_SHORT}" PARENT_SCOPE)
    set(LSB_RELEASE_CODENAME_SHORT "${LSB_RELEASE_CODENAME_SHORT}" PARENT_SCOPE)
endfunction()

message(STATUS "COMMITTER_FULLSHA ${COMMITTER_FULLSHA}")
message(STATUS "COMMITTER_SHORTSHA ${COMMITTER_SHORTSHA}")
message(STATUS "COMMITTER_DATE ${COMMITTER_DATE}")
message(STATUS "PROJECT_VERSION ${PROJECT_VERSION}")
if(CMAKE_SYSTEM_NAME MATCHES "Linux")
    get_linux_lsb_release_information()
    message(STATUS "OS Linux ${LSB_RELEASE_ID_SHORT} ${LSB_RELEASE_VERSION_SHORT} ${LSB_RELEASE_CODENAME_SHORT}")
else()
    message(STATUS "OS ${CMAKE_SYSTEM_NAME}")
    message(STATUS "OS VERSION ${CMAKE_SYSTEM_VERSION}")
endif()


add_library(btree INTERFACE
    include/fc/comp.h 
    include/fc/disk_fixed_alloc.h 
    include/fc/mmfile.h 
    include/fc/btree.h 
    include/fc/disk_btree.h
    include/fc/mmfile_win.h
    include/fc/mmfile_nix.h
    include/fc/details.h)

target_include_directories(
    btree
    INTERFACE
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
        "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>" )

add_library(btree::lib ALIAS btree)

add_executable(fc_tests
    test/fc_catch2.h
    test/unittest.cpp
    test/rwtest.cpp
    test/perftest.cpp)

if (${Catch_VER} EQUAL 2)
    message(STATUS "Used old target for catch2")
    if (NOT TARGET Catch2::Catch2WithMain)
        set(CATCH_LIBS_ALIASES Catch2::Catch2)
    endif()
    target_compile_definitions(fc_tests PRIVATE CATCH2_OLD)
endif()
target_compile_definitions(fc_tests PRIVATE CATCH_CONFIG_ENABLE_BENCHMARKING)
target_include_directories(fc_tests PRIVATE ${CMAKE_CURRENT_LIST_DIR})
target_link_libraries(fc_tests PRIVATE ${CATCH_LIBS_ALIASES} btree::lib)

if(MSVC)
    target_compile_options(fc_tests PRIVATE /W4 /WX /nologo /MDd /EHsc /std:c++latest /experimental:module)
    if (CMAKE_BUILD_TYPE STREQUAL "Release")
        add_compile_options(/Ox)
    endif()
else()
    target_compile_options(fc_tests PRIVATE -Wall -Wextra -Wpedantic -Werror -march=native)
endif()

include(CTest)
include(Catch)
catch_discover_tests(fc_tests)
