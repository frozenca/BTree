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
find_package(Catch2 REQUIRED)
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

add_executable(fc_tests
    test/unittest.cpp
    test/rwtest.cpp
    fc_comp.h 
    fc_disk_fixed_alloc.h 
    fc_mmfile.h fc_btree.h 
    fc_disk_btree.h
    fc_mmfile_win.h
    fc_mmfile_nix.h
    fc_details.h)

if (NOT TARGET Catch2::Catch2WithMain)
    message(STATUS "Used old target for catch2")
    set(CATCH_LIBS_ALIASES Catch2::Catch2)
    set_source_files_properties(unittest.cpp PROPERTIES COMPILE_DEFINITIONS CATCH_CONFIG_MAIN)
endif()
target_include_directories(fc_tests PRIVATE ${CMAKE_CURRENT_LIST_DIR})
target_link_libraries(fc_tests PRIVATE ${CATCH_LIBS_ALIASES})

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