cmake_minimum_required(VERSION 3.16)
project(btree)

set(CMAKE_CXX_STANDARD 20)

find_package(Catch2 3 REQUIRED)

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
target_include_directories(fc_tests PRIVATE ${CMAKE_CURRENT_LIST_DIR})
target_link_libraries(fc_tests PRIVATE Catch2::Catch2 Catch2::Catch2WithMain)

if(MSVC)
    target_compile_options(fc_tests PRIVATE /W4 /WX /nologo /MDd /EHsc /std:c++latest /experimental:module)
    if (CMAKE_BUILD_TYPE STREQUAL "Release")
        add_compile_options(/Ox)
    endif()
else()
    target_compile_options(fc_tests PRIVATE -Wall -Wextra -Wpedantic -Werror -march=native)
endif()
