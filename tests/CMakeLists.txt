cmake_minimum_required(VERSION 3.28)
set(CMAKE_CXX_STANDARD 17)

# External dependencies
## OpenMP
find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

# Local
set(SOURCE_FILES
    main.cpp
	BruteforceNSearch.h
	BruteforceNSearch.cpp
	tests.cpp
	tests.h
	)
   
add_executable(tests ${SOURCE_FILES})
target_link_libraries(tests PUBLIC TreeNSearch)

# GPU (Metal) backend validation + benchmark. Apple only.
if (APPLE)
	add_executable(gpu_tests gpu_tests.cpp gpu_resident_check.mm BruteforceNSearch.cpp BruteforceNSearch.h)
	target_link_libraries(gpu_tests PUBLIC TreeNSearch)   # Metal/Foundation inherited (PUBLIC)
	target_include_directories(gpu_tests PRIVATE ${CMAKE_SOURCE_DIR}/TreeNSearch/source)
	# ARC for the Objective-C++ resident-solve test (uses __bridge casts).
	set_source_files_properties(gpu_resident_check.mm PROPERTIES COMPILE_FLAGS "-fobjc-arc")

	# Timing harness for the GPU-resident solve (wall time only, no correctness checks).
	add_executable(gpu_bench gpu_bench.mm)
	target_link_libraries(gpu_bench PUBLIC TreeNSearch)
	target_include_directories(gpu_bench PRIVATE ${CMAKE_SOURCE_DIR}/TreeNSearch/source)
	set_source_files_properties(gpu_bench.mm PROPERTIES COMPILE_FLAGS "-fobjc-arc")
endif()
