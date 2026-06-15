set(INTERNAL_FILES
	source/internals/vectors_internals.h
	source/internals/octree_internals.h
	source/internals/shuffle_lut.h
)

set(INCLUDE_FILES
	source/NeighborList.h
	source/TreeNSearch.h
	)

set(SOURCE_FILES
	source/TreeNSearch.cpp
	)

# Apple Silicon: GPU (Metal) backend, compiled as Objective-C++.
if (APPLE)
	list(APPEND INTERNAL_FILES source/internals/metal_nsearch.h)
	list(APPEND SOURCE_FILES   source/internals/metal_nsearch.mm)
endif()

# Create a library version so it can be linked with other modules
add_library(TreeNSearch ${INTERNAL_FILES} ${SOURCE_FILES} ${INCLUDE_FILES})
target_include_directories(TreeNSearch PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/extern)
target_include_directories(TreeNSearch INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)

if (APPLE)
	find_library(METAL_LIBRARY Metal REQUIRED)
	find_library(FOUNDATION_LIBRARY Foundation REQUIRED)
	target_link_libraries(TreeNSearch PUBLIC ${METAL_LIBRARY} ${FOUNDATION_LIBRARY})
	# ARC for the Objective-C++ translation unit.
	set_source_files_properties(source/internals/metal_nsearch.mm PROPERTIES
		COMPILE_FLAGS "-fobjc-arc")
endif()

install(FILES ${HEADER_FILES} DESTINATION include/)
install(FILES ${INTERNAL_FILES} DESTINATION include/internals)

install(TARGETS TreeNSearch
	RUNTIME DESTINATION bin
	LIBRARY DESTINATION lib
	ARCHIVE DESTINATION lib
	)

source_group("internals" FILES ${INTERNAL_FILES})
source_group("include" FILES ${INCLUDE_FILES})
source_group("source" FILES ${SOURCE_FILES})

find_package(OpenGL REQUIRED)
find_package(glfw3 REQUIRED HINTS /opt/homebrew/lib/cmake/glfw3)

add_executable(demo ${CMAKE_CURRENT_SOURCE_DIR}/demo/main.cpp)

# include is relative to THIS CMakeLists.txt (TreeNSearch/include)
target_include_directories(demo PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/source)
target_link_libraries(demo PRIVATE TreeNSearch OpenGL::GL glfw)