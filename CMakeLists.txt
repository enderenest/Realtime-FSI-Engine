#https://github.com/meemknight/GLFWCMakeSetup
#Version 1.0.0

cmake_minimum_required(VERSION 3.16)

set(CMAKE_CXX_STANDARD 17)
# vckpg integration, make sure to change this to the correct path on your machine
set(CMAKE_TOOLCHAIN_FILE "C:/Programming/vcpkg/scripts/buildsystems/vcpkg.cmake" CACHE STRING "" FORCE)

#! ! ! ! ! ! !
#set this to true to ship the game!
#basically this will change the RESOURCES_PATH to be the local path
#! ! ! ! ! ! !
#DELETE THE OUT FOLDER AFTER CHANGING THIS BECAUSE VISUAL STUDIO DOESN'T SEEM TO RECOGNIZE THIS CHANGE AND REBUILD!
option(PRODUCTION_BUILD "Make this a production build" OFF)
#DELETE THE OUT FOLDER AFTER CHANGING THIS BECAUSE VISUAL STUDIO DOESN'T SEEM TO RECOGNIZE THIS CHANGE AND REBUILD!


set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Release>:Release>")
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")

set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE) #link time optimization

if(MSVC) 
add_compile_options(/arch:AVX2) #make sure SIMD optimizations take place
endif()


project(fsi)


set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GLFW_INSTALL OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)


add_subdirectory(thirdparty/glad)					#opengl 4.6 loader — must come before polyscope so its if(NOT TARGET glad) guard uses ours
add_subdirectory(thirdparty/glfw-3.3.2)				#window opener
add_subdirectory(thirdparty/polyscope)				#viewer
# add_subdirectory(thirdparty/glm)					#math
# add_subdirectory(thirdparty/imgui-docking)		#ui
add_subdirectory(thirdparty/openmesh)
add_subdirectory(thirdparty/libigl)


# MY_SOURCES is defined to be a list of all the source files for my game 
# DON'T ADD THE SOURCES BY HAND, they are already added with this macro
file(GLOB_RECURSE MY_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")

# Exclude legacy raw-OpenGL rendering files (superseded by Polyscope)
# GPU compute wrappers (SSBO, ComputeShader, UBO) are still active and must stay in the build
list(FILTER MY_SOURCES EXCLUDE REGEX "src/opengl/VAO\\.cpp")
list(FILTER MY_SOURCES EXCLUDE REGEX "src/opengl/VBO\\.cpp")
list(FILTER MY_SOURCES EXCLUDE REGEX "src/opengl/EBO\\.cpp")
list(FILTER MY_SOURCES EXCLUDE REGEX "src/opengl/openglErrorReporting\\.cpp")
list(FILTER MY_SOURCES EXCLUDE REGEX "src/interaction/Gizmo\\.cpp")


add_executable("${CMAKE_PROJECT_NAME}")

set_property(TARGET "${CMAKE_PROJECT_NAME}" PROPERTY CXX_STANDARD 17)


target_compile_definitions("${CMAKE_PROJECT_NAME}" PUBLIC GLFW_INCLUDE_NONE=1) 

if(PRODUCTION_BUILD)
	# setup the ASSETS_PATH macro to be in the root folder of your exe
	target_compile_definitions("${CMAKE_PROJECT_NAME}" PUBLIC RESOURCES_PATH="./resources/") 

	target_compile_definitions("${CMAKE_PROJECT_NAME}" PUBLIC PRODUCTION_BUILD=1) 

else()
	# This is useful to get an ASSETS_PATH in your IDE during development
	target_compile_definitions("${CMAKE_PROJECT_NAME}" PUBLIC RESOURCES_PATH="${CMAKE_CURRENT_SOURCE_DIR}/resources/")
	target_compile_definitions("${CMAKE_PROJECT_NAME}" PUBLIC PRODUCTION_BUILD=0) 

endif()

target_sources("${CMAKE_PROJECT_NAME}" PRIVATE ${MY_SOURCES} )


if(MSVC) # If using the VS compiler...

	target_compile_definitions("${CMAKE_PROJECT_NAME}" PUBLIC _CRT_SECURE_NO_WARNINGS)

	# Add this line to fix the OpenMesh math error:
	target_compile_definitions("${CMAKE_PROJECT_NAME}" PUBLIC _USE_MATH_DEFINES)

	#add this line if you want to remove the console!
	#set_target_properties("${CMAKE_PROJECT_NAME}" PROPERTIES LINK_FLAGS "/SUBSYSTEM:WINDOWS /ENTRY:mainCRTStartup") #no console
	
	#set_property(TARGET "${CMAKE_PROJECT_NAME}" PROPERTY MSVC_RUNTIME_LIBRARY "MultiThreadedDebug<$<CONFIG:Debug>:Debug>")
	#set_property(TARGET "${CMAKE_PROJECT_NAME}" PROPERTY MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Release>:Release>")

endif()

# Ask vcpkg to find CHOLMOD (from SuiteSparse)
find_package(CHOLMOD CONFIG REQUIRED)

# Tell Eigen it is allowed to use CHOLMOD
target_compile_definitions("${CMAKE_PROJECT_NAME}" PUBLIC EIGEN_USE_CHOLMOD)

target_include_directories("${CMAKE_PROJECT_NAME}" PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include/")
target_include_directories("${CMAKE_PROJECT_NAME}" PRIVATE 
    thirdparty/eigen 
    thirdparty/openmesh/src
	thirdparty/libigl/include
)

target_link_libraries("${CMAKE_PROJECT_NAME}" PRIVATE glad polyscope OpenMeshCore OpenMeshTools SuiteSparse::CHOLMOD)


