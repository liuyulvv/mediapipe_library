cmake_minimum_required(VERSION 3.15)

project(mediapipe_library_demo LANGUAGES CXX VERSION 0.1)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_INCLUDE_CURRENT_DIR ON)
SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR})

file(GLOB srcs RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")

include_directories(
    "./include"
    "./include/mediapipe"
)

if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")

    find_package(PkgConfig REQUIRED)
    pkg_check_modules(opencv REQUIRED IMPORTED_TARGET opencv4)

    link_directories(
        "./lib/macos"
    )

    foreach(mainfile IN LISTS srcs)
        get_filename_component(src_name ${mainfile} NAME_WE)
        add_executable(${src_name} ${mainfile})
        target_link_libraries(${src_name}
            PkgConfig::opencv
            libmediapipe
        )
    endforeach()

elseif(${CMAKE_SYSTEM_NAME} MATCHES "Windows")

    # set opencv dir
    set (OpenCV_DIR "D:/OpenCV/build/x64/vc15/lib")
    find_package(OpenCV REQUIRED)

    include_directories(
        ${OpenCV_INCLUDE_DIRS}
    )

    link_directories(
        "./lib/windows"
    )

    foreach(mainfile IN LISTS srcs)
        get_filename_component(src_name ${mainfile} NAME_WE)
        add_executable(${src_name} ${mainfile})
        target_link_libraries(${src_name}
            ${OpenCV_LIBS}
            libmediapipe
        )
    endforeach()

else()

    include_directories(
        "/usr/local/include/opencv4"
        "/usr/local/include/opencv4/opencv2"
    )
    link_directories(
        "./lib/linux"
        "/usr/local/lib"
    )
    foreach(mainfile IN LISTS srcs)
        get_filename_component(src_name ${mainfile} NAME_WE)
        add_executable(${src_name} ${mainfile})
        target_link_libraries(${src_name}
            libmediapipe
            opencv_core
            opencv_calib3d
            opencv_features2d
            opencv_highgui
            opencv_imgcodecs
            opencv_imgproc
            opencv_video
            opencv_videoio
        )
    endforeach()

endif()