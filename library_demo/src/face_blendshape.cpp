#include <chrono>
#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <thread>
#include <vector>

#include "mediapipe_library.h"

#define _CRTDBG_MAP_ALLOC
#include <crtdbg.h>
#include <stdlib.h>

cv::Mat camera_bgr_frame;

float *blend_shape_list = nullptr;
int blend_shape_size = 0;

void BlendShapeCallback(float *blend_shapes, size_t size) {
    blend_shape_size = size;
    blend_shape_list = new float[size];
    memcpy(blend_shape_list, blend_shapes, sizeof(float) * size);
}

const std::string GRAPH_PATH = "mediapipe/graphs/face_blendshape/face_blendshape_desktop_live.pbtxt";

int main() {
    CreateFaceBlendShapeInterface(GRAPH_PATH.c_str());

    cv::namedWindow("MediaPipeLibrary");
    cv::VideoCapture capture;
    // capture.open(0);
    // bool is_camera = true;
    capture.open("D:/video/cxk.mp4");
    bool is_camera = false;

    bool grab_frame = true;
    if (!capture.isOpened()) {
        return -1;
    }

    SetFaceBlendShapeCallback(BlendShapeCallback);

    ObserveFaceBlendShape();

    StartFaceBlendShape();

    while (grab_frame) {
        capture >> camera_bgr_frame;
        if (is_camera) {
            cv::flip(camera_bgr_frame, camera_bgr_frame, 1);
        }
        if (camera_bgr_frame.empty()) {
            break;
        }
        cv::Mat camera_rgb_frame;
        cv::cvtColor(camera_bgr_frame, camera_rgb_frame, cv::COLOR_BGR2RGB);
        FaceMeshProcess(&camera_rgb_frame);

        if (camera_bgr_frame.cols > 0) {
            for (int i = 0; i < blend_shape_size; ++i) {
                std::cout << blend_shape_list[i] << " ";
            }
            std::cout << std::endl;
        }
        // delete[] landmark_lists;
        // blend_shape_size = 0;
        int pressed_key = cv::waitKey(30);
        if (pressed_key >= 0 && pressed_key != 255) grab_frame = false;
    }
    StopFaceMesh();
    ReleaseFaceMeshInterface();

    _CrtDumpMemoryLeaks();

    return 0;
}