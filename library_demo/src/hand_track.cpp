#include <chrono>
#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <thread>
#include <vector>

#include "mediapipe_library.h"

cv::Mat camera_bgr_frame;

NormalizedLandmark *landmark_lists = nullptr;
int landmark_lists_size = 0;

void LandmarkCallback(NormalizedLandmark *normalized_landmark_lists, size_t size) {
    int width = camera_bgr_frame.cols;
    int height = camera_bgr_frame.rows;
    landmark_lists_size = size;
    landmark_lists = normalized_landmark_lists;
    for (int i = 0; i < size; ++i) {
        auto &landmark = landmark_lists[i];
        landmark.x_ *= width;
        landmark.y_ *= height;
    }
}

const std::string GRAPH_PATH = "mediapipe/graphs/hand_tracking/hand_tracking_desktop_live.pbtxt";

int main() {
    CreateHandTrackInterface(GRAPH_PATH.c_str());

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

    SetHandTrackObserveCallback(LandmarkCallback);

    ObserveHandTrack();

    StartHandTrack();

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
        HandTrackProcess(&camera_rgb_frame);

        if (camera_bgr_frame.cols > 0) {
            for (int i = 0; i < landmark_lists_size; ++i) {
                auto &landmark = landmark_lists[i];
                cv::circle(camera_bgr_frame, cv::Point2f(landmark.x_, landmark.y_), 2, cv::Scalar(255, 0, 0));
            }
            cv::imshow("MediaPipeLibrary", camera_bgr_frame);
        }
        int pressed_key = cv::waitKey(30);
        if (pressed_key >= 0 && pressed_key != 255) grab_frame = false;
    }
    StopHandTrack();
    ReleaseHandTrackInterface();
    return 0;
}