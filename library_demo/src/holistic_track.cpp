#include <chrono>
#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <thread>
#include <vector>

#include "mediapipe_library.h"

cv::Mat camera_bgr_frame;

NormalizedLandmark *pose_landmark_lists = nullptr;
NormalizedLandmark *face_landmark_lists = nullptr;
NormalizedLandmark *left_hand_landmark_lists = nullptr;
NormalizedLandmark *right_hand_landmark_lists = nullptr;

int pose_landmark_lists_size = 0;
int face_landmark_lists_size = 0;
int left_hand_landmark_lists_size = 0;
int right_hand_landmark_lists_size = 0;

void PoseLandmarkCallback(NormalizedLandmark *normalized_landmark_lists, size_t size) {
    int width = camera_bgr_frame.cols;
    int height = camera_bgr_frame.rows;
    pose_landmark_lists_size = size;
    pose_landmark_lists = new NormalizedLandmark[size];
    memcpy(pose_landmark_lists, normalized_landmark_lists, sizeof(NormalizedLandmark) * size);
    for (int i = 0; i < size; ++i) {
        auto &landmark = pose_landmark_lists[i];
        landmark.x_ *= width;
        landmark.y_ *= height;
    }
}

void FaceLandmarkCallback(NormalizedLandmark *normalized_landmark_lists, size_t size) {
    int width = camera_bgr_frame.cols;
    int height = camera_bgr_frame.rows;
    face_landmark_lists_size = size;
    face_landmark_lists = new NormalizedLandmark[size];
    memcpy(face_landmark_lists, normalized_landmark_lists, sizeof(NormalizedLandmark) * size);
    for (int i = 0; i < size; ++i) {
        auto &landmark = face_landmark_lists[i];
        landmark.x_ *= width;
        landmark.y_ *= height;
    }
}

void LeftHandLandmarkCallback(NormalizedLandmark *normalized_landmark_lists, size_t size) {
    int width = camera_bgr_frame.cols;
    int height = camera_bgr_frame.rows;
    left_hand_landmark_lists_size = size;
    left_hand_landmark_lists = new NormalizedLandmark[size];
    memcpy(left_hand_landmark_lists, normalized_landmark_lists, sizeof(NormalizedLandmark) * size);
    for (int i = 0; i < size; ++i) {
        auto &landmark = left_hand_landmark_lists[i];
        landmark.x_ *= width;
        landmark.y_ *= height;
    }
}

void RightHandLandmarkCallback(NormalizedLandmark *normalized_landmark_lists, size_t size) {
    int width = camera_bgr_frame.cols;
    int height = camera_bgr_frame.rows;
    right_hand_landmark_lists_size = size;
    right_hand_landmark_lists = new NormalizedLandmark[size];
    memcpy(right_hand_landmark_lists, normalized_landmark_lists, sizeof(NormalizedLandmark) * size);
    for (int i = 0; i < size; ++i) {
        auto &landmark = right_hand_landmark_lists[i];
        landmark.x_ *= width;
        landmark.y_ *= height;
    }
}

const std::string GRAPH_PATH = "mediapipe/graphs/holistic_tracking/holistic_tracking_cpu.pbtxt";

int main() {
    CreateHolisticTrackInterface(GRAPH_PATH.c_str());

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

    SetHolisticTrackObserveCallback(PoseLandmarkCallback, HolisticCallbackType::POSE);
    SetHolisticTrackObserveCallback(FaceLandmarkCallback, HolisticCallbackType::FACE);
    SetHolisticTrackObserveCallback(LeftHandLandmarkCallback, HolisticCallbackType::LEFT_HAND);
    SetHolisticTrackObserveCallback(RightHandLandmarkCallback, HolisticCallbackType::RIGHT_HAND);

    ObserveHolisticTrack();

    StartHolisticTrack();

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
            for (int i = 0; i < pose_landmark_lists_size; ++i) {
                auto &landmark = pose_landmark_lists[i];
                cv::circle(camera_bgr_frame, cv::Point2f(landmark.x_, landmark.y_), 2, cv::Scalar(0, 0, 255));
            }
            for (int i = 0; i < face_landmark_lists_size; ++i) {
                auto &landmark = face_landmark_lists[i];
                cv::circle(camera_bgr_frame, cv::Point2f(landmark.x_, landmark.y_), 2, cv::Scalar(255, 0, 0));
            }
            for (int i = 0; i < left_hand_landmark_lists_size; ++i) {
                auto &landmark = left_hand_landmark_lists[i];
                cv::circle(camera_bgr_frame, cv::Point2f(landmark.x_, landmark.y_), 2, cv::Scalar(127, 127, 127));
            }
            for (int i = 0; i < right_hand_landmark_lists_size; ++i) {
                auto &landmark = right_hand_landmark_lists[i];
                cv::circle(camera_bgr_frame, cv::Point2f(landmark.x_, landmark.y_), 2, cv::Scalar(127, 127, 127));
            }
            cv::imshow("MediaPipeLibrary", camera_bgr_frame);
        }
        int pressed_key = cv::waitKey(30);
        if (pressed_key >= 0 && pressed_key != 255) grab_frame = false;
    }
    StopHolisticTrack();
    ReleaseHolisticTrackInterface();
    return 0;
}