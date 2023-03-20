#include <chrono>
#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <thread>
#include <vector>

#include "mediapipe_library.h"

class Logger : public MediapipeLogger {
public:
    virtual void Log(const std::string& content) const override {
        std::cout << content << std::endl;
    }
};

const std::string GRAPH_PATH = "graph/";
const std::string GRAPH_NAME = "hand_tracking_desktop_live.pbtxt";

int main() {
    // init mediapipe and logger
    auto logger = std::make_shared<Logger>();
    auto interface = CreateHandTrackInterface();
    interface->SetLogger(logger);

    cv::namedWindow("MediaPipe");
    cv::namedWindow("MediaPipeLibrary");
    cv::VideoCapture capture;
    // capture.open(0);
    // bool is_camera = true;
    capture.open("D:/video/cxk.mp4");
    bool is_camera = false;

    cv::Mat output_bgr_frame;
    cv::Mat camera_bgr_frame;

    bool grab_frame = true;
    if (!capture.isOpened()) {
        logger->Log("VideoCapture is not open");
        return -1;
    }

    interface->SetGraph(GRAPH_PATH + GRAPH_NAME);

    auto mat_callback = [&](const cv::Mat& frame) {
        cv::cvtColor(frame, output_bgr_frame, cv::COLOR_RGB2BGR);
    };
    interface->SetPreviewCallback(mat_callback);
    interface->Preview();

    std::vector<LandmarkList> landmark_lists;

    auto landmark_callback = [&](std::vector<NormalizedLandmarkList>& normalized_landmark_lists) {
        int width = camera_bgr_frame.cols;
        int height = camera_bgr_frame.rows;
        landmark_lists = normalized_landmark_lists;
        logger->Log(std::to_string(landmark_lists.size()));
        for(auto &landmark_list : landmark_lists) {
            for(auto &landmark : landmark_list) {
                landmark.x_ *= width;
                landmark.y_ *= height;
            }
        }
    };

    interface->SetObserveCallback(landmark_callback);
    interface->Observe();

    interface->Start();

    while (grab_frame) {
        capture >> camera_bgr_frame;
        if (is_camera) {
            cv::flip(camera_bgr_frame, camera_bgr_frame, 1);
        }
        if (camera_bgr_frame.empty()) {
            logger->Log("Empty frame.");
            break;
        }
        cv::Mat camera_rgb_frame;
        cv::cvtColor(camera_bgr_frame, camera_rgb_frame, cv::COLOR_BGR2RGB);
        interface->Detect(camera_rgb_frame);

        if (output_bgr_frame.cols > 0) {
            cv::imshow("MediaPipe", output_bgr_frame);
        }

        if (camera_bgr_frame.cols > 0) {
            for(auto &landmark_list : landmark_lists) {
                for(auto &landmark : landmark_list) {
                    cv::circle(camera_bgr_frame, cv::Point2f(landmark.x_, landmark.y_), 2, cv::Scalar(255, 0, 0));
                }
            }
            cv::imshow("MediaPipeLibrary", camera_bgr_frame);
        }

        int pressed_key = cv::waitKey(30);
        if (pressed_key >= 0 && pressed_key != 255) grab_frame = false;
    }
    interface->Stop();
    delete interface;
    return 0;
}