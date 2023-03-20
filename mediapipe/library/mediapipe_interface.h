#ifndef MEDIAPIPE_INTERFACE_H_
#define MEDIAPIPE_INTERFACE_H_

#include "mediapipe_log.h"
#include "mediapipe_struct.h"
#include <opencv2/opencv.hpp>
#include <functional>
#include <memory>
#include <string>
#include <vector>

class MediapipeInterface {
public:
    MediapipeInterface() = default;
    virtual ~MediapipeInterface() = default;

    using MatCallback = std::function<void(const cv::Mat& frame)>;

    virtual void SetLogger(const std::shared_ptr<MediapipeLogger>& logger) = 0;
    virtual void SetGraph(const std::string& path) = 0;
    virtual void SetPreviewCallback(const MatCallback& callback) = 0;
    virtual void Preview() = 0;
    virtual void Start() = 0;
    virtual void Detect(const cv::Mat& frame) = 0;
    virtual void Stop() = 0;
};

class FaceDetectInterface : public MediapipeInterface {
public:
    FaceDetectInterface() = default;
    virtual ~FaceDetectInterface() = default;

    using DetectionCallback = std::function<void(std::vector<Detection>& detection)>;
    virtual void SetObserveCallback(const DetectionCallback& callback) = 0;
    virtual void Observe() = 0;
};

#endif