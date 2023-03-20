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

class FaceMeshInterface : public MediapipeInterface {
public:
    FaceMeshInterface() = default;
    virtual ~FaceMeshInterface() = default;

    using NormalizedLandmarkCallback = std::function<void(std::vector<NormalizedLandmarkList>& normalized_landmark_lists)>;
    virtual void SetObserveCallback(const NormalizedLandmarkCallback& callback) = 0;
    virtual void Observe() = 0;
};

class HandTrackInterface : public MediapipeInterface {
public:
    HandTrackInterface() = default;
    virtual ~HandTrackInterface() = default;

    using NormalizedLandmarkCallback = std::function<void(std::vector<NormalizedLandmarkList>& normalized_landmark_lists)>;
    virtual void SetObserveCallback(const NormalizedLandmarkCallback& callback) = 0;
    virtual void Observe() = 0;
};

class PoseTrackInterface : public MediapipeInterface {
public:
    PoseTrackInterface() = default;
    virtual ~PoseTrackInterface() = default;

    using NormalizedLandmarkCallback = std::function<void(NormalizedLandmarkList& normalized_landmark_list)>;
    virtual void SetObserveCallback(const NormalizedLandmarkCallback& callback) = 0;
    virtual void Observe() = 0;
};

class HolisticTrackInterface : public MediapipeInterface {
public:
    HolisticTrackInterface() = default;
    virtual ~HolisticTrackInterface() = default;

    using NormalizedLandmarkCallback = std::function<void(NormalizedLandmarkList& normalized_landmark_list)>;

    virtual void SetObserveCallback(const NormalizedLandmarkCallback& callback, const HolisticCallbackType& type) = 0;
    virtual void Observe() = 0;
};

#endif