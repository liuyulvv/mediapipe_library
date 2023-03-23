#ifndef MEDIAPIPE_INTERFACE_HPP_
#define MEDIAPIPE_INTERFACE_HPP_

#include <opencv2/opencv.hpp>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "mediapipe_struct.h"
#include "mediapipe/framework/calculator_framework.h"

class MediapipeInterface {
public:
    MediapipeInterface();
    virtual ~MediapipeInterface() = default;

public:
    using MatCallback = std::function<void(const cv::Mat& frame)>;

    void SetGraph(const std::string& graph_name);
    void Start();
    void Process(const cv::Mat& frame);
    void Stop();

protected:
    // only for developer
    void SetPreviewCallback(const MatCallback& callback);
    void Preview();

protected:
    const std::string INPUT_STREAM_ = "input_video";
    const std::string OUTPUT_STREAM_ = "output_video";
    mediapipe::CalculatorGraph graph_;
    MatCallback preview_callback_;
};

class FaceMeshInterface final : public MediapipeInterface {
public:
    FaceMeshInterface() = default;
    ~FaceMeshInterface() = default;

    void SetObserveCallback(const landmark_callback& callback);
    void Observe();

private:
    landmark_callback observe_callback_;
};

class HandTrackInterface final : public MediapipeInterface {
public:
    HandTrackInterface() = default;
    ~HandTrackInterface() = default;

    void SetObserveCallback(const landmark_callback& callback);
    void Observe();

private:
    landmark_callback observe_callback_;
};

class PoseTrackInterface final : public MediapipeInterface {
public:
    PoseTrackInterface() = default;
    ~PoseTrackInterface() = default;

    void SetObserveCallback(const landmark_callback& callback);
    void Observe();

private:
    landmark_callback observe_callback_;
};

class HolisticTrackInterface final : public MediapipeInterface {
public:
    HolisticTrackInterface() = default;
    ~HolisticTrackInterface() = default;

    void SetObserveCallback(const landmark_callback& callback, const HolisticCallbackType& type);
    void Observe();

private:
    landmark_callback pose_callback_;
    landmark_callback face_callback_;
    landmark_callback left_hand_callback_;
    landmark_callback right_hand_callback_;
};

class FaceBlendShapeInterface final : public MediapipeInterface {
public:
    FaceBlendShapeInterface() = default;
    ~FaceBlendShapeInterface() = default;

    void SetObserveCallback(const blend_shape_callback& callback);
    void Observe();

private:
    blend_shape_callback observe_callback_;
};

#endif