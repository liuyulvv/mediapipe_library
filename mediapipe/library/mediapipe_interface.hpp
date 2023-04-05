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

    void AddOutputStreamPoller();
    void GetOutput(NormalizedLandmark* normalized_landmark_list, size_t size);

private:
    landmark_callback observe_callback_;
    std::shared_ptr<mediapipe::OutputStreamPoller> landmark_poller_{nullptr};
    std::shared_ptr<mediapipe::OutputStreamPoller> presence_poller_{nullptr};
};

class HandTrackInterface final : public MediapipeInterface {
public:
    HandTrackInterface() = default;
    ~HandTrackInterface() = default;

    void SetObserveCallback(const landmark_callback& callback);
    void Observe();
    
    void AddOutputStreamPoller();
    void GetOutput(NormalizedLandmark* normalized_landmark_list, size_t size);

private:
    landmark_callback observe_callback_;
    std::shared_ptr<mediapipe::OutputStreamPoller> poller_{nullptr};
};

class PoseTrackInterface final : public MediapipeInterface {
public:
    PoseTrackInterface() = default;
    ~PoseTrackInterface() = default;

    void SetObserveCallback(const landmark_callback& callback);
    void Observe();

    void AddOutputStreamPoller();
    void GetOutput(NormalizedLandmark* normalized_landmark_list, size_t size);

private:
    landmark_callback observe_callback_;
    std::shared_ptr<mediapipe::OutputStreamPoller> poller_{nullptr};
};

class HolisticTrackInterface final : public MediapipeInterface {
public:
    HolisticTrackInterface() = default;
    ~HolisticTrackInterface() = default;

    void SetObserveCallback(const landmark_callback& callback, const HolisticCallbackType& type);
    void Observe();

    // void AddOutputStreamPoller(const std::string& stream_name);
    // void GetOutput(NormalizedLandmark* normalized_landmark_list, size_t* size);

private:
    landmark_callback pose_callback_;
    landmark_callback face_callback_;
    landmark_callback left_hand_callback_;
    landmark_callback right_hand_callback_;

    // mediapipe::OutputStreamPoller pose_poller_;
    // mediapipe::OutputStreamPoller face_poller_;
    // mediapipe::OutputStreamPoller left_hand_poller_;
    // mediapipe::OutputStreamPoller right_hand_poller_;
};

class FaceBlendShapeInterface final : public MediapipeInterface {
public:
    FaceBlendShapeInterface() = default;
    ~FaceBlendShapeInterface() = default;

    void SetObserveCallback(const blend_shape_callback& callback);
    void Observe();

    void AddOutputStreamPoller();
    void GetOutput(float* blend_shape_list, size_t size);

private:
    blend_shape_callback observe_callback_;
    std::shared_ptr<mediapipe::OutputStreamPoller> poller_{nullptr};
    std::shared_ptr<mediapipe::OutputStreamPoller> presence_poller_{nullptr};
};

#endif