#ifndef MEDIAPIPE_LIBRARY_H_
#define MEDIAPIPE_LIBRARY_H_

#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/opencv_core_inc.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe_interface.h"
#include "mediapipe_log.h"
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

class MediapipeLibrary : public MediapipeInterface {
public:
    MediapipeLibrary(); 
    virtual ~MediapipeLibrary() = default;

    virtual void SetLogger(const std::shared_ptr<MediapipeLogger>& logger) override;
    virtual void SetGraph(const std::string& path) override;
    virtual void SetPreviewCallback(const MatCallback& callback) override;
    virtual void Preview() override;
    virtual void Start() override;
    virtual void Detect(const cv::Mat& frame) override;
    virtual void Stop() override;

    const std::string INPUT_STREAM = "input_video";
    const std::string OUTPUT_STREAM = "output_video";
    std::shared_ptr<MediapipeLogger> logger_;
    mediapipe::CalculatorGraph graph_;
    MatCallback preview_callback_;
};

class FaceDetectLibrary : public FaceDetectInterface {
public:
    FaceDetectLibrary();

    virtual void SetLogger(const std::shared_ptr<MediapipeLogger>& logger) override;
    virtual void SetGraph(const std::string& path) override;
    virtual void SetPreviewCallback(const MatCallback& callback) override;
    virtual void SetObserveCallback(const DetectionCallback& callback) override;
    virtual void Preview() override;
    virtual void Observe() override;
    virtual void Start() override;
    virtual void Detect(const cv::Mat& frame) override;
    virtual void Stop() override;

private:
    std::unique_ptr<MediapipeLibrary> interface_;
    DetectionCallback observe_callback_;
};

#endif