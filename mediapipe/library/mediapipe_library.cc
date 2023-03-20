#include "mediapipe_library.h"
#include "mediapipe_struct.h"
#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "mediapipe/framework/formats/classification.pb.h"
#include "mediapipe/framework/formats/detection.pb.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/port/file_helpers.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/util/resource_util.h"
#include <iostream>
#include <string>

ABSL_DECLARE_FLAG(std::string, resource_root_dir);

MediapipeLibrary::MediapipeLibrary() {
    absl::SetFlag(&FLAGS_resource_root_dir, "");
}

void MediapipeLibrary::SetLogger(const std::shared_ptr<MediapipeLogger>& logger) {
    logger_ = logger;
}

void MediapipeLibrary::SetGraph(const std::string& path) {
    std::string graph_content;
    mediapipe::file::GetContents(path, &graph_content);
    auto config = mediapipe::ParseTextProtoOrDie<mediapipe::CalculatorGraphConfig>(graph_content);
    auto status = graph_.Initialize(config);
    if (!status.ok()) {
        logger_->Log(status.ToString());
    }
}

void MediapipeLibrary::SetPreviewCallback(const MatCallback& callback) {
    preview_callback_ = callback;
}

void MediapipeLibrary::Preview() {
    auto mat_callback = [&](const mediapipe::Packet& packet) {
        auto& output_frame = packet.Get<mediapipe::ImageFrame>();
        auto output_mat = mediapipe::formats::MatView(&output_frame);
        preview_callback_(output_mat);
        return absl::OkStatus();
    };
    graph_.ObserveOutputStream(OUTPUT_STREAM, mat_callback);
}

void MediapipeLibrary::Start() {
    auto status = graph_.StartRun({});
    if (!status.ok()) {
        logger_->Log(status.ToString());
    }
}

void MediapipeLibrary::Detect(const cv::Mat& input) {
    auto input_frame = absl::make_unique<mediapipe::ImageFrame>(mediapipe::ImageFormat::SRGB, input.cols, input.rows, mediapipe::ImageFrame::kDefaultAlignmentBoundary);
    auto input_frame_mat = mediapipe::formats::MatView(input_frame.get());
    input.copyTo(input_frame_mat);
    size_t frameTimestampUs = static_cast<double>(cv::getTickCount()) / static_cast<double>(cv::getTickFrequency()) * 1e6;
    auto status = graph_.AddPacketToInputStream(INPUT_STREAM, mediapipe::Adopt(input_frame.release()).At(mediapipe::Timestamp(frameTimestampUs)));
    if (!status.ok()) {
        logger_->Log(status.ToString());
    }
}

void MediapipeLibrary::Stop() {
    graph_.CloseInputStream(INPUT_STREAM);
    graph_.WaitUntilDone();
}

FaceDetectLibrary::FaceDetectLibrary() {
    interface_ = std::make_unique<MediapipeLibrary>();
}

void FaceDetectLibrary::SetLogger(const std::shared_ptr<MediapipeLogger>& logger) {
    interface_->SetLogger(logger);
}

void FaceDetectLibrary::SetGraph(const std::string& path) {
    interface_->SetGraph(path);
}

void FaceDetectLibrary::SetPreviewCallback(const MatCallback& callback) {
    interface_->SetPreviewCallback(callback);
}

void FaceDetectLibrary::SetObserveCallback(const DetectionCallback & callback) {
    observe_callback_ = callback;
}

void FaceDetectLibrary::Preview() {
    interface_->Preview();
}

void FaceDetectLibrary::Observe() {
    auto packet_callback = [&](const mediapipe::Packet& packet) {
        auto& detections = packet.Get<std::vector<mediapipe::Detection>>();
        std::vector<Detection> ret;
        for (const auto& mediapipe_detection : detections) {
            Detection detection;
            detection.feature_tag_ = mediapipe_detection.feature_tag();
            detection.track_id_ = mediapipe_detection.track_id();
            detection.detection_id_ = mediapipe_detection.detection_id();
            detection.timestamp_usec_ = mediapipe_detection.timestamp_usec();
            for (int i = 0; i < mediapipe_detection.label_size(); ++i) {
                detection.label_.push_back(mediapipe_detection.label(i));
                detection.label_id_.push_back(mediapipe_detection.label_id(i));
                detection.score_.push_back(mediapipe_detection.score(i));
                detection.display_name_.push_back(mediapipe_detection.display_name(i));
                const auto& associated_detections = mediapipe_detection.associated_detections(i);
                detection.associated_detections_.push_back({associated_detections.id(), associated_detections.confidence()});
            }
            const auto& location = mediapipe_detection.location_data();
            detection.location_data_.format_ = static_cast<LocationDataFormat>(location.format());
            const auto& bounding_box = location.bounding_box();
            detection.location_data_.bounding_box_ = {bounding_box.xmin(), bounding_box.ymin(), bounding_box.width(), bounding_box.height()};
            const auto& relative_bounding_box = location.relative_bounding_box();
            detection.location_data_.relative_bounding_box_ = {relative_bounding_box.xmin(), relative_bounding_box.ymin(), relative_bounding_box.width(), relative_bounding_box.height()};
            const auto& mask = location.mask();
            const auto& rasterization = mask.rasterization();
            detection.location_data_.mask_.width_ = mask.width();
            detection.location_data_.mask_.height_ = mask.height();
            for (int i = 0; i < rasterization.interval_size(); ++i) {
                const auto& interval = rasterization.interval(i);
                detection.location_data_.mask_.rasterization_.interval_.push_back({interval.y(), interval.left_x(), interval.right_x()});
            }
            for (int i = 0; i < location.relative_keypoints_size(); ++i) {
                const auto& keypoint = location.relative_keypoints(i);
                detection.location_data_.relative_keypoints_.push_back({keypoint.x(), keypoint.y(), keypoint.keypoint_label(), keypoint.score()});
            }
            ret.push_back(detection);
        }
        observe_callback_(ret);
        return absl::OkStatus();
    };
    interface_->graph_.ObserveOutputStream("face_detections", packet_callback);
}

void FaceDetectLibrary::Start() {
    interface_->Start();
}

void FaceDetectLibrary::Detect(const cv::Mat& frame) {
    interface_->Detect(frame);
}

void FaceDetectLibrary::Stop() {
    interface_->Stop();
}