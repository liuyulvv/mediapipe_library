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
    static_cast<void>(mediapipe::file::GetContents(path, &graph_content));
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
    static_cast<void>(graph_.ObserveOutputStream(OUTPUT_STREAM, mat_callback));
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
    static_cast<void>(graph_.CloseInputStream(INPUT_STREAM));
    static_cast<void>(graph_.WaitUntilDone());
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
    static_cast<void>(interface_->graph_.ObserveOutputStream("face_detections", packet_callback));
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

FaceMeshLibrary::FaceMeshLibrary() {
    interface_ = std::make_unique<MediapipeLibrary>();
}

void FaceMeshLibrary::SetLogger(const std::shared_ptr<MediapipeLogger>& logger) {
    interface_->SetLogger(logger);
}

void FaceMeshLibrary::SetGraph(const std::string & path) {
    interface_->SetGraph(path);
}

void FaceMeshLibrary::SetPreviewCallback(const MatCallback & callback) {
    interface_->SetPreviewCallback(callback);
}

void FaceMeshLibrary::SetObserveCallback(const NormalizedLandmarkCallback& callback) {
    observe_callback_ = callback;
}

void FaceMeshLibrary::Preview() {
    interface_->Preview();
}

void FaceMeshLibrary::Observe() {
    auto packet_callback = [&](const mediapipe::Packet& packet) {
        auto& multi_face_landmarks = packet.Get<std::vector<mediapipe::NormalizedLandmarkList>>();
        std::vector<NormalizedLandmarkList> ret;
        for (const auto& face_landmarks : multi_face_landmarks) {
            NormalizedLandmarkList normalized_landmark_list;
            for(int i = 0; i< face_landmarks.landmark_size(); ++i) {
                const auto& face_landmark = face_landmarks.landmark(i);
                NormalizedLandmark normalized_landmark {face_landmark.x(), face_landmark.y(), face_landmark.z(), face_landmark.visibility(), face_landmark.presence()};
                normalized_landmark_list.push_back(normalized_landmark);
            }
            ret.push_back(normalized_landmark_list);
        }
        observe_callback_(ret);
        return absl::OkStatus();
    };
    static_cast<void>(interface_->graph_.ObserveOutputStream("multi_face_landmarks", packet_callback));
}

void FaceMeshLibrary::Start() {
    interface_->Start();
}

void FaceMeshLibrary::Detect(const cv::Mat & frame) {
    interface_->Detect(frame);
}

void FaceMeshLibrary::Stop() {
    interface_->Stop();
}

HandTrackLibrary::HandTrackLibrary() {
    interface_ = std::make_unique<MediapipeLibrary>();
}

void HandTrackLibrary::SetLogger(const std::shared_ptr<MediapipeLogger>& logger) {
    interface_->SetLogger(logger);
}

void HandTrackLibrary::SetGraph(const std::string & path) {
    interface_->SetGraph(path);
}

void HandTrackLibrary::SetPreviewCallback(const MatCallback & callback) {
    interface_->SetPreviewCallback(callback);
}

void HandTrackLibrary::SetObserveCallback(const NormalizedLandmarkCallback& callback) {
    observe_callback_ = callback;
}

void HandTrackLibrary::Preview() {
    interface_->Preview();
}

void HandTrackLibrary::Observe() {
    auto packet_callback = [&](const mediapipe::Packet& packet) {
        auto& multi_hand_landmarks = packet.Get<std::vector<mediapipe::NormalizedLandmarkList>>();
        std::vector<NormalizedLandmarkList> ret;
        for (const auto& hand_landmarks : multi_hand_landmarks) {
            NormalizedLandmarkList normalized_landmark_list;
            for(int i = 0; i< hand_landmarks.landmark_size(); ++i) {
                const auto& hand_landmark = hand_landmarks.landmark(i);
                NormalizedLandmark normalized_landmark {hand_landmark.x(), hand_landmark.y(), hand_landmark.z(), hand_landmark.visibility(), hand_landmark.presence()};
                normalized_landmark_list.push_back(normalized_landmark);
            }
            ret.push_back(normalized_landmark_list);
        }
        observe_callback_(ret);
        return absl::OkStatus();
    };
    static_cast<void>(interface_->graph_.ObserveOutputStream("multi_hand_landmarks", packet_callback));
}

void HandTrackLibrary::Start() {
    interface_->Start();
}

void HandTrackLibrary::Detect(const cv::Mat & frame) {
    interface_->Detect(frame);
}

void HandTrackLibrary::Stop() {
    interface_->Stop();
}

PoseTrackLibrary::PoseTrackLibrary() {
    interface_ = std::make_unique<MediapipeLibrary>();
}

void PoseTrackLibrary::SetLogger(const std::shared_ptr<MediapipeLogger>& logger) {
    interface_->SetLogger(logger);
}

void PoseTrackLibrary::SetGraph(const std::string & path) {
    interface_->SetGraph(path);
}

void PoseTrackLibrary::SetPreviewCallback(const MatCallback & callback) {
    interface_->SetPreviewCallback(callback);
}

void PoseTrackLibrary::SetObserveCallback(const NormalizedLandmarkCallback& callback) {
    observe_callback_ = callback;
}

void PoseTrackLibrary::Preview() {
    interface_->Preview();
}

void PoseTrackLibrary::Observe() {
    auto packet_callback = [&](const mediapipe::Packet& packet) {
        auto& pose_landmarks = packet.Get<mediapipe::NormalizedLandmarkList>();
        NormalizedLandmarkList ret;
        for(int i = 0; i< pose_landmarks.landmark_size(); ++i) {
            const auto& pose_landmark = pose_landmarks.landmark(i);
            NormalizedLandmark normalized_landmark {pose_landmark.x(), pose_landmark.y(), pose_landmark.z(), pose_landmark.visibility(), pose_landmark.presence()};
            ret.push_back(normalized_landmark);
        }
        observe_callback_(ret);
        return absl::OkStatus();
    };
    static_cast<void>(interface_->graph_.ObserveOutputStream("pose_landmarks", packet_callback));
}

void PoseTrackLibrary::Start() {
    interface_->Start();
}

void PoseTrackLibrary::Detect(const cv::Mat & frame) {
    interface_->Detect(frame);
}

void PoseTrackLibrary::Stop() {
    interface_->Stop();
}

HolisticTrackLibrary::HolisticTrackLibrary() {
    interface_ = std::make_unique<MediapipeLibrary>();
}

void HolisticTrackLibrary::SetLogger(const std::shared_ptr<MediapipeLogger>& logger) {
    interface_->SetLogger(logger);
}

void HolisticTrackLibrary::SetGraph(const std::string & path) {
    interface_->SetGraph(path);
}

void HolisticTrackLibrary::SetPreviewCallback(const MatCallback & callback) {
    interface_->SetPreviewCallback(callback);
}

void HolisticTrackLibrary::SetObserveCallback(const NormalizedLandmarkCallback & callback, const HolisticCallbackType& type) {
    switch (type) {
        case HolisticCallbackType::POSE:
            pose_callback_ = callback;
            break;
        case HolisticCallbackType::FACE:
            face_callback_ = callback;
            break;
        case HolisticCallbackType::LEFT_HAND:
            left_hand_callback_ = callback;
            break;
        case HolisticCallbackType::RIGHT_HAND:
            right_hand_callback_ = callback;
            break;
    }
}

void HolisticTrackLibrary::Preview() {
    interface_->Preview();
}

void HolisticTrackLibrary::Observe() {
    if (pose_callback_) {
        auto packet_callback = [&](const mediapipe::Packet& packet) {
            auto& pose_landmarks = packet.Get<mediapipe::NormalizedLandmarkList>();
            NormalizedLandmarkList ret;
            for(int i = 0; i< pose_landmarks.landmark_size(); ++i) {
                const auto& pose_landmark = pose_landmarks.landmark(i);
                NormalizedLandmark normalized_landmark {pose_landmark.x(), pose_landmark.y(), pose_landmark.z(), pose_landmark.visibility(), pose_landmark.presence()};
                ret.push_back(normalized_landmark);
            }
            pose_callback_(ret);
            return absl::OkStatus();
        };
        static_cast<void>(interface_->graph_.ObserveOutputStream("pose_landmarks", packet_callback));
    }
    if (face_callback_) {
        auto packet_callback = [&](const mediapipe::Packet& packet) {
            auto& pose_landmarks = packet.Get<mediapipe::NormalizedLandmarkList>();
            NormalizedLandmarkList ret;
            for(int i = 0; i< pose_landmarks.landmark_size(); ++i) {
                const auto& pose_landmark = pose_landmarks.landmark(i);
                NormalizedLandmark normalized_landmark {pose_landmark.x(), pose_landmark.y(), pose_landmark.z(), pose_landmark.visibility(), pose_landmark.presence()};
                ret.push_back(normalized_landmark);
            }
            face_callback_(ret);
            return absl::OkStatus();
        };
        static_cast<void>(interface_->graph_.ObserveOutputStream("face_landmarks", packet_callback));
    }
    if (left_hand_callback_) {
        auto packet_callback = [&](const mediapipe::Packet& packet) {
            auto& pose_landmarks = packet.Get<mediapipe::NormalizedLandmarkList>();
            NormalizedLandmarkList ret;
            for(int i = 0; i< pose_landmarks.landmark_size(); ++i) {
                const auto& pose_landmark = pose_landmarks.landmark(i);
                NormalizedLandmark normalized_landmark {pose_landmark.x(), pose_landmark.y(), pose_landmark.z(), pose_landmark.visibility(), pose_landmark.presence()};
                ret.push_back(normalized_landmark);
            }
            left_hand_callback_(ret);
            return absl::OkStatus();
        };
        static_cast<void>(interface_->graph_.ObserveOutputStream("left_hand_landmarks", packet_callback));
    }
    if (right_hand_callback_) {
        auto packet_callback = [&](const mediapipe::Packet& packet) {
            auto& pose_landmarks = packet.Get<mediapipe::NormalizedLandmarkList>();
            NormalizedLandmarkList ret;
            for(int i = 0; i< pose_landmarks.landmark_size(); ++i) {
                const auto& pose_landmark = pose_landmarks.landmark(i);
                NormalizedLandmark normalized_landmark {pose_landmark.x(), pose_landmark.y(), pose_landmark.z(), pose_landmark.visibility(), pose_landmark.presence()};
                ret.push_back(normalized_landmark);
            }
            right_hand_callback_(ret);
            return absl::OkStatus();
        };
        static_cast<void>(interface_->graph_.ObserveOutputStream("right_hand_landmarks", packet_callback));
    }
}

void HolisticTrackLibrary::Start() {
    interface_->Start();
}

void HolisticTrackLibrary::Detect(const cv::Mat & frame) {
    interface_->Detect(frame);
}

void HolisticTrackLibrary::Stop() {
    interface_->Stop();
}
