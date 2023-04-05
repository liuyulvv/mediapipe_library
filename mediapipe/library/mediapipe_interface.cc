#include "mediapipe_interface.hpp"
#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/classification.pb.h"
#include "mediapipe/framework/formats/detection.pb.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/port/file_helpers.h"
#include "mediapipe/framework/port/opencv_core_inc.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/util/resource_util.h"
#include <exception>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include <iostream>

ABSL_DECLARE_FLAG(std::string, resource_root_dir);

MediapipeInterface::MediapipeInterface() {
    absl::SetFlag(&FLAGS_resource_root_dir, "");
    
}

void MediapipeInterface::SetGraph(const std::string& graph_name) {
    std::string graph_content;
    auto status = mediapipe::file::GetContents(graph_name, &graph_content);
    if(!status.ok()){
        std::cout << status.ToString() << std::endl ;
        throw std::runtime_error(status.ToString());
    }
    auto config = mediapipe::ParseTextProtoOrDie<mediapipe::CalculatorGraphConfig>(graph_content);
    status = graph_.Initialize(config);
    if (!status.ok()) {
        std::cout << status.ToString() << std::endl ;
        throw std::runtime_error(status.ToString());
    }
}

void MediapipeInterface::Start() {
    auto status = graph_.StartRun({});
    if (!status.ok()) {
        std::cout << status.ToString() << std::endl ;
        throw std::runtime_error(status.ToString());
    }
}

void MediapipeInterface::Process(const cv::Mat& input) {
    auto input_frame = absl::make_unique<mediapipe::ImageFrame>(mediapipe::ImageFormat::SRGB, input.cols, input.rows, mediapipe::ImageFrame::kDefaultAlignmentBoundary);
    auto input_frame_mat = mediapipe::formats::MatView(input_frame.get());
    input.copyTo(input_frame_mat);
    size_t frameTimestampUs = static_cast<double>(cv::getTickCount()) / static_cast<double>(cv::getTickFrequency()) * 1e6;
    auto status = graph_.AddPacketToInputStream(INPUT_STREAM_, mediapipe::Adopt(input_frame.release()).At(mediapipe::Timestamp(frameTimestampUs)));
    if (!status.ok()) {
        std::cout << status.ToString() << std::endl ;
        throw std::runtime_error(status.ToString());
    }
}

void MediapipeInterface::Stop() {
    static_cast<void>(graph_.CloseInputStream(INPUT_STREAM_));
    static_cast<void>(graph_.WaitUntilDone());
}

void MediapipeInterface::SetPreviewCallback(const MatCallback& callback) {
    preview_callback_ = callback;
}

void MediapipeInterface::Preview() {
    auto mat_callback = [&](const mediapipe::Packet& packet) {
        auto& output_frame = packet.Get<mediapipe::ImageFrame>();
        auto output_mat = mediapipe::formats::MatView(&output_frame);
        preview_callback_(output_mat);
        return absl::OkStatus();
    };
    auto status = graph_.ObserveOutputStream(OUTPUT_STREAM_, mat_callback);
    if (!status.ok()) {
        std::cout << status.ToString() << std::endl ;
        throw std::runtime_error(status.ToString());
    }
}

void FaceMeshInterface::SetObserveCallback(const landmark_callback& callback) {
    observe_callback_ = callback;
}

void FaceMeshInterface::Observe() {
    auto packet_callback = [&](const mediapipe::Packet& packet) {
        auto& multi_face_landmarks = packet.Get<std::vector<mediapipe::NormalizedLandmarkList>>();
        for (const auto& face_landmarks : multi_face_landmarks) {
            auto normalized_landmark_list = new NormalizedLandmark[face_landmarks.landmark_size()];
            for(int i = 0; i< face_landmarks.landmark_size(); ++i) {
                const auto& face_landmark = face_landmarks.landmark(i);
                normalized_landmark_list[i].x_ = face_landmark.x();
                normalized_landmark_list[i].y_ = face_landmark.y();
                normalized_landmark_list[i].z_ = face_landmark.z();
                normalized_landmark_list[i].visibility_ = face_landmark.visibility();
                normalized_landmark_list[i].presence_ = face_landmark.presence();
            }
            observe_callback_(normalized_landmark_list, face_landmarks.landmark_size());
            delete[] normalized_landmark_list;
            return absl::OkStatus();
        }
        return absl::OkStatus();
    };
    auto status = graph_.ObserveOutputStream("multi_face_landmarks", packet_callback);
    if (!status.ok()) {
        std::cout << status.ToString() << std::endl ;
        throw std::runtime_error(status.ToString());
    }
}

void FaceMeshInterface::AddOutputStreamPoller() {
    auto landmark_poller_or_status =  graph_.AddOutputStreamPoller("multi_face_landmarks");
    auto presence_poller_or_status =  graph_.AddOutputStreamPoller("multi_landmarks_presence");
    landmark_poller_ = std::make_shared<mediapipe::OutputStreamPoller>(std::move(landmark_poller_or_status.value()));
    presence_poller_ = std::make_shared<mediapipe::OutputStreamPoller>(std::move(presence_poller_or_status.value()));
}

void FaceMeshInterface::GetOutput(NormalizedLandmark * normalized_landmark_list, size_t size) {
    mediapipe::Packet packet;
    if (presence_poller_ && presence_poller_->Next(&packet)) {
        auto have_landmark = packet.Get<bool>();
        if (have_landmark) {
            if(landmark_poller_ && landmark_poller_->Next(&packet)) {
                auto& multi_face_landmarks = packet.Get<std::vector<mediapipe::NormalizedLandmarkList>>();
                // only one
                for (const auto& face_landmarks : multi_face_landmarks) {
                    assert(size == face_landmarks.landmark_size());
                    for(int i = 0; i < size; ++i) {
                        const auto& face_landmark = face_landmarks.landmark(i);
                        normalized_landmark_list[i].x_ = face_landmark.x();
                        normalized_landmark_list[i].y_ = face_landmark.y();
                        normalized_landmark_list[i].z_ = face_landmark.z();
                        normalized_landmark_list[i].visibility_ = face_landmark.visibility();
                        normalized_landmark_list[i].presence_ = face_landmark.presence();
                    }
                }
            }
        }
    }
}

void HandTrackInterface::SetObserveCallback(const landmark_callback& callback) {
    observe_callback_ = callback;
}

void HandTrackInterface::Observe() {
    auto packet_callback = [&](const mediapipe::Packet& packet) {
        auto& multi_hand_landmarks = packet.Get<std::vector<mediapipe::NormalizedLandmarkList>>();
        for (const auto& hand_landmarks : multi_hand_landmarks) {
            auto normalized_landmark_list = new NormalizedLandmark[hand_landmarks.landmark_size()];
            for(int i = 0; i< hand_landmarks.landmark_size(); ++i) {
                const auto& hand_landmark = hand_landmarks.landmark(i);
                normalized_landmark_list[i].x_ = hand_landmark.x();
                normalized_landmark_list[i].y_ = hand_landmark.y();
                normalized_landmark_list[i].z_ = hand_landmark.z();
                normalized_landmark_list[i].visibility_ = hand_landmark.visibility();
                normalized_landmark_list[i].presence_ = hand_landmark.presence();
            }
            observe_callback_(normalized_landmark_list, hand_landmarks.landmark_size());
            delete[] normalized_landmark_list;
            return absl::OkStatus();
        }
        return absl::OkStatus();
    };
    auto status = graph_.ObserveOutputStream("landmarks", packet_callback);
    if (!status.ok()) {
        std::cout << status.ToString() << std::endl ;
        throw std::runtime_error(status.ToString());
    }
}

void HandTrackInterface::AddOutputStreamPoller() {
    auto poller_or_status =  graph_.AddOutputStreamPoller("landmarks");
    poller_ = std::make_shared<mediapipe::OutputStreamPoller>(std::move(poller_or_status.value()));
}

void HandTrackInterface::GetOutput(NormalizedLandmark * normalized_landmark_list, size_t size) {
    mediapipe::Packet packet;
    if(poller_ && poller_->Next(&packet)) {
        auto& multi_face_landmarks = packet.Get<std::vector<mediapipe::NormalizedLandmarkList>>();
        // only one
        for (const auto& face_landmarks : multi_face_landmarks) {
            assert(size = face_landmarks.landmark_size());
            for(int i = 0; i < size; ++i) {
                const auto& face_landmark = face_landmarks.landmark(i);
                normalized_landmark_list[i].x_ = face_landmark.x();
                normalized_landmark_list[i].y_ = face_landmark.y();
                normalized_landmark_list[i].z_ = face_landmark.z();
                normalized_landmark_list[i].visibility_ = face_landmark.visibility();
                normalized_landmark_list[i].presence_ = face_landmark.presence();
            }
        }
    }
}

void PoseTrackInterface::SetObserveCallback(const landmark_callback& callback) {
    observe_callback_ = callback;
}

void PoseTrackInterface::Observe() {
    auto packet_callback = [&](const mediapipe::Packet& packet) {
        auto& pose_landmarks = packet.Get<mediapipe::NormalizedLandmarkList>();
        auto normalized_landmark_list = new NormalizedLandmark[pose_landmarks.landmark_size()];
        for(int i = 0; i< pose_landmarks.landmark_size(); ++i) {
            const auto& pose_landmark = pose_landmarks.landmark(i);
            normalized_landmark_list[i].x_ = pose_landmark.x();
            normalized_landmark_list[i].y_ = pose_landmark.y();
            normalized_landmark_list[i].z_ = pose_landmark.z();
            normalized_landmark_list[i].visibility_ = pose_landmark.visibility();
            normalized_landmark_list[i].presence_ = pose_landmark.presence();
        }
        observe_callback_(normalized_landmark_list, pose_landmarks.landmark_size());
        delete[] normalized_landmark_list;
        return absl::OkStatus();
    };
    auto status = graph_.ObserveOutputStream("pose_landmarks", packet_callback);
    if (!status.ok()) {
        std::cout << status.ToString() << std::endl ;
        throw std::runtime_error(status.ToString());
    }
}

void PoseTrackInterface::AddOutputStreamPoller() {
    auto poller_or_status =  graph_.AddOutputStreamPoller("pose_landmarks");
    poller_ = std::make_shared<mediapipe::OutputStreamPoller>(std::move(poller_or_status.value()));
}

void PoseTrackInterface::GetOutput(NormalizedLandmark * normalized_landmark_list, size_t size) {
    mediapipe::Packet packet;
    if(poller_ && poller_->Next(&packet)) {
        auto& multi_face_landmarks = packet.Get<std::vector<mediapipe::NormalizedLandmarkList>>();
        // only one
        for (const auto& face_landmarks : multi_face_landmarks) {
            assert(size = face_landmarks.landmark_size());
            for(int i = 0; i < size; ++i) {
                const auto& face_landmark = face_landmarks.landmark(i);
                normalized_landmark_list[i].x_ = face_landmark.x();
                normalized_landmark_list[i].y_ = face_landmark.y();
                normalized_landmark_list[i].z_ = face_landmark.z();
                normalized_landmark_list[i].visibility_ = face_landmark.visibility();
                normalized_landmark_list[i].presence_ = face_landmark.presence();
            }
        }
    }
}

void HolisticTrackInterface::SetObserveCallback(const landmark_callback & callback, const HolisticCallbackType& type) {
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

void HolisticTrackInterface::Observe() {
    if (pose_callback_) {
        auto packet_callback = [&](const mediapipe::Packet& packet) {
            auto& pose_landmarks = packet.Get<mediapipe::NormalizedLandmarkList>();
            auto normalized_landmark_list = new NormalizedLandmark[pose_landmarks.landmark_size()];
            for(int i = 0; i< pose_landmarks.landmark_size(); ++i) {
                const auto& pose_landmark = pose_landmarks.landmark(i);
                normalized_landmark_list[i].x_ = pose_landmark.x();
                normalized_landmark_list[i].y_ = pose_landmark.y();
                normalized_landmark_list[i].z_ = pose_landmark.z();
                normalized_landmark_list[i].visibility_ = pose_landmark.visibility();
                normalized_landmark_list[i].presence_ = pose_landmark.presence();
            }
            pose_callback_(normalized_landmark_list, pose_landmarks.landmark_size());
            delete[] normalized_landmark_list;
            return absl::OkStatus();
        };
        auto status = graph_.ObserveOutputStream("pose_landmarks", packet_callback);
        if (!status.ok()) {
            std::cout << status.ToString() << std::endl ;
            throw std::runtime_error(status.ToString());
        }
    }
    if (face_callback_) {
        auto packet_callback = [&](const mediapipe::Packet& packet) {
            auto& face_landmarks = packet.Get<mediapipe::NormalizedLandmarkList>();
            auto normalized_landmark_list = new NormalizedLandmark[face_landmarks.landmark_size()];
            for(int i = 0; i< face_landmarks.landmark_size(); ++i) {
                const auto& face_landmark = face_landmarks.landmark(i);
                normalized_landmark_list[i].x_ = face_landmark.x();
                normalized_landmark_list[i].y_ = face_landmark.y();
                normalized_landmark_list[i].z_ = face_landmark.z();
                normalized_landmark_list[i].visibility_ = face_landmark.visibility();
                normalized_landmark_list[i].presence_ = face_landmark.presence();
            }
            pose_callback_(normalized_landmark_list, face_landmarks.landmark_size());
            delete[] normalized_landmark_list;
            return absl::OkStatus();
        };
        auto status = graph_.ObserveOutputStream("face_landmarks", packet_callback);
        if (!status.ok()) {
            std::cout << status.ToString() << std::endl ;
            throw std::runtime_error(status.ToString());
        }
    }
    if (left_hand_callback_) {
        auto packet_callback = [&](const mediapipe::Packet& packet) {
            auto& left_hand_landmarks = packet.Get<mediapipe::NormalizedLandmarkList>();
            auto normalized_landmark_list = new NormalizedLandmark[left_hand_landmarks.landmark_size()];
            for(int i = 0; i< left_hand_landmarks.landmark_size(); ++i) {
                const auto& left_hand_landmark = left_hand_landmarks.landmark(i);
                normalized_landmark_list[i].x_ = left_hand_landmark.x();
                normalized_landmark_list[i].y_ = left_hand_landmark.y();
                normalized_landmark_list[i].z_ = left_hand_landmark.z();
                normalized_landmark_list[i].visibility_ = left_hand_landmark.visibility();
                normalized_landmark_list[i].presence_ = left_hand_landmark.presence();
            }
            pose_callback_(normalized_landmark_list, left_hand_landmarks.landmark_size());
            delete[] normalized_landmark_list;
            return absl::OkStatus();
        };
        auto status = graph_.ObserveOutputStream("left_hand_landmarks", packet_callback);
        if (!status.ok()) {
            std::cout << status.ToString() << std::endl ;
            throw std::runtime_error(status.ToString());
        }
    }
    if (right_hand_callback_) {
        auto packet_callback = [&](const mediapipe::Packet& packet) {
            auto& right_hand_landmarks = packet.Get<mediapipe::NormalizedLandmarkList>();
            auto normalized_landmark_list = new NormalizedLandmark[right_hand_landmarks.landmark_size()];
            for(int i = 0; i< right_hand_landmarks.landmark_size(); ++i) {
                const auto& right_hand_landmark = right_hand_landmarks.landmark(i);
                normalized_landmark_list[i].x_ = right_hand_landmark.x();
                normalized_landmark_list[i].y_ = right_hand_landmark.y();
                normalized_landmark_list[i].z_ = right_hand_landmark.z();
                normalized_landmark_list[i].visibility_ = right_hand_landmark.visibility();
                normalized_landmark_list[i].presence_ = right_hand_landmark.presence();
            }
            pose_callback_(normalized_landmark_list, right_hand_landmarks.landmark_size());
            delete[] normalized_landmark_list;
            return absl::OkStatus();
        };
        auto status = graph_.ObserveOutputStream("right_hand_landmarks", packet_callback);
        if (!status.ok()) {
            std::cout << status.ToString() << std::endl ;
            throw std::runtime_error(status.ToString());
        }
    }
}

void FaceBlendShapeInterface::SetObserveCallback(const blend_shape_callback & callback) {
    observe_callback_ = callback;
}

void FaceBlendShapeInterface::Observe() {
    auto packet_callback = [&](const mediapipe::Packet& packet) {
        auto& blend_shapes = packet.Get<mediapipe::ClassificationList>();
        auto blend_shape_list = new float[blend_shapes.classification_size()];
        for(int i = 0; i< blend_shapes.classification_size(); ++i) {
            const auto& blend_shape = blend_shapes.classification(i);
            blend_shape_list[i] = blend_shape.score();
        }
        observe_callback_(blend_shape_list, blend_shapes.classification_size());
        delete[] blend_shape_list;
        return absl::OkStatus();
    };
    auto status = graph_.ObserveOutputStream("blendshapes", packet_callback);
    if (!status.ok()) {
        std::cout << status.ToString() << std::endl ;
        throw std::runtime_error(status.ToString());
    }
}

void FaceBlendShapeInterface::AddOutputStreamPoller() {
    auto blend_shape_poller_or_status =  graph_.AddOutputStreamPoller("blendshapes");
    auto presence_poller_or_status =  graph_.AddOutputStreamPoller("landmarks_presence");
    poller_ = std::make_shared<mediapipe::OutputStreamPoller>(std::move(blend_shape_poller_or_status.value()));
    presence_poller_ = std::make_shared<mediapipe::OutputStreamPoller>(std::move(presence_poller_or_status.value()));
}

void FaceBlendShapeInterface::GetOutput(float * blend_shape_list, size_t size) {
    mediapipe::Packet packet;
    if(presence_poller_ && presence_poller_->Next(&packet)) {
        auto have = packet.Get<bool>();
        if(have){
            if(poller_ && poller_->Next(&packet)) {
                auto& blend_shapes = packet.Get<mediapipe::ClassificationList>();
                assert(size = blend_shapes.classification_size());
                for(int i = 0; i< size; ++i) {
                    const auto& blend_shape = blend_shapes.classification(i);
                    blend_shape_list[i] = blend_shape.score();
                }
            }
        }
    }

}

