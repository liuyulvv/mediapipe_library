#ifndef MEDIAPIPE_STRUCT_H_
#define MEDIAPIPE_STRUCT_H_

#include <string>
#include <vector>

enum class LocationDataFormat {
    GLOBAL = 0,
    BOUND_BOX,
    RELATIVE_BOUNDING_BOX,
    MASK
};

struct BoundingBox {
    int x_min_ = 0;
    int y_min_ = 0;
    int width_ = 0;
    int height_ = 0;
};

struct RelativeBoundingBox {
    float x_min_ = 0.f;
    float y_min_ = 0.f;
    float width_ = 0.f;
    float height_ = 0.f;
};

struct RasterizationInterval {
    int y_ = 0;
    int left_x_ = 0;
    int right_x_ = 0;
};

struct Rasterization {
    std::vector<RasterizationInterval> interval_{};
};

struct BinaryMask {
    int width_ = 0;
    int height_ = 0;
    Rasterization rasterization_{};
};

struct RelativeKeypoint {
    float x_ = 0.f;
    float y_ = 0.f;
    std::string keypoint_label_{};
    float score_ = 0.f;
};

struct LocationData {
    LocationDataFormat format_{};
    BoundingBox bounding_box_{};
    RelativeBoundingBox relative_bounding_box_{};
    BinaryMask mask_{};
    std::vector<RelativeKeypoint> relative_keypoints_{};
};

struct AssociateDetection {
    int id_ = 0;
    float confidence_ = 0.f;
};

struct Detection {
    std::vector<std::string> label_{};
    std::vector<int> label_id_{};
    std::vector<float> score_{};
    LocationData location_data_{};
    std::string feature_tag_{};
    std::string track_id_{};
    int64_t detection_id_ = 0;
    std::vector<AssociateDetection> associated_detections_{};
    std::vector<std::string> display_name_{};
    int64_t timestamp_usec_ = 10;
};

#endif