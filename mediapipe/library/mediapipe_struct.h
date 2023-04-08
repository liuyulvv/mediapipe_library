#ifndef MEDIAPIPE_STRUCT_H_
#define MEDIAPIPE_STRUCT_H_

#ifdef __cplusplus
extern "C" {
#endif

struct Landmark {
    float x_;
    float y_;
    float z_;
    float visibility_;
    float presence_;
};

typedef Landmark NormalizedLandmark;

typedef void (*landmark_callback)(NormalizedLandmark* normalized_landmark_list, unsigned size);

typedef void (*blend_shape_callback)(float* blend_shape, unsigned size);

enum HolisticCallbackType {
    POSE,
    FACE,
    LEFT_HAND,
    RIGHT_HAND
};

#ifdef __cplusplus
}
#endif

#endif