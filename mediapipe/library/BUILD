cc_binary(
    name = "libmediapipe.dylib",
    srcs = [
        "mediapipe_api.cc",
        "mediapipe_api.h",
        "mediapipe_interface.h",
        "mediapipe_library.cc",
        "mediapipe_library.h",
        "mediapipe_log.h",
        "mediapipe_struct.h",
    ],
    linkshared = True,
    deps = [
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/formats:image_frame",
        "//mediapipe/framework/formats:image_frame_opencv",
        "//mediapipe/framework/formats:landmark_cc_proto",
        "//mediapipe/framework/port:file_helpers",
        "//mediapipe/framework/port:opencv_core",
        "//mediapipe/framework/port:parse_text_proto",
        "//mediapipe/framework/port:status",
        "//mediapipe/util:resource_util",
        "//third_party:opencv",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        # face_detection
        "//mediapipe/graphs/face_detection:desktop_live_calculators",
        # face_mesh
        "//mediapipe/graphs/face_mesh:desktop_live_calculators",
        # hand_tracking
        "//mediapipe/graphs/hand_tracking:desktop_tflite_calculators",
        # pose_tracking
        "//mediapipe/graphs/pose_tracking:pose_tracking_cpu_deps",
        # holistic_tracking
        "//mediapipe/graphs/holistic_tracking:holistic_tracking_cpu_graph_deps",
    ],
)

cc_binary(
    name = "libmediapipe.so",
    srcs = [
        "mediapipe_api.cc",
        "mediapipe_api.h",
        "mediapipe_interface.h",
        "mediapipe_library.cc",
        "mediapipe_library.h",
        "mediapipe_log.h",
        "mediapipe_struct.h",
    ],
    linkshared = True,
    deps = [
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/formats:image_frame",
        "//mediapipe/framework/formats:image_frame_opencv",
        "//mediapipe/framework/formats:landmark_cc_proto",
        "//mediapipe/framework/port:file_helpers",
        "//mediapipe/framework/port:opencv_core",
        "//mediapipe/framework/port:parse_text_proto",
        "//mediapipe/framework/port:status",
        "//mediapipe/util:resource_util",
        "//third_party:opencv",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        # face_detection
        "//mediapipe/graphs/face_detection:desktop_live_calculators",
        # face_mesh
        "//mediapipe/graphs/face_mesh:desktop_live_calculators",
        # hand_tracking
        "//mediapipe/graphs/hand_tracking:desktop_tflite_calculators",
        # pose_tracking
        "//mediapipe/graphs/pose_tracking:pose_tracking_cpu_deps",
        # holistic_tracking
        "//mediapipe/graphs/holistic_tracking:holistic_tracking_cpu_graph_deps",
    ],
)

cc_binary(
    name = "libmediapipe.dll",
    srcs = [
        "mediapipe_api.cc",
        "mediapipe_api.h",
        "mediapipe_interface.h",
        "mediapipe_library.cc",
        "mediapipe_library.h",
        "mediapipe_log.h",
        "mediapipe_struct.h",
    ],
    linkshared = True,
    deps = [
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/formats:image_frame",
        "//mediapipe/framework/formats:image_frame_opencv",
        "//mediapipe/framework/formats:landmark_cc_proto",
        "//mediapipe/framework/port:file_helpers",
        "//mediapipe/framework/port:opencv_core",
        "//mediapipe/framework/port:parse_text_proto",
        "//mediapipe/framework/port:status",
        "//mediapipe/util:resource_util",
        "//third_party:opencv",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        # face_detection
        "//mediapipe/graphs/face_detection:desktop_live_calculators",
        # face_mesh
        "//mediapipe/graphs/face_mesh:desktop_live_calculators",
        # hand_tracking
        "//mediapipe/graphs/hand_tracking:desktop_tflite_calculators",
        # pose_tracking
        "//mediapipe/graphs/pose_tracking:pose_tracking_cpu_deps",
        # holistic_tracking
        "//mediapipe/graphs/holistic_tracking:holistic_tracking_cpu_graph_deps",
    ],
)
