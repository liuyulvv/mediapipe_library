# Copyright 2019 The MediaPipe Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

licenses(["notice"])

package(default_visibility = ["//mediapipe/examples:__subpackages__"])

cc_binary(
    name = "face_blendshape_cpu",
    data = ["//mediapipe/modules/face_blendshape:face_blendshapes.tflite"],
    deps = [
        "//mediapipe/examples/desktop:demo_run_graph_main",
        "//mediapipe/tasks/cc/vision/face_landmarker:face_blendshapes_graph",
        "//mediapipe/calculators/core:constant_side_packet_calculator",
        "//mediapipe/calculators/core:flow_limiter_calculator",
        "//mediapipe/graphs/face_mesh/subgraphs:face_renderer_cpu",
        "//mediapipe/modules/face_landmark:face_landmark_front_cpu",
    ],
)