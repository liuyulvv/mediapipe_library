/* Copyright 2023 The MediaPipe Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef MEDIAPIPE_TASKS_CC_VISION_FACE_STYLIZER_FACE_STYLIZER_H_
#define MEDIAPIPE_TASKS_CC_VISION_FACE_STYLIZER_FACE_STYLIZER_H_

#include <functional>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/tasks/cc/core/base_options.h"
#include "mediapipe/tasks/cc/vision/core/base_vision_task_api.h"
#include "mediapipe/tasks/cc/vision/core/image_processing_options.h"
#include "tensorflow/lite/kernels/register.h"

namespace mediapipe {
namespace tasks {
namespace vision {
namespace face_stylizer {

// The options for configuring a mediapipe face stylizer task.
struct FaceStylizerOptions {
  // Base options for configuring MediaPipe Tasks, such as specifying the model
  // file with metadata, accelerator options, op resolver, etc.
  tasks::core::BaseOptions base_options;

  // The running mode of the task. Default to the image mode.
  // Face stylizer has three running modes:
  // 1) The image mode for stylizing faces on single image inputs.
  // 2) The video mode for stylizing faces on the decoded frames of a video.
  // 3) The live stream mode for stylizing faces on the live stream of input
  // data, such as from camera. In this mode, the "result_callback" below must
  // be specified to receive the stylization results asynchronously.
  core::RunningMode running_mode = core::RunningMode::IMAGE;

  // The user-defined result callback for processing live stream data.
  // The result callback should only be specified when the running mode is set
  // to RunningMode::LIVE_STREAM.
  std::function<void(absl::StatusOr<mediapipe::Image>, const Image&, int64_t)>
      result_callback = nullptr;
};

// Performs face stylization on images.
class FaceStylizer : tasks::vision::core::BaseVisionTaskApi {
 public:
  using BaseVisionTaskApi::BaseVisionTaskApi;

  // Creates a FaceStylizer from the provided options.
  static absl::StatusOr<std::unique_ptr<FaceStylizer>> Create(
      std::unique_ptr<FaceStylizerOptions> options);

  // Performs face stylization on the provided single image.
  //
  // The optional 'image_processing_options' parameter can be used to specify:
  //   - the rotation to apply to the image before performing stylization, by
  //     setting its 'rotation_degrees' field.
  //   and/or
  //   - the region-of-interest on which to perform stylization, by setting its
  //   'region_of_interest' field. If not specified, the full image is used.
  // If both are specified, the crop around the region-of-interest is extracted
  // first, then the specified rotation is applied to the crop.
  //
  // Only use this method when the FaceStylizer is created with the image
  // running mode.
  //
  // The input image can be of any size with format RGB or RGBA.
  // To ensure that the output image has reasonable quality, the stylized output
  // image size is the smaller of the model output size and the size of the
  // 'region_of_interest' specified in 'image_processing_options'.
  absl::StatusOr<mediapipe::Image> Stylize(
      mediapipe::Image image,
      std::optional<core::ImageProcessingOptions> image_processing_options =
          std::nullopt);

  // Performs face stylization on the provided video frame.
  //
  // The optional 'image_processing_options' parameter can be used to specify:
  //   - the rotation to apply to the image before performing stylization, by
  //     setting its 'rotation_degrees' field.
  //   and/or
  //   - the region-of-interest on which to perform stylization, by setting its
  //   'region_of_interest' field. If not specified, the full image is used.
  // If both are specified, the crop around the region-of-interest is extracted
  // first, then the specified rotation is applied to the crop.
  //
  // Only use this method when the FaceStylizer is created with the video
  // running mode.
  //
  // The image can be of any size with format RGB or RGBA. It's required to
  // provide the video frame's timestamp (in milliseconds). The input timestamps
  // must be monotonically increasing.
  // To ensure that the output image has reasonable quality, the stylized output
  // image size is the smaller of the model output size and the size of the
  // 'region_of_interest' specified in 'image_processing_options'.
  absl::StatusOr<mediapipe::Image> StylizeForVideo(
      mediapipe::Image image, int64_t timestamp_ms,
      std::optional<core::ImageProcessingOptions> image_processing_options =
          std::nullopt);

  // Sends live image data to perform face stylization, and the results will
  // be available via the "result_callback" provided in the
  // FaceStylizerOptions.
  //
  // The optional 'image_processing_options' parameter can be used to specify:
  //   - the rotation to apply to the image before performing stylization, by
  //     setting its 'rotation_degrees' field.
  //   and/or
  //   - the region-of-interest on which to perform stylization, by setting its
  //   'region_of_interest' field. If not specified, the full image is used.
  // If both are specified, the crop around the region-of-interest is extracted
  // first, then the specified rotation is applied to the crop.
  //
  // Only use this method when the FaceStylizer is created with the live stream
  // running mode.
  //
  // The image can be of any size with format RGB or RGBA. It's required to
  // provide a timestamp (in milliseconds) to indicate when the input image is
  // sent to the face stylizer. The input timestamps must be monotonically
  // increasing.
  //
  // The "result_callback" provides:
  //   - The stylized image which size is the smaller of the model output size
  //     and the size of the 'region_of_interest' specified in
  //     'image_processing_options'.
  //   - The input timestamp in milliseconds.
  absl::Status StylizeAsync(mediapipe::Image image, int64_t timestamp_ms,
                            std::optional<core::ImageProcessingOptions>
                                image_processing_options = std::nullopt);

  // Shuts down the FaceStylizer when all works are done.
  absl::Status Close() { return runner_->Close(); }
};

}  // namespace face_stylizer
}  // namespace vision
}  // namespace tasks
}  // namespace mediapipe

#endif  // MEDIAPIPE_TASKS_CC_VISION_FACE_STYLIZER_FACE_STYLIZER_H_
