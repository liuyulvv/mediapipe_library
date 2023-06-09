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

#include "mediapipe/tasks/cc/vision/interactive_segmenter/interactive_segmenter.h"

#include <memory>
#include <string>

#include "absl/flags/flag.h"
#include "mediapipe/framework/deps/file_path.h"
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/opencv_core_inc.h"
#include "mediapipe/framework/port/opencv_imgcodecs_inc.h"
#include "mediapipe/framework/port/status_matchers.h"
#include "mediapipe/tasks/cc/components/containers/keypoint.h"
#include "mediapipe/tasks/cc/components/containers/rect.h"
#include "mediapipe/tasks/cc/core/proto/base_options.pb.h"
#include "mediapipe/tasks/cc/core/proto/external_file.pb.h"
#include "mediapipe/tasks/cc/vision/core/image_processing_options.h"
#include "mediapipe/tasks/cc/vision/image_segmenter/calculators/tensors_to_segmentation_calculator.pb.h"
#include "mediapipe/tasks/cc/vision/image_segmenter/proto/image_segmenter_graph_options.pb.h"
#include "mediapipe/tasks/cc/vision/utils/image_utils.h"
#include "tensorflow/lite/core/shims/cc/shims_test_util.h"
#include "tensorflow/lite/kernels/builtin_op_kernels.h"
#include "tensorflow/lite/mutable_op_resolver.h"

namespace mediapipe {
namespace tasks {
namespace vision {
namespace interactive_segmenter {
namespace {

using ::mediapipe::Image;
using ::mediapipe::file::JoinPath;
using ::mediapipe::tasks::components::containers::NormalizedKeypoint;
using ::mediapipe::tasks::components::containers::RectF;
using ::mediapipe::tasks::vision::core::ImageProcessingOptions;
using ::testing::HasSubstr;
using ::testing::Optional;

constexpr char kTestDataDirectory[] = "/mediapipe/tasks/testdata/vision/";
constexpr char kPtmModel[] = "ptm_512_hdt_ptm_woid.tflite";
constexpr char kCatsAndDogsJpg[] = "cats_and_dogs.jpg";
// Golden mask for the dogs in cats_and_dogs.jpg.
constexpr char kCatsAndDogsMaskDog1[] = "cats_and_dogs_mask_dog1.png";
constexpr char kCatsAndDogsMaskDog2[] = "cats_and_dogs_mask_dog2.png";

constexpr float kGoldenMaskSimilarity = 0.97;

// Magnification factor used when creating the golden category masks to make
// them more human-friendly. Since interactive segmenter has only 2 categories,
// the golden mask uses 0 or 255 for each pixel.
constexpr int kGoldenMaskMagnificationFactor = 255;

// Intentionally converting output into CV_8UC1 and then again into CV_32FC1
// as expected outputs are stored in CV_8UC1, so this conversion allows to do
// fair comparison.
cv::Mat PostProcessResultMask(const cv::Mat& mask) {
  cv::Mat mask_float;
  mask.convertTo(mask_float, CV_8UC1, 255);
  mask_float.convertTo(mask_float, CV_32FC1, 1 / 255.f);
  return mask_float;
}

double CalculateSum(const cv::Mat& m) {
  double sum = 0.0;
  cv::Scalar s = cv::sum(m);
  for (int i = 0; i < m.channels(); ++i) {
    sum += s.val[i];
  }
  return sum;
}

double CalculateSoftIOU(const cv::Mat& m1, const cv::Mat& m2) {
  cv::Mat intersection;
  cv::multiply(m1, m2, intersection);
  double intersection_value = CalculateSum(intersection);
  double union_value =
      CalculateSum(m1.mul(m1)) + CalculateSum(m2.mul(m2)) - intersection_value;
  return union_value > 0.0 ? intersection_value / union_value : 0.0;
}

MATCHER_P2(SimilarToFloatMask, expected_mask, similarity_threshold, "") {
  cv::Mat actual_mask = PostProcessResultMask(arg);
  return arg.rows == expected_mask.rows && arg.cols == expected_mask.cols &&
         CalculateSoftIOU(arg, expected_mask) > similarity_threshold;
}

MATCHER_P3(SimilarToUint8Mask, expected_mask, similarity_threshold,
           magnification_factor, "") {
  if (arg.rows != expected_mask.rows || arg.cols != expected_mask.cols) {
    return false;
  }
  int consistent_pixels = 0;
  const int num_pixels = expected_mask.rows * expected_mask.cols;
  for (int i = 0; i < num_pixels; ++i) {
    consistent_pixels +=
        (arg.data[i] * magnification_factor == expected_mask.data[i]);
  }
  return static_cast<float>(consistent_pixels) / num_pixels >=
         similarity_threshold;
}

class CreateFromOptionsTest : public tflite_shims::testing::Test {};

class DeepLabOpResolverMissingOps : public ::tflite::MutableOpResolver {
 public:
  DeepLabOpResolverMissingOps() {
    AddBuiltin(::tflite::BuiltinOperator_ADD,
               ::tflite::ops::builtin::Register_ADD());
  }

  DeepLabOpResolverMissingOps(const DeepLabOpResolverMissingOps& r) = delete;
};

TEST_F(CreateFromOptionsTest, FailsWithSelectiveOpResolverMissingOps) {
  auto options = std::make_unique<InteractiveSegmenterOptions>();
  options->base_options.model_asset_path =
      JoinPath("./", kTestDataDirectory, kPtmModel);
  options->base_options.op_resolver =
      absl::make_unique<DeepLabOpResolverMissingOps>();
  auto segmenter_or = InteractiveSegmenter::Create(std::move(options));
  // TODO: Make MediaPipe InferenceCalculator report the detailed
  // interpreter errors (e.g., "Encountered unresolved custom op").
  EXPECT_EQ(segmenter_or.status().code(), absl::StatusCode::kInternal);
  EXPECT_THAT(
      segmenter_or.status().message(),
      testing::HasSubstr("interpreter_builder(&interpreter) == kTfLiteOk"));
}

TEST_F(CreateFromOptionsTest, FailsWithMissingModel) {
  absl::StatusOr<std::unique_ptr<InteractiveSegmenter>> segmenter_or =
      InteractiveSegmenter::Create(
          std::make_unique<InteractiveSegmenterOptions>());

  EXPECT_EQ(segmenter_or.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(
      segmenter_or.status().message(),
      HasSubstr("ExternalFile must specify at least one of 'file_content', "
                "'file_name', 'file_pointer_meta' or 'file_descriptor_meta'."));
  EXPECT_THAT(segmenter_or.status().GetPayload(kMediaPipeTasksPayload),
              Optional(absl::Cord(absl::StrCat(
                  MediaPipeTasksStatus::kRunnerInitializationError))));
}

struct InteractiveSegmenterTestParams {
  std::string test_name;
  RegionOfInterest::Format format;
  NormalizedKeypoint roi;
  std::string golden_mask_file;
  float similarity_threshold;
};

using SucceedSegmentationWithRoi =
    ::testing::TestWithParam<InteractiveSegmenterTestParams>;

TEST_P(SucceedSegmentationWithRoi, SucceedsWithCategoryMask) {
  const InteractiveSegmenterTestParams& params = GetParam();
  MP_ASSERT_OK_AND_ASSIGN(
      Image image,
      DecodeImageFromFile(JoinPath("./", kTestDataDirectory, kCatsAndDogsJpg)));
  RegionOfInterest interaction_roi;
  interaction_roi.format = params.format;
  interaction_roi.keypoint = params.roi;
  auto options = std::make_unique<InteractiveSegmenterOptions>();
  options->base_options.model_asset_path =
      JoinPath("./", kTestDataDirectory, kPtmModel);
  options->output_type = InteractiveSegmenterOptions::OutputType::CATEGORY_MASK;

  MP_ASSERT_OK_AND_ASSIGN(std::unique_ptr<InteractiveSegmenter> segmenter,
                          InteractiveSegmenter::Create(std::move(options)));
  MP_ASSERT_OK_AND_ASSIGN(auto category_masks,
                          segmenter->Segment(image, interaction_roi));
  EXPECT_EQ(category_masks.size(), 1);

  cv::Mat actual_mask = mediapipe::formats::MatView(
      category_masks[0].GetImageFrameSharedPtr().get());

  cv::Mat expected_mask =
      cv::imread(JoinPath("./", kTestDataDirectory, params.golden_mask_file),
                 cv::IMREAD_GRAYSCALE);
  EXPECT_THAT(actual_mask,
              SimilarToUint8Mask(expected_mask, params.similarity_threshold,
                                 kGoldenMaskMagnificationFactor));
}

TEST_P(SucceedSegmentationWithRoi, SucceedsWithConfidenceMask) {
  const auto& params = GetParam();
  MP_ASSERT_OK_AND_ASSIGN(
      Image image,
      DecodeImageFromFile(JoinPath("./", kTestDataDirectory, kCatsAndDogsJpg)));
  RegionOfInterest interaction_roi;
  interaction_roi.format = params.format;
  interaction_roi.keypoint = params.roi;
  auto options = std::make_unique<InteractiveSegmenterOptions>();
  options->base_options.model_asset_path =
      JoinPath("./", kTestDataDirectory, kPtmModel);
  options->output_type =
      InteractiveSegmenterOptions::OutputType::CONFIDENCE_MASK;

  MP_ASSERT_OK_AND_ASSIGN(std::unique_ptr<InteractiveSegmenter> segmenter,
                          InteractiveSegmenter::Create(std::move(options)));
  MP_ASSERT_OK_AND_ASSIGN(auto confidence_masks,
                          segmenter->Segment(image, interaction_roi));
  EXPECT_EQ(confidence_masks.size(), 2);

  cv::Mat expected_mask =
      cv::imread(JoinPath("./", kTestDataDirectory, params.golden_mask_file),
                 cv::IMREAD_GRAYSCALE);
  cv::Mat expected_mask_float;
  expected_mask.convertTo(expected_mask_float, CV_32FC1, 1 / 255.f);

  cv::Mat actual_mask = mediapipe::formats::MatView(
      confidence_masks[1].GetImageFrameSharedPtr().get());
  EXPECT_THAT(actual_mask, SimilarToFloatMask(expected_mask_float,
                                              params.similarity_threshold));
}

INSTANTIATE_TEST_SUITE_P(
    SucceedSegmentationWithRoiTest, SucceedSegmentationWithRoi,
    ::testing::ValuesIn<InteractiveSegmenterTestParams>(
        {{"PointToDog1", RegionOfInterest::KEYPOINT,
          NormalizedKeypoint{0.44, 0.70}, kCatsAndDogsMaskDog1, 0.84f},
         {"PointToDog2", RegionOfInterest::KEYPOINT,
          NormalizedKeypoint{0.66, 0.66}, kCatsAndDogsMaskDog2,
          kGoldenMaskSimilarity}}),
    [](const ::testing::TestParamInfo<SucceedSegmentationWithRoi::ParamType>&
           info) { return info.param.test_name; });

class ImageModeTest : public tflite_shims::testing::Test {};

// TODO: fix this unit test after image segmenter handled post
// processing correctly with rotated image.
TEST_F(ImageModeTest, DISABLED_SucceedsWithRotation) {
  MP_ASSERT_OK_AND_ASSIGN(
      Image image,
      DecodeImageFromFile(JoinPath("./", kTestDataDirectory, kCatsAndDogsJpg)));
  RegionOfInterest interaction_roi;
  interaction_roi.format = RegionOfInterest::KEYPOINT;
  interaction_roi.keypoint = NormalizedKeypoint{0.66, 0.66};
  auto options = std::make_unique<InteractiveSegmenterOptions>();
  options->base_options.model_asset_path =
      JoinPath("./", kTestDataDirectory, kPtmModel);
  options->output_type =
      InteractiveSegmenterOptions::OutputType::CONFIDENCE_MASK;

  MP_ASSERT_OK_AND_ASSIGN(std::unique_ptr<InteractiveSegmenter> segmenter,
                          InteractiveSegmenter::Create(std::move(options)));
  ImageProcessingOptions image_processing_options;
  image_processing_options.rotation_degrees = -90;
  MP_ASSERT_OK_AND_ASSIGN(
      auto confidence_masks,
      segmenter->Segment(image, interaction_roi, image_processing_options));
  EXPECT_EQ(confidence_masks.size(), 2);
}

TEST_F(ImageModeTest, FailsWithRegionOfInterest) {
  MP_ASSERT_OK_AND_ASSIGN(
      Image image,
      DecodeImageFromFile(JoinPath("./", kTestDataDirectory, kCatsAndDogsJpg)));
  RegionOfInterest interaction_roi;
  interaction_roi.format = RegionOfInterest::KEYPOINT;
  interaction_roi.keypoint = NormalizedKeypoint{0.66, 0.66};
  auto options = std::make_unique<InteractiveSegmenterOptions>();
  options->base_options.model_asset_path =
      JoinPath("./", kTestDataDirectory, kPtmModel);
  options->output_type =
      InteractiveSegmenterOptions::OutputType::CONFIDENCE_MASK;

  MP_ASSERT_OK_AND_ASSIGN(std::unique_ptr<InteractiveSegmenter> segmenter,
                          InteractiveSegmenter::Create(std::move(options)));
  RectF roi{/*left=*/0.1, /*top=*/0, /*right=*/0.9, /*bottom=*/1};
  ImageProcessingOptions image_processing_options{roi, /*rotation_degrees=*/0};

  auto results =
      segmenter->Segment(image, interaction_roi, image_processing_options);
  EXPECT_EQ(results.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(results.status().message(),
              HasSubstr("This task doesn't support region-of-interest"));
  EXPECT_THAT(
      results.status().GetPayload(kMediaPipeTasksPayload),
      Optional(absl::Cord(absl::StrCat(
          MediaPipeTasksStatus::kImageProcessingInvalidArgumentError))));
}

}  // namespace
}  // namespace interactive_segmenter
}  // namespace vision
}  // namespace tasks
}  // namespace mediapipe
