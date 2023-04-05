/* Copyright 2022 The MediaPipe Authors. All Rights Reserved.

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

#include <memory>
#include <optional>
#include <type_traits>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "mediapipe/calculators/image/image_clone_calculator.pb.h"
#include "mediapipe/calculators/image/image_transformation_calculator.pb.h"
#include "mediapipe/calculators/tensor/tensor_converter_calculator.pb.h"
#include "mediapipe/framework/api2/builder.h"
#include "mediapipe/framework/api2/port.h"
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/framework/formats/rect.pb.h"
#include "mediapipe/framework/formats/tensor.h"
#include "mediapipe/framework/port/status_macros.h"
#include "mediapipe/tasks/cc/common.h"
#include "mediapipe/tasks/cc/components/processors/image_preprocessing_graph.h"
#include "mediapipe/tasks/cc/components/processors/proto/image_preprocessing_graph_options.pb.h"
#include "mediapipe/tasks/cc/core/model_resources.h"
#include "mediapipe/tasks/cc/core/model_task_graph.h"
#include "mediapipe/tasks/cc/core/proto/inference_subgraph.pb.h"
#include "mediapipe/tasks/cc/metadata/metadata_extractor.h"
#include "mediapipe/tasks/cc/vision/image_segmenter/calculators/tensors_to_segmentation_calculator.pb.h"
#include "mediapipe/tasks/cc/vision/image_segmenter/proto/image_segmenter_graph_options.pb.h"
#include "mediapipe/tasks/cc/vision/image_segmenter/proto/segmenter_options.pb.h"
#include "mediapipe/tasks/cc/vision/utils/image_tensor_specs.h"
#include "mediapipe/tasks/metadata/image_segmenter_metadata_schema_generated.h"
#include "mediapipe/tasks/metadata/metadata_schema_generated.h"
#include "mediapipe/util/graph_builder_utils.h"
#include "mediapipe/util/label_map.pb.h"
#include "mediapipe/util/label_map_util.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace mediapipe {
namespace tasks {
namespace vision {
namespace image_segmenter {

namespace {

using ::mediapipe::Image;
using ::mediapipe::NormalizedRect;
using ::mediapipe::api2::Input;
using ::mediapipe::api2::Output;
using ::mediapipe::api2::builder::Graph;
using ::mediapipe::api2::builder::MultiSource;
using ::mediapipe::api2::builder::Source;
using ::mediapipe::tasks::metadata::ModelMetadataExtractor;
using ::mediapipe::tasks::vision::image_segmenter::proto::
    ImageSegmenterGraphOptions;
using ::mediapipe::tasks::vision::image_segmenter::proto::SegmenterOptions;
using ::tflite::TensorMetadata;
using LabelItems = mediapipe::proto_ns::Map<int64_t, ::mediapipe::LabelMapItem>;

constexpr char kSegmentationTag[] = "SEGMENTATION";
constexpr char kGroupedSegmentationTag[] = "GROUPED_SEGMENTATION";
constexpr char kConfidenceMaskTag[] = "CONFIDENCE_MASK";
constexpr char kConfidenceMasksTag[] = "CONFIDENCE_MASKS";
constexpr char kCategoryMaskTag[] = "CATEGORY_MASK";
constexpr char kImageTag[] = "IMAGE";
constexpr char kImageCpuTag[] = "IMAGE_CPU";
constexpr char kImageGpuTag[] = "IMAGE_GPU";
constexpr char kNormRectTag[] = "NORM_RECT";
constexpr char kTensorsTag[] = "TENSORS";
constexpr char kOutputSizeTag[] = "OUTPUT_SIZE";
constexpr char kSegmentationMetadataName[] = "SEGMENTER_METADATA";

// Struct holding the different output streams produced by the image segmenter
// subgraph.
struct ImageSegmenterOutputs {
  std::optional<std::vector<Source<Image>>> segmented_masks;
  std::optional<std::vector<Source<Image>>> confidence_masks;
  std::optional<Source<Image>> category_mask;
  // The same as the input image, mainly used for live stream mode.
  Source<Image> image;
};

// Struct holding the image and input tensors after image preprocessing and
// transferred to the requested device.
struct ImageAndTensorsOnDevice {
  Source<Image> image;
  Source<std::vector<Tensor>> tensors;
};

}  // namespace

absl::Status SanityCheckOptions(const ImageSegmenterGraphOptions& options) {
  // TODO: remove deprecated output type support.
  if (options.segmenter_options().has_output_type() &&
      options.segmenter_options().output_type() ==
          SegmenterOptions::UNSPECIFIED) {
    return CreateStatusWithPayload(absl::StatusCode::kInvalidArgument,
                                   "`output_type` must not be UNSPECIFIED",
                                   MediaPipeTasksStatus::kInvalidArgumentError);
  }
  return absl::OkStatus();
}

absl::StatusOr<LabelItems> GetLabelItemsIfAny(
    const ModelMetadataExtractor& metadata_extractor,
    const TensorMetadata& tensor_metadata, absl::string_view locale) {
  const std::string labels_filename =
      ModelMetadataExtractor::FindFirstAssociatedFileName(
          tensor_metadata, tflite::AssociatedFileType_TENSOR_AXIS_LABELS);
  if (labels_filename.empty()) {
    LabelItems empty_label_items;
    return empty_label_items;
  }
  ASSIGN_OR_RETURN(absl::string_view labels_file,
                   metadata_extractor.GetAssociatedFile(labels_filename));
  const std::string display_names_filename =
      ModelMetadataExtractor::FindFirstAssociatedFileName(
          tensor_metadata, tflite::AssociatedFileType_TENSOR_AXIS_LABELS,
          locale);
  absl::string_view display_names_file;
  if (!display_names_filename.empty()) {
    ASSIGN_OR_RETURN(display_names_file, metadata_extractor.GetAssociatedFile(
                                             display_names_filename));
  }
  return mediapipe::BuildLabelMapFromFiles(labels_file, display_names_file);
}

absl::Status ConfigureTensorsToSegmentationCalculator(
    const ImageSegmenterGraphOptions& segmenter_option,
    const core::ModelResources& model_resources,
    TensorsToSegmentationCalculatorOptions* options) {
  // Set default activation function NONE
  options->mutable_segmenter_options()->CopyFrom(
      segmenter_option.segmenter_options());
  // Find the custom metadata of ImageSegmenterOptions type in model metadata.
  const auto* metadata_extractor = model_resources.GetMetadataExtractor();
  bool found_activation_in_metadata = false;
  if (metadata_extractor->GetCustomMetadataList() != nullptr &&
      metadata_extractor->GetCustomMetadataList()->size() > 0) {
    for (const auto& custom_metadata :
         *metadata_extractor->GetCustomMetadataList()) {
      if (custom_metadata->name()->str() == kSegmentationMetadataName) {
        found_activation_in_metadata = true;
        auto activation_fb =
            GetImageSegmenterOptions(custom_metadata->data()->data())
                ->activation();
        switch (activation_fb) {
          case Activation_NONE:
            options->mutable_segmenter_options()->set_activation(
                SegmenterOptions::NONE);
            break;
          case Activation_SIGMOID:
            options->mutable_segmenter_options()->set_activation(
                SegmenterOptions::SIGMOID);
            break;
          case Activation_SOFTMAX:
            options->mutable_segmenter_options()->set_activation(
                SegmenterOptions::SOFTMAX);
            break;
          default:
            return CreateStatusWithPayload(
                absl::StatusCode::kInvalidArgument,
                "Invalid activation type found in CustomMetadata of "
                "ImageSegmenterOptions type.");
        }
      }
    }
  }
  if (!found_activation_in_metadata) {
    LOG(WARNING)
        << "No activation type is found in model metadata. Use NONE for "
           "ImageSegmenterGraph.";
  }
  const tflite::Model& model = *model_resources.GetTfLiteModel();
  if (model.subgraphs()->size() != 1) {
    return CreateStatusWithPayload(
        absl::StatusCode::kInvalidArgument,
        "Segmentation tflite models are assumed to have a single subgraph.",
        MediaPipeTasksStatus::kInvalidArgumentError);
  }
  const auto* primary_subgraph = (*model.subgraphs())[0];
  if (primary_subgraph->outputs()->size() != 1) {
    return CreateStatusWithPayload(
        absl::StatusCode::kInvalidArgument,
        "Segmentation tflite models are assumed to have a single output.",
        MediaPipeTasksStatus::kInvalidArgumentError);
  }

  ASSIGN_OR_RETURN(
      *options->mutable_label_items(),
      GetLabelItemsIfAny(*metadata_extractor,
                         *metadata_extractor->GetOutputTensorMetadata()->Get(0),
                         segmenter_option.display_names_locale()));
  return absl::OkStatus();
}

// Get the output tensor from the tflite model of given model resources.
absl::StatusOr<const tflite::Tensor*> GetOutputTensor(
    const core::ModelResources& model_resources) {
  const tflite::Model& model = *model_resources.GetTfLiteModel();
  const auto* primary_subgraph = (*model.subgraphs())[0];
  const auto* output_tensor =
      (*primary_subgraph->tensors())[(*primary_subgraph->outputs())[0]];
  return output_tensor;
}

// Get the input tensor from the tflite model of given model resources.
absl::StatusOr<const tflite::Tensor*> GetInputTensor(
    const core::ModelResources& model_resources) {
  const tflite::Model& model = *model_resources.GetTfLiteModel();
  const auto* primary_subgraph = (*model.subgraphs())[0];
  const auto* input_tensor =
      (*primary_subgraph->tensors())[(*primary_subgraph->inputs())[0]];
  return input_tensor;
}

// Configure the ImageTransformationCalculator according to the input tensor.
void ConfigureImageTransformationCalculator(
    const tflite::Tensor& tflite_input_tensor,
    mediapipe::ImageTransformationCalculatorOptions& options) {
  options.set_output_height(tflite_input_tensor.shape()->data()[1]);
  options.set_output_width(tflite_input_tensor.shape()->data()[2]);
}

// Configure the TensorConverterCalculator to convert the image to tensor.
void ConfigureTensorConverterCalculator(
    const ImageTensorSpecs& image_tensor_specs,
    mediapipe::TensorConverterCalculatorOptions& options) {
  float mean = image_tensor_specs.normalization_options->mean_values[0];
  float std = image_tensor_specs.normalization_options->std_values[0];
  options.set_max_num_channels(4);
  options.mutable_output_tensor_float_range()->set_min((0.0f - mean) / std);
  options.mutable_output_tensor_float_range()->set_max((255.0f - mean) / std);
}

// Image preprocessing step to convert the given image to the input tensors for
// the tflite model.
absl::StatusOr<ImageAndTensorsOnDevice> ConvertImageToTensors(
    Source<Image> image_in, Source<NormalizedRect> norm_rect_in, bool use_gpu,
    const core::ModelResources& model_resources, Graph& graph) {
  ASSIGN_OR_RETURN(const tflite::Tensor* tflite_input_tensor,
                   GetInputTensor(model_resources));
  if (tflite_input_tensor->shape()->size() != 4) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Expect segmentation model has input image tensor to "
                        "be 4 dims. Got input tensor with "
                        "dims: %d",
                        tflite_input_tensor->shape()->size()));
  }
  const int input_tensor_channel = tflite_input_tensor->shape()->data()[3];
  if (input_tensor_channel != 3 && input_tensor_channel != 4) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Expect segmentation model has input image tensor with channels = 3 or "
        "4. Get "
        "channel = %d",
        tflite_input_tensor->shape()->data()[3]));
  } else if (input_tensor_channel == 3) {
    // ImagePreprocessingGraph is backed by ImageToTensorCalculator which only
    // supports Tensor with channel = 3.
    auto& preprocessing = graph.AddNode(
        "mediapipe.tasks.components.processors.ImagePreprocessingGraph");
    MP_RETURN_IF_ERROR(components::processors::ConfigureImagePreprocessingGraph(
        model_resources, use_gpu,
        &preprocessing.GetOptions<tasks::components::processors::proto::
                                      ImagePreprocessingGraphOptions>()));
    image_in >> preprocessing.In(kImageTag);
    norm_rect_in >> preprocessing.In(kNormRectTag);
    return {{preprocessing.Out(kImageTag).Cast<Image>(),
             preprocessing.Out(kTensorsTag).Cast<std::vector<Tensor>>()}};
  } else {
    // TODO Remove legacy preprocessing calculators.
    // For segmentation model with input Tensor with channel = 4, use legacy
    // TfLite preprocessing calculators

    // Upload image to GPU if requested to use gpu.
    auto& image_clone = graph.AddNode("ImageCloneCalculator");
    image_clone.GetOptions<mediapipe::ImageCloneCalculatorOptions>()
        .set_output_on_gpu(use_gpu);
    image_in >> image_clone.In("");
    Source<Image> image_on_device = image_clone.Out("").Cast<Image>();

    // Convert from Image to legacy ImageFrame or GpuBuffer.
    auto& from_image = graph.AddNode("FromImageCalculator");
    image_on_device >> from_image.In(kImageTag);
    auto image_cpu_or_gpu =
        from_image.Out(use_gpu ? kImageGpuTag : kImageCpuTag);

    // Resize the input image to the model input size.
    auto& image_transformation = graph.AddNode("ImageTransformationCalculator");
    ConfigureImageTransformationCalculator(
        *tflite_input_tensor,
        image_transformation
            .GetOptions<mediapipe::ImageTransformationCalculatorOptions>());
    const absl::string_view image_or_image_gpu_tag =
        use_gpu ? kImageGpuTag : kImageTag;
    image_cpu_or_gpu >> image_transformation.In(image_or_image_gpu_tag);
    auto transformed_image = image_transformation.Out(image_or_image_gpu_tag);

    // Convert image to mediapipe tensor.
    auto& tensor_converter = graph.AddNode("TensorConverterCalculator");
    ASSIGN_OR_RETURN(auto image_tensor_specs,
                     vision::BuildInputImageTensorSpecs(model_resources));
    ConfigureTensorConverterCalculator(
        image_tensor_specs,
        tensor_converter
            .GetOptions<mediapipe::TensorConverterCalculatorOptions>());

    transformed_image >> tensor_converter.In(image_or_image_gpu_tag);
    auto tensors =
        tensor_converter.Out(kTensorsTag).Cast<std::vector<Tensor>>();

    return {{image_on_device, tensors}};
  }
}

// An "mediapipe.tasks.vision.image_segmenter.ImageSegmenterGraph" performs
// semantic segmentation. The graph always output confidence masks, and an
// optional category mask if CATEGORY_MASK is connected.
//
//  Two kinds of outputs for confidence mask are provided: CONFIDENCE_MASK and
//  CONFIDENCE_MASKS. Users can retrieve segmented mask of only particular
//  category/channel from CONFIDENCE_MASK, and users can also get all segmented
//  confidence masks from CONFIDENCE_MASKS.
// - Accepts CPU input images and outputs segmented masks on CPU.
//
// Inputs:
//   IMAGE - Image
//     Image to perform segmentation on.
//   NORM_RECT - NormalizedRect @Optional
//     Describes image rotation and region of image to perform detection
//     on.
//     @Optional: rect covering the whole image is used if not specified.
//
// Outputs:
//   CONFIDENCE_MASK - mediapipe::Image @Multiple
//     Confidence masks for individual category. Confidence mask of single
//     category can be accessed by index based output stream.
//   CONFIDENCE_MASKS - std::vector<mediapipe::Image>
//     The output confidence masks grouped in a vector.
//   CATEGORY_MASK - mediapipe::Image @Optional
//     Optional Category mask.
//   IMAGE - mediapipe::Image
//     The image that image segmenter runs on.
//
// Example:
// node {
//   calculator: "mediapipe.tasks.vision.ImageSegmenterGraph"
//   input_stream: "IMAGE:image"
//   output_stream: "SEGMENTATION:segmented_masks"
//   options {
//     [mediapipe.tasks.vision.image_segmenter.proto.ImageSegmenterGraphOptions.ext]
//     {
//       base_options {
//         model_asset {
//           file_name: "/path/to/model.tflite"
//         }
//       }
//       segmenter_options {
//         output_type: CONFIDENCE_MASK
//         activation: SOFTMAX
//       }
//     }
//   }
// }
class ImageSegmenterGraph : public core::ModelTaskGraph {
 public:
  absl::StatusOr<mediapipe::CalculatorGraphConfig> GetConfig(
      mediapipe::SubgraphContext* sc) override {
    ASSIGN_OR_RETURN(const auto* model_resources,
                     CreateModelResources<ImageSegmenterGraphOptions>(sc));
    Graph graph;
    const auto& options = sc->Options<ImageSegmenterGraphOptions>();
    ASSIGN_OR_RETURN(
        auto output_streams,
        BuildSegmentationTask(
            options, *model_resources, graph[Input<Image>(kImageTag)],
            graph[Input<NormalizedRect>::Optional(kNormRectTag)],
            HasOutput(sc->OriginalNode(), kCategoryMaskTag), graph));

    auto& merge_images_to_vector =
        graph.AddNode("MergeImagesToVectorCalculator");
    // TODO: remove deprecated output type support.
    if (options.segmenter_options().has_output_type()) {
      for (int i = 0; i < output_streams.segmented_masks->size(); ++i) {
        output_streams.segmented_masks->at(i) >>
            merge_images_to_vector[Input<Image>::Multiple("")][i];
        output_streams.segmented_masks->at(i) >>
            graph[Output<Image>::Multiple(kSegmentationTag)][i];
      }
      merge_images_to_vector.Out("") >>
          graph[Output<std::vector<Image>>(kGroupedSegmentationTag)];
    } else {
      for (int i = 0; i < output_streams.confidence_masks->size(); ++i) {
        output_streams.confidence_masks->at(i) >>
            merge_images_to_vector[Input<Image>::Multiple("")][i];
        output_streams.confidence_masks->at(i) >>
            graph[Output<Image>::Multiple(kConfidenceMaskTag)][i];
      }
      merge_images_to_vector.Out("") >>
          graph[Output<std::vector<Image>>(kConfidenceMasksTag)];
      if (output_streams.category_mask) {
        *output_streams.category_mask >> graph[Output<Image>(kCategoryMaskTag)];
      }
    }
    output_streams.image >> graph[Output<Image>(kImageTag)];
    return graph.GetConfig();
  }

 private:
  // Adds a mediapipe image segmentation task pipeline graph into the provided
  // builder::Graph instance. The segmentation pipeline takes images
  // (mediapipe::Image) as the input and returns segmented image mask as output.
  //
  // task_options: the mediapipe tasks ImageSegmenterGraphOptions proto.
  // model_resources: the ModelSources object initialized from a segmentation
  // model file with model metadata.
  // image_in: (mediapipe::Image) stream to run segmentation on.
  // graph: the mediapipe builder::Graph instance to be updated.
  absl::StatusOr<ImageSegmenterOutputs> BuildSegmentationTask(
      const ImageSegmenterGraphOptions& task_options,
      const core::ModelResources& model_resources, Source<Image> image_in,
      Source<NormalizedRect> norm_rect_in, bool output_category_mask,
      Graph& graph) {
    MP_RETURN_IF_ERROR(SanityCheckOptions(task_options));

    // Adds preprocessing calculators and connects them to the graph input image
    // stream.
    bool use_gpu =
        components::processors::DetermineImagePreprocessingGpuBackend(
            task_options.base_options().acceleration());
    ASSIGN_OR_RETURN(auto image_and_tensors,
                     ConvertImageToTensors(image_in, norm_rect_in, use_gpu,
                                           model_resources, graph));
    // Adds inference subgraph and connects its input stream to the output
    // tensors produced by the ImageToTensorCalculator.
    auto& inference = AddInference(
        model_resources, task_options.base_options().acceleration(), graph);
    image_and_tensors.tensors >> inference.In(kTensorsTag);

    // Adds segmentation calculators for output streams.
    auto& tensor_to_images =
        graph.AddNode("mediapipe.tasks.TensorsToSegmentationCalculator");
    RET_CHECK_OK(ConfigureTensorsToSegmentationCalculator(
        task_options, model_resources,
        &tensor_to_images
             .GetOptions<TensorsToSegmentationCalculatorOptions>()));
    inference.Out(kTensorsTag) >> tensor_to_images.In(kTensorsTag);

    // Adds image property calculator for output size.
    auto& image_properties = graph.AddNode("ImagePropertiesCalculator");
    image_in >> image_properties.In("IMAGE");
    image_properties.Out("SIZE") >> tensor_to_images.In(kOutputSizeTag);

    // Exports multiple segmented masks.
    // TODO: remove deprecated output type support.
    if (task_options.segmenter_options().has_output_type()) {
      std::vector<Source<Image>> segmented_masks;
      if (task_options.segmenter_options().output_type() ==
          SegmenterOptions::CATEGORY_MASK) {
        segmented_masks.push_back(
            Source<Image>(tensor_to_images[Output<Image>(kSegmentationTag)]));
      } else {
        ASSIGN_OR_RETURN(const tflite::Tensor* output_tensor,
                         GetOutputTensor(model_resources));
        int segmentation_streams_num = *output_tensor->shape()->rbegin();
        for (int i = 0; i < segmentation_streams_num; ++i) {
          segmented_masks.push_back(Source<Image>(
              tensor_to_images[Output<Image>::Multiple(kSegmentationTag)][i]));
        }
      }
      return ImageSegmenterOutputs{/*segmented_masks=*/segmented_masks,
                                   /*confidence_masks=*/std::nullopt,
                                   /*category_mask=*/std::nullopt,
                                   /*image=*/image_and_tensors.image};
    } else {
      ASSIGN_OR_RETURN(const tflite::Tensor* output_tensor,
                       GetOutputTensor(model_resources));
      int segmentation_streams_num = *output_tensor->shape()->rbegin();
      std::vector<Source<Image>> confidence_masks;
      confidence_masks.reserve(segmentation_streams_num);
      for (int i = 0; i < segmentation_streams_num; ++i) {
        confidence_masks.push_back(Source<Image>(
            tensor_to_images[Output<Image>::Multiple(kConfidenceMaskTag)][i]));
      }
      return ImageSegmenterOutputs{
          /*segmented_masks=*/std::nullopt,
          /*confidence_masks=*/confidence_masks,
          /*category_mask=*/
          output_category_mask
              ? std::make_optional(
                    tensor_to_images[Output<Image>(kCategoryMaskTag)])
              : std::nullopt,
          /*image=*/image_and_tensors.image};
    }
  }
};

REGISTER_MEDIAPIPE_GRAPH(
    ::mediapipe::tasks::vision::image_segmenter::ImageSegmenterGraph);

}  // namespace image_segmenter
}  // namespace vision
}  // namespace tasks
}  // namespace mediapipe
