/**
 * Copyright 2023 The MediaPipe Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import {CalculatorGraphConfig} from '../../../../framework/calculator_pb';
import {CalculatorOptions} from '../../../../framework/calculator_options_pb';
import {BaseOptions as BaseOptionsProto} from '../../../../tasks/cc/core/proto/base_options_pb';
import {ImageSegmenterGraphOptions as ImageSegmenterGraphOptionsProto} from '../../../../tasks/cc/vision/image_segmenter/proto/image_segmenter_graph_options_pb';
import {SegmenterOptions as SegmenterOptionsProto} from '../../../../tasks/cc/vision/image_segmenter/proto/segmenter_options_pb';
import {WasmFileset} from '../../../../tasks/web/core/wasm_fileset';
import {ImageProcessingOptions} from '../../../../tasks/web/vision/core/image_processing_options';
import {RegionOfInterest, SegmentationMask, SegmentationMaskCallback} from '../../../../tasks/web/vision/core/types';
import {VisionGraphRunner, VisionTaskRunner} from '../../../../tasks/web/vision/core/vision_task_runner';
import {Color as ColorProto} from '../../../../util/color_pb';
import {RenderAnnotation as RenderAnnotationProto, RenderData as RenderDataProto} from '../../../../util/render_data_pb';
import {ImageSource, WasmModule} from '../../../../web/graph_runner/graph_runner';
// Placeholder for internal dependency on trusted resource url

import {InteractiveSegmenterOptions} from './interactive_segmenter_options';

export * from './interactive_segmenter_options';
export {SegmentationMask, SegmentationMaskCallback, RegionOfInterest};
export {ImageSource};

const IMAGE_IN_STREAM = 'image_in';
const NORM_RECT_IN_STREAM = 'norm_rect_in';
const ROI_IN_STREAM = 'roi_in';
const IMAGE_OUT_STREAM = 'image_out';
const IMAGEA_SEGMENTER_GRAPH =
    'mediapipe.tasks.vision.interactive_segmenter.InteractiveSegmenterGraph';

// The OSS JS API does not support the builder pattern.
// tslint:disable:jspb-use-builder-pattern

/**
 * Performs interactive segmentation on images.
 *
 * Users can represent user interaction through `RegionOfInterest`, which gives
 * a hint to InteractiveSegmenter to perform segmentation focusing on the given
 * region of interest.
 *
 * The API expects a TFLite model with mandatory TFLite Model Metadata.
 *
 * Input tensor:
 *   (kTfLiteUInt8/kTfLiteFloat32)
 *   - image input of size `[batch x height x width x channels]`.
 *   - batch inference is not supported (`batch` is required to be 1).
 *   - RGB inputs is supported (`channels` is required to be 3).
 *   - if type is kTfLiteFloat32, NormalizationOptions are required to be
 *     attached to the metadata for input normalization.
 * Output tensors:
 *  (kTfLiteUInt8/kTfLiteFloat32)
 *   - list of segmented masks.
 *   - if `output_type` is CATEGORY_MASK, uint8 Image, Image vector of size 1.
 *   - if `output_type` is CONFIDENCE_MASK, float32 Image list of size
 *     `channels`.
 *   - batch is always 1
 */
export class InteractiveSegmenter extends VisionTaskRunner {
  private userCallback: SegmentationMaskCallback = () => {};
  private readonly options: ImageSegmenterGraphOptionsProto;
  private readonly segmenterOptions: SegmenterOptionsProto;

  /**
   * Initializes the Wasm runtime and creates a new interactive segmenter from
   * the provided options.
   * @param wasmFileset A configuration object that provides the location of
   *     the Wasm binary and its loader.
   * @param interactiveSegmenterOptions The options for the Interactive
   *     Segmenter. Note that either a path to the model asset or a model buffer
   *     needs to be provided (via `baseOptions`).
   * @return A new `InteractiveSegmenter`.
   */
  static createFromOptions(
      wasmFileset: WasmFileset,
      interactiveSegmenterOptions: InteractiveSegmenterOptions):
      Promise<InteractiveSegmenter> {
    return VisionTaskRunner.createVisionInstance(
        InteractiveSegmenter, wasmFileset, interactiveSegmenterOptions);
  }

  /**
   * Initializes the Wasm runtime and creates a new interactive segmenter based
   * on the provided model asset buffer.
   * @param wasmFileset A configuration object that provides the location of
   *     the Wasm binary and its loader.
   * @param modelAssetBuffer A binary representation of the model.
   * @return A new `InteractiveSegmenter`.
   */
  static createFromModelBuffer(
      wasmFileset: WasmFileset,
      modelAssetBuffer: Uint8Array): Promise<InteractiveSegmenter> {
    return VisionTaskRunner.createVisionInstance(
        InteractiveSegmenter, wasmFileset, {baseOptions: {modelAssetBuffer}});
  }

  /**
   * Initializes the Wasm runtime and creates a new interactive segmenter based
   * on the path to the model asset.
   * @param wasmFileset A configuration object that provides the location of
   *     the Wasm binary and its loader.
   * @param modelAssetPath The path to the model asset.
   * @return A new `InteractiveSegmenter`.
   */
  static createFromModelPath(
      wasmFileset: WasmFileset,
      modelAssetPath: string): Promise<InteractiveSegmenter> {
    return VisionTaskRunner.createVisionInstance(
        InteractiveSegmenter, wasmFileset, {baseOptions: {modelAssetPath}});
  }

  /** @hideconstructor */
  constructor(
      wasmModule: WasmModule,
      glCanvas?: HTMLCanvasElement|OffscreenCanvas|null) {
    super(
        new VisionGraphRunner(wasmModule, glCanvas), IMAGE_IN_STREAM,
        NORM_RECT_IN_STREAM, /* roiAllowed= */ false);
    this.options = new ImageSegmenterGraphOptionsProto();
    this.segmenterOptions = new SegmenterOptionsProto();
    this.options.setSegmenterOptions(this.segmenterOptions);
    this.options.setBaseOptions(new BaseOptionsProto());
  }


  protected override get baseOptions(): BaseOptionsProto {
    return this.options.getBaseOptions()!;
  }

  protected override set baseOptions(proto: BaseOptionsProto) {
    this.options.setBaseOptions(proto);
  }

  /**
   * Sets new options for the interactive segmenter.
   *
   * Calling `setOptions()` with a subset of options only affects those
   * options. You can reset an option back to its default value by
   * explicitly setting it to `undefined`.
   *
   * @param options The options for the interactive segmenter.
   * @return A Promise that resolves when the settings have been applied.
   */
  override setOptions(options: InteractiveSegmenterOptions): Promise<void> {
    if (options.outputType === 'CONFIDENCE_MASK') {
      this.segmenterOptions.setOutputType(
          SegmenterOptionsProto.OutputType.CONFIDENCE_MASK);
    } else {
      this.segmenterOptions.setOutputType(
          SegmenterOptionsProto.OutputType.CATEGORY_MASK);
    }

    return super.applyOptions(options);
  }

  /**
   * Performs interactive segmentation on the provided single image and invokes
   * the callback with the response.  The `roi` parameter is used to represent a
   * user's region of interest for segmentation.
   *
   * If the output_type is `CATEGORY_MASK`, the callback is invoked with vector
   * of images that represent per-category segmented image mask. If the
   * output_type is `CONFIDENCE_MASK`, the callback is invoked with a vector of
   * images that contains only one confidence image mask. The method returns
   * synchronously once the callback returns.
   *
   * @param image An image to process.
   * @param roi The region of interest for segmentation.
   * @param callback The callback that is invoked with the segmented masks. The
   *    lifetime of the returned data is only guaranteed for the duration of the
   *    callback.
   */
  segment(
      image: ImageSource, roi: RegionOfInterest,
      callback: SegmentationMaskCallback): void;
  /**
   * Performs interactive segmentation on the provided single image and invokes
   * the callback with the response. The `roi` parameter is used to represent a
   * user's region of interest for segmentation.
   *
   * The 'image_processing_options' parameter can be used to specify the
   * rotation to apply to the image before performing segmentation, by setting
   * its 'rotationDegrees' field. Note that specifying a region-of-interest
   * using the 'regionOfInterest' field is NOT supported and will result in an
   * error.
   *
   * If the output_type is `CATEGORY_MASK`, the callback is invoked with vector
   * of images that represent per-category segmented image mask. If the
   * output_type is `CONFIDENCE_MASK`, the callback is invoked with a vector of
   * images that contains only one confidence image mask. The method returns
   * synchronously once the callback returns.
   *
   * @param image An image to process.
   * @param roi The region of interest for segmentation.
   * @param imageProcessingOptions the `ImageProcessingOptions` specifying how
   *    to process the input image before running inference.
   * @param callback The callback that is invoked with the segmented masks. The
   *    lifetime of the returned data is only guaranteed for the duration of the
   *    callback.
   */
  segment(
      image: ImageSource, roi: RegionOfInterest,
      imageProcessingOptions: ImageProcessingOptions,
      callback: SegmentationMaskCallback): void;
  segment(
      image: ImageSource, roi: RegionOfInterest,
      imageProcessingOptionsOrCallback: ImageProcessingOptions|
      SegmentationMaskCallback,
      callback?: SegmentationMaskCallback): void {
    const imageProcessingOptions =
        typeof imageProcessingOptionsOrCallback !== 'function' ?
        imageProcessingOptionsOrCallback :
        {};

    this.userCallback = typeof imageProcessingOptionsOrCallback === 'function' ?
        imageProcessingOptionsOrCallback :
        callback!;

    this.processRenderData(roi, this.getSynctheticTimestamp());
    this.processImageData(image, imageProcessingOptions);
    this.userCallback = () => {};
  }

  /** Updates the MediaPipe graph configuration. */
  protected override refreshGraph(): void {
    const graphConfig = new CalculatorGraphConfig();
    graphConfig.addInputStream(IMAGE_IN_STREAM);
    graphConfig.addInputStream(ROI_IN_STREAM);
    graphConfig.addInputStream(NORM_RECT_IN_STREAM);
    graphConfig.addOutputStream(IMAGE_OUT_STREAM);

    const calculatorOptions = new CalculatorOptions();
    calculatorOptions.setExtension(
        ImageSegmenterGraphOptionsProto.ext, this.options);

    const segmenterNode = new CalculatorGraphConfig.Node();
    segmenterNode.setCalculator(IMAGEA_SEGMENTER_GRAPH);
    segmenterNode.addInputStream('IMAGE:' + IMAGE_IN_STREAM);
    segmenterNode.addInputStream('ROI:' + ROI_IN_STREAM);
    segmenterNode.addInputStream('NORM_RECT:' + NORM_RECT_IN_STREAM);
    segmenterNode.addOutputStream('GROUPED_SEGMENTATION:' + IMAGE_OUT_STREAM);
    segmenterNode.setOptions(calculatorOptions);

    graphConfig.addNode(segmenterNode);

    this.graphRunner.attachImageVectorListener(
        IMAGE_OUT_STREAM, (masks, timestamp) => {
          if (masks.length === 0) {
            this.userCallback([], 0, 0);
          } else {
            this.userCallback(
                masks.map(m => m.data), masks[0].width, masks[0].height);
          }
          this.setLatestOutputTimestamp(timestamp);
        });
    this.graphRunner.attachEmptyPacketListener(IMAGE_OUT_STREAM, timestamp => {
      this.setLatestOutputTimestamp(timestamp);
    });

    const binaryGraph = graphConfig.serializeBinary();
    this.setGraph(new Uint8Array(binaryGraph), /* isBinary= */ true);
  }

  /**
   * Converts the user-facing RegionOfInterest message to the RenderData proto
   * and sends it to the graph
   */
  private processRenderData(roi: RegionOfInterest, timestamp: number): void {
    const renderData = new RenderDataProto();

    const renderAnnotation = new RenderAnnotationProto();

    const color = new ColorProto();
    color.setR(255);
    renderAnnotation.setColor(color);

    const point = new RenderAnnotationProto.Point();
    point.setNormalized(true);
    point.setX(roi.keypoint.x);
    point.setY(roi.keypoint.y);
    renderAnnotation.setPoint(point);

    renderData.addRenderAnnotations(renderAnnotation);

    this.graphRunner.addProtoToStream(
        renderData.serializeBinary(), 'mediapipe.RenderData', ROI_IN_STREAM,
        timestamp);
  }
}


