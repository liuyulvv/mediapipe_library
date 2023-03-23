/**
 * Copyright 2022 The MediaPipe Authors. All Rights Reserved.
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
import {TensorsToSegmentationCalculatorOptions} from '../../../../tasks/cc/vision/image_segmenter/calculators/tensors_to_segmentation_calculator_pb';
import {ImageSegmenterGraphOptions as ImageSegmenterGraphOptionsProto} from '../../../../tasks/cc/vision/image_segmenter/proto/image_segmenter_graph_options_pb';
import {SegmenterOptions as SegmenterOptionsProto} from '../../../../tasks/cc/vision/image_segmenter/proto/segmenter_options_pb';
import {WasmFileset} from '../../../../tasks/web/core/wasm_fileset';
import {ImageProcessingOptions} from '../../../../tasks/web/vision/core/image_processing_options';
import {SegmentationMask, SegmentationMaskCallback} from '../../../../tasks/web/vision/core/types';
import {VisionGraphRunner, VisionTaskRunner} from '../../../../tasks/web/vision/core/vision_task_runner';
import {LabelMapItem} from '../../../../util/label_map_pb';
import {ImageSource, WasmModule} from '../../../../web/graph_runner/graph_runner';
// Placeholder for internal dependency on trusted resource url

import {ImageSegmenterOptions} from './image_segmenter_options';

export * from './image_segmenter_options';
export {SegmentationMask, SegmentationMaskCallback};
export {ImageSource};  // Used in the public API

const IMAGE_STREAM = 'image_in';
const NORM_RECT_STREAM = 'norm_rect';
const GROUPED_SEGMENTATIONS_STREAM = 'segmented_masks';
const IMAGE_SEGMENTER_GRAPH =
    'mediapipe.tasks.vision.image_segmenter.ImageSegmenterGraph';
const TENSORS_TO_SEGMENTATION_CALCULATOR_NAME =
    'mediapipe.tasks.TensorsToSegmentationCalculator';

// The OSS JS API does not support the builder pattern.
// tslint:disable:jspb-use-builder-pattern

/** Performs image segmentation on images. */
export class ImageSegmenter extends VisionTaskRunner {
  private userCallback: SegmentationMaskCallback = () => {};
  private labels: string[] = [];
  private readonly options: ImageSegmenterGraphOptionsProto;
  private readonly segmenterOptions: SegmenterOptionsProto;

  /**
   * Initializes the Wasm runtime and creates a new image segmenter from the
   * provided options.
   * @param wasmFileset A configuration object that provides the location of
   *     the Wasm binary and its loader.
   * @param imageSegmenterOptions The options for the Image Segmenter. Note
   *     that either a path to the model asset or a model buffer needs to be
   *     provided (via `baseOptions`).
   */
  static createFromOptions(
      wasmFileset: WasmFileset,
      imageSegmenterOptions: ImageSegmenterOptions): Promise<ImageSegmenter> {
    return VisionTaskRunner.createInstance(
        ImageSegmenter, /* initializeCanvas= */ true, wasmFileset,
        imageSegmenterOptions);
  }

  /**
   * Initializes the Wasm runtime and creates a new image segmenter based on
   * the provided model asset buffer.
   * @param wasmFileset A configuration object that provides the location of
   *     the Wasm binary and its loader.
   * @param modelAssetBuffer A binary representation of the model.
   */
  static createFromModelBuffer(
      wasmFileset: WasmFileset,
      modelAssetBuffer: Uint8Array): Promise<ImageSegmenter> {
    return VisionTaskRunner.createInstance(
        ImageSegmenter, /* initializeCanvas= */ true, wasmFileset,
        {baseOptions: {modelAssetBuffer}});
  }

  /**
   * Initializes the Wasm runtime and creates a new image segmenter based on
   * the path to the model asset.
   * @param wasmFileset A configuration object that provides the location of
   *     the Wasm binary and its loader.
   * @param modelAssetPath The path to the model asset.
   */
  static createFromModelPath(
      wasmFileset: WasmFileset,
      modelAssetPath: string): Promise<ImageSegmenter> {
    return VisionTaskRunner.createInstance(
        ImageSegmenter, /* initializeCanvas= */ true, wasmFileset,
        {baseOptions: {modelAssetPath}});
  }

  /** @hideconstructor */
  constructor(
      wasmModule: WasmModule,
      glCanvas?: HTMLCanvasElement|OffscreenCanvas|null) {
    super(
        new VisionGraphRunner(wasmModule, glCanvas), IMAGE_STREAM,
        NORM_RECT_STREAM, /* roiAllowed= */ false);
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
   * Sets new options for the image segmenter.
   *
   * Calling `setOptions()` with a subset of options only affects those
   * options. You can reset an option back to its default value by
   * explicitly setting it to `undefined`.
   *
   * @param options The options for the image segmenter.
   */
  override setOptions(options: ImageSegmenterOptions): Promise<void> {
    // Note that we have to support both JSPB and ProtobufJS, hence we
    // have to expliclity clear the values instead of setting them to
    // `undefined`.
    if (options.displayNamesLocale !== undefined) {
      this.options.setDisplayNamesLocale(options.displayNamesLocale);
    } else if ('displayNamesLocale' in options) {  // Check for undefined
      this.options.clearDisplayNamesLocale();
    }

    if (options.outputType === 'CONFIDENCE_MASK') {
      this.segmenterOptions.setOutputType(
          SegmenterOptionsProto.OutputType.CONFIDENCE_MASK);
    } else {
      this.segmenterOptions.setOutputType(
          SegmenterOptionsProto.OutputType.CATEGORY_MASK);
    }

    return super.applyOptions(options);
  }

  protected override onGraphRefreshed(): void {
    this.populateLabels();
  }

  /**
   * Populate the labelMap in TensorsToSegmentationCalculator to labels field.
   * @throws Exception if there is an error during finding
   *     TensorsToSegmentationCalculator.
   */
  private populateLabels(): void {
    const graphConfig = this.getCalculatorGraphConfig();
    const tensorsToSegmentationCalculators = graphConfig.getNodeList().filter(
        (n: CalculatorGraphConfig.Node) =>
            n.getName().includes(TENSORS_TO_SEGMENTATION_CALCULATOR_NAME));

    this.labels = [];
    if (tensorsToSegmentationCalculators.length > 1) {
      throw new Error(`The graph has more than one ${
          TENSORS_TO_SEGMENTATION_CALCULATOR_NAME}.`);
    } else if (tensorsToSegmentationCalculators.length === 1) {
      const labelItems =
          tensorsToSegmentationCalculators[0]
              .getOptions()
              ?.getExtension(TensorsToSegmentationCalculatorOptions.ext)
              ?.getLabelItemsMap() ??
          new Map<string, LabelMapItem>();
      labelItems.forEach((value, index) => {
        // tslint:disable-next-line:no-unnecessary-type-assertion
        this.labels[Number(index)] = value.getName()!;
      });
    }
  }

  /**
   * Performs image segmentation on the provided single image and invokes the
   * callback with the response. The method returns synchronously once the
   * callback returns. Only use this method when the ImageSegmenter is
   * created with running mode `image`.
   *
   * @param image An image to process.
   * @param callback The callback that is invoked with the segmented masks. The
   *    lifetime of the returned data is only guaranteed for the duration of the
   *    callback.
   */
  segment(image: ImageSource, callback: SegmentationMaskCallback): void;
  /**
   * Performs image segmentation on the provided single image and invokes the
   * callback with the response. The method returns synchronously once the
   * callback returns. Only use this method when the ImageSegmenter is
   * created with running mode `image`.
   *
   * @param image An image to process.
   * @param imageProcessingOptions the `ImageProcessingOptions` specifying how
   *    to process the input image before running inference.
   * @param callback The callback that is invoked with the segmented masks. The
   *    lifetime of the returned data is only guaranteed for the duration of the
   *    callback.
   */
  segment(
      image: ImageSource, imageProcessingOptions: ImageProcessingOptions,
      callback: SegmentationMaskCallback): void;
  segment(
      image: ImageSource,
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
    this.processImageData(image, imageProcessingOptions);
    this.userCallback = () => {};
  }

  /**
   * Get the category label list of the ImageSegmenter can recognize. For
   * `CATEGORY_MASK` type, the index in the category mask corresponds to the
   * category in the label list. For `CONFIDENCE_MASK` type, the output mask
   * list at index corresponds to the category in the label list.
   *
   * If there is no labelmap provided in the model file, empty label array is
   * returned.
   *
   * @return The labels used by the current model.
   */
  getLabels(): string[] {
    return this.labels;
  }

  /**
   * Performs image segmentation on the provided video frame and invokes the
   * callback with the response. The method returns synchronously once the
   * callback returns. Only use this method when the ImageSegmenter is
   * created with running mode `video`.
   *
   * @param videoFrame A video frame to process.
   * @param timestamp The timestamp of the current frame, in ms.
   * @param callback The callback that is invoked with the segmented masks. The
   *    lifetime of the returned data is only guaranteed for the duration of the
   *    callback.
   */
  segmentForVideo(
      videoFrame: ImageSource, timestamp: number,
      callback: SegmentationMaskCallback): void;
  /**
   * Performs image segmentation on the provided video frame and invokes the
   * callback with the response. The method returns synchronously once the
   * callback returns. Only use this method when the ImageSegmenter is
   * created with running mode `video`.
   *
   * @param videoFrame A video frame to process.
   * @param imageProcessingOptions the `ImageProcessingOptions` specifying how
   *    to process the input image before running inference.
   * @param timestamp The timestamp of the current frame, in ms.
   * @param callback The callback that is invoked with the segmented masks. The
   *    lifetime of the returned data is only guaranteed for the duration of the
   *    callback.
   */
  segmentForVideo(
      videoFrame: ImageSource, imageProcessingOptions: ImageProcessingOptions,
      timestamp: number, callback: SegmentationMaskCallback): void;
  segmentForVideo(
      videoFrame: ImageSource,
      timestampOrImageProcessingOptions: number|ImageProcessingOptions,
      timestampOrCallback: number|SegmentationMaskCallback,
      callback?: SegmentationMaskCallback): void {
    const imageProcessingOptions =
        typeof timestampOrImageProcessingOptions !== 'number' ?
        timestampOrImageProcessingOptions :
        {};
    const timestamp = typeof timestampOrImageProcessingOptions === 'number' ?
        timestampOrImageProcessingOptions :
        timestampOrCallback as number;

    this.userCallback = typeof timestampOrCallback === 'function' ?
        timestampOrCallback :
        callback!;
    this.processVideoData(videoFrame, imageProcessingOptions, timestamp);
    this.userCallback = () => {};
  }

  /** Updates the MediaPipe graph configuration. */
  protected override refreshGraph(): void {
    const graphConfig = new CalculatorGraphConfig();
    graphConfig.addInputStream(IMAGE_STREAM);
    graphConfig.addInputStream(NORM_RECT_STREAM);
    graphConfig.addOutputStream(GROUPED_SEGMENTATIONS_STREAM);

    const calculatorOptions = new CalculatorOptions();
    calculatorOptions.setExtension(
        ImageSegmenterGraphOptionsProto.ext, this.options);

    const segmenterNode = new CalculatorGraphConfig.Node();
    segmenterNode.setCalculator(IMAGE_SEGMENTER_GRAPH);
    segmenterNode.addInputStream('IMAGE:' + IMAGE_STREAM);
    segmenterNode.addInputStream('NORM_RECT:' + NORM_RECT_STREAM);
    segmenterNode.addOutputStream(
        'GROUPED_SEGMENTATION:' + GROUPED_SEGMENTATIONS_STREAM);
    segmenterNode.setOptions(calculatorOptions);

    graphConfig.addNode(segmenterNode);

    this.graphRunner.attachImageVectorListener(
        GROUPED_SEGMENTATIONS_STREAM, (masks, timestamp) => {
          if (masks.length === 0) {
            this.userCallback([], 0, 0);
          } else {
            this.userCallback(
                masks.map(m => m.data), masks[0].width, masks[0].height);
          }
          this.setLatestOutputTimestamp(timestamp);
        });
    this.graphRunner.attachEmptyPacketListener(
        GROUPED_SEGMENTATIONS_STREAM, timestamp => {
          this.setLatestOutputTimestamp(timestamp);
        });

    const binaryGraph = graphConfig.serializeBinary();
    this.setGraph(new Uint8Array(binaryGraph), /* isBinary= */ true);
  }
}


