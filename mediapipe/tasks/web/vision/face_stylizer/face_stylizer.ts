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
import {FaceStylizerGraphOptions as FaceStylizerGraphOptionsProto} from '../../../../tasks/cc/vision/face_stylizer/proto/face_stylizer_graph_options_pb';
import {WasmFileset} from '../../../../tasks/web/core/wasm_fileset';
import {ImageProcessingOptions} from '../../../../tasks/web/vision/core/image_processing_options';
import {ImageCallback} from '../../../../tasks/web/vision/core/types';
import {VisionGraphRunner, VisionTaskRunner} from '../../../../tasks/web/vision/core/vision_task_runner';
import {ImageSource, WasmModule} from '../../../../web/graph_runner/graph_runner';
// Placeholder for internal dependency on trusted resource url

import {FaceStylizerOptions} from './face_stylizer_options';

export * from './face_stylizer_options';
export {ImageSource};  // Used in the public API

const IMAGE_STREAM = 'image_in';
const NORM_RECT_STREAM = 'norm_rect';
const STYLIZED_IMAGE_STREAM = 'stylized_image';
const FACE_STYLIZER_GRAPH =
    'mediapipe.tasks.vision.face_stylizer.FaceStylizerGraph';

// The OSS JS API does not support the builder pattern.
// tslint:disable:jspb-use-builder-pattern

export {ImageCallback};

/** Performs face stylization on images. */
export class FaceStylizer extends VisionTaskRunner {
  private userCallback: ImageCallback = () => {};
  private readonly options: FaceStylizerGraphOptionsProto;

  /**
   * Initializes the Wasm runtime and creates a new Face Stylizer from the
   * provided options.
   * @param wasmFileset A configuration object that provides the location of
   *     the Wasm binary and its loader.
   * @param faceStylizerOptions The options for the Face Stylizer. Note
   *     that either a path to the model asset or a model buffer needs to be
   *     provided (via `baseOptions`).
   */
  static createFromOptions(
      wasmFileset: WasmFileset,
      faceStylizerOptions: FaceStylizerOptions): Promise<FaceStylizer> {
    return VisionTaskRunner.createInstance(
        FaceStylizer, /* initializeCanvas= */ true, wasmFileset,
        faceStylizerOptions);
  }

  /**
   * Initializes the Wasm runtime and creates a new Face Stylizer based on
   * the provided model asset buffer.
   * @param wasmFileset A configuration object that provides the location of
   *     the Wasm binary and its loader.
   * @param modelAssetBuffer A binary representation of the model.
   */
  static createFromModelBuffer(
      wasmFileset: WasmFileset,
      modelAssetBuffer: Uint8Array): Promise<FaceStylizer> {
    return VisionTaskRunner.createInstance(
        FaceStylizer, /* initializeCanvas= */ true, wasmFileset,
        {baseOptions: {modelAssetBuffer}});
  }

  /**
   * Initializes the Wasm runtime and creates a new Face Stylizer based on
   * the path to the model asset.
   * @param wasmFileset A configuration object that provides the location of
   *     the Wasm binary and its loader.
   * @param modelAssetPath The path to the model asset.
   */
  static createFromModelPath(
      wasmFileset: WasmFileset,
      modelAssetPath: string): Promise<FaceStylizer> {
    return VisionTaskRunner.createInstance(
        FaceStylizer, /* initializeCanvas= */ true, wasmFileset,
        {baseOptions: {modelAssetPath}});
  }

  /** @hideconstructor */
  constructor(
      wasmModule: WasmModule,
      glCanvas?: HTMLCanvasElement|OffscreenCanvas|null) {
    super(
        new VisionGraphRunner(wasmModule, glCanvas), IMAGE_STREAM,
        NORM_RECT_STREAM, /* roiAllowed= */ true);
    this.options = new FaceStylizerGraphOptionsProto();
    this.options.setBaseOptions(new BaseOptionsProto());
  }


  protected override get baseOptions(): BaseOptionsProto {
    return this.options.getBaseOptions()!;
  }

  protected override set baseOptions(proto: BaseOptionsProto) {
    this.options.setBaseOptions(proto);
  }

  /**
   * Sets new options for the Face Stylizer.
   *
   * Calling `setOptions()` with a subset of options only affects those
   * options. You can reset an option back to its default value by
   * explicitly setting it to `undefined`.
   *
   * @param options The options for the Face Stylizer.
   */
  override setOptions(options: FaceStylizerOptions): Promise<void> {
    return super.applyOptions(options);
  }


  /**
   * Performs face stylization on the provided single image. The method returns
   * synchronously once the callback returns. Only use this method when the
   * FaceStylizer is created with the image running mode.
   *
   * The input image can be of any size. To ensure that the output image has
   * reasonable quailty, the stylized output image size is determined by the
   * model output size.
   *
   * @param image An image to process.
   * @param callback The callback that is invoked with the stylized image. The
   *    lifetime of the returned data is only guaranteed for the duration of the
   *    callback.
   */
  stylize(image: ImageSource, callback: ImageCallback): void;
  /**
   * Performs face stylization on the provided single image. The method returns
   * synchronously once the callback returns. Only use this method when the
   * FaceStylizer is created with the image running mode.
   *
   * The 'imageProcessingOptions' parameter can be used to specify one or all
   * of:
   *  - the rotation to apply to the image before performing stylization, by
   *    setting its 'rotationDegrees' property.
   *  - the region-of-interest on which to perform stylization, by setting its
   *   'regionOfInterest' property. If not specified, the full image is used.
   *  If both are specified, the crop around the region-of-interest is extracted
   *  first, then the specified rotation is applied to the crop.
   *
   * The input image can be of any size. To ensure that the output image has
   * reasonable quailty, the stylized output image size is the smaller of the
   * model output size and the size of the 'regionOfInterest' specified in
   * 'imageProcessingOptions'.
   *
   * @param image An image to process.
   * @param imageProcessingOptions the `ImageProcessingOptions` specifying how
   *    to process the input image before running inference.
   * @param callback The callback that is invoked with the stylized image. The
   *    lifetime of the returned data is only guaranteed for the duration of the
   *    callback.
   */
  stylize(
      image: ImageSource, imageProcessingOptions: ImageProcessingOptions,
      callback: ImageCallback): void;
  stylize(
      image: ImageSource,
      imageProcessingOptionsOrCallback: ImageProcessingOptions|ImageCallback,
      callback?: ImageCallback): void {
    const imageProcessingOptions =
        typeof imageProcessingOptionsOrCallback !== 'function' ?
        imageProcessingOptionsOrCallback :
        {};

    this.userCallback = typeof imageProcessingOptionsOrCallback === 'function' ?
        imageProcessingOptionsOrCallback :
        callback!;
    this.processImageData(image, imageProcessingOptions ?? {});
    this.userCallback = () => {};
  }

  /**
   * Performs face stylization on the provided video frame. Only use this method
   * when the FaceStylizer is created with the video running mode.
   *
   * The input frame can be of any size. It's required to provide the video
   * frame's timestamp (in milliseconds). The input timestamps must be
   * monotonically increasing.
   *
   * To ensure that the output image has reasonable quality, the stylized
   * output image size is determined by the model output size.
   *
   * @param videoFrame A video frame to process.
   * @param timestamp The timestamp of the current frame, in ms.
   * @param callback The callback that is invoked with the stylized image. The
   *    lifetime of the returned data is only guaranteed for the duration of
   * the callback.
   */
  stylizeForVideo(
      videoFrame: ImageSource, timestamp: number,
      callback: ImageCallback): void;
  /**
   * Performs face stylization on the provided video frame. Only use this
   * method when the FaceStylizer is created with the video running mode.
   *
   * The 'imageProcessingOptions' parameter can be used to specify one or all
   * of:
   *  - the rotation to apply to the image before performing stylization, by
   *    setting its 'rotationDegrees' property.
   *  - the region-of-interest on which to perform stylization, by setting its
   *   'regionOfInterest' property. If not specified, the full image is used.
   *  If both are specified, the crop around the region-of-interest is
   * extracted first, then the specified rotation is applied to the crop.
   *
   * The input frame can be of any size. It's required to provide the video
   * frame's timestamp (in milliseconds). The input timestamps must be
   * monotonically increasing.
   *
   * To ensure that the output image has reasonable quailty, the stylized
   * output image size is the smaller of the model output size and the size of
   * the 'regionOfInterest' specified in 'imageProcessingOptions'.
   *
   * @param videoFrame A video frame to process.
   * @param imageProcessingOptions the `ImageProcessingOptions` specifying how
   *    to process the input image before running inference.
   * @param timestamp The timestamp of the current frame, in ms.
   * @param callback The callback that is invoked with the stylized image. The
   *    lifetime of the returned data is only guaranteed for the duration of
   * the callback.
   */
  stylizeForVideo(
      videoFrame: ImageSource, imageProcessingOptions: ImageProcessingOptions,
      timestamp: number, callback: ImageCallback): void;
  stylizeForVideo(
      videoFrame: ImageSource,
      timestampOrImageProcessingOptions: number|ImageProcessingOptions,
      timestampOrCallback: number|ImageCallback,
      callback?: ImageCallback): void {
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
    graphConfig.addOutputStream(STYLIZED_IMAGE_STREAM);

    const calculatorOptions = new CalculatorOptions();
    calculatorOptions.setExtension(
        FaceStylizerGraphOptionsProto.ext, this.options);

    const segmenterNode = new CalculatorGraphConfig.Node();
    segmenterNode.setCalculator(FACE_STYLIZER_GRAPH);
    segmenterNode.addInputStream('IMAGE:' + IMAGE_STREAM);
    segmenterNode.addInputStream('NORM_RECT:' + NORM_RECT_STREAM);
    segmenterNode.addOutputStream('STYLIZED_IMAGE:' + STYLIZED_IMAGE_STREAM);
    segmenterNode.setOptions(calculatorOptions);

    graphConfig.addNode(segmenterNode);

    this.graphRunner.attachImageListener(
        STYLIZED_IMAGE_STREAM, (image, timestamp) => {
          const imageData = this.convertToImageData(image);
          this.userCallback(imageData, image.width, image.height);
          this.setLatestOutputTimestamp(timestamp);
        });
    this.graphRunner.attachEmptyPacketListener(
        STYLIZED_IMAGE_STREAM, timestamp => {
          this.setLatestOutputTimestamp(timestamp);
        });

    const binaryGraph = graphConfig.serializeBinary();
    this.setGraph(new Uint8Array(binaryGraph), /* isBinary= */ true);
  }
}


