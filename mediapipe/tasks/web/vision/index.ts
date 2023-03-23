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

import {FilesetResolver as FilesetResolverImpl} from '../../../tasks/web/core/fileset_resolver';
import {FaceStylizer as FaceStylizerImpl} from '../../../tasks/web/vision/face_stylizer/face_stylizer';
import {GestureRecognizer as GestureRecognizerImpl} from '../../../tasks/web/vision/gesture_recognizer/gesture_recognizer';
import {HandLandmarker as HandLandmarkerImpl} from '../../../tasks/web/vision/hand_landmarker/hand_landmarker';
import {ImageClassifier as ImageClassifierImpl} from '../../../tasks/web/vision/image_classifier/image_classifier';
import {ImageEmbedder as ImageEmbedderImpl} from '../../../tasks/web/vision/image_embedder/image_embedder';
import {ImageSegmenter as ImageSegementerImpl} from '../../../tasks/web/vision/image_segmenter/image_segmenter';
import {InteractiveSegmenter as InteractiveSegmenterImpl} from '../../../tasks/web/vision/interactive_segmenter/interactive_segmenter';
import {ObjectDetector as ObjectDetectorImpl} from '../../../tasks/web/vision/object_detector/object_detector';

// Declare the variables locally so that Rollup in OSS includes them explicitly
// as exports.
const FilesetResolver = FilesetResolverImpl;
const FaceStylizer = FaceStylizerImpl;
const GestureRecognizer = GestureRecognizerImpl;
const HandLandmarker = HandLandmarkerImpl;
const ImageClassifier = ImageClassifierImpl;
const ImageEmbedder = ImageEmbedderImpl;
const ImageSegmenter = ImageSegementerImpl;
const InteractiveSegmenter = InteractiveSegmenterImpl;
const ObjectDetector = ObjectDetectorImpl;

export {
  FilesetResolver,
  FaceStylizer,
  GestureRecognizer,
  HandLandmarker,
  ImageClassifier,
  ImageEmbedder,
  ImageSegmenter,
  InteractiveSegmenter,
  ObjectDetector
};
