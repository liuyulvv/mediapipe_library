// Copyright 2023 The MediaPipe Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package com.google.mediapipe.tasks.vision.interactivesegmenter;

import static com.google.common.truth.Truth.assertThat;

import android.content.res.AssetManager;
import android.graphics.BitmapFactory;
import androidx.test.core.app.ApplicationProvider;
import androidx.test.ext.junit.runners.AndroidJUnit4;
import com.google.mediapipe.framework.image.BitmapImageBuilder;
import com.google.mediapipe.framework.image.MPImage;
import com.google.mediapipe.tasks.components.containers.NormalizedKeypoint;
import com.google.mediapipe.tasks.core.BaseOptions;
import com.google.mediapipe.tasks.vision.imagesegmenter.ImageSegmenterResult;
import com.google.mediapipe.tasks.vision.interactivesegmenter.InteractiveSegmenter.InteractiveSegmenterOptions;
import java.io.InputStream;
import java.util.List;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Suite;
import org.junit.runners.Suite.SuiteClasses;

/** Test for {@link InteractiveSegmenter}. */
@RunWith(Suite.class)
@SuiteClasses({
  InteractiveSegmenterTest.General.class,
})
public class InteractiveSegmenterTest {
  private static final String DEEPLAB_MODEL_FILE = "ptm_512_hdt_ptm_woid.tflite";
  private static final String CATS_AND_DOGS_IMAGE = "cats_and_dogs.jpg";
  private static final int MAGNIFICATION_FACTOR = 10;

  @RunWith(AndroidJUnit4.class)
  public static final class General extends InteractiveSegmenterTest {
    @Test
    public void segment_successWithCategoryMask() throws Exception {
      final String inputImageName = CATS_AND_DOGS_IMAGE;
      final InteractiveSegmenter.RegionOfInterest roi =
          InteractiveSegmenter.RegionOfInterest.create(NormalizedKeypoint.create(0.25f, 0.9f));
      InteractiveSegmenterOptions options =
          InteractiveSegmenterOptions.builder()
              .setBaseOptions(BaseOptions.builder().setModelAssetPath(DEEPLAB_MODEL_FILE).build())
              .setOutputType(InteractiveSegmenterOptions.OutputType.CATEGORY_MASK)
              .build();
      InteractiveSegmenter imageSegmenter =
          InteractiveSegmenter.createFromOptions(
              ApplicationProvider.getApplicationContext(), options);
      MPImage image = getImageFromAsset(inputImageName);
      ImageSegmenterResult actualResult = imageSegmenter.segment(image, roi);
      List<MPImage> segmentations = actualResult.segmentations();
      assertThat(segmentations.size()).isEqualTo(1);
    }

    @Test
    public void segment_successWithConfidenceMask() throws Exception {
      final String inputImageName = CATS_AND_DOGS_IMAGE;
      final InteractiveSegmenter.RegionOfInterest roi =
          InteractiveSegmenter.RegionOfInterest.create(NormalizedKeypoint.create(0.25f, 0.9f));
      InteractiveSegmenterOptions options =
          InteractiveSegmenterOptions.builder()
              .setBaseOptions(BaseOptions.builder().setModelAssetPath(DEEPLAB_MODEL_FILE).build())
              .setOutputType(InteractiveSegmenterOptions.OutputType.CONFIDENCE_MASK)
              .build();
      InteractiveSegmenter imageSegmenter =
          InteractiveSegmenter.createFromOptions(
              ApplicationProvider.getApplicationContext(), options);
      ImageSegmenterResult actualResult =
          imageSegmenter.segment(getImageFromAsset(inputImageName), roi);
      List<MPImage> segmentations = actualResult.segmentations();
      assertThat(segmentations.size()).isEqualTo(2);
    }
  }

  private static MPImage getImageFromAsset(String filePath) throws Exception {
    AssetManager assetManager = ApplicationProvider.getApplicationContext().getAssets();
    InputStream istr = assetManager.open(filePath);
    return new BitmapImageBuilder(BitmapFactory.decodeStream(istr)).build();
  }
}
