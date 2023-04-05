#include "mediapipe/tasks/cc/vision/image_segmenter/calculators/segmentation_postprocessor_gl.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "mediapipe/framework/port/status_macros.h"
#include "mediapipe/gpu/gl_simple_shaders.h"
#include "mediapipe/gpu/shader_util.h"
#include "mediapipe/tasks/cc/vision/image_segmenter/proto/segmenter_options.pb.h"

namespace mediapipe {
namespace tasks {
namespace {

using mediapipe::kBasicSquareVertices;
using mediapipe::kBasicTextureVertices;
using mediapipe::kBasicVertexShader;
using ::mediapipe::tasks::vision::Shape;
using ::mediapipe::tasks::vision::image_segmenter::proto::SegmenterOptions;

// TODO: This part of the setup code is so common, we should really
// refactor to a helper utility.
enum { ATTRIB_VERTEX, ATTRIB_TEXTURE_POSITION, NUM_ATTRIBUTES };
const GLint attr_location[NUM_ATTRIBUTES] = {
    ATTRIB_VERTEX,
    ATTRIB_TEXTURE_POSITION,
};
const GLchar* attr_name[NUM_ATTRIBUTES] = {
    "position",
    "texture_coordinate",
};

// We assume ES3.0+ for some of our shaders here so we can make liberal use of
// MRT easily.
static constexpr char kEs30RequirementHeader[] = "#version 300 es\n";

static constexpr char kActivationFragmentShader[] = R"(
DEFAULT_PRECISION(mediump, float)
in vec2 sample_coordinate;
uniform sampler2D input_texture;

void main() {
  vec4 in_value = texture2D(input_texture, sample_coordinate);

  // Run activation function over all 4 channels at once.
  %s

  gl_FragColor = out_value;
})";

// Trivial passthrough fragment shader; do splitting in a custom vertex shader.
static constexpr char kPassthroughShader[] = R"(
DEFAULT_PRECISION(mediump, float)
in vec2 sample_coordinate;
uniform sampler2D input_texture;

void main() {
  gl_FragColor = texture2D(input_texture, sample_coordinate);
})";

// Vertex shader for splitting; kLayoutAligned means we just move across x-axis.
static constexpr char kSplitVertexShader[] = R"(
DEFAULT_PRECISION(highp, float)
attribute vec4 position;
attribute vec4 texture_coordinate;
varying vec2 sample_coordinate;

// We assume kLayoutAligned for now. Everything will be scaled properly, so just
// need offset for decimation iterations.
uniform float x_offset;

void main() {
  sample_coordinate = vec2(texture_coordinate.x + x_offset, texture_coordinate.y);
  gl_Position = position;
})";

// TODO: Consider using MRT to speed this up in the future.
static constexpr char kChannelSelectShader[] = R"(
DEFAULT_PRECISION(mediump, float)
in vec2 sample_coordinate;
uniform sampler2D input_texture;
uniform int channel_select;

void main() {
  vec4 in_value = texture2D(input_texture, sample_coordinate);
  float out_value;
  if (channel_select == 0) {
    out_value = in_value.r;
  } else if (channel_select == 1) {
    out_value = in_value.g;
  } else if (channel_select == 2) {
    out_value = in_value.b;
  } else {
    out_value = in_value.a;
  }
  gl_FragColor = vec4(out_value, out_value, out_value, out_value);
})";

// Hard-coded for max of 3 textures for now, so num classes must be <= 12, and
// the cost of this shader will be higher than necessary for smaller numbers of
// classes.
// TODO: Improve this.
static constexpr char kArgmaxShader[] = R"(
DEFAULT_PRECISION(mediump, float)
in vec2 sample_coordinate;
uniform sampler2D input_texture0;
uniform sampler2D input_texture1;
uniform sampler2D input_texture2;

int argmax4(vec4 vec) {
  float aMax = max(vec.x, vec.y);
  float bMax = max(vec.z, vec.w);
  if (aMax >= bMax) {
    if (vec.x >= vec.y) return 0;
    return 1;
  } else if (vec.z >= vec.w) return 2;
  return 3;
}

float max4(vec4 vec) {
  return max(max(vec.x, vec.y), max(vec.z, vec.w));
}

void main() {
  // Grab all vecs
  vec4 pixel0 = texture2D(input_texture0, sample_coordinate);
  vec4 pixel1 = texture2D(input_texture1, sample_coordinate);
  vec4 pixel2 = texture2D(input_texture2, sample_coordinate);

  // Find vector which contains maximum value, and return its argmax
  float max0 = max4(pixel0);
  float max1 = max4(pixel1);
  float max2 = max4(pixel2);

  int argmax;
  float out_value;
  if (max0 >= max1) {
    if (max0 >= max2) {
      argmax = argmax4(pixel0);
    } else {
      argmax = argmax4(pixel2) + 8;
    }
  } else if (max1 >= max2) {
    argmax = argmax4(pixel1) + 4;
  } else {
    argmax = argmax4(pixel2) + 8;
  }

  out_value = float(argmax) / 255.0;
  gl_FragColor = vec4(out_value, out_value, out_value, out_value);
})";

// Softmax is in 3 steps:
// - First we find max over all masks
// - Then we transform all masks to be exp(val - maxval), and also add to
//   cumulative-sum image with MRT
// - Then we normalize all masks by cumulative-sum image

// Part one: max shader
// To start with, we just do this chunk by chunk, using GL_MAX blend mode so we
// don't need to tap into the max-so-far texture.
static constexpr char kMaxShader[] = R"(
DEFAULT_PRECISION(mediump, float)
in vec2 sample_coordinate;
uniform sampler2D current_chunk;
uniform int num_channels;  // how many channels from current chunk to use (1-4)

float max4(vec4 vec) {
  return max(max(vec.x, vec.y), max(vec.z, vec.w));
}
float max3(vec4 vec) {
  return max(max(vec.x, vec.y), vec.z);
}
float max2(vec4 vec) {
  return max(vec.x, vec.y);
}
void main() {
    vec4 chunk_pixel = texture2D(current_chunk, sample_coordinate);
    float new_max;
    if (num_channels == 1) {
      new_max = chunk_pixel.x;
    } else if (num_channels == 2) {
      new_max = max2(chunk_pixel);
    } else if (num_channels == 3) {
      new_max = max3(chunk_pixel);
    } else {
      new_max = max4(chunk_pixel);
    }
    gl_FragColor = vec4(new_max, 0.0, 0.0, 1.0);
})";

// Part two: transform-and-sum shader
// We use GL blending so we can more easily render a cumulative sum texture, and
// this only costs us a glClear for the output chunk (needed since using MRT).
static constexpr char kTransformAndSumShader[] = R"(
DEFAULT_PRECISION(highp, float)
in vec2 sample_coordinate;
uniform sampler2D max_value_texture;
uniform sampler2D current_chunk;
uniform int num_channels;  // how many channels from current chunk to use (1-4)

layout(location = 0) out vec4 cumulative_sum_texture;
layout(location = 1) out vec4 out_chunk_texture;

void main() {
    float max_pixel = texture(max_value_texture, sample_coordinate).r;
    vec4 chunk_pixel = texture(current_chunk, sample_coordinate);
    vec4 new_chunk_pixel = exp(chunk_pixel - max_pixel);

    float sum_so_far;
    if (num_channels == 1) {
      sum_so_far = new_chunk_pixel.x;
    } else if (num_channels == 2) {
      sum_so_far = dot(vec2(1.0, 1.0), new_chunk_pixel.xy);
    } else if (num_channels == 3) {
      sum_so_far = dot(vec3(1.0, 1.0, 1.0), new_chunk_pixel.xyz);
    } else {
      sum_so_far = dot(vec4(1.0, 1.0, 1.0, 1.0), new_chunk_pixel);
    }

    cumulative_sum_texture = vec4(sum_so_far, 0.0, 0.0, 1.0);
    out_chunk_texture = new_chunk_pixel;
})";

// Part three: normalization shader
static constexpr char kNormalizationShader[] = R"(
DEFAULT_PRECISION(mediump, float)
in vec2 sample_coordinate;
uniform sampler2D sum_texture;  // cumulative summation value (to normalize by)
uniform sampler2D current_chunk;  // current chunk

void main() {
    float sum_pixel = texture2D(sum_texture, sample_coordinate).r;
    vec4 chunk_pixel = texture2D(current_chunk, sample_coordinate);

    // NOTE: We assume non-zero sum_pixel here, which is a safe assumption for
    // result of an exp transform, but not if this shader is extended to other
    // uses.
    gl_FragColor = chunk_pixel / sum_pixel;
})";

}  // namespace

// static
absl::Status SegmentationPostprocessorGl::UpdateContract(
    CalculatorContract* cc) {
  return GlCalculatorHelper::UpdateContract(cc);
}

absl::Status SegmentationPostprocessorGl::Initialize(
    CalculatorContext* cc,
    TensorsToSegmentationCalculatorOptions const& options) {
  options_ = options;  // Just copy for now
  MP_RETURN_IF_ERROR(helper_.Open(cc));
  MP_RETURN_IF_ERROR(GlInit());
  return absl::OkStatus();
}

absl::Status SegmentationPostprocessorGl::CreateBasicFragmentShaderProgram(
    std::string const& program_name, std::string const& fragment_shader_source,
    std::vector<std::string> const& uniform_names, GlShader* shader_struct_ptr,
    bool is_es30_only = false) {
  // Format source and create basic ES3.0+ fragment-shader-only program
  const std::string frag_shader_source =
      absl::StrCat(is_es30_only ? std::string(kEs30RequirementHeader) : "",
                   std::string(mediapipe::kMediaPipeFragmentShaderPreamble),
                   std::string(fragment_shader_source));
  const std::string vert_shader_source =
      absl::StrCat(is_es30_only ? std::string(kEs30RequirementHeader) : "",
                   std::string(kBasicVertexShader));
  mediapipe::GlhCreateProgram(
      vert_shader_source.c_str(), frag_shader_source.c_str(), NUM_ATTRIBUTES,
      &attr_name[0], attr_location, &shader_struct_ptr->program,
      /* force_log_errors */ true);
  RET_CHECK(shader_struct_ptr->program)
      << "Problem initializing the " << program_name << " program.";

  // Hook up all desired uniforms
  for (const auto& uniform_name : uniform_names) {
    shader_struct_ptr->uniforms[uniform_name] =
        glGetUniformLocation(shader_struct_ptr->program, uniform_name.c_str());
    RET_CHECK(shader_struct_ptr->uniforms[uniform_name] > 0)
        << uniform_name << " uniform not found for " << program_name
        << " program";
  }
  return absl::OkStatus();
}

absl::Status SegmentationPostprocessorGl::GlInit() {
  return helper_.RunInGlContext([this]() -> absl::Status {
    // Default to passthrough/NONE
    std::string activation_fn = "vec4 out_value = in_value;";
    switch (options_.segmenter_options().activation()) {
      case SegmenterOptions::SIGMOID:
        LOG(INFO) << "SIGMOID activation function chosen on GPU";
        activation_fn = "vec4 out_value = 1.0 / (exp(-in_value) + 1.0);";
        break;
      case SegmenterOptions::SOFTMAX:
        LOG(WARNING) << "SOFTMAX activation function not yet efficient on GPU";
        break;
      case SegmenterOptions::NONE:
        LOG(INFO) << "NONE activation function chosen on GPU";
        break;
    }

    // TODO: Skip activation step entirely for "NONE" to save a full
    //     renderpass.  (And same applies for CATEGORY_MASK mode).
    bool is_category_mask = options_.segmenter_options().output_type() ==
                            SegmenterOptions::CATEGORY_MASK;
    if (is_category_mask) {
      LOG(INFO) << "CATEGORY_MASK requested; using NONE activation function.";
    }

    const std::string activation_shader_source =
        absl::StrCat(std::string(mediapipe::kMediaPipeFragmentShaderPreamble),
                     absl::StrFormat(kActivationFragmentShader, activation_fn));

    const std::string split_fragment_shader_source =
        absl::StrCat(std::string(mediapipe::kMediaPipeFragmentShaderPreamble),
                     std::string(kPassthroughShader));
    const std::string split_vertex_shader_source =
        absl::StrCat(std::string(mediapipe::kMediaPipeVertexShaderPreamble),
                     std::string(kSplitVertexShader));

    const std::string channel_select_shader_source =
        absl::StrCat(std::string(mediapipe::kMediaPipeFragmentShaderPreamble),
                     std::string(kChannelSelectShader));

    const std::string argmax_shader_source =
        absl::StrCat(std::string(mediapipe::kMediaPipeFragmentShaderPreamble),
                     std::string(kArgmaxShader));

    // Softmax shaders (Max, Transform+Sum, and Normalization)
    MP_RETURN_IF_ERROR(CreateBasicFragmentShaderProgram(
        "softmax max", kMaxShader, {"current_chunk", "num_channels"},
        &softmax_max_shader_));
    MP_RETURN_IF_ERROR(CreateBasicFragmentShaderProgram(
        "softmax transform-and-sum", kTransformAndSumShader,
        {"max_value_texture", "current_chunk", "num_channels"},
        &softmax_transform_and_sum_shader_, true /* is_es30_only */));
    MP_RETURN_IF_ERROR(CreateBasicFragmentShaderProgram(
        "softmax normalization", kNormalizationShader,
        {"sum_texture", "current_chunk"}, &softmax_normalization_shader_));

    // Compile all our shader programs.
    // Note: we enable `force_log_errors` so that we get full debugging error
    //   messages when compiling shaders on web, where normally such errors are
    //   suppressed. See //mediapipe/gpu/shader_util.cc for more
    //   info.
    mediapipe::GlhCreateProgram(
        kBasicVertexShader, activation_shader_source.c_str(), NUM_ATTRIBUTES,
        &attr_name[0], attr_location, &activation_program_,
        /* force_log_errors */ true);
    RET_CHECK(activation_program_)
        << "Problem initializing the activation program.";

    mediapipe::GlhCreateProgram(split_vertex_shader_source.c_str(),
                                split_fragment_shader_source.c_str(),
                                NUM_ATTRIBUTES, &attr_name[0], attr_location,
                                &split_program_,
                                /* force_log_errors */ true);
    RET_CHECK(split_program_) << "Problem initializing the split program.";

    mediapipe::GlhCreateProgram(
        kBasicVertexShader, channel_select_shader_source.c_str(),
        NUM_ATTRIBUTES, &attr_name[0], attr_location, &channel_select_program_,
        /* force_log_errors */ true);
    RET_CHECK(channel_select_program_)
        << "Problem initializing the channel select program.";

    mediapipe::GlhCreateProgram(kBasicVertexShader,
                                argmax_shader_source.c_str(), NUM_ATTRIBUTES,
                                &attr_name[0], attr_location, &argmax_program_,
                                /* force_log_errors */ true);
    RET_CHECK(argmax_program_) << "Problem initializing the argmax program.";

    // Get uniform locations.
    activation_texture_uniform_ =
        glGetUniformLocation(activation_program_, "input_texture");
    RET_CHECK(activation_texture_uniform_ > 0)
        << "activation input_texture uniform not found.";

    split_texture_uniform_ =
        glGetUniformLocation(split_program_, "input_texture");
    RET_CHECK(split_texture_uniform_ > 0)
        << "split input_texture uniform not found.";
    split_x_offset_uniform_ = glGetUniformLocation(split_program_, "x_offset");
    RET_CHECK(split_x_offset_uniform_ > 0)
        << "split x_offset uniform not found.";

    channel_select_texture_uniform_ =
        glGetUniformLocation(channel_select_program_, "input_texture");
    RET_CHECK(channel_select_texture_uniform_ > 0)
        << "channel select input_texture uniform not found.";
    channel_select_index_uniform_ =
        glGetUniformLocation(channel_select_program_, "channel_select");
    RET_CHECK(channel_select_index_uniform_ > 0)
        << "channel select indexing uniform not found.";

    argmax_texture0_uniform_ =
        glGetUniformLocation(argmax_program_, "input_texture0");
    RET_CHECK(argmax_texture0_uniform_ > 0)
        << "argmax input_texture0 uniform not found.";
    argmax_texture1_uniform_ =
        glGetUniformLocation(argmax_program_, "input_texture1");
    RET_CHECK(argmax_texture1_uniform_ > 0)
        << "argmax input_texture1 uniform not found.";
    argmax_texture2_uniform_ =
        glGetUniformLocation(argmax_program_, "input_texture2");
    RET_CHECK(argmax_texture2_uniform_ > 0)
        << "argmax input_texture2 uniform not found.";

    // TODO: If ES3.0+ only, switch to VAO for handling attributes.
    glGenBuffers(1, &square_vertices_);
    glBindBuffer(GL_ARRAY_BUFFER, square_vertices_);
    glBufferData(GL_ARRAY_BUFFER, sizeof(kBasicSquareVertices),
                 kBasicSquareVertices, GL_STATIC_DRAW);

    glGenBuffers(1, &texture_vertices_);
    glBindBuffer(GL_ARRAY_BUFFER, texture_vertices_);
    glBufferData(GL_ARRAY_BUFFER, sizeof(kBasicTextureVertices),
                 kBasicTextureVertices, GL_STATIC_DRAW);

    glBindBuffer(GL_ARRAY_BUFFER, 0);

    return absl::OkStatus();
  });
}

std::vector<std::unique_ptr<Image>>
SegmentationPostprocessorGl::GetSegmentationResultGpu(const Shape& input_shape,
                                                      const Shape& output_shape,
                                                      const Tensor& tensor) {
  std::vector<std::unique_ptr<Image>> image_outputs;
  auto status = helper_.RunInGlContext([this, &input_shape, &output_shape,
                                        &tensor,
                                        &image_outputs]() -> absl::Status {
    // Get Tensor input and image output parameters
    int input_width, input_height;

    if (!tensor.ready_as_opengl_texture_2d()) {
      LOG(WARNING) << "Tensor wasn't ready on GPU; using slow workaround.";
      (void)tensor.GetCpuReadView();
    }

    const auto layout = tensor.GetOpenGlTexture2dReadView().GetLayoutDimensions(
        tensor.shape(), &input_width, &input_height);
    if (layout != Tensor::OpenGlTexture2dView::Layout::kAligned) {
      LOG(ERROR) << "Tensor layout not kAligned! Cannot handle.";
    }

    bool is_category_mask = options_.segmenter_options().output_type() ==
                            SegmenterOptions::CATEGORY_MASK;
    bool is_softmax =
        options_.segmenter_options().activation() == SegmenterOptions::SOFTMAX;

    const GpuBufferFormat activation_output_format =
        GpuBufferFormat::kRGBAFloat128;
    const GpuBufferFormat chunk_output_format = GpuBufferFormat::kRGBAFloat128;

    // Uint8 pipeline and conversions are lacking, so for now we just use F32
    // textures even for category masks.
    // TODO: Also, some platforms (like certain iOS devices) do not
    //   allow for rendering to RGBAF32 textures, so we should switch to using
    //   F16 textures in those instances.
    const GpuBufferFormat final_output_format = GpuBufferFormat::kGrayFloat32;
    const Tensor::OpenGlTexture2dView read_view =
        tensor.GetOpenGlTexture2dReadView();

    const int width = input_shape.width;           // Slice width from shape
    const int height = input_shape.height;         // Slice height from chape
    const int num_outputs = input_shape.channels;  // One output per channel
    const int num_chunks = (input_shape.channels + 3) / 4;  // ceil(channels/4)
    const int output_width = output_shape.width;    // Final output width
    const int output_height = output_shape.height;  // Final output height

    // We disable blending or else our alpha channel may destroy our other
    // channels' data.
    glDisable(GL_BLEND);

    // Step 0: bind buffers / textures
    glBindBuffer(GL_ARRAY_BUFFER, square_vertices_);
    glVertexAttribPointer(ATTRIB_VERTEX, 2, GL_FLOAT, 0, 0, nullptr);
    glEnableVertexAttribArray(ATTRIB_VERTEX);

    glBindBuffer(GL_ARRAY_BUFFER, texture_vertices_);
    glVertexAttribPointer(ATTRIB_TEXTURE_POSITION, 2, GL_FLOAT, 0, 0, nullptr);
    glEnableVertexAttribArray(ATTRIB_TEXTURE_POSITION);

    // Step 1: apply activation pass
    glUseProgram(activation_program_);
    glUniform1i(activation_texture_uniform_, 1);
    GlTexture activated_texture = helper_.CreateDestinationTexture(
        input_width, input_height, activation_output_format);
    helper_.BindFramebuffer(activated_texture);

    // All our input source textures are just simple GL_TEXTURE_2D types.
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, read_view.name());

    // Render
    glClear(GL_COLOR_BUFFER_BIT);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

    // Step 2: split megatexture into 4-chunks (assume kLayoutAligned for now).
    std::vector<GlTexture> chunks;
    // # chunks: offset in pixels at which taps must be made
    // 1 chunk: 0
    // 2 chunks: -0.5, +0.5
    // 3 chunks: -1,0,1
    // 4 chunks: -1.5, -.5, .5, 1.5
    // ...
    // Step is always 1 pixel, while initial offset is (1 - N) * 0.5
    glUseProgram(split_program_);
    glUniform1i(split_texture_uniform_, 1);
    const float tex_offset = 0.5 * (1.0 - (float)num_chunks);
    for (int i = 0; i < num_chunks; i++) {
      chunks.push_back(
          helper_.CreateDestinationTexture(width, height, chunk_output_format));
      helper_.BindFramebuffer(chunks.back());
      glUniform1f(split_x_offset_uniform_,
                  ((float)i + tex_offset) / (float)(input_width));
      // Technically duplicated, but fine for now; we want this after the bind
      glBindTexture(GL_TEXTURE_2D, activated_texture.name());
      // Disable hardware GPU interpolation
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
      // Render
      glClear(GL_COLOR_BUFFER_BIT);
      glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    }

    std::vector<GlTexture> softmax_chunks;
    if (is_softmax) {
      // Step 2.5: For SOFTMAX, apply softmax shaders (max, transformAndSum, and
      // normalization) to create softmax-transformed chunks before channel
      // extraction.
      // NOTE: exp(x-C) / sum_over_x(exp(x-C)) = exp(x) / sum_over_x(exp(x)). So
      //   theoretically we can skip the max shader step entirely. However,
      //   applying it does bring all our values into a nice (0, 1] range, so it
      //   will likely be better for precision, especially when dealing with an
      //   exponential function on arbitrary values. Therefore, we keep it, but
      //   this is potentially a skippable step for known "good" models, if we
      //   ever want to provide that as an option.
      // TODO: For a tiny bit more efficiency, could combine channel
      // extraction into last step of this via MRT.

      // Max
      glUseProgram(softmax_max_shader_.program);
      glUniform1i(softmax_max_shader_.uniforms["current_chunk"], 1);

      // We just need one channel, so format will match final output confidence
      // masks
      auto max_texture =
          helper_.CreateDestinationTexture(width, height, final_output_format);
      helper_.BindFramebuffer(max_texture);

      // We clear our newly-created destination texture to a reasonable minimum.
      glClearColor(0.0, 0.0, 0.0, 0.0);
      glClear(GL_COLOR_BUFFER_BIT);

      // We will use hardware GPU blending to apply max to all our writes.
      glEnable(GL_BLEND);
      glBlendEquation(GL_MAX);

      glActiveTexture(GL_TEXTURE1);
      for (int i = 0; i < num_chunks; i++) {
        int num_channels = 4;
        if ((i + 1) * 4 > num_outputs) num_channels = num_outputs % 4;
        glUniform1i(softmax_max_shader_.uniforms["num_channels"], num_channels);
        glBindTexture(GL_TEXTURE_2D, chunks[i].name());
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
      }

      // Transform & Sum
      std::vector<GlTexture> unnormalized_softmax_chunks;
      glUseProgram(softmax_transform_and_sum_shader_.program);
      glUniform1i(softmax_transform_and_sum_shader_.uniforms["current_chunk"],
                  1);
      glUniform1i(
          softmax_transform_and_sum_shader_.uniforms["max_value_texture"], 2);

      auto sum_texture =
          helper_.CreateDestinationTexture(width, height, final_output_format);
      helper_.BindFramebuffer(sum_texture);
      glClear(GL_COLOR_BUFFER_BIT);

      glActiveTexture(GL_TEXTURE2);
      glBindTexture(GL_TEXTURE_2D, max_texture.name());

      glBlendEquation(GL_FUNC_ADD);
      glBlendFunc(GL_ONE, GL_ONE);
      glActiveTexture(GL_TEXTURE1);

      // We use glDrawBuffers to clear only the new texture, then again to
      // draw to both textures simultaneously for rendering.
      GLuint both_attachments[2] = {GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1};
      GLuint one_attachment[2] = {GL_NONE, GL_COLOR_ATTACHMENT1};
      for (int i = 0; i < num_chunks; i++) {
        int num_channels = 4;
        if ((i + 1) * 4 > num_outputs) num_channels = num_outputs % 4;
        glUniform1i(softmax_transform_and_sum_shader_.uniforms["num_channels"],
                    num_channels);
        unnormalized_softmax_chunks.push_back(helper_.CreateDestinationTexture(
            width, height, chunk_output_format));
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1,
                               GL_TEXTURE_2D,
                               unnormalized_softmax_chunks.back().name(), 0);

        // Note that we must bind AFTER the CreateDestinationTexture, or else we
        // end up with (0, 0, 0, 1) data being read from an unbound texture
        // unit.
        glBindTexture(GL_TEXTURE_2D, chunks[i].name());

        // Clear *only* the new chunk
        glDrawBuffers(2, one_attachment);
        glClear(GL_COLOR_BUFFER_BIT);

        // Then draw into both
        glDrawBuffers(2, both_attachments);
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
      }

      // Turn off MRT and blending, and unbind second color attachment
      glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1,
                             GL_TEXTURE_2D, 0, 0);
      glDrawBuffers(1, both_attachments);
      glDisable(GL_BLEND);

      // Normalize each chunk into a new chunk as our final step
      glUseProgram(softmax_normalization_shader_.program);
      glUniform1i(softmax_normalization_shader_.uniforms["current_chunk"], 1);
      glUniform1i(softmax_normalization_shader_.uniforms["sum_texture"], 2);

      glActiveTexture(GL_TEXTURE2);
      glBindTexture(GL_TEXTURE_2D, sum_texture.name());
      glActiveTexture(GL_TEXTURE1);

      for (int i = 0; i < num_chunks; i++) {
        softmax_chunks.push_back(helper_.CreateDestinationTexture(
            width, height, chunk_output_format));
        helper_.BindFramebuffer(softmax_chunks.back());
        glBindTexture(GL_TEXTURE_2D, unnormalized_softmax_chunks[i].name());
        glClear(GL_COLOR_BUFFER_BIT);
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
      }

      // Unbind textures here
      glActiveTexture(GL_TEXTURE2);
      glBindTexture(GL_TEXTURE_2D, 0);
      // We make sure to switch back to texture unit 1, since our confidence
      // mask extraction code assumes that's our default.
      glActiveTexture(GL_TEXTURE1);
      glBindTexture(GL_TEXTURE_2D, 0);
    }

    std::vector<GlTexture> outputs;
    if (is_category_mask) {
      // Step 3: For CATEGORY, apply argmax shader with up to 3 textures to
      // extract final index mask.
      RET_CHECK(num_chunks <= 3)
          << "Cannot handle more than 12 classes in argmax shader.";

      glUseProgram(argmax_program_);
      glUniform1i(argmax_texture0_uniform_, 1);
      glUniform1i(argmax_texture1_uniform_, 2);
      glUniform1i(argmax_texture2_uniform_, 3);
      outputs.push_back(helper_.CreateDestinationTexture(
          output_width, output_height, final_output_format));
      helper_.BindFramebuffer(outputs.back());

      // Bind however many chunks we have
      for (int i = 0; i < num_chunks; ++i) {
        glActiveTexture(GL_TEXTURE1 + i);
        glBindTexture(GL_TEXTURE_2D, chunks[i].name());
      }

      for (int i = num_chunks; i < 3; ++i) {  // 3 is hard-coded max chunks
        glActiveTexture(GL_TEXTURE1 + i);
        // If texture is unbound, sampling from it should always give zeros.
        // This is not ideal, but is ok for now for not polluting the argmax
        // shader results too much.
        glBindTexture(GL_TEXTURE_2D, 0);
      }

      glClear(GL_COLOR_BUFFER_BIT);
      glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

      // Unbind the extra textures here.
      for (int i = 0; i < num_chunks; ++i) {
        glActiveTexture(GL_TEXTURE1 + i);
        glBindTexture(GL_TEXTURE_2D, 0);
      }
    } else {
      // Step 3: For CONFIDENCE, apply channel-select repeatedly to extract
      // final textures.
      glUseProgram(channel_select_program_);
      glUniform1i(channel_select_texture_uniform_, 1);
      for (int i = 0; i < num_outputs; i++) {
        glUniform1i(channel_select_index_uniform_, (i % 4));
        outputs.push_back(helper_.CreateDestinationTexture(
            output_width, output_height, final_output_format));
        helper_.BindFramebuffer(outputs.back());

        // We have to rebind constantly because BindFramebuffer seems to
        // interfere with this.
        if (is_softmax) {
          glBindTexture(GL_TEXTURE_2D, softmax_chunks[i / 4].name());
        } else {
          glBindTexture(GL_TEXTURE_2D, chunks[i / 4].name());
        }

        glClear(GL_COLOR_BUFFER_BIT);
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
      }
    }

    // Unbind everything
    glDisableVertexAttribArray(ATTRIB_VERTEX);
    glDisableVertexAttribArray(ATTRIB_TEXTURE_POSITION);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glBindTexture(GL_TEXTURE_2D, 0);

    // Get Image vector from GlTexture vector
    for (auto& output_texture : outputs) {
      image_outputs.push_back(output_texture.GetFrame<Image>());
    }

    return absl::OkStatus();
  });

  if (!status.ok()) {
    LOG(ERROR) << "Error with rendering: " << status;
  }

  return image_outputs;
}

// Cleanup OpenGL resources on destruction
SegmentationPostprocessorGl::~SegmentationPostprocessorGl() {
  helper_.RunInGlContext([this] {
    glDeleteProgram(activation_program_);
    glDeleteProgram(argmax_program_);
    glDeleteProgram(channel_select_program_);
    glDeleteProgram(split_program_);
    glDeleteBuffers(1, &square_vertices_);
    glDeleteBuffers(1, &texture_vertices_);
    activation_program_ = 0;
    argmax_program_ = 0;
    channel_select_program_ = 0;
    split_program_ = 0;
    square_vertices_ = 0;
    texture_vertices_ = 0;

    glDeleteProgram(softmax_max_shader_.program);
    glDeleteProgram(softmax_transform_and_sum_shader_.program);
    glDeleteProgram(softmax_normalization_shader_.program);
  });
}

}  // namespace tasks
}  // namespace mediapipe
