#pragma once

// Specialized WaveNet fast path for the A2 standard (Channels=8) and
// A2 nano (Channels=3) models. Shares the same architecture shape; only
// the channel count differs.
//
// When NAM_ENABLE_A2_FAST is defined at build time, wavenet::create_config
// consults is_a2_shape() on every incoming WaveNet config and, on match,
// instantiates an A2FastModel<Channels> instead of the generic WaveNet.
//
// The baseline here is correct-but-unoptimized (plain column-major loops).
// Follow-up optimizations (unrolled GEMV, tap-major nest, factored
// per-kernel-size helpers) plug into the same class.

#if defined(NAM_ENABLE_A2_FAST)

#include <array>
#include <memory>

#include "../model_config.h"
#include "json.hpp"

namespace nam
{
namespace wavenet
{
struct WaveNetConfig;

namespace a2_fast
{

/// \brief Number of layers in an A2 layer array.
constexpr int kNumLayers = 23;
/// \brief Kernel size of the layer-array head rechannel convolution.
constexpr int kHeadKernelSize = 16;
/// \brief Head scale factor used by every A2 model.
constexpr float kHeadScale = 0.01f;
/// \brief LeakyReLU negative-slope used by every layer.
constexpr float kLeakySlope = 0.01f;

/// \brief Per-layer kernel sizes (fixed pattern shared by A2 standard + nano).
inline constexpr std::array<int, kNumLayers> kKernelSizes = {
  6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 15, 15, 6, 6, 6, 6, 6, 6, 6};

/// \brief Per-layer dilations (fixed pattern shared by A2 standard + nano).
inline constexpr std::array<int, kNumLayers> kDilations = {
  1, 3, 7, 17, 41, 101, 239, 1, 3, 7, 17, 41, 101, 239, 1, 13, 1, 3, 7, 17, 41, 101, 239};

/// \brief Strict detector: returns true iff config matches the A2 shape.
/// \param config   The "config" sub-object from a .nam WaveNet entry.
/// \param channels Out-param set to 3 (A2 nano) or 8 (A2 standard) on match.
/// \return true if every architectural knob matches the A2 signature exactly.
bool is_a2_shape(const nlohmann::json& config, int* channels);

/// \brief Strict detector against an already-parsed WaveNetConfig.
/// Used to route both .nam (JSON) and .namb (binary) loaders through the
/// same A2 fast-path: the binary loader produces a WaveNetConfig directly
/// without ever touching JSON, so we must be able to detect A2 shape
/// from the parsed struct as well.
///
/// Note: distinct name (not an overload of is_a2_shape) on purpose. nlohmann
/// json has implicit conversion operators that, during overload resolution,
/// instantiate type traits over the candidate parameter types. If this were
/// declared as `is_a2_shape(const WaveNetConfig&, ...)`, every consumer of
/// the JSON variant would need WaveNetConfig to be a complete type at the
/// call site (not just forward-declared), polluting the include graph.
/// \param config   Parsed WaveNet configuration.
/// \param channels Out-param set to 3 (A2 nano) or 8 (A2 standard) on match.
/// \return true if every architectural knob matches the A2 signature exactly.
bool is_a2_shape_from_parsed(const WaveNetConfig& config, int* channels);

/// \brief Build a ModelConfig that instantiates the A2 fast path.
/// \pre is_a2_shape(config, ...) returned true.
std::unique_ptr<ModelConfig> create_a2_fast_config(const nlohmann::json& config, double sampleRate);

/// \brief Create an A2 fast-path DSP directly from weights.
/// \param channels Must be 3 (nano) or 8 (standard).
/// \param weights Model weights vector.
/// \param sampleRate Expected sample rate.
/// \return Unique pointer to DSP, or nullptr if channels is unsupported.
std::unique_ptr<DSP> create_a2_fast(int channels, std::vector<float> weights, double sampleRate);

} // namespace a2_fast
} // namespace wavenet
} // namespace nam

#endif // NAM_ENABLE_A2_FAST
