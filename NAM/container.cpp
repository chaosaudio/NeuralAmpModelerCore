#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <sstream>

#include "container.h"
#include "get_dsp.h"
#include "model_config.h"

namespace nam
{
namespace container
{

// =============================================================================
// ContainerModel
// =============================================================================

ContainerModel::ContainerModel(std::vector<Submodel> submodels, const double expected_sample_rate)
: DSP(1, 1, expected_sample_rate)
, _submodels(std::move(submodels))
{
  if (_submodels.empty())
    throw std::runtime_error("ContainerModel: no submodels provided");

  // Validate ordering and that final max_value covers 1.0
  for (size_t i = 1; i < _submodels.size(); ++i)
  {
    if (_submodels[i].max_value <= _submodels[i - 1].max_value)
      throw std::runtime_error("ContainerModel: submodels must be sorted by ascending max_value");
  }
  if (_submodels.back().max_value < 1.0)
    throw std::runtime_error("ContainerModel: last submodel max_value must be >= 1.0");

  // Validate all submodels have the same expected sample rate
  for (const auto& sm : _submodels)
  {
    double sr = sm.model->GetExpectedSampleRate();
    if (sr != expected_sample_rate && sr != NAM_UNKNOWN_EXPECTED_SAMPLE_RATE
        && expected_sample_rate != NAM_UNKNOWN_EXPECTED_SAMPLE_RATE)
    {
      std::stringstream ss;
      ss << "ContainerModel: submodel sample rate mismatch (expected " << expected_sample_rate << ", got " << sr << ")";
      throw std::runtime_error(ss.str());
    }
  }

  // Default to full size (last submodel)
  _active_index = _submodels.size() - 1;
}

void ContainerModel::process(NAM_SAMPLE** input, NAM_SAMPLE** output, const int num_frames)
{
  const size_t active_index = _active_index.load(std::memory_order_acquire);
  _submodels[active_index].model->process(input, output, num_frames);
}

void ContainerModel::prewarm()
{
  for (auto& sm : _submodels)
    sm.model->prewarm();
}

void ContainerModel::Reset(const double sampleRate, const int maxBufferSize)
{
  DSP::Reset(sampleRate, maxBufferSize);
  for (auto& sm : _submodels)
    sm.model->Reset(sampleRate, maxBufferSize);
}

void ContainerModel::ResetStateOnly(const int maxBufferSize)
{
  // The base DSP::ResetStateOnly just calls SetMaxBufferSize(), which on
  // the container itself does nothing useful (the container has no
  // per-layer state — it's a thin selector between submodels). The actual
  // state lives in each submodel, so we forward.
  DSP::ResetStateOnly(maxBufferSize);
  for (auto& sm : _submodels)
    sm.model->ResetStateOnly(maxBufferSize);
}

void ContainerModel::SetSlimmableSize(const double val)
{
  size_t active_index = _submodels.size() - 1;
  for (size_t i = 0; i < _submodels.size(); ++i)
  {
    if (val < _submodels[i].max_value)
    {
      active_index = i;
      break;
    }
  }

  // Fast path: no change to active model.
  if (active_index == _active_index.load(std::memory_order_acquire))
  {
    return;
  }

  // Plugin host can deliver param changes from both UI/controller and processor paths.
  // Serialize reset+prewarm so only one thread can perform model activation at a time.
  std::lock_guard<std::mutex> lock(_slim_set_mutex);
  if (active_index == _active_index.load(std::memory_order_acquire))
  {
    return;
  }
  // Setting _active_index puts the model in the RT path, so prewarm before doing that
  const double sr = mHaveExternalSampleRate ? mExternalSampleRate : mExpectedSampleRate;
  _submodels[active_index].model->ResetAndPrewarm(sr, GetMaxBufferSize());

  // Finally set when we're ready:
  _active_index.store(active_index, std::memory_order_release);
}

// =============================================================================
// Config / factory
// =============================================================================

std::unique_ptr<DSP> ContainerConfig::create(std::vector<float> weights, double sampleRate)
{
  (void)weights; // Container has no top-level weights

  auto submodels_json = raw_config["submodels"];
  if (!submodels_json.is_array() || submodels_json.empty())
    throw std::runtime_error("SlimmableContainer: 'submodels' must be a non-empty array");

#if defined(NAM_SLIMMABLE_KEEP_SMALLEST_ONLY)
  // Stratus only ever pins SlimmableContainer models to the smallest submodel
  // via SetSlimmableSize(0.0) at load time (e.g., the 3-channel A2 Nano in an
  // A2 model that also ships an 8-channel Standard). The 8-channel kernel
  // costs ~6.4 MB of layer-history allocation plus prewarm time and is not
  // real-time on Cortex-A8 1 GHz anyway. Build only the smallest submodel.
  //
  // Force the kept submodel's max_value to 1.0 so ContainerModel's
  // constructor validation (last submodel must cover up to 1.0) passes.
  // ContainerModel's _active_index defaults to size()-1; with one submodel
  // that's index 0, so the subsequent SetSlimmableSize(0.0) on the host side
  // becomes a fast-path no-op rather than a redundant Reset+Prewarm.
  size_t smallest_idx = 0;
  double smallest_max = std::numeric_limits<double>::infinity();
  for (size_t i = 0; i < submodels_json.size(); ++i)
  {
    double mv = submodels_json[i].at("max_value").get<double>();
    if (mv < smallest_max)
    {
      smallest_max = mv;
      smallest_idx = i;
    }
  }
  std::vector<Submodel> submodels;
  submodels.reserve(1);
  const auto& entry = submodels_json[smallest_idx];
  const auto& model_json = entry.at("model");
  auto dsp = get_dsp(model_json);
  submodels.push_back({1.0, std::move(dsp)});
#else
  std::vector<Submodel> submodels;
  submodels.reserve(submodels_json.size());

  for (const auto& entry : submodels_json)
  {
    double max_val = entry.at("max_value").get<double>();
    const auto& model_json = entry.at("model");

    // Each submodel is a full NAM model spec (has architecture, config, weights, etc.)
    auto dsp = get_dsp(model_json);

    submodels.push_back({max_val, std::move(dsp)});
  }
#endif

  return std::make_unique<ContainerModel>(std::move(submodels), sampleRate);
}

std::unique_ptr<ModelConfig> create_config(const nlohmann::json& config, double sampleRate)
{
  auto c = std::make_unique<ContainerConfig>();
  c->raw_config = config;
  c->sample_rate = sampleRate;
  return c;
}

// Auto-register
static ConfigParserHelper _register_SlimmableContainer("SlimmableContainer", create_config);

} // namespace container
} // namespace nam
