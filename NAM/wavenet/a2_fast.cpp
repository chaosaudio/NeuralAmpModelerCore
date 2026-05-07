#if defined(NAM_ENABLE_A2_FAST)

// Ring-buffer strategy:
//   0 = linear memmove-rewind (variable worst-case latency, sporadic spikes)
//   1 = pow2 + tail mirror (constant per-block work, branchless reads)
// Controlled externally with -DNAM_A2_RING_MODE=0 for head-to-head comparison.
#ifndef NAM_A2_RING_MODE
#define NAM_A2_RING_MODE 1
#endif

#include "a2_fast.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <iterator>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Dense>

#ifdef __ARM_NEON__
#include <arm_neon.h>
#endif

#ifdef STRATUS_NAM_LAYER_PROFILING
#include <cstdint>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <limits>
#endif

#include "../dsp.h"

namespace nam
{
namespace wavenet
{
namespace a2_fast
{

namespace
{

// =============================================================================
// A2FastModel<Channels>
//
// Skeleton implementation: correct but not yet optimized.
//
// Architectural invariants (checked once by is_a2_shape before we get here):
//   - single layer array with 23 layers
//   - Bottleneck == Channels
//   - condition_size == input_size == out_channels == 1
//   - LeakyReLU(0.01) on every layer, no gating, no FiLM, no head1x1
//   - layer1x1 active (groups=1), head rechannel conv k=16 bias=true
//   - head_scale == 0.01, no post-stack head
//
// Weight storage: column-major per kernel tap. For a (out_ch × in_ch) matrix
// at tap k, element (row=i, col=j) lives at w[k][j * out_ch + i]. All 1×1
// and K×1 convolutions follow the same convention (with K = 1 for 1×1).
// =============================================================================
template <int Channels>
class A2FastModel : public DSP
{
public:
  static constexpr int kChannels = Channels;
  static constexpr int kBottleneck = Channels;
  static constexpr int kHeadIn = Channels;

  A2FastModel(std::vector<float> weights, double expected_sample_rate);
  ~A2FastModel() override = default;

  void process(NAM_SAMPLE** input, NAM_SAMPLE** output, int num_frames) override;

protected:
  void SetMaxBufferSize(int maxBufferSize) override;
  int PrewarmSamples() override { return _prewarm_samples; }

private:
  struct Layer
  {
    int kernel_size = 0;
    int dilation = 0;
    int max_lookback = 0; // (kernel_size - 1) * dilation

    // Dilated conv (Channels -> Bottleneck), column-major per tap.
    // Flat size = kernel_size * Channels * Bottleneck.
    std::vector<float> conv_w;
    std::array<float, Channels> conv_b{};

    // Input mixin (cond_size=1 -> Bottleneck), no bias.
    std::array<float, Channels> mixin_w{};

    // layer1x1 (Bottleneck -> Channels), with bias. Column-major (Channels × Bottleneck).
    std::array<float, Channels * Channels> l1x1_w{};
    std::array<float, Channels> l1x1_b{};

    // Conv1D input history ring buffer, column-major (Channels rows).
    // Points into A2FastModel::_history_arena (a single allocation that
    // holds all 23 layers' history buffers). Sequential placement inside
    // the arena naturally spreads each layer's base mod-8K (the L1 set
    // stride on Cortex-A8) because per-layer slot size mod 8 KB ≠ 0, so
    // 23 layers land at 23 distinct mod-8K positions roughly evenly. This
    // is the cache-coloring step — it doesn't reduce memory traffic but
    // it does reduce L1 set-conflict misses, especially for the dilation-
    // 239 layers whose tap-read range is wide enough to brush up against
    // _layer_in / _head_history mod-8K positions on a poorly-colored layout.
    float* history = nullptr;
#if NAM_A2_RING_MODE == 1
    // pow2 ring + tail mirror. Storage = (pow2_size + max_buffer_size) cols.
    // write_pos is kept in [0, pow2_size), reads use (pos & pow2_mask) and are
    // always contiguous because cols [pow2_size, pow2_size + max_buffer_size)
    // mirror cols [0, max_buffer_size).
    int pow2_size = 0;
    int pow2_mask = 0;
    int write_pos = 0;
#else
    // Linear ring with sporadic memmove-rewind. history_cols = 2*max_lookback +
    // max_buffer_size; write_pos grows monotonically until rewind fires.
    int history_cols = 0;
    int write_pos = 0;
#endif
  };

  std::array<Layer, kNumLayers> _layers;

  // Rechannel (input_size=1 -> Channels), no bias.
  std::array<float, Channels> _rechannel_w{};

  // Head rechannel (Bottleneck -> 1), kernel=16, bias. Column-major per tap.
  // At each tap, matrix is (1 × Channels) col-major -> Channels floats.
  std::array<std::array<float, Channels>, kHeadKernelSize> _head_w{};
  float _head_b = 0.0f;

  // Head scale is stored as the trailing float in the weights stream (the generic
  // WaveNet reads it the same way, overriding the JSON head_scale field).
  float _head_scale = kHeadScale;

  // Head ring buffer (Channels rows, col-major). Same ring layout as per-layer.
  std::vector<float> _head_history;
#if NAM_A2_RING_MODE == 1
  int _head_pow2_size = 0;
  int _head_pow2_mask = 0;
  int _head_write_pos = 0;
#else
  int _head_history_cols = 0;
  int _head_write_pos = 0;
#endif

  // Single arena for all 23 layers' history buffers. Per-layer Layer::history
  // pointers index into this arena. The point is cache-coloring: when each
  // layer is its own std::vector<float> the heap allocator picks addresses
  // that often cluster mod 8 KB (Cortex-A8 L1 set stride), causing conflict
  // misses on the dilation-239 layers where tap reads, _layer_in writes, and
  // _head_history writes all need cache room simultaneously. Allocating one
  // arena and placing layers sequentially naturally spreads bases across the
  // mod-8K range because per-layer slot size mod 8 KB ≠ 0.
  std::vector<float> _history_arena;

  // Working buffers (all Channels rows, max_buffer_size cols, col-major).
  std::vector<float> _layer_in; // current layer input / next layer input (in-place residual)
  std::vector<float> _z;        // per-layer conv output accumulator (tap-major); used by Channels=8 path only
  std::vector<float> _cond;     // float32 copy of the double NAM_SAMPLE input, reused each block
  std::vector<float> _head_out; // float32 head output before writing to NAM_SAMPLE

  int _prewarm_samples = 0;

#ifdef STRATUS_NAM_LAYER_PROFILING
  // Per-layer wall-clock stats (CLOCK_MONOTONIC_RAW). Reset every dump.
  // Per-call timer cost on Bela's Cortex-A8 measures ~1.3 µs (the kernel's
  // clock_gettime is not VDSO-fast on this build), so 23 layers × 2 calls ≈
  // 60 µs/buffer ≈ 5% CPU overhead at -p 128 / 44.1 kHz. Off in production
  // by default; subtract ~5pt from the observed CPU% for a true reading.
  struct LayerStats
  {
    double total_ns = 0.0;
    double min_ns = std::numeric_limits<double>::max();
    double max_ns = 0.0;
    uint64_t calls = 0;
  };
  std::array<LayerStats, kNumLayers> _layer_stats{};
  uint64_t _layer_stats_buffer_count = 0;
  static constexpr uint64_t kLayerStatsDumpInterval = 3440;
  bool _history_addrs_logged = false;

  void _dump_layer_stats();
  void _log_history_addresses();
#endif

  void _load_weights(std::vector<float>& weights);
  void _ring_write(Layer& L, int num_frames);
  void _layer_forward(int layer_idx, const float* cond, int num_frames, bool is_first, int head_wp);
  void _head_forward(float* output, int num_frames);

  // Compile-time-specialized per-layer kernel. KernelSize is lifted to a
  // template parameter so the K tap loop becomes a compile-time constant;
  // the compiler can unroll and schedule FMAs across taps. For the A2 shape
  // we only need K=6 and K=15.
  //
  // IsFirst selects whether the head accumulator write uses `=` (first layer,
  // initializing the recent _head_history slots) or `+=` (subsequent layers).
  // This lets us drop the explicit memset that a separate _head_sum buffer
  // would otherwise need.
  //
  // head_wp is the snapshot of _head_write_pos at process() entry; layers
  // write the head accumulator directly into _head_history[head_wp + f] (no
  // intermediate _head_sum buffer). Caller (process()) ensures head_wp +
  // num_frames does not overflow the writable contiguous range by rewinding
  // when needed.
  template <int KernelSize, bool IsFirst>
  void _layer_forward_k(Layer& L, const float* cond, int num_frames, int head_wp);
};

// -----------------------------------------------------------------------------
// Construction
// -----------------------------------------------------------------------------
template <int Channels>
A2FastModel<Channels>::A2FastModel(std::vector<float> weights, double expected_sample_rate)
: DSP(/*in_channels=*/1, /*out_channels=*/1, expected_sample_rate)
{
  for (int i = 0; i < kNumLayers; i++)
  {
    _layers[i].kernel_size = kKernelSizes[i];
    _layers[i].dilation = kDilations[i];
    _layers[i].max_lookback = (kKernelSizes[i] - 1) * kDilations[i];
    _layers[i].conv_w.assign(static_cast<size_t>(kKernelSizes[i]) * Channels * Channels, 0.0f);
  }

  _load_weights(weights);

  int prewarm = 0;
  for (int i = 0; i < kNumLayers; i++)
    prewarm += _layers[i].max_lookback;
  prewarm += kHeadKernelSize - 1;
  _prewarm_samples = prewarm;
}

// -----------------------------------------------------------------------------
// Weight loader
//
// Reproduces the generic path's weight-reading order exactly:
//   - LayerArray::set_weights_:
//       _rechannel (Conv1x1 1 -> Channels, no bias)
//       for each layer:
//           _conv (Conv1D Channels -> Bottleneck, K × C × B + B bias)
//           _input_mixin (Conv1x1 1 -> Bottleneck, no bias)
//           _layer1x1 (Conv1x1 Bottleneck -> Channels, with bias)
//       _head_rechannel (Conv1D Bottleneck -> 1, K=16, bias)
//
// Generic Conv1D loader order: for i in out_ch: for j in in_ch: for k in taps.
// Generic Conv1x1 loader order: for i in out_ch: for j in in_ch.
// We permute into column-major per-tap storage while reading.
// -----------------------------------------------------------------------------
template <int Channels>
void A2FastModel<Channels>::_load_weights(std::vector<float>& weights)
{
  auto it = weights.begin();
  const auto end = weights.end();

  auto take = [&]() -> float {
    if (it == end)
      throw std::runtime_error("A2FastModel: weight stream exhausted");
    return *it++;
  };

  // Rechannel: 1 -> Channels, no bias. Read order: for i in Channels: for j in 1.
  for (int i = 0; i < Channels; i++)
    _rechannel_w[i] = take();

  for (int li = 0; li < kNumLayers; li++)
  {
    Layer& L = _layers[li];
    const int K = L.kernel_size;

    // Conv1D: Channels -> Bottleneck, kernel=K, bias.
    // Read order: for i in Bottleneck: for j in Channels: for k in K.
    // Store at conv_w[k * C * B + j * B + i] (col-major (B × C) per tap).
    for (int i = 0; i < Channels; i++) // row (out)
    {
      for (int j = 0; j < Channels; j++) // col (in)
      {
        for (int k = 0; k < K; k++)
        {
          L.conv_w[k * Channels * Channels + j * Channels + i] = take();
        }
      }
    }
    for (int i = 0; i < Channels; i++)
      L.conv_b[i] = take();

    // Input mixin: 1 -> Bottleneck, no bias. Read order: for i in Bottleneck: for j in 1.
    for (int i = 0; i < Channels; i++)
      L.mixin_w[i] = take();

    // layer1x1: Bottleneck -> Channels, with bias. Read order: for i in Channels: for j in Bottleneck.
    // Store at l1x1_w[j * Channels + i] (col-major Channels × Bottleneck).
    for (int i = 0; i < Channels; i++) // row (out = Channels)
    {
      for (int j = 0; j < Channels; j++) // col (in = Bottleneck)
      {
        L.l1x1_w[j * Channels + i] = take();
      }
    }
    for (int i = 0; i < Channels; i++)
      L.l1x1_b[i] = take();
  }

  // Head rechannel: Bottleneck -> 1, kernel=16, bias.
  // Read order: for i in 1: for j in Bottleneck: for k in 16.
  // Store at _head_w[k][j] (row=0 since out=1, column-major => just Channels floats per tap).
  for (int j = 0; j < Channels; j++)
  {
    for (int k = 0; k < kHeadKernelSize; k++)
    {
      _head_w[k][j] = take();
    }
  }
  _head_b = take();

  // Matches WaveNet::set_weights_: the last value in the stream is head_scale.
  _head_scale = take();

  if (it != end)
  {
    std::stringstream ss;
    ss << "A2FastModel: weight stream has " << std::distance(it, end) << " trailing bytes";
    throw std::runtime_error(ss.str());
  }
}

// -----------------------------------------------------------------------------
// Buffer sizing
// -----------------------------------------------------------------------------
namespace
{
// Smallest power of 2 >= v (v > 0).
int next_pow2(int v)
{
  int p = 1;
  while (p < v)
    p <<= 1;
  return p;
}
} // namespace

template <int Channels>
void A2FastModel<Channels>::SetMaxBufferSize(int maxBufferSize)
{
  DSP::SetMaxBufferSize(maxBufferSize);

  _layer_in.assign(static_cast<size_t>(Channels) * maxBufferSize, 0.0f);
  _z.assign(static_cast<size_t>(Channels) * maxBufferSize, 0.0f);
  _cond.assign(static_cast<size_t>(maxBufferSize), 0.0f);
  _head_out.assign(static_cast<size_t>(maxBufferSize), 0.0f);

  // First pass: compute per-layer pow2_size / history_cols and per-layer slot
  // size (in floats) so we know how big the arena needs to be.
  std::array<size_t, kNumLayers> slot_sizes{};
  size_t arena_total_floats = 0;
  for (int i = 0; i < kNumLayers; i++)
  {
    Layer& L = _layers[i];
#if NAM_A2_RING_MODE == 1
    L.pow2_size = next_pow2(L.max_lookback + maxBufferSize);
    L.pow2_mask = L.pow2_size - 1;
    slot_sizes[i] = static_cast<size_t>(Channels) * (L.pow2_size + maxBufferSize);
#else
    L.history_cols = 2 * L.max_lookback + maxBufferSize;
    slot_sizes[i] = static_cast<size_t>(Channels) * L.history_cols;
#endif
    L.write_pos = L.max_lookback;
    arena_total_floats += slot_sizes[i];
  }

  // Allocate the arena. We don't add any deliberate stagger between layers:
  // each per-layer slot size is on the order of (pow2_size + maxBufferSize) *
  // Channels floats — large enough that slot_size mod 8 KB is essentially
  // arbitrary and coprime with 8192 in practice. As a result, sequentially-
  // placed layer bases naturally land at 23 distinct mod-8K positions roughly
  // evenly spread across [0, 8192). That's the cache-coloring win we wanted.
  //
  // (Compare to the prior-per-layer std::vector allocations: the heap
  // allocator on Bela was clustering layers at mod-8K = 0x0008..0x0048 and
  // 0x1748..0x1768 — many layers in just two narrow ranges. See the Bela
  // layer-profiling dump 2026-05.)
  _history_arena.assign(arena_total_floats, 0.0f);
  size_t cur_offset = 0;
  for (int i = 0; i < kNumLayers; i++)
  {
    _layers[i].history = _history_arena.data() + cur_offset;
    cur_offset += slot_sizes[i];
  }

  const int head_lookback = kHeadKernelSize - 1;
#if NAM_A2_RING_MODE == 1
  _head_pow2_size = next_pow2(head_lookback + maxBufferSize);
  _head_pow2_mask = _head_pow2_size - 1;
  _head_history.assign(static_cast<size_t>(Channels) * (_head_pow2_size + maxBufferSize), 0.0f);
  _head_write_pos = head_lookback;
#else
  _head_history_cols = 2 * head_lookback + maxBufferSize;
  _head_history.assign(static_cast<size_t>(Channels) * _head_history_cols, 0.0f);
  _head_write_pos = head_lookback;
#endif
}

// -----------------------------------------------------------------------------
// Ring-write helpers.
//   Mode 1: pow2 + tail mirror, mirror-on-demand. The conv loop reads
//   `tap_base + f` unmasked, so the mirror at [pow2_size, pow2_size + N)
//   covers reads that overflow past pow2_size. Predict next-call read
//   ranges per tap from new_wp + dilation; mirror only the bytes actually
//   needed (frequently zero — only the layers + write positions where a
//   tap's read range straddles the wrap). Saves the full mbs * Channels *
//   sizeof(float) memcpy per layer per block when no tap will wrap.
//   Mode 0: linear with periodic memmove rewind. When write_pos nears the
//   end of history, memmove the trailing max_lookback cols back to offset 0
//   and reset write_pos. That memmove is the jitter spike we're measuring.
// -----------------------------------------------------------------------------
template <int Channels>
void A2FastModel<Channels>::_ring_write(Layer& L, int num_frames)
{
#if NAM_A2_RING_MODE == 1
  float* const hist = L.history;
  const float* const src = _layer_in.data();
  const int wp = L.write_pos;
  const int first = std::min(num_frames, L.pow2_size - wp);
  std::memcpy(hist + static_cast<size_t>(wp) * Channels, src,
              static_cast<size_t>(first) * Channels * sizeof(float));
  if (first < num_frames)
  {
    std::memcpy(hist, src + static_cast<size_t>(first) * Channels,
                static_cast<size_t>(num_frames - first) * Channels * sizeof(float));
  }

  const int new_wp = (wp + num_frames) & L.pow2_mask;

  // Mirror-on-demand: predict the next call's read addresses for each tap
  // and find the max overflow past pow2_size. Mirror only that many entries.
  // (Assumes num_frames is stable across calls, which it is on Stratus —
  //  the audio buffer size is fixed at runtime.)
  int mirror_needed = 0;
  const int K = L.kernel_size;
  const int D = L.dilation;
  for (int k = 0; k < K; k++)
  {
    const int taps_back = K - 1 - k;
    const int tap_base = (new_wp - num_frames - taps_back * D) & L.pow2_mask;
    const int read_end = tap_base + num_frames - 1;
    if (read_end >= L.pow2_size)
    {
      const int overflow = read_end - L.pow2_size + 1;
      if (overflow > mirror_needed)
        mirror_needed = overflow;
    }
  }
  if (mirror_needed > 0)
  {
    std::memcpy(hist + static_cast<size_t>(L.pow2_size) * Channels, hist,
                static_cast<size_t>(mirror_needed) * Channels * sizeof(float));
  }

  L.write_pos = new_wp;
#else
  if (L.write_pos + num_frames > L.history_cols)
  {
    const int keep = L.max_lookback;
    std::memmove(L.history, L.history + static_cast<size_t>(L.write_pos - keep) * Channels,
                 static_cast<size_t>(keep) * Channels * sizeof(float));
    L.write_pos = keep;
  }
  std::memcpy(L.history + static_cast<size_t>(L.write_pos) * Channels, _layer_in.data(),
              static_cast<size_t>(num_frames) * Channels * sizeof(float));
  L.write_pos += num_frames;
#endif
}

// -----------------------------------------------------------------------------
// Per-layer forward pass. Reads current _layer_in, writes back into _layer_in
// after applying dilated conv + mixin + LeakyReLU + layer1x1 residual, and
// accumulates activations into _head_sum.
// -----------------------------------------------------------------------------
// Compile-time-specialized per-layer kernel. KernelSize is a template param
// so the K tap loop + per-tap weight offsets become compile-time constants;
// clang fully unrolls and can schedule FMAs across taps. Called from the
// runtime dispatcher below for each A2 kernel size (6 and 15).
template <int Channels>
template <int KernelSize, bool IsFirst>
void A2FastModel<Channels>::_layer_forward_k(Layer& L, const float* cond, int num_frames, int head_wp)
{
  constexpr int K = KernelSize;
  const int D = L.dilation;
  // Physical ring position of this block's first frame, offset by `taps_back *
  // D` samples into the past. In pow2 mode the position is wrapped by mask and
  // reads spanning the wrap land in the tail mirror; in linear mode write_pos
  // is monotonic and arithmetic is plain.
#if NAM_A2_RING_MODE == 1
  const int mask = L.pow2_mask;
  auto tap_base_phys = [&](int taps_back) {
    return (L.write_pos - num_frames - taps_back * D) & mask;
  };
#else
  const int base = L.write_pos - num_frames;
  auto tap_base_phys = [&](int taps_back) { return base - taps_back * D; };
#endif

  // Two conv strategies, dispatched at compile time on Channels:
  //
  //   - Channels <= 4 (A2 nano): full-block tap-major. The z accumulator lives
  //     in the heap buffer across all taps, and for each tap the inner f-loop
  //     iterates over all num_frames. This gives clang frame-level
  //     parallelism — it vectorizes across 4 frames at a time, which matters
  //     more than weight-reload cost when the b-loop (3 wide) can't saturate
  //     NEON lanes on its own.
  //
  //   - Channels >= 8 (A2 standard): frame-tiled tap-major with T=4. ztile
  //     stays in NEON registers across all K taps, amortizing weight loads
  //     over 4 frames — equivalent to what a GEMM kernel does. Weight reuse
  //     matters here because the b-loop (8 wide) already saturates SIMD, so
  //     frame-level parallelism gives no extra headroom. The 1x1 residual is
  //     also tiled over the same T=4 frames so W1x1 loads are amortized.

  if constexpr (Channels == 3)
  {
    // -------------------------------------------------------------------------
    // CRITICAL: this kernel's loop nest (k outer, f inner over all num_frames)
    // is what lets the compiler's loop vectorizer (LV) pack the inner f-loop
    // into NEON Q-form FMAs. DO NOT switch to a frame-tile-outer / k-inner
    // structure to "eliminate _z heap traffic". Two attempts have failed:
    //
    //   1. 2026-05: Fully-unrolled the T=4 tile into 12 named scalars (no
    //      inner loop). Dropped clang/GCC from LV to SLP, which bailed to
    //      VFP scalar code on the 3-channel stride-3 history loads. Cost
    //      +94pt CPU (38% → 132%).
    //   2. 2026-05 (later): kept the inner j-loop with `float a0[T]` arrays
    //      and added `#pragma clang loop unroll(disable) vectorize(enable)`.
    //      Host clang's LV did engage (verified via -Rpass=loop-vectorize
    //      reports on Apple Silicon). But the FIRMWARE TOOLCHAIN IS GCC 10
    //      (per Toolchain.cmake — arm-linux-gnueabihf-gcc-10), and clang
    //      pragmas are silently ignored by GCC. Same +94pt regression as
    //      attempt 1 on the Bela device. Lesson: clang pragmas don't
    //      transfer to gcc; portable forcing requires `#pragma omp simd`
    //      with `-fopenmp-simd`, or hand-NEON intrinsics with the unpadded
    //      layout, or trusting GCC to engage LV on a sufficiently-large
    //      tile (T=8 or larger).
    //
    // The cache-pollution concern around _z is real (per Bela layer-
    // profiling 2026-05: _z mod-8K = 0x1740 overlaps dilation-239 layers'
    // tap-4 reads at mod-8K ≈ 0x1688), but eliminating it via loop-nest
    // swap is fragile across toolchains. If you need to attack _z again,
    // the safest path is cache-coloring _z's allocation address — NOT
    // restructuring the kernel.
    // -------------------------------------------------------------------------
    //
    // Inner 3x3 GEMV fully unrolled: all 9 weights lifted into named consts
    // before the frame loop, the c-reduction kept in scalar temps a0/a1/a2 so
    // the compiler keeps them in FP registers across the frame loop. Mirrors
    // the nam2c --fused structure.
    //
    // NOTE: An NEON 4-frame-tiled kernel was tried and measured ~32% SLOWER
    // on Cortex-A8 than this scalar code. Two reasons:
    //   1. vld3q_f32 / vst3q_f32 on A8 take ~10 cycles each (deinterleaved
    //      load/store is not free). At 3 channels they're issued on every
    //      tile, every tap.
    //   2. NEON Q-form VMLA.F32 has the same 9-cycle latency and 1/cycle
    //      throughput as VFP scalar VFMA. Wider parallelism only wins when
    //      you can pack independent FMAs — the 3x3 GEMV has only 3 parallel
    //      chains of length 3, so NEON doesn't expose more parallelism than
    //      the compiler's auto-unrolled scalar code already exploits.
    // For 8 channels the tradeoff flips (vld1q is cheap, more parallel
    // chains, deeper chains) and the hand-rolled NEON kernel wins.
    //
    // CLEAN PLD DIAGNOSTIC: one __builtin_prefetch per tap entry, placed
    // before the inner frame loop. No in-loop branching, no "next tap"
    // speculation, no multi-line prefetch. If even this minimal pattern
    // doesn't deliver savings, the path is definitively compute-bound and
    // W16A16 quantization (a memory-bandwidth play) won't help either.
    float* z = _z.data();

    // Tap 0: seed z with conv_b (saves the memset-to-zero pass) and fold in
    // the first tap's FMAs.
    {
      const float* wk = &L.conv_w[0];
      const int tap_base = tap_base_phys(K - 1);
      __builtin_prefetch(&L.history[static_cast<size_t>(tap_base) * 3], 0, 3);
      const float w0 = wk[0], w1 = wk[1], w2 = wk[2];
      const float w3 = wk[3], w4 = wk[4], w5 = wk[5];
      const float w6 = wk[6], w7 = wk[7], w8 = wk[8];
      const float cb0 = L.conv_b[0], cb1 = L.conv_b[1], cb2 = L.conv_b[2];
      for (int f = 0; f < num_frames; f++)
      {
        const float* src = &L.history[static_cast<size_t>(tap_base + f) * 3];
        float a0 = cb0 + w0 * src[0];
        float a1 = cb1 + w1 * src[0];
        float a2 = cb2 + w2 * src[0];
        a0 += w3 * src[1];
        a1 += w4 * src[1];
        a2 += w5 * src[1];
        a0 += w6 * src[2];
        a1 += w7 * src[2];
        a2 += w8 * src[2];
        float* zf = z + static_cast<size_t>(f) * 3;
        zf[0] = a0;
        zf[1] = a1;
        zf[2] = a2;
      }
    }

    // Taps 1..K-2: accumulate into z with the same unrolled inner kernel.
    for (int k = 1; k < K - 1; k++)
    {
      const float* wk = &L.conv_w[static_cast<size_t>(k) * 9];
      const int tap_base = tap_base_phys(K - 1 - k);
      __builtin_prefetch(&L.history[static_cast<size_t>(tap_base) * 3], 0, 3);
      const float w0 = wk[0], w1 = wk[1], w2 = wk[2];
      const float w3 = wk[3], w4 = wk[4], w5 = wk[5];
      const float w6 = wk[6], w7 = wk[7], w8 = wk[8];
      for (int f = 0; f < num_frames; f++)
      {
        const float* src = &L.history[static_cast<size_t>(tap_base + f) * 3];
        float* zf = z + static_cast<size_t>(f) * 3;
        float a0 = zf[0] + w0 * src[0];
        float a1 = zf[1] + w1 * src[0];
        float a2 = zf[2] + w2 * src[0];
        a0 += w3 * src[1];
        a1 += w4 * src[1];
        a2 += w5 * src[1];
        a0 += w6 * src[2];
        a1 += w7 * src[2];
        a2 += w8 * src[2];
        zf[0] = a0;
        zf[1] = a1;
        zf[2] = a2;
      }
    }

    // Final tap (K-1, offset 0) fully inlined with the post-conv tail.
    // Everything runs on register-resident scalars:
    //   conv tap K-1 -> mixin -> LeakyReLU -> head_sum += -> layer1x1 residual.
    const float* wk_last = &L.conv_w[static_cast<size_t>(K - 1) * 9];
    const int tap_base_last = tap_base_phys(0);
    __builtin_prefetch(&L.history[static_cast<size_t>(tap_base_last) * 3], 0, 3);
    const float cw0 = wk_last[0], cw1 = wk_last[1], cw2 = wk_last[2];
    const float cw3 = wk_last[3], cw4 = wk_last[4], cw5 = wk_last[5];
    const float cw6 = wk_last[6], cw7 = wk_last[7], cw8 = wk_last[8];
    const float mw0 = L.mixin_w[0], mw1 = L.mixin_w[1], mw2 = L.mixin_w[2];
    // layer1x1 col-major: lw[b*3 + c] is weight from bottleneck b to output c.
    const float lw00 = L.l1x1_w[0], lw01 = L.l1x1_w[1], lw02 = L.l1x1_w[2];
    const float lw10 = L.l1x1_w[3], lw11 = L.l1x1_w[4], lw12 = L.l1x1_w[5];
    const float lw20 = L.l1x1_w[6], lw21 = L.l1x1_w[7], lw22 = L.l1x1_w[8];
    const float lb0 = L.l1x1_b[0], lb1 = L.l1x1_b[1], lb2 = L.l1x1_b[2];
    for (int f = 0; f < num_frames; f++)
    {
      const float* src = &L.history[static_cast<size_t>(tap_base_last + f) * 3];
      const float* zf_mem = z + static_cast<size_t>(f) * 3;
      // Final tap GEMV.
      float a0 = zf_mem[0] + cw0 * src[0];
      float a1 = zf_mem[1] + cw1 * src[0];
      float a2 = zf_mem[2] + cw2 * src[0];
      a0 += cw3 * src[1];
      a1 += cw4 * src[1];
      a2 += cw5 * src[1];
      a0 += cw6 * src[2];
      a1 += cw7 * src[2];
      a2 += cw8 * src[2];
      // Mixin + LeakyReLU.
      const float cf = cond[f];
      a0 += mw0 * cf;
      a1 += mw1 * cf;
      a2 += mw2 * cf;
      a0 = (a0 >= 0.0f) ? a0 : a0 * kLeakySlope;
      a1 = (a1 >= 0.0f) ? a1 : a1 * kLeakySlope;
      a2 = (a2 >= 0.0f) ? a2 : a2 * kLeakySlope;
      // Head accumulator: write straight into _head_history at the current
      // ring position (head_wp + f). Assign on the first layer (overwriting
      // whatever stale data was at that slot), accumulate on subsequent
      // layers. if constexpr resolves at compile time — no runtime branch.
      // process() guarantees head_wp + num_frames stays within the writable
      // contiguous range (rewinds when needed).
      float* hsum = &_head_history[static_cast<size_t>(head_wp + f) * 3];
      if constexpr (IsFirst) {
        hsum[0] = a0;
        hsum[1] = a1;
        hsum[2] = a2;
      } else {
        hsum[0] += a0;
        hsum[1] += a1;
        hsum[2] += a2;
      }
      // layer1x1 residual.
      float* lin = &_layer_in[static_cast<size_t>(f) * 3];
      lin[0] += lb0 + lw00 * a0 + lw10 * a1 + lw20 * a2;
      lin[1] += lb1 + lw01 * a0 + lw11 * a1 + lw21 * a2;
      lin[2] += lb2 + lw02 * a0 + lw12 * a1 + lw22 * a2;
    }
  }
  else
  {
    // =========================================================================
    // !!! KNOWN REGRESSION — 8-CHANNEL PATH IS CURRENTLY SLOWER THAN ORIGINAL !!!
    // =========================================================================
    // Status (2026-05): Channels=8 measures ~517% CPU on Cortex-A8 1 GHz at
    // -p 128. Before the hand-rolled NEON 8x4 microkernel was added below,
    // this same path used Eigen's `ztile.noalias() = W * input_block` GEMM
    // and ran at ~249% CPU.
    //
    // FIX PATH when 8-channel becomes load-bearing:
    //   1. Delete the `#if defined(__ARM_NEON__) ... #else ... #endif` block
    //      below. Keep ONLY the Eigen `for (int k = 0; k < K; k++) {...}`
    //      loop currently in the #else branch — that's the original path.
    //   2. Re-benchmark. Expect ~249% CPU (still not real-time on A8).
    //   3. Real fix: smaller A2 Standard variant or faster silicon (A53 /
    //      A55 / A72). 8ch on 1GHz A8 will never be real-time at -p 128.
    //
    // KEEP THESE — they help 8ch too:
    //   - mirror-on-demand pow2 ring (_ring_write)
    //   - ztile zero-fill drop (tap 0 seeds via vdupq_n_f32(0))
    //   - _head_sum elimination (writes go straight to _head_history)
    //
    // DO NOT TRY:
    //   - PLD/__builtin_prefetch attacks (cache-pollution-bound, made worse).
    //   - NAM_CHUNK_SIZE < 128 (27% regression at 64).
    //   - 4-channel-padded NEON refactor (+9pt on 3-channel; padding bloated
    //     working set 33%).
    //   - Loop-nest swap (k-outer/f-inner → f-tile-outer/k-inner) to drop _z.
    //     Both fully-unrolled and pragma-driven attempts cost +94pt because
    //     GCC silently ignored clang pragmas and SLP can't replace LV.
    //
    // See NAM_BENCHMARK.md "Known regression: 8-channel A2 Standard kernel".
    // =========================================================================
    //
    // Use Eigen's tuned 8x8 × 8xN GEMM for the whole block at once. Unlike a
    // small-tile version, this hits Eigen's actual GEMM kernel (tuned for
    // inner dimensions of ~64) rather than its tiny-matrix fallback path.
    //
    // Compile-time improvements over the generic WaveNet path:
    //   - Channels and Bottleneck are template constants (no dynamic shape).
    //   - Per-layer buffers are pre-sized at SetMaxBufferSize; nothing resizes
    //     during process().
    //   - No FiLM / gating / head1x1 / grouped-conv branches.
    //   - No virtual dispatch / conditional on optional layer features.
    //   - All conv + post-conv ops operate on the full block — even the
    //     mixin, bias, activation, and 1x1 residual are Eigen block ops so
    //     they vectorize the same way the GEMMs do.
    using MatCC = Eigen::Matrix<float, Channels, Channels>;
    using MatCDyn = Eigen::Matrix<float, Channels, Eigen::Dynamic>;
    using VecC = Eigen::Matrix<float, Channels, 1>;
    using RowDyn = Eigen::Matrix<float, 1, Eigen::Dynamic>;

    Eigen::Map<const VecC> conv_b_vec(L.conv_b.data());
    Eigen::Map<const VecC> mixin_vec(L.mixin_w.data());
    Eigen::Map<const MatCC> l1x1_mat(L.l1x1_w.data());
    Eigen::Map<const VecC> l1x1_b_vec(L.l1x1_b.data());
    Eigen::Map<const RowDyn> cond_row(cond, 1, num_frames);

    Eigen::Map<MatCDyn> ztile(_z.data(), Channels, num_frames);
    // Map directly into _head_history at the current ring position. process()
    // guarantees head_wp + num_frames stays within the writable contiguous
    // range so this map cannot overflow.
    Eigen::Map<MatCDyn> hsum_block(&_head_history[static_cast<size_t>(head_wp) * Channels], Channels, num_frames);
    Eigen::Map<MatCDyn> lin_block(_layer_in.data(), Channels, num_frames);

    // No ztile.setZero() — tap 0 below initializes ztile via vdupq_n_f32(0)
    // accumulator seeding. Skips Channels * num_frames * 4 bytes of writes
    // per layer per buffer (~94 KB at K=8 / mbs=128 / 23 layers) that would
    // otherwise pollute L1 cache lines aliased to conv working data.

#if defined(__ARM_NEON__)
    // -----------------------------------------------------------------------
    // Conv: hand-rolled NEON 8x4 microkernel.
    //
    // For each of K taps, accumulate ztile[c, f] += W[c, c'] * H[c', f]
    // (column-major, c' = input channel, c = output channel). Tile by T=4
    // frames so 8 output channels × 4 frames live in 8 NEON Q-registers
    // across the c' inner loop. For each c' in [0,8): load W[:,c'] once
    // (2 Q-regs reused across all 4 frames), then 4× vmlaq_n_f32 broadcasts
    // of H[c', f]. Inner loop has no register-level dependency between the
    // 8 accumulators, so the in-order Cortex-A8 NEON pipeline can issue an
    // FMA every cycle.
    //
    // Tap 0 seeds ztile (accumulators init to zero, no load from _z and no
    // upfront setZero); taps 1..K-1 load+accumulate as usual.
    //
    // 64 fp32 FMAs per tile × num_frames/4 tiles per tap × K taps per layer.
    // Replaces Eigen's 8x8 × 8xN GEMM, which dispatches to a small-matrix
    // path that hadn't kept its accumulators register-resident across taps.
    // -----------------------------------------------------------------------
    {
      float* z = _z.data();
      const float* hist_base = L.history;
      const float* W_all = L.conv_w.data();
      const int neonF = (num_frames / 4) * 4;

      for (int k = 0; k < K; k++)
      {
        const int tap_base = tap_base_phys(K - 1 - k);
        const float* W = W_all + static_cast<size_t>(k) * Channels * Channels;
        const float* hist = hist_base + static_cast<size_t>(tap_base) * Channels;
        const bool seed = (k == 0);  // tap 0 seeds ztile, taps >=1 accumulate

        int f = 0;
        for (; f < neonF; f += 4)
        {
          float* z_base = z + static_cast<size_t>(f) * Channels;
          const float* h_base = hist + static_cast<size_t>(f) * Channels;

          // Initialize accumulators: zero on tap 0 (seed), running sum from
          // _z on taps >=1. The branch is loop-invariant within the K loop
          // and predicted-not-taken K-1 of K times.
          float32x4_t a0_lo, a0_hi, a1_lo, a1_hi, a2_lo, a2_hi, a3_lo, a3_hi;
          if (seed)
          {
            const float32x4_t zero = vdupq_n_f32(0.0f);
            a0_lo = a0_hi = a1_lo = a1_hi = a2_lo = a2_hi = a3_lo = a3_hi = zero;
          }
          else
          {
            a0_lo = vld1q_f32(z_base + 0 * Channels + 0);
            a0_hi = vld1q_f32(z_base + 0 * Channels + 4);
            a1_lo = vld1q_f32(z_base + 1 * Channels + 0);
            a1_hi = vld1q_f32(z_base + 1 * Channels + 4);
            a2_lo = vld1q_f32(z_base + 2 * Channels + 0);
            a2_hi = vld1q_f32(z_base + 2 * Channels + 4);
            a3_lo = vld1q_f32(z_base + 3 * Channels + 0);
            a3_hi = vld1q_f32(z_base + 3 * Channels + 4);
          }

          // Accumulate W[:,c'] * h[c',f] for each input channel c'.
          for (int cp = 0; cp < Channels; cp++)
          {
            float32x4_t w_lo = vld1q_f32(W + cp * Channels + 0);
            float32x4_t w_hi = vld1q_f32(W + cp * Channels + 4);

            const float i0 = h_base[0 * Channels + cp];
            const float i1 = h_base[1 * Channels + cp];
            const float i2 = h_base[2 * Channels + cp];
            const float i3 = h_base[3 * Channels + cp];

            a0_lo = vmlaq_n_f32(a0_lo, w_lo, i0);
            a0_hi = vmlaq_n_f32(a0_hi, w_hi, i0);
            a1_lo = vmlaq_n_f32(a1_lo, w_lo, i1);
            a1_hi = vmlaq_n_f32(a1_hi, w_hi, i1);
            a2_lo = vmlaq_n_f32(a2_lo, w_lo, i2);
            a2_hi = vmlaq_n_f32(a2_hi, w_hi, i2);
            a3_lo = vmlaq_n_f32(a3_lo, w_lo, i3);
            a3_hi = vmlaq_n_f32(a3_hi, w_hi, i3);
          }

          vst1q_f32(z_base + 0 * Channels + 0, a0_lo);
          vst1q_f32(z_base + 0 * Channels + 4, a0_hi);
          vst1q_f32(z_base + 1 * Channels + 0, a1_lo);
          vst1q_f32(z_base + 1 * Channels + 4, a1_hi);
          vst1q_f32(z_base + 2 * Channels + 0, a2_lo);
          vst1q_f32(z_base + 2 * Channels + 4, a2_hi);
          vst1q_f32(z_base + 3 * Channels + 0, a3_lo);
          vst1q_f32(z_base + 3 * Channels + 4, a3_hi);
        }

        // Scalar tail for any frames past the multiple-of-4 boundary.
        for (; f < num_frames; f++)
        {
          float* z_col = z + static_cast<size_t>(f) * Channels;
          const float* h_col = hist + static_cast<size_t>(f) * Channels;
          for (int o = 0; o < Channels; o++)
          {
            // Tap 0 seeds (sum=0); subsequent taps accumulate on top.
            float sum = seed ? 0.0f : z_col[o];
            for (int cp = 0; cp < Channels; cp++)
              sum += W[cp * Channels + o] * h_col[cp];
            z_col[o] = sum;
          }
        }
      }
    }
#else
    // Eigen fallback for non-ARM builds (host-side dev / tests). Tap 0
    // assigns into ztile (replaces the dropped setZero); taps >=1 accumulate.
    for (int k = 0; k < K; k++)
    {
      const int tap_base = tap_base_phys(K - 1 - k);
      Eigen::Map<const MatCC> W(&L.conv_w[static_cast<size_t>(k) * Channels * Channels]);
      Eigen::Map<const MatCDyn> input_block(&L.history[static_cast<size_t>(tap_base) * Channels], Channels, num_frames);
      if (k == 0)
        ztile.noalias() = W * input_block;
      else
        ztile.noalias() += W * input_block;
    }
#endif

    // Post-conv: bias, mixin, LeakyReLU, head_sum, 1x1 residual — all block ops.
    ztile.colwise() += conv_b_vec;
    ztile.noalias() += mixin_vec * cond_row;                               // rank-1 outer product
    ztile = (ztile.array() < 0.0f).select(ztile.array() * kLeakySlope, ztile.array());
    // First-layer assigns into _head_history at the current ring window
    // (replaces stale data); subsequent layers accumulate. Compile-time branch.
    if constexpr (IsFirst)
      hsum_block = ztile;
    else
      hsum_block += ztile;
    lin_block.noalias() += l1x1_mat * ztile;                               // 8x8 × 8xN GEMM
    lin_block.colwise() += l1x1_b_vec;
  }
}

// Runtime dispatcher: selects the K-specialized kernel for this layer.
// For the A2 shape the detector only admits K in {6, 15}; any other value
// here means something passed the detector that shouldn't have.
//
// is_first selects the layer-0-only code path that initializes the recent
// _head_history slots via assignment (rather than accumulating into them).
// Branched at runtime here, but inside the kernel the choice is a compile-
// time constant.
//
// head_wp is the snapshot of _head_write_pos at process() entry; layers
// write the head accumulator directly into _head_history[head_wp + f].
template <int Channels>
void A2FastModel<Channels>::_layer_forward(int layer_idx, const float* cond, int num_frames, bool is_first, int head_wp)
{
  Layer& L = _layers[layer_idx];
  _ring_write(L, num_frames);
  switch (L.kernel_size)
  {
    case 6:
      if (is_first) _layer_forward_k<6, true>(L, cond, num_frames, head_wp);
      else          _layer_forward_k<6, false>(L, cond, num_frames, head_wp);
      break;
    case 15:
      if (is_first) _layer_forward_k<15, true>(L, cond, num_frames, head_wp);
      else          _layer_forward_k<15, false>(L, cond, num_frames, head_wp);
      break;
    default:
      throw std::runtime_error("A2FastModel: unexpected kernel_size "
                               + std::to_string(L.kernel_size));
  }
}

// -----------------------------------------------------------------------------
// Head: K=16 dilation-1 conv from Channels to 1, plus bias + scale.
//
// Caller (process()) must have advanced _head_write_pos to (head_wp +
// num_frames) before invoking this function — that's the "post-write"
// position the col_of() lambda assumes. Layers have already written this
// buffer's head accumulators directly into _head_history[head_wp .. head_wp +
// num_frames - 1].
// -----------------------------------------------------------------------------
template <int Channels>
void A2FastModel<Channels>::_head_forward(float* output, int num_frames)
{
#if NAM_A2_RING_MODE == 1
  const int mask = _head_pow2_mask;
  auto col_of = [&](int f, int k) {
    return (_head_write_pos - num_frames + f - (kHeadKernelSize - 1 - k)) & mask;
  };
#else
  const int base = _head_write_pos - num_frames;
  auto col_of = [&](int f, int k) { return base + f - (kHeadKernelSize - 1 - k); };
#endif

#if defined(__ARM_NEON__)
  if constexpr (Channels == 3)
  {
    // 4-frame-tiled NEON kernel for the 3-channel path. process()'s rewind
    // guarantees head_wp + num_frames ≤ head_pow2_size, so cols within a
    // buffer are always contiguous in _head_history — no ring wrap mid-buffer.
    // That lets us compute base_col once per tile (via col_of(f, 0)) and walk
    // linearly through the 16 taps without re-masking. vld3q deinterleaves 4
    // consecutive 3-channel cols into 3 frame-vectors (one per channel),
    // matching the inner reduction's natural axis: y[f] += sum_b W[k][b] *
    // H[col_of(f,k)][b].
    //
    // Per tile: 16 vld3q + 48 vmlaq_n + 1 vmulq_n + 1 vst1q. vld3q.32 on
    // Cortex-A8 takes ~10 cycles each (LSU-bound) but pipelines behind the
    // FMA chain. Vs the scalar path's 192 FMAs/tile at VFP scalar throughput,
    // expected speedup ~3-3.5×.
    const float scale_const = _head_scale;
    const float bias = _head_b;
    const float32x4_t bias_vec = vdupq_n_f32(bias);
    const int neonF = (num_frames / 4) * 4;
    int f = 0;
    for (; f < neonF; f += 4)
    {
      float32x4_t y = bias_vec;
      const int base_col = col_of(f, 0);
      for (int k = 0; k < kHeadKernelSize; k++)
      {
        const float* src = &_head_history[static_cast<size_t>(base_col + k) * Channels];
        const float32x4x3_t H = vld3q_f32(src);
        const float w0 = _head_w[k][0];
        const float w1 = _head_w[k][1];
        const float w2 = _head_w[k][2];
        y = vmlaq_n_f32(y, H.val[0], w0);
        y = vmlaq_n_f32(y, H.val[1], w1);
        y = vmlaq_n_f32(y, H.val[2], w2);
      }
      y = vmulq_n_f32(y, scale_const);
      vst1q_f32(output + f, y);
    }
    // Scalar tail for any frames past the multiple-of-4 boundary.
    for (; f < num_frames; f++)
    {
      float y = bias;
      for (int k = 0; k < kHeadKernelSize; k++)
      {
        const int col = col_of(f, k);
        const float* src = &_head_history[static_cast<size_t>(col) * Channels];
        const float* wk = _head_w[k].data();
        for (int b = 0; b < Channels; b++)
          y += wk[b] * src[b];
      }
      output[f] = y * scale_const;
    }
    return;
  }
#endif

  // Scalar path: 8-channel and any non-NEON build.
  for (int f = 0; f < num_frames; f++)
  {
    float y = _head_b;
    for (int k = 0; k < kHeadKernelSize; k++)
    {
      const int col = col_of(f, k);
      const float* src = &_head_history[static_cast<size_t>(col) * Channels];
      const float* wk = _head_w[k].data();
      for (int b = 0; b < Channels; b++)
        y += wk[b] * src[b];
    }
    output[f] = y * _head_scale;
  }
}

#ifdef STRATUS_NAM_LAYER_PROFILING
// -----------------------------------------------------------------------------
// Per-layer profiling output (compiled in only with STRATUS_NAM_LAYER_PROFILING).
// _dump_layer_stats prints min/mean/max per layer plus an aggregate summary;
// _log_history_addresses prints each layer's history-buffer base address mod
// 8 KB so we can spot Cortex-A8 L1 set-aliasing collisions.
// -----------------------------------------------------------------------------
template <int Channels>
void A2FastModel<Channels>::_dump_layer_stats()
{
  std::ostringstream ss;
  ss << "[NAM A2 layer-profiling] per-layer timing (ns), Channels=" << Channels << ":\n";
  ss << "  idx   K   D    calls       min      mean       max  share%\n";
  double total_ns_all = 0.0;
  for (const auto& s : _layer_stats)
    total_ns_all += s.total_ns;
  for (int i = 0; i < kNumLayers; i++)
  {
    const auto& s = _layer_stats[i];
    if (s.calls == 0) continue;
    const double mean_ns = s.total_ns / static_cast<double>(s.calls);
    const double share = (total_ns_all > 0.0) ? 100.0 * s.total_ns / total_ns_all : 0.0;
    ss << "  " << std::setw(3) << i
       << "  " << std::setw(2) << _layers[i].kernel_size
       << "  " << std::setw(3) << _layers[i].dilation
       << "  " << std::setw(7) << s.calls
       << "  " << std::setw(8) << static_cast<long long>(s.min_ns)
       << "  " << std::setw(8) << static_cast<long long>(mean_ns)
       << "  " << std::setw(8) << static_cast<long long>(s.max_ns)
       << "   " << std::fixed << std::setprecision(1) << std::setw(5) << share
       << std::defaultfloat
       << "\n";
  }
  if (_layer_stats_buffer_count > 0)
  {
    const double total_per_buf = total_ns_all / static_cast<double>(_layer_stats_buffer_count);
    ss << "  TOTAL per-buffer mean across " << _layer_stats_buffer_count
       << " buffers: " << static_cast<long long>(total_per_buf) << " ns\n";
  }
  std::cerr << ss.str();
}

template <int Channels>
void A2FastModel<Channels>::_log_history_addresses()
{
  std::ostringstream ss;
  ss << "[NAM A2 layer-profiling] history base addresses (Channels=" << Channels << "):\n";
  ss << "  idx   K   D     pow2    base addr  base mod 8K\n";
  for (int i = 0; i < kNumLayers; i++)
  {
    const auto& L = _layers[i];
    const uintptr_t addr = reinterpret_cast<uintptr_t>(L.history);
#if NAM_A2_RING_MODE == 1
    const int pow2 = L.pow2_size;
#else
    const int pow2 = L.history_cols;
#endif
    ss << "  " << std::setw(3) << i
       << "  " << std::setw(2) << L.kernel_size
       << "  " << std::setw(3) << L.dilation
       << "  " << std::setw(7) << pow2
       << "  0x" << std::hex << std::setw(8) << std::setfill('0') << addr
       << "       0x" << std::setw(4) << (addr & 0x1FFFu)
       << std::dec << std::setfill(' ')
       << "\n";
  }
  const uintptr_t z_addr   = reinterpret_cast<uintptr_t>(_z.data());
  const uintptr_t lin_addr = reinterpret_cast<uintptr_t>(_layer_in.data());
  const uintptr_t hh_addr  = reinterpret_cast<uintptr_t>(_head_history.data());
  ss << "  scratch:\n"
     << "    _z            0x" << std::hex << std::setw(8) << std::setfill('0') << z_addr
     << "       0x" << std::setw(4) << (z_addr & 0x1FFFu)
     << std::dec << std::setfill(' ') << "\n"
     << "    _layer_in     0x" << std::hex << std::setw(8) << std::setfill('0') << lin_addr
     << "       0x" << std::setw(4) << (lin_addr & 0x1FFFu)
     << std::dec << std::setfill(' ') << "\n"
     << "    _head_history 0x" << std::hex << std::setw(8) << std::setfill('0') << hh_addr
     << "       0x" << std::setw(4) << (hh_addr & 0x1FFFu)
     << std::dec << std::setfill(' ') << "\n";
  std::cerr << ss.str();
}
#endif

// -----------------------------------------------------------------------------
// DSP::process override
// -----------------------------------------------------------------------------
template <int Channels>
void A2FastModel<Channels>::process(NAM_SAMPLE** input, NAM_SAMPLE** output, int num_frames)
{
  if (num_frames > GetMaxBufferSize())
    SetMaxBufferSize(num_frames);

  const NAM_SAMPLE* in0 = input[0];
  NAM_SAMPLE* out0 = output[0];

  // Rechannel: layer_in[c, f] = _rechannel_w[c] * input[f] for c in Channels.
  // Also prepare float cond buffer (input copied to float for inner loops).
  float* cond = _cond.data();
  for (int f = 0; f < num_frames; f++)
  {
    const float x = static_cast<float>(in0[f]);
    cond[f] = x;
    float* lin = &_layer_in[static_cast<size_t>(f) * Channels];
    for (int c = 0; c < Channels; c++)
      lin[c] = _rechannel_w[c] * x;
  }

  // Head accumulator: written directly into _head_history (no separate
  // _head_sum buffer + memcpy). Snapshot the current write position and
  // rewind the ring if writing num_frames more would overflow the writable
  // contiguous range. The kept frames preserve the head's K=16 look-back
  // window. _head_forward()'s col_of() lambda already masks every read, so
  // reads continue to work correctly whether or not the rewind fired.
  const int head_keep = kHeadKernelSize - 1;
#if NAM_A2_RING_MODE == 1
  const int head_capacity = _head_pow2_size;
#else
  const int head_capacity = _head_history_cols;
#endif
  if (_head_write_pos + num_frames > head_capacity)
  {
    std::memmove(_head_history.data(),
                 _head_history.data() + static_cast<size_t>(_head_write_pos - head_keep) * Channels,
                 static_cast<size_t>(head_keep) * Channels * sizeof(float));
    _head_write_pos = head_keep;
  }
  const int head_wp = _head_write_pos;

#ifdef STRATUS_NAM_LAYER_PROFILING
  if (!_history_addrs_logged)
  {
    _log_history_addresses();
    _history_addrs_logged = true;
  }

  for (int li = 0; li < kNumLayers; li++)
  {
    timespec t0{}, t1{};
    clock_gettime(CLOCK_MONOTONIC_RAW, &t0);
    _layer_forward(li, cond, num_frames, /*is_first=*/(li == 0), head_wp);
    clock_gettime(CLOCK_MONOTONIC_RAW, &t1);
    const double ns = static_cast<double>(t1.tv_sec - t0.tv_sec) * 1e9
                    + static_cast<double>(t1.tv_nsec - t0.tv_nsec);
    auto& s = _layer_stats[li];
    s.total_ns += ns;
    s.calls++;
    if (ns < s.min_ns) s.min_ns = ns;
    if (ns > s.max_ns) s.max_ns = ns;
  }

  if (++_layer_stats_buffer_count >= kLayerStatsDumpInterval)
  {
    _dump_layer_stats();
    for (auto& s : _layer_stats) s = LayerStats{};
    _layer_stats_buffer_count = 0;
  }
#else
  _layer_forward(0, cond, num_frames, /*is_first=*/true, head_wp);
  for (int li = 1; li < kNumLayers; li++)
    _layer_forward(li, cond, num_frames, /*is_first=*/false, head_wp);
#endif

  // Advance _head_write_pos past this buffer's writes. _head_forward's
  // col_of() lambda assumes _head_write_pos is the post-write position.
#if NAM_A2_RING_MODE == 1
  _head_write_pos = (head_wp + num_frames) & _head_pow2_mask;
#else
  _head_write_pos = head_wp + num_frames;
#endif

  // Output.
  float* head_out = _head_out.data();
  _head_forward(head_out, num_frames);
  for (int f = 0; f < num_frames; f++)
    out0[f] = static_cast<NAM_SAMPLE>(head_out[f]);
}

// -----------------------------------------------------------------------------
// A2FastConfig — wraps the constructed DSP behind the ModelConfig interface.
// -----------------------------------------------------------------------------
struct A2FastConfig : public ModelConfig
{
  int channels = 0;

  std::unique_ptr<DSP> create(std::vector<float> weights, double sampleRate) override
  {
    if (channels == 3)
      return std::make_unique<A2FastModel<3>>(std::move(weights), sampleRate);
    if (channels == 8)
      return std::make_unique<A2FastModel<8>>(std::move(weights), sampleRate);
    throw std::runtime_error("A2FastConfig: unsupported channel count " + std::to_string(channels));
  }
};

// -----------------------------------------------------------------------------
// Detector helpers
// -----------------------------------------------------------------------------
bool close_to(float v, float target)
{
  return std::fabs(v - target) <= 1e-7f;
}

bool all_none_strings(const nlohmann::json& j)
{
  if (!j.is_array())
    return false;
  for (const auto& e : j)
  {
    if (!e.is_string() || e.get<std::string>() != "none")
      return false;
  }
  return true;
}

bool all_null(const nlohmann::json& j)
{
  if (!j.is_array())
    return false;
  for (const auto& e : j)
  {
    if (!e.is_null())
      return false;
  }
  return true;
}

bool film_inactive(const nlohmann::json& layer, const char* key)
{
  auto it = layer.find(key);
  if (it == layer.end() || it->is_null())
    return true;
  if (it->is_boolean())
    return !it->get<bool>();
  if (it->is_object())
    return !it->value("active", false);
  return false;
}

} // namespace

// -----------------------------------------------------------------------------
// Public API
// -----------------------------------------------------------------------------
bool is_a2_shape(const nlohmann::json& config, int* channels)
{
  // Exactly one layer array
  auto layers_it = config.find("layers");
  if (layers_it == config.end() || !layers_it->is_array() || layers_it->size() != 1)
    return false;

  // No post-stack head
  auto head_it = config.find("head");
  if (head_it != config.end() && !head_it->is_null())
    return false;

  // head_scale must be exactly 0.01
  auto hs_it = config.find("head_scale");
  if (hs_it == config.end() || !hs_it->is_number())
    return false;
  if (!close_to(hs_it->get<float>(), kHeadScale))
    return false;

  // in_channels defaults to 1, must be 1
  if (config.value("in_channels", 1) != 1)
    return false;

  const auto& la = (*layers_it)[0];

  if (la.value("input_size", 0) != 1)
    return false;
  if (la.value("condition_size", 0) != 1)
    return false;

  const int ch = la.value("channels", 0);
  const int bn = la.value("bottleneck", 0);
  if (ch != bn)
    return false;
  if (ch != 3 && ch != 8)
    return false;

  // kernel_sizes must match kKernelSizes exactly
  auto ks_it = la.find("kernel_sizes");
  if (ks_it == la.end() || !ks_it->is_array() || ks_it->size() != kNumLayers)
    return false;
  for (int i = 0; i < kNumLayers; i++)
  {
    if (!(*ks_it)[i].is_number_integer() || (*ks_it)[i].get<int>() != kKernelSizes[i])
      return false;
  }

  // dilations must match kDilations exactly
  auto dl_it = la.find("dilations");
  if (dl_it == la.end() || !dl_it->is_array() || dl_it->size() != kNumLayers)
    return false;
  for (int i = 0; i < kNumLayers; i++)
  {
    if (!(*dl_it)[i].is_number_integer() || (*dl_it)[i].get<int>() != kDilations[i])
      return false;
  }

  // activation: all LeakyReLU(0.01)
  auto act_it = la.find("activation");
  if (act_it == la.end() || !act_it->is_array() || act_it->size() != kNumLayers)
    return false;
  for (const auto& a : *act_it)
  {
    if (!a.is_object() || a.value("type", std::string()) != "LeakyReLU")
      return false;
    if (!close_to(a.value("negative_slope", 0.0f), kLeakySlope))
      return false;
  }

  // gating_mode: all "none" (or field absent)
  auto gm_it = la.find("gating_mode");
  if (gm_it != la.end() && !gm_it->is_null())
  {
    if (!all_none_strings(*gm_it) || gm_it->size() != kNumLayers)
      return false;
  }

  // secondary_activation: all null (or field absent)
  auto sa_it = la.find("secondary_activation");
  if (sa_it != la.end() && !sa_it->is_null())
  {
    if (!all_null(*sa_it) || sa_it->size() != kNumLayers)
      return false;
  }

  // head1x1 inactive
  auto h1x1_it = la.find("head1x1");
  if (h1x1_it != la.end() && h1x1_it->is_object() && h1x1_it->value("active", false))
    return false;

  // layer1x1 active with groups=1
  auto l1x1_it = la.find("layer1x1");
  if (l1x1_it == la.end() || !l1x1_it->is_object())
    return false;
  if (!l1x1_it->value("active", false))
    return false;
  if (l1x1_it->value("groups", 1) != 1)
    return false;

  // Layer-array head rechannel: k=16, out_channels=1, bias=true
  auto lah_it = la.find("head");
  if (lah_it == la.end() || !lah_it->is_object())
    return false;
  if (lah_it->value("out_channels", 0) != 1)
    return false;
  if (lah_it->value("kernel_size", 0) != kHeadKernelSize)
    return false;
  if (!lah_it->value("bias", false))
    return false;

  // No FiLM anywhere
  for (const char* key :
       {"conv_pre_film", "conv_post_film", "input_mixin_pre_film", "input_mixin_post_film", "activation_pre_film",
        "activation_post_film", "layer1x1_post_film", "head1x1_post_film"})
  {
    if (!film_inactive(la, key))
      return false;
  }

  // No grouped convolutions
  if (la.value("groups_input", 1) != 1)
    return false;
  if (la.value("groups_input_mixin", 1) != 1)
    return false;

  // Not slimmable
  auto slim_it = la.find("slimmable");
  if (slim_it != la.end() && !slim_it->is_null())
    return false;

  if (channels)
    *channels = ch;
  return true;
}

std::unique_ptr<ModelConfig> create_a2_fast_config(const nlohmann::json& config, double sampleRate)
{
  (void)sampleRate;
  int ch = 0;
  if (!is_a2_shape(config, &ch))
    throw std::runtime_error("create_a2_fast_config: config does not match A2 shape");
  std::cerr << "[NAM] A2 fast-path: creating A2FastModel with channels=" << ch << std::endl;
  auto out = std::make_unique<A2FastConfig>();
  out->channels = ch;
  return out;
}

std::unique_ptr<DSP> create_a2_fast(int channels, std::vector<float> weights, double sampleRate)
{
  if (channels == 3)
    return std::make_unique<A2FastModel<3>>(std::move(weights), sampleRate);
  if (channels == 8)
    return std::make_unique<A2FastModel<8>>(std::move(weights), sampleRate);
  return nullptr;
}

} // namespace a2_fast
} // namespace wavenet
} // namespace nam

#endif // NAM_ENABLE_A2_FAST
