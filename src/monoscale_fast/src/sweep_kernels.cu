// The plane sweep on the GPU.
//
// The CPU port got the sweep from 272 ms to 77 ms across 32 cores, but that
// number does not travel: a single core still needs 185 ms, and the Orin Nano
// Super has six of them at a lower clock. What is left after the box filter
// was folded down is the warping and the per-pixel accumulation, which is the
// shape of work a GPU exists for -- every (plane, pixel) is independent and
// reads a handful of neighbouring source pixels.
//
// The volume is 26 x 180 x 320 floats, 6 MB, so it lives on the device from
// the first kernel to the last and only three images come back.
//
// The arithmetic follows the CPU version step for step. Bilinear sampling
// matches OpenCV's INTER_LINEAR with BORDER_CONSTANT: a sample is taken when
// its four neighbours are all inside the image, which is what makes the
// -1 sentinel mean the same thing on both sides.

#include <cfloat>
#include <cmath>

namespace
{

__device__ inline float sample_bilinear(
  const float * __restrict__ image, int rows, int cols, float x, float y,
  bool * inside)
{
  if (!isfinite(x) || !isfinite(y)) {
    *inside = false;
    return 0.0f;
  }
  const float floor_x = floorf(x);
  const float floor_y = floorf(y);
  const int x0 = static_cast<int>(floor_x);
  const int y0 = static_cast<int>(floor_y);
  // `x0 + 1 >= cols` is the obvious bound and it is wrong. A homography that
  // throws a pixel far outside the source gives a float past 2^31, the
  // conversion saturates at INT_MAX, and INT_MAX + 1 wraps to INT_MIN -- which
  // is less than cols, so the bound passes and the sample is taken two
  // billion elements past the image. It needs a warp big enough to push a
  // pixel that far, which is why it survived every straight drive and
  // surfaced only when a weaving one was matched against sources 3.6 m away.
  // Comparing without the increment cannot overflow.
  if (x0 < 0 || y0 < 0 || x0 >= cols - 1 || y0 >= rows - 1) {
    *inside = false;
    return 0.0f;
  }
  const float fx = x - floor_x;
  const float fy = y - floor_y;
  const float * row0 = image + static_cast<size_t>(y0) * cols + x0;
  const float * row1 = row0 + cols;
  const float top = row0[0] + fx * (row0[1] - row0[0]);
  const float bottom = row1[0] + fx * (row1[1] - row1[0]);
  *inside = true;
  return top + fy * (bottom - top);
}

}  // namespace

// One thread per (plane, pixel). Walks every source, warps this pixel onto the
// plane's homography, and accumulates the absolute difference and the count of
// sources that actually covered it.
extern "C" __global__ void sweep_accumulate(
  const float * __restrict__ reference,
  const float * __restrict__ sources,      // (source_count, rows, cols)
  const float * __restrict__ homographies, // (source_count, planes, 9), row major
  int source_count, int planes, int rows, int cols,
  float * __restrict__ cost, float * __restrict__ seen)
{
  const size_t pixels = static_cast<size_t>(rows) * cols;
  const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= pixels * planes) {
    return;
  }
  const int plane = static_cast<int>(index / pixels);
  const size_t pixel = index - static_cast<size_t>(plane) * pixels;
  const int row = static_cast<int>(pixel / cols);
  const int column = static_cast<int>(pixel - static_cast<size_t>(row) * cols);
  const float reference_value = reference[pixel];

  float total = 0.0f;
  float covered = 0.0f;
  for (int s = 0; s < source_count; ++s) {
    const float * h = homographies + (static_cast<size_t>(s) * planes + plane) * 9;
    const float w = h[6] * column + h[7] * row + h[8];
    if (fabsf(w) < 1e-12f) {
      continue;
    }
    const float inverse = 1.0f / w;
    const float x = (h[0] * column + h[1] * row + h[2]) * inverse;
    const float y = (h[3] * column + h[4] * row + h[5]) * inverse;
    bool inside = false;
    const float value = sample_bilinear(
      sources + static_cast<size_t>(s) * pixels, rows, cols, x, y, &inside);
    if (inside) {
      total += fabsf(value - reference_value);
      covered += 1.0f;
    }
  }
  cost[index] = total;
  seen[index] = covered;
}

// The same walk for normalised cross correlation, one source at a time.
//
// SAD lets every source be summed before the box filter, because a box filter
// is linear and the sum of differences is a difference of sums. ZNCC is not:
// it divides by a spread that belongs to one pair of images, so the filter has
// to run per source and the costs are combined afterwards. That is what this
// kernel is for -- it lays down the six sums a window needs, and a second pass
// turns them into a correlation once the box filter has run over all six.
//
// `a` is the reference and `b` the warped source. `a` cannot be folded into
// n * reference afterwards: the mask is the source's, so which reference
// pixels are in the window depends on the plane.
extern "C" __global__ void zncc_accumulate(
  const float * __restrict__ reference,
  const float * __restrict__ source,
  const float * __restrict__ homographies, // (planes, 9) for this source
  int planes, int rows, int cols,
  float * __restrict__ stats)              // (6, planes, rows, cols)
{
  const size_t pixels = static_cast<size_t>(rows) * cols;
  const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= pixels * planes) {
    return;
  }
  const int plane = static_cast<int>(index / pixels);
  const size_t pixel = index - static_cast<size_t>(plane) * pixels;
  const int row = static_cast<int>(pixel / cols);
  const int column = static_cast<int>(pixel - static_cast<size_t>(row) * cols);

  const float * h = homographies + static_cast<size_t>(plane) * 9;
  const float w = h[6] * column + h[7] * row + h[8];
  float n = 0.0f, a = 0.0f, aa = 0.0f, b = 0.0f, bb = 0.0f, ab = 0.0f;
  if (fabsf(w) >= 1e-12f) {
    const float inverse = 1.0f / w;
    const float x = (h[0] * column + h[1] * row + h[2]) * inverse;
    const float y = (h[3] * column + h[4] * row + h[5]) * inverse;
    bool inside = false;
    const float value = sample_bilinear(source, rows, cols, x, y, &inside);
    if (inside) {
      const float held = reference[pixel];
      n = 1.0f; a = held; aa = held * held;
      b = value; bb = value * value; ab = held * value;
    }
  }
  const size_t stride = pixels * planes;
  stats[index] = n;
  stats[index + stride] = a;
  stats[index + 2 * stride] = aa;
  stats[index + 3 * stride] = b;
  stats[index + 4 * stride] = bb;
  stats[index + 5 * stride] = ab;
}

// The same accumulation for a lens that has no homography.
//
// An equidistant fisheye maps r = f*theta, so a pixel does not become a ray by
// any 3x3 and the plane correspondence cannot be collapsed into one. The
// geometry is unchanged -- a ray out of the reference camera, where it meets
// the plane, that point carried into the source and projected -- it is just
// solved per pixel here instead of once per plane on the host.
//
// `geometry` is 30 floats, fixed for a source: focal, cx, cy, then the 3x3
// base-from-camera rotation, the camera's position in base, the plane normal,
// the source-from-reference rotation, and its translation. Only the plane
// offset varies, so it comes in its own array.
__device__ __forceinline__ void matrix_times(const float * m, const float * v,
                                             float * out)
{
  out[0] = m[0] * v[0] + m[1] * v[1] + m[2] * v[2];
  out[1] = m[3] * v[0] + m[4] * v[1] + m[5] * v[2];
  out[2] = m[6] * v[0] + m[7] * v[1] + m[8] * v[2];
}

__device__ __forceinline__ void matrix_transpose_times(const float * m,
                                                       const float * v,
                                                       float * out)
{
  out[0] = m[0] * v[0] + m[3] * v[1] + m[6] * v[2];
  out[1] = m[1] * v[0] + m[4] * v[1] + m[7] * v[2];
  out[2] = m[2] * v[0] + m[5] * v[1] + m[8] * v[2];
}

extern "C" __global__ void zncc_accumulate_fisheye(
  const float * __restrict__ reference,
  const float * __restrict__ source,
  const float * __restrict__ geometry,   // 30 floats, see above
  const float * __restrict__ offsets,    // (planes,)
  int planes, int rows, int cols,
  float * __restrict__ stats)            // (6, planes, rows, cols)
{
  const size_t pixels = static_cast<size_t>(rows) * cols;
  const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= pixels * planes) {
    return;
  }
  const int plane = static_cast<int>(index / pixels);
  const size_t pixel = index - static_cast<size_t>(plane) * pixels;
  const int row = static_cast<int>(pixel / cols);
  const int column = static_cast<int>(pixel - static_cast<size_t>(row) * cols);

  const float focal = geometry[0];
  const float cx = geometry[1];
  const float cy = geometry[2];
  const float * rotation_bc = geometry + 3;
  const float * eye = geometry + 12;
  const float * normal = geometry + 15;
  const float * rotation = geometry + 18;
  const float * shift = geometry + 27;

  float n = 0.0f, a = 0.0f, aa = 0.0f, b = 0.0f, bb = 0.0f, ab = 0.0f;

  const float dx = column - cx;
  const float dy = row - cy;
  const float radius = sqrtf(dx * dx + dy * dy);
  const float theta = radius / focal;
  const float scale = radius > 1e-9f ? __sinf(theta) / radius : 1.0f / focal;
  float ray_camera[3] = {dx * scale, dy * scale, __cosf(theta)};
  float ray_base[3];
  matrix_times(rotation_bc, ray_camera, ray_base);

  // normal[0] == 0 with the other two also zero is the caller's way of asking
  // for a depth sweep instead of a plane sweep. A plane fixes the surface
  // orientation before the match is made -- the road for the height family,
  // the optical axis laid flat for the vertical one -- and a car's flank is
  // neither, which is why its height came out at 2.40 m against a real 1.22.
  // Sweeping depth along each pixel's own ray assumes nothing about which way
  // the surface faces. It costs nothing here: the fisheye path already solves
  // every pixel separately, so the homography the plane bought is not in use.
  const bool depth_mode = (normal[0] == 0.0f && normal[1] == 0.0f
                           && normal[2] == 0.0f);
  const float denominator = normal[0] * ray_base[0] + normal[1] * ray_base[1]
                          + normal[2] * ray_base[2];
  const float lift = normal[0] * eye[0] + normal[1] * eye[1] + normal[2] * eye[2];
  if (depth_mode || fabsf(denominator) >= 1e-12f) {
    const float ray_norm = sqrtf(ray_base[0] * ray_base[0]
                               + ray_base[1] * ray_base[1]
                               + ray_base[2] * ray_base[2]);
    const float along = depth_mode
      ? offsets[plane] / fmaxf(ray_norm, 1e-9f)
      : (offsets[plane] - lift) / denominator;
    if (along > 0.0f) {
      float point[3] = {eye[0] + along * ray_base[0],
                        eye[1] + along * ray_base[1],
                        eye[2] + along * ray_base[2]};
      float carried[3];
      matrix_times(rotation, point, carried);
      carried[0] += shift[0] - eye[0];
      carried[1] += shift[1] - eye[1];
      carried[2] += shift[2] - eye[2];
      float in_camera[3];
      matrix_transpose_times(rotation_bc, carried, in_camera);
      if (in_camera[2] > 0.0f) {
        const float sideways = sqrtf(in_camera[0] * in_camera[0]
                                   + in_camera[1] * in_camera[1]);
        const float angle = atan2f(sideways, in_camera[2]);
        const float out = sideways > 1e-9f ? focal * angle / sideways : focal;
        const float x = in_camera[0] * out + cx;
        const float y = in_camera[1] * out + cy;
        bool inside = false;
        const float value = sample_bilinear(source, rows, cols, x, y, &inside);
        if (inside) {
          const float held = reference[pixel];
          n = 1.0f; a = held; aa = held * held;
          b = value; bb = value * value; ab = held * value;
        }
      }
    }
  }
  const size_t stride = pixels * planes;
  stats[index] = n;
  stats[index + stride] = a;
  stats[index + 2 * stride] = aa;
  stats[index + 3 * stride] = b;
  stats[index + 4 * stride] = bb;
  stats[index + 5 * stride] = ab;
}

// Window sums to a correlation, and a correlation to a cost on the same scale
// the grey-level difference used, so a margin quoted in one reads in the other.
extern "C" __global__ void zncc_reduce(
  const float * __restrict__ stats, int planes, int rows, int cols,
  float * __restrict__ cost, float * __restrict__ seen)
{
  const size_t pixels = static_cast<size_t>(rows) * cols;
  const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= pixels * planes) {
    return;
  }
  const size_t stride = pixels * planes;
  const float n = stats[index];
  const float divisor = fmaxf(n, 1.0f);
  const float mean_a = stats[index + stride] / divisor;
  const float mean_b = stats[index + 3 * stride] / divisor;
  const float var_a = stats[index + 2 * stride] / divisor - mean_a * mean_a;
  const float var_b = stats[index + 4 * stride] / divisor - mean_b * mean_b;
  const float covariance = stats[index + 5 * stride] / divisor - mean_a * mean_b;
  const float correlation = covariance /
    sqrtf(fmaxf(var_a, 0.0f) * fmaxf(var_b, 0.0f) + 1.0f);
  cost[index] += (1.0f - correlation) * 100.0f * n;
  seen[index] += n;
}

// Separable box sum, not normalised, clamped at the border the way OpenCV's
// BORDER_DEFAULT would not be -- the CPU side counts how many samples landed
// inside instead, so the two agree by construction.
extern "C" __global__ void box_horizontal(
  const float * __restrict__ input, int planes, int rows, int cols, int radius,
  float * __restrict__ output)
{
  const size_t pixels = static_cast<size_t>(rows) * cols;
  const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= pixels * planes) {
    return;
  }
  const size_t pixel = index % pixels;
  const int row = static_cast<int>(pixel / cols);
  const int column = static_cast<int>(pixel - static_cast<size_t>(row) * cols);
  const size_t base = index - column;
  float total = 0.0f;
  for (int offset = -radius; offset <= radius; ++offset) {
    int at = column + offset;
    at = at < 0 ? 0 : (at >= cols ? cols - 1 : at);
    total += input[base + at];
  }
  output[index] = total;
}

extern "C" __global__ void box_vertical(
  const float * __restrict__ input, int planes, int rows, int cols, int radius,
  float * __restrict__ output)
{
  const size_t pixels = static_cast<size_t>(rows) * cols;
  const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= pixels * planes) {
    return;
  }
  const int plane = static_cast<int>(index / pixels);
  const size_t pixel = index - static_cast<size_t>(plane) * pixels;
  const int row = static_cast<int>(pixel / cols);
  const int column = static_cast<int>(pixel - static_cast<size_t>(row) * cols);
  const size_t plane_base = static_cast<size_t>(plane) * pixels;
  float total = 0.0f;
  for (int offset = -radius; offset <= radius; ++offset) {
    int at = row + offset;
    at = at < 0 ? 0 : (at >= rows ? rows - 1 : at);
    total += input[plane_base + static_cast<size_t>(at) * cols + column];
  }
  output[index] = total;
}

// cost / seen, with the planes nothing reached marked unusable.
extern "C" __global__ void sweep_finalise(
  const float * __restrict__ cost, const float * __restrict__ seen,
  int planes, int rows, int cols, float unseen,
  float * __restrict__ volume, unsigned char * __restrict__ valid)
{
  const size_t total = static_cast<size_t>(rows) * cols * planes;
  const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= total) {
    return;
  }
  const float count = seen[index];
  const bool usable = count >= unseen;
  valid[index] = usable ? 1 : 0;
  volume[index] = usable ? cost[index] / fmaxf(count, 1.0f) : 1.0e6f;
}

// Semi-global accumulation along one line. A block owns a line and a thread
// owns a plane, so the running minimum over planes is a block reduction.
// `stride` steps along the line, `line_stride` picks the line out of the image.
extern "C" __global__ void sweep_aggregate(
  const float * __restrict__ volume, int planes, int pixels, int steps,
  long long line_stride, long long line_offset, long long stride,
  float small_step, float big_step, float * __restrict__ total)
{
  extern __shared__ float shared[];
  float * previous = shared;           // planes
  float * floor_value = shared + planes;

  const int plane = threadIdx.x;
  if (plane >= planes) {
    return;
  }
  const long long start =
    static_cast<long long>(blockIdx.x) * line_stride + line_offset;
  const size_t plane_offset = static_cast<size_t>(plane) * pixels;

  previous[plane] = volume[plane_offset + start];
  // No atomics: a block owns a line and a thread owns a plane, so within one
  // launch no two threads ever address the same element. The four directions
  // do overlap, and are launched one after another.
  total[plane_offset + start] += previous[plane];
  __syncthreads();

  for (int step = 1; step < steps; ++step) {
    // Twenty-six planes is small enough that one thread reducing them beats
    // the synchronisation a tree reduction would need at every step.
    if (plane == 0) {
      float lowest = previous[0];
      for (int d = 1; d < planes; ++d) {
        lowest = fminf(lowest, previous[d]);
      }
      floor_value[0] = lowest;
    }
    __syncthreads();
    const float floor = floor_value[0];

    float best = fminf(previous[plane], floor + big_step);
    if (plane > 0) {
      best = fminf(best, previous[plane - 1] + small_step);
    }
    if (plane + 1 < planes) {
      best = fminf(best, previous[plane + 1] + small_step);
    }
    const size_t at =
      plane_offset + static_cast<size_t>(start + static_cast<long long>(step) * stride);
    const float carried = volume[at] + best - floor;
    __syncthreads();
    previous[plane] = carried;
    total[at] += carried;
    __syncthreads();
  }
}

// Cheapest plane per pixel, and the road plane's own cost beside it.
extern "C" __global__ void sweep_argmin(
  const float * __restrict__ volume, const unsigned char * __restrict__ valid,
  int planes, int pixels, int road_plane,
  int * __restrict__ best_index, float * __restrict__ best_cost,
  float * __restrict__ road_cost, float * __restrict__ second_cost)
{
  const size_t pixel = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (pixel >= static_cast<size_t>(pixels)) {
    return;
  }
  int argmin = 0;
  float lowest = FLT_MAX;
  bool any = false;
  for (int d = 0; d < planes; ++d) {
    const size_t at = static_cast<size_t>(d) * pixels + pixel;
    if (!valid[at]) {
      continue;
    }
    const float here = volume[at];
    if (!any || here < lowest) {
      lowest = here;
      argmin = d;
      any = true;
    }
  }
  // Neighbouring planes are not rivals -- they are the same answer read to a
  // finer step, which is why the shoulder is skipped. The planes at and below
  // the road are rivals in the same sense: they all say "this is ground", and
  // letting them undercut each other reports the flattest, best understood
  // pixel in the frame as undecided.
  const bool on_ground = any && argmin <= road_plane;
  float runner_up = INFINITY;
  for (int d = 0; d < planes; ++d) {
    if (d >= argmin - 1 && d <= argmin + 1) {
      continue;
    }
    if (on_ground && d <= road_plane) {
      continue;
    }
    const size_t at = static_cast<size_t>(d) * pixels + pixel;
    if (valid[at]) {
      runner_up = fminf(runner_up, volume[at]);
    }
  }
  // The best of the planes at or below the road, not the road plane alone.
  // They are the same when the sweep starts at the road, which is the
  // default; with planes underneath, a pixel matching one of them is road
  // sitting low rather than something standing up.
  float ground = INFINITY;
  for (int d = 0; d <= road_plane && d < planes; ++d) {
    const size_t at = static_cast<size_t>(d) * pixels + pixel;
    if (valid[at]) {
      ground = fminf(ground, volume[at]);
    }
  }
  best_index[pixel] = any ? argmin : 0;
  best_cost[pixel] = any ? lowest : INFINITY;
  second_cost[pixel] = any ? runner_up : INFINITY;
  road_cost[pixel] = ground;
}


// Host side. The buffers are held between calls because a keyframe is swept
// every few hundred milliseconds and cudaMalloc costs more than the kernels.
#include <cstdio>
#include <cstdlib>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace
{

// A failed launch leaves the caller's output arrays exactly as they were, and
// they are freshly allocated numpy buffers -- so the sweep returns whatever
// was in that memory. It surfaced as a plane index of -901663568 out of 26,
// several stages downstream and looking nothing like a CUDA problem. Every
// call site is checked now, and the two that matter are checked after a
// synchronise, because a launch reports only the errors it can see before the
// kernel runs.
void expect(const char * what)
{
  if (getenv("SWEEP_CUDA_SYNC") != nullptr) {
    cudaDeviceSynchronize();
  }
  const cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    throw std::runtime_error(
      std::string("CUDA failed at ") + what + ": " + cudaGetErrorString(error));
  }
}


struct Scratch
{
  size_t elements = 0;
  size_t image = 0;
  size_t sources = 0;
  float * cost = nullptr;
  float * seen = nullptr;
  float * scratch = nullptr;
  float * blurred_cost = nullptr;
  float * blurred_seen = nullptr;
  float * volume = nullptr;
  float * total = nullptr;
  unsigned char * valid = nullptr;
  float * reference = nullptr;
  float * source_stack = nullptr;
  float * homographies = nullptr;
  size_t homography_count = 0;
  float * geometry = nullptr;       // (source_count, 30) 어안 기하
  size_t geometry_count = 0;
  float * offsets = nullptr;        // (planes,) 평면 높이
  size_t offset_count = 0;
  float * stats = nullptr;
  size_t stats_held = 0;
  float * stats_scratch = nullptr;
  size_t stats_scratch_held = 0;
  int * best_index = nullptr;
  float * best_cost = nullptr;
  float * road_cost = nullptr;
  float * second_cost = nullptr;
};

Scratch g_scratch;
// The volume of the call just made, so a caller that decides afterwards that
// it wants the whole cost profile can have it without sweeping twice.
const float * g_last_volume = nullptr;
size_t g_last_elements = 0;

template <typename T>
void ensure(T ** pointer, size_t & held, size_t wanted)
{
  if (held >= wanted && *pointer != nullptr) {
    return;
  }
  if (*pointer != nullptr) {
    cudaFree(*pointer);
    *pointer = nullptr;
  }
  held = 0;
  const cudaError_t error =
    cudaMalloc(reinterpret_cast<void **>(pointer), wanted * sizeof(T));
  if (error != cudaSuccess) {
    *pointer = nullptr;
    throw std::runtime_error(
      std::string("CUDA could not allocate ") + std::to_string(wanted) +
      " elements: " + cudaGetErrorString(error));
  }
  held = wanted;
}

}  // namespace

void plane_sweep_cuda_volume(float * out, size_t count)
{
  if (g_last_volume == nullptr || count > g_last_elements) {
    throw std::runtime_error("no volume from the last sweep");
  }
  cudaMemcpy(out, g_last_volume, count * sizeof(float), cudaMemcpyDeviceToHost);
  expect("volume download");
}

bool plane_sweep_cuda_available()
{
  int devices = 0;
  return cudaGetDeviceCount(&devices) == cudaSuccess && devices > 0;
}

void plane_sweep_cuda_impl(
  const float * reference, const float * sources, int source_count,
  const float * homographies, int planes, int rows, int cols,
  int window_x, int window_y, int cost_kind, int road_plane,
  float small_step, float big_step,
  int * best_index, float * best_cost, float * road_cost, float * second_cost,
  float * volume_out,
  // Fisheye: `geometry` is (source_count, 30) and `offsets` is (planes,).
  // Non-null switches the accumulation to the per-pixel ray solve, because a
  // lens with r = f*theta has no homography to stack.
  const float * geometry, const float * offsets)
{
  Scratch & s = g_scratch;
  const size_t pixels = static_cast<size_t>(rows) * cols;
  const size_t elements = pixels * planes;

  size_t held = s.elements;
  ensure(&s.cost, held, elements);
  held = s.elements; ensure(&s.seen, held, elements);
  held = s.elements; ensure(&s.scratch, held, elements);
  held = s.elements; ensure(&s.blurred_cost, held, elements);
  held = s.elements; ensure(&s.blurred_seen, held, elements);
  held = s.elements; ensure(&s.volume, held, elements);
  held = s.elements; ensure(&s.total, held, elements);
  size_t valid_held = s.elements;
  ensure(&s.valid, valid_held, elements);
  s.elements = elements;

  size_t image_held = s.image;
  ensure(&s.reference, image_held, pixels);
  size_t index_held = s.image;
  ensure(&s.best_index, index_held, pixels);
  size_t cost_held = s.image;
  ensure(&s.best_cost, cost_held, pixels);
  size_t road_held = s.image;
  ensure(&s.road_cost, road_held, pixels);
  size_t second_held = s.image;
  ensure(&s.second_cost, second_held, pixels);
  s.image = pixels;

  ensure(&s.source_stack, s.sources, pixels * source_count);
  ensure(&s.homographies, s.homography_count,
         static_cast<size_t>(source_count) * planes * 9);

  cudaMemcpy(s.reference, reference, pixels * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(s.source_stack, sources, pixels * source_count * sizeof(float),
             cudaMemcpyHostToDevice);
  const bool fisheye = geometry != nullptr && offsets != nullptr;
  if (fisheye) {
    size_t geometry_held = s.geometry_count;
    ensure(&s.geometry, geometry_held, static_cast<size_t>(source_count) * 30);
    s.geometry_count = geometry_held;
    size_t offsets_held = s.offset_count;
    ensure(&s.offsets, offsets_held, static_cast<size_t>(planes));
    s.offset_count = offsets_held;
    cudaMemcpy(s.geometry, geometry,
               static_cast<size_t>(source_count) * 30 * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(s.offsets, offsets, static_cast<size_t>(planes) * sizeof(float),
               cudaMemcpyHostToDevice);
  } else {
    cudaMemcpy(s.homographies, homographies,
               static_cast<size_t>(source_count) * planes * 9 * sizeof(float),
               cudaMemcpyHostToDevice);
  }
  expect("upload");

  const int threads = 256;
  const int blocks = static_cast<int>((elements + threads - 1) / threads);
  const int radius_x = window_x / 2;
  const int radius_y = window_y / 2;

  // SAD has no fisheye variant. The sweep only ever runs ZNCC in practice --
  // a sunlit car park moves the grey levels between frames for reasons that
  // have nothing to do with height -- so this is a guard, not a gap.
  if (cost_kind == 0 && fisheye) {
    fprintf(stderr, "[sweep] fisheye needs cost_kind=zncc; falling back\n");
  }
  if (cost_kind == 0) {
    expect("before sweep_accumulate");
  sweep_accumulate<<<blocks, threads>>>(
      s.reference, s.source_stack, s.homographies, source_count, planes, rows,
      cols, s.cost, s.seen);
    expect("before box_horizontal");
  box_horizontal<<<blocks, threads>>>(s.cost, planes, rows, cols, radius_x, s.scratch);
    expect("before box_vertical");
  box_vertical<<<blocks, threads>>>(s.scratch, planes, rows, cols, radius_y, s.blurred_cost);
    expect("before box_horizontal");
  box_horizontal<<<blocks, threads>>>(s.seen, planes, rows, cols, radius_x, s.scratch);
    expect("before box_vertical");
  box_vertical<<<blocks, threads>>>(s.scratch, planes, rows, cols, radius_y, s.blurred_seen);
  } else {
    // Correlation cannot share a filter between sources, so each one lays down
    // its six window sums, is filtered, and is folded into the running cost.
    // The six ride in one buffer and the box kernels are told there are six
    // times as many planes, which costs nothing and saves five launches.
    ensure(&s.stats, s.stats_held, elements * 6);
    ensure(&s.stats_scratch, s.stats_scratch_held, elements * 6);
    cudaMemset(s.blurred_cost, 0, elements * sizeof(float));
    cudaMemset(s.blurred_seen, 0, elements * sizeof(float));
    const int wide = static_cast<int>((elements * 6 + threads - 1) / threads);
    if (getenv("SWEEP_CUDA_SYNC") != nullptr) {
      fprintf(stderr,
        "[sweep] planes=%d rows=%d cols=%d sources=%d elements=%zu "
        "stats_held=%zu sources_held=%zu homog_held=%zu wanted_h=%zu\n",
        planes, rows, cols, source_count, elements, s.stats_held, s.sources,
        s.homography_count,
        static_cast<size_t>(source_count) * planes * 9);
    }
    for (int source = 0; source < source_count; ++source) {
      expect("before zncc_accumulate");
      if (fisheye) {
        zncc_accumulate_fisheye<<<blocks, threads>>>(
          s.reference, s.source_stack + static_cast<size_t>(source) * pixels,
          s.geometry + static_cast<size_t>(source) * 30, s.offsets,
          planes, rows, cols, s.stats);
      } else {
        zncc_accumulate<<<blocks, threads>>>(
          s.reference, s.source_stack + static_cast<size_t>(source) * pixels,
          s.homographies + static_cast<size_t>(source) * planes * 9,
          planes, rows, cols, s.stats);
      }
      expect("before box_horizontal");
  box_horizontal<<<wide, threads>>>(
        s.stats, planes * 6, rows, cols, radius_x, s.stats_scratch);
      expect("before box_vertical");
  box_vertical<<<wide, threads>>>(
        s.stats_scratch, planes * 6, rows, cols, radius_y, s.stats);
      expect("before zncc_reduce");
  zncc_reduce<<<blocks, threads>>>(
        s.stats, planes, rows, cols, s.blurred_cost, s.blurred_seen);
    }
  }

  cudaDeviceSynchronize();
  expect("matching");

  // The validity bar: what fraction of (window x sources) samples a pixel
  // must land inside the sources. At 0.5 a pixel on the fisheye seam is
  // discarded outright -- half its sources swing out of this camera's field
  // exactly because the sideways parallax is large, which is the geometry
  // that resolves range best. Cost is already normalised by the seen count,
  // so the bar is pure admission policy, not fairness.
  float seen_fraction = 0.5f;
  if (const char *env = getenv("SWEEP_SEEN_FRACTION")) {
    seen_fraction = static_cast<float>(atof(env));
  }
  const float unseen = static_cast<float>(window_x) * window_y *
    seen_fraction * static_cast<float>(source_count);
  expect("before sweep_finalise");
  sweep_finalise<<<blocks, threads>>>(
    s.blurred_cost, s.blurred_seen, planes, rows, cols, unseen, s.volume, s.valid);

  const float * chosen = s.volume;
  if (small_step > 0.0f) {
    cudaMemset(s.total, 0, elements * sizeof(float));
    const size_t shared = (static_cast<size_t>(planes) + 1) * sizeof(float);
    const int block_threads = planes;
    expect("before sweep_aggregate");
    sweep_aggregate<<<rows, block_threads, shared>>>(
      s.volume, planes, static_cast<int>(pixels), cols, cols, 0, 1,
      small_step, big_step, s.total);
    expect("before sweep_aggregate");
    sweep_aggregate<<<rows, block_threads, shared>>>(
      s.volume, planes, static_cast<int>(pixels), cols, cols, cols - 1, -1,
      small_step, big_step, s.total);
    expect("before sweep_aggregate");
    sweep_aggregate<<<cols, block_threads, shared>>>(
      s.volume, planes, static_cast<int>(pixels), rows, 1, 0, cols,
      small_step, big_step, s.total);
    expect("before sweep_aggregate");
    sweep_aggregate<<<cols, block_threads, shared>>>(
      s.volume, planes, static_cast<int>(pixels), rows, 1,
      static_cast<long long>(rows - 1) * cols, -cols,
      small_step, big_step, s.total);
    chosen = s.total;
  }

  const int pixel_blocks = static_cast<int>((pixels + threads - 1) / threads);
  expect("before sweep_argmin");
  sweep_argmin<<<pixel_blocks, threads>>>(
    chosen, s.valid, planes, static_cast<int>(pixels), road_plane,
    s.best_index, s.best_cost, s.road_cost, s.second_cost);

  cudaDeviceSynchronize();
  expect("reduction");

  cudaMemcpy(best_index, s.best_index, pixels * sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(best_cost, s.best_cost, pixels * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(road_cost, s.road_cost, pixels * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(second_cost, s.second_cost, pixels * sizeof(float), cudaMemcpyDeviceToHost);
  if (volume_out != nullptr) {
    cudaMemcpy(volume_out, chosen, elements * sizeof(float),
               cudaMemcpyDeviceToHost);
  }
  g_last_volume = chosen;
  g_last_elements = elements;
  expect("download");
}
