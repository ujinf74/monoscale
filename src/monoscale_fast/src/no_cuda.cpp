// Stands in for the CUDA translation unit where there is no nvcc, so the
// module still builds and the caller finds out by asking rather than by
// failing to import.
#include <stdexcept>

bool plane_sweep_cuda_available()
{
  return false;
}

// The signature has to track fast.cpp's declaration exactly. It did not: the
// geometry and offset arrays were added there and not here, so on a machine
// without nvcc the module linked against a name that nothing defined and
// failed to import outright -- the one situation this file exists to prevent.
void plane_sweep_cuda_impl(
  const float *, const float *, int, const float *, int, int, int, int, int,
  int, int, float, float, int *, float *, float *, float *, float *,
  const float *, const float *)
{
  throw std::runtime_error("built without CUDA");
}

void plane_sweep_cuda_volume(float *, size_t)
{
  throw std::runtime_error("built without CUDA");
}
