#include "monoscale_sweep/sweep.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

#include <opencv2/imgproc.hpp>

namespace monoscale_sweep
{

// The CUDA backend, in cuda_backend.cpp. Fills best/costs (and the volume when
// asked) from the monoscale_fast kernel; returns false when CUDA is not linked.
bool cuda_match(
  const Lens & lens, const SweepSettings & settings,
  const cv::Mat & reference32, const std::vector<cv::Mat> & source32,
  const Pose5 & reference_pose, const std::vector<Pose5> & source_poses,
  const std::vector<double> & heights, int road,
  cv::Mat & best, cv::Mat & best_cost, cv::Mat & road_cost, cv::Mat & second_cost,
  std::vector<cv::Mat> & volume, bool want_volume);

namespace
{
constexpr double kInf = std::numeric_limits<double>::infinity();
const Eigen::Vector3d kEz(0.0, 0.0, 1.0);
}  // namespace

Eigen::Matrix3d attitude(const Pose5 & pose)
{
  const double cz = std::cos(pose.yaw);
  const double sz = std::sin(pose.yaw);
  Eigen::Matrix3d yaw;
  yaw << cz, -sz, 0.0, sz, cz, 0.0, 0.0, 0.0, 1.0;
  if (pose.roll == 0.0 && pose.pitch == 0.0) {
    return yaw;
  }
  const double cy = std::cos(pose.pitch);
  const double sy = std::sin(pose.pitch);
  const double cx = std::cos(pose.roll);
  const double sx = std::sin(pose.roll);
  Eigen::Matrix3d pitch;
  pitch << cy, 0.0, sy, 0.0, 1.0, 0.0, -sy, 0.0, cy;
  Eigen::Matrix3d roll;
  roll << 1.0, 0.0, 0.0, 0.0, cx, -sx, 0.0, sx, cx;
  return yaw * pitch * roll;
}

void CameraGrid::reset(const SweepSettings & s)
{
  log_odds = cv::Mat::zeros(s.grid_height, s.grid_width, CV_32F);
  observed = cv::Mat::zeros(s.grid_height, s.grid_width, CV_8U);
  slab_free[0] = cv::Mat::zeros(s.grid_height, s.grid_width, CV_16S);
  slab_free[1] = cv::Mat::zeros(s.grid_height, s.grid_width, CV_16S);
}

Sweep::Sweep(const SweepSettings & settings, const Lens & lens)
: settings_(settings), lens_(lens) {}

void Sweep::ensure_rays(int width, int height) const
{
  if (!ray_base_.empty() && ray_base_.cols == width && ray_base_.rows == height) {
    return;
  }
  // Unproject every pixel to a unit ray in camera axes, then rotate into
  // base_link. Equidistant inverse: theta = radius / focal.
  ray_base_.create(height, width, CV_64FC3);
  const Eigen::Matrix3d & rbc = lens_.rotation_base_from_camera;
  for (int v = 0; v < height; ++v) {
    auto * row = ray_base_.ptr<cv::Vec3d>(v);
    for (int u = 0; u < width; ++u) {
      const double x = u - lens_.cx;
      const double y = v - lens_.cy;
      const double radius = std::hypot(x, y);
      const double theta = radius / lens_.focal;
      const double scale = radius > 1e-9 ? std::sin(theta) / radius : 1.0 / lens_.focal;
      const Eigen::Vector3d cam(x * scale, y * scale, std::cos(theta));
      const Eigen::Vector3d base = rbc * cam;
      row[u] = cv::Vec3d(base.x(), base.y(), base.z());
    }
  }
}

void Sweep::warp_maps(
  const Pose5 & reference, const Pose5 & source, const Eigen::Vector3d & normal,
  double offset, const cv::Mat & dot_normal, cv::Mat & map_x, cv::Mat & map_y) const
{
  const Eigen::Matrix3d forward = attitude(reference);
  const Eigen::Matrix3d backward = attitude(source);
  const Eigen::Matrix3d rotation = backward.transpose() * forward;
  const Eigen::Vector3d shift = backward.transpose() *
    Eigen::Vector3d(reference.x - source.x, reference.y - source.y, 0.0);
  const Eigen::Vector3d & eye = lens_.translation_base_from_camera;
  const Eigen::Matrix3d & rbc = lens_.rotation_base_from_camera;
  const double normal_eye = normal.dot(eye);

  map_x.create(height_, width_, CV_32F);
  map_y.create(height_, width_, CV_32F);
#pragma omp parallel for schedule(static)
  for (int v = 0; v < height_; ++v) {
    const auto * rray = ray_base_.ptr<cv::Vec3d>(v);
    const double * rden = dot_normal.ptr<double>(v);
    float * mx = map_x.ptr<float>(v);
    float * my = map_y.ptr<float>(v);
    for (int u = 0; u < width_; ++u) {
      const double denom = rden[u];
      const double along = (offset - normal_eye) / denom;
      const cv::Vec3d & rb = rray[u];
      const Eigen::Vector3d point_base(
        eye.x() + along * rb[0], eye.y() + along * rb[1], eye.z() + along * rb[2]);
      const Eigen::Vector3d point_source = rotation * point_base + shift;
      const Eigen::Vector3d point_camera = rbc.transpose() * (point_source - eye);
      bool bad = !std::isfinite(along) || along <= 0.0 || point_camera.z() <= 0.0;
      if (bad) {
        mx[u] = -1.0f;
        my[u] = -1.0f;
        continue;
      }
      // project(): equidistant forward.
      const double sideways = std::hypot(point_camera.x(), point_camera.y());
      const double th = std::atan2(sideways, point_camera.z());
      const double scale = sideways > 1e-9 ? lens_.focal * th / sideways : lens_.focal;
      mx[u] = static_cast<float>(point_camera.x() * scale + lens_.cx);
      my[u] = static_cast<float>(point_camera.y() * scale + lens_.cy);
    }
  }
}

void Sweep::keyframe(
  const cv::Mat & reference_gray, const Pose5 & reference_pose,
  const std::vector<cv::Mat> & source_grays,
  const std::vector<Pose5> & source_poses,
  CameraGrid & grid) const
{
  if (source_grays.size() < 2) {
    return;
  }
  const_cast<Sweep *>(this)->width_ = reference_gray.cols;
  const_cast<Sweep *>(this)->height_ = reference_gray.rows;
  ensure_rays(width_, height_);

  // The ladder of world-horizontal planes, and its road rung.
  std::vector<double> heights;
  for (double h = settings_.height_min;
    h <= settings_.obstacle_max_height + 1e-9; h += settings_.height_step)
  {
    heights.push_back(h);
  }
  const int planes = static_cast<int>(heights.size());
  int road = 0;
  for (int i = 1; i < planes; ++i) {
    if (std::abs(heights[i]) < std::abs(heights[road])) {road = i;}
  }

  const Eigen::Vector3d normal = attitude(reference_pose).transpose() * kEz;
  // normal . ray_base, per pixel: fixed across the ladder.
  cv::Mat dot_normal(height_, width_, CV_64F);
  for (int v = 0; v < height_; ++v) {
    const auto * rray = ray_base_.ptr<cv::Vec3d>(v);
    double * dn = dot_normal.ptr<double>(v);
    for (int u = 0; u < width_; ++u) {
      dn[u] = normal.x() * rray[u][0] + normal.y() * rray[u][1] + normal.z() * rray[u][2];
    }
  }

  const cv::Mat reference32_full = reference_gray;
  cv::Mat reference32;
  reference_gray.convertTo(reference32, CV_32F);
  const cv::Size box(settings_.window_x, settings_.window_y);
  cv::Mat sq_ref;
  cv::multiply(reference32, reference32, sq_ref);

  std::vector<cv::Mat> volume;
  cv::Mat best;
  cv::Mat best_cost;
  cv::Mat road_cost;
  cv::Mat second_cost;

  std::vector<cv::Mat> source32;
  source32.reserve(source_grays.size());
  for (const auto & g : source_grays) {
    cv::Mat f;
    g.convertTo(f, CV_32F);
    source32.push_back(f);
  }

  // The CUDA kernel does the whole match-aggregate-reduce when it is linked --
  // the same one plane_sweep.py runs, so this is not a re-implementation but a
  // shared arithmetic. It also returns the aggregated volume, which subplane
  // needs. The CPU path below is the parity reference and the fallback.
  const bool did_cuda = settings_.use_cuda && cuda_match(
    lens_, settings_, reference32, source32, reference_pose, source_poses,
    heights, road, best, best_cost, road_cost, second_cost, volume,
    settings_.subplane);

  if (!did_cuda) {
    // ZNCC cost volume, summed over sources, x100 * count -- the same scaling
    // the python kernel uses so the SGM steps and margin read on one scale.
    volume.assign(planes, cv::Mat());
    std::vector<cv::Mat> seen(planes);
    for (int p = 0; p < planes; ++p) {
      volume[p] = cv::Mat::zeros(height_, width_, CV_32F);
      seen[p] = cv::Mat::zeros(height_, width_, CV_32F);
    }
    cv::Mat map_x;
    cv::Mat map_y;
    cv::Mat warped;
    cv::Mat inside;
    cv::Mat count;
    cv::Mat mean_a;
    cv::Mat mean_b;
    cv::Mat var_a;
    cv::Mat var_b;
    cv::Mat cov;
    cv::Mat held;
    for (size_t s = 0; s < source32.size(); ++s) {
      for (int p = 0; p < planes; ++p) {
        warp_maps(reference_pose, source_poses[s], normal, heights[p], dot_normal,
          map_x, map_y);
        cv::remap(source32[s], warped, map_x, map_y, cv::INTER_LINEAR,
          cv::BORDER_CONSTANT, cv::Scalar(-1.0));
        inside = (warped >= 0.0f);
        inside.convertTo(inside, CV_32F, 1.0 / 255.0);
        cv::boxFilter(inside, count, -1, box, cv::Point(-1, -1), false);
        cv::multiply(warped, inside, held);
        cv::Mat total = cv::max(count, 1.0f);
        cv::boxFilter(reference32.mul(inside), mean_a, -1, box, cv::Point(-1, -1), false);
        cv::divide(mean_a, total, mean_a);
        cv::boxFilter(held, mean_b, -1, box, cv::Point(-1, -1), false);
        cv::divide(mean_b, total, mean_b);
        cv::boxFilter(sq_ref.mul(inside), var_a, -1, box, cv::Point(-1, -1), false);
        cv::divide(var_a, total, var_a);
        var_a -= mean_a.mul(mean_a);
        cv::boxFilter(held.mul(held), var_b, -1, box, cv::Point(-1, -1), false);
        cv::divide(var_b, total, var_b);
        var_b -= mean_b.mul(mean_b);
        cv::boxFilter(reference32.mul(held), cov, -1, box, cv::Point(-1, -1), false);
        cv::divide(cov, total, cov);
        cov -= mean_a.mul(mean_b);
        cv::Mat denom;
        cv::sqrt(cv::max(var_a, 0.0f).mul(cv::max(var_b, 0.0f)) + 1.0f, denom);
        cv::Mat correlation;
        cv::divide(cov, denom, correlation);
        cv::Mat cost = (1.0f - correlation) * 100.0f;
        cost = cost.mul(count);
        volume[p] += cost;
        seen[p] += count;
      }
    }
    const double seen_bar = static_cast<double>(box.width) * box.height *
      settings_.seen_fraction * static_cast<double>(source32.size());
    for (int p = 0; p < planes; ++p) {
      cv::Mat safe = cv::max(seen[p], 1.0f);
      cv::divide(volume[p], safe, volume[p]);
      volume[p].setTo(kInf, seen[p] < static_cast<float>(seen_bar));
    }
    aggregate(volume, settings_.small_step, settings_.big_step);
    best.create(height_, width_, CV_32S);
    best_cost.create(height_, width_, CV_32F);
    road_cost.create(height_, width_, CV_32F);
    second_cost.create(height_, width_, CV_32F);
    reduce_volume(volume, road, best, best_cost, road_cost, second_cost);
  }

  Verdict verdict;
  build_verdict(
    reference32, best, best_cost, road_cost, second_cost, heights, road,
    reference_pose, source_poses, dot_normal, normal, verdict);

  // Sub-plane refinement of the surface height, from the aggregated cost
  // curve: a parabola through the winning rung and its two neighbours. The
  // placement gate `surface >= low` in build_verdict already ran on the rung
  // itself, so this only moves where the point lands, which is what --subplane
  // does in plane_sweep.run. (It measured a wash for placement but is part of
  // the adopted flag set; kept for parity, not for a gain.)
  if (settings_.subplane) {
    const double step = settings_.height_step;
    for (int v = 0; v < height_; ++v) {
      const int * bt = best.ptr<int>(v);
      const float * bc = best_cost.ptr<float>(v);
      float * sf = verdict.surface.ptr<float>(v);
      for (int u = 0; u < width_; ++u) {
        const int p = bt[u];
        const int pb = std::max(p - 1, 0);
        const int pa = std::min(p + 1, planes - 1);
        const float below = volume[pb].at<float>(v, u);
        const float above = volume[pa].at<float>(v, u);
        const float curve = below - 2.0f * bc[u] + above;
        double shift = 0.0;
        if (std::isfinite(below) && std::isfinite(above) && curve > 1e-9f) {
          shift = std::clamp(0.5 * (below - above) / curve, -0.5, 0.5);
        }
        sf[u] = static_cast<float>(heights[p] + shift * step);
      }
    }
  }

  integrate(reference_pose, heights, road, best, best_cost, second_cost,
    verdict, grid);
}

void Sweep::aggregate(
  std::vector<cv::Mat> & volume, double small_step, double big_step) const
{
  // Parity target: plane_sweep.aggregate. Four directional passes of the walk
  // recurrence, summed in float32 in the order numpy adds them.
  const int planes = static_cast<int>(volume.size());
  const int rows = volume[0].rows;
  const int cols = volume[0].cols;
  const float small = static_cast<float>(small_step);
  const float big = static_cast<float>(big_step);

  // Unseen cells walk as 1e6, not inf, so a path may cross them at a price;
  // only the finished volume gets its infinities back.
  std::vector<cv::Mat> filled(planes);
  std::vector<cv::Mat> total(planes);
  for (int p = 0; p < planes; ++p) {
    filled[p].create(rows, cols, CV_32F);
    for (int y = 0; y < rows; ++y) {
      const float * src = volume[p].ptr<float>(y);
      float * dst = filled[p].ptr<float>(y);
      for (int x = 0; x < cols; ++x) {
        dst[x] = std::isfinite(src[x]) ? src[x] : 1e6f;
      }
    }
    total[p] = cv::Mat::zeros(rows, cols, CV_32F);
  }

  std::vector<cv::Mat> out(planes);
  for (int p = 0; p < planes; ++p) {
    out[p].create(rows, cols, CV_32F);
  }
  std::vector<float> previous(planes);

  // One pass: columns are the walk axis when `horizontal`, rows otherwise;
  // `reverse` runs against the axis. Each step pays its own cost plus the
  // cheapest continuation, minus the running floor that keeps sums bounded.
  auto walk = [&](bool horizontal, bool reverse) {
      const int length = horizontal ? cols : rows;
      const int breadth = horizontal ? rows : cols;
      auto index = [&](int j, int s, int & y, int & x) {
          const int along = reverse ? length - 1 - s : s;
          y = horizontal ? j : along;
          x = horizontal ? along : j;
        };
      int y;
      int x;
      for (int j = 0; j < breadth; ++j) {
        index(j, 0, y, x);
        for (int p = 0; p < planes; ++p) {
          out[p].at<float>(y, x) = filled[p].at<float>(y, x);
        }
      }
      for (int s = 1; s < length; ++s) {
        for (int j = 0; j < breadth; ++j) {
          int py;
          int px;
          index(j, s - 1, py, px);
          index(j, s, y, x);
          float floor = std::numeric_limits<float>::infinity();
          for (int p = 0; p < planes; ++p) {
            previous[p] = out[p].at<float>(py, px);
            floor = std::min(floor, previous[p]);
          }
          for (int p = 0; p < planes; ++p) {
            // Stay on the plane, jump anywhere for big_step, or slide one
            // rung (clamped at the ends, as numpy's edge padding does) for
            // small_step.
            float best = std::min(previous[p], floor + big);
            if (planes > 1) {
              const float up = previous[std::min(p + 1, planes - 1)];
              const float down = previous[std::max(p - 1, 0)];
              best = std::min(best, std::min(up, down) + small);
            }
            out[p].at<float>(y, x) = filled[p].at<float>(y, x) + best - floor;
          }
        }
      }
      for (int p = 0; p < planes; ++p) {
        total[p] += out[p];
      }
    };

  walk(true, false);
  walk(true, true);
  walk(false, false);
  walk(false, true);

  // The sum replaces the volume in place; cells nothing ever saw stay inf.
  for (int p = 0; p < planes; ++p) {
    for (int y = 0; y < rows; ++y) {
      const float * sum = total[p].ptr<float>(y);
      float * dst = volume[p].ptr<float>(y);
      for (int x = 0; x < cols; ++x) {
        dst[x] = std::isfinite(dst[x]) ? sum[x] : static_cast<float>(kInf);
      }
    }
  }
}

void Sweep::reduce_volume(
  const std::vector<cv::Mat> & volume, int road, cv::Mat & best,
  cv::Mat & best_cost, cv::Mat & road_cost, cv::Mat & second_cost) const
{
  // Parity target: the argmin / road_cost / shoulder block of plane_sweep.run.
  const int planes = static_cast<int>(volume.size());
  const int rows = volume[0].rows;
  const int cols = volume[0].cols;
  best.create(rows, cols, CV_32S);
  best_cost.create(rows, cols, CV_32F);
  road_cost.create(rows, cols, CV_32F);
  second_cost.create(rows, cols, CV_32F);
  for (int y = 0; y < rows; ++y) {
    int * best_row = best.ptr<int>(y);
    float * best_cost_row = best_cost.ptr<float>(y);
    float * road_row = road_cost.ptr<float>(y);
    float * second_row = second_cost.ptr<float>(y);
    for (int x = 0; x < cols; ++x) {
      // First minimum wins ties, as np.argmin does; an all-inf pixel keeps
      // index 0 the same way.
      int arg = 0;
      float lowest = volume[0].at<float>(y, x);
      for (int p = 1; p < planes; ++p) {
        const float value = volume[p].at<float>(y, x);
        if (value < lowest) {
          lowest = value;
          arg = p;
        }
      }
      best_row[x] = arg;
      best_cost_row[x] = lowest;
      // The best of the planes at or below the road, not the road plane
      // alone: road that sits low is still road.
      float under = volume[0].at<float>(y, x);
      for (int p = 1; p <= road; ++p) {
        under = std::min(under, volume[p].at<float>(y, x));
      }
      road_row[x] = under;
      // Second best outside the winner's +/-1 shoulder; when every plane sits
      // on the shoulder it stays infinite.
      float second = static_cast<float>(kInf);
      for (int p = 0; p < planes; ++p) {
        if (p < arg - 1 || p > arg + 1) {
          second = std::min(second, volume[p].at<float>(y, x));
        }
      }
      second_row[x] = second;
    }
  }
}

void Sweep::build_verdict(
  const cv::Mat & reference32, const cv::Mat & best, const cv::Mat & best_cost,
  const cv::Mat & road_cost, const cv::Mat & second_cost,
  const std::vector<double> & heights, int road,
  const Pose5 & reference_pose, const std::vector<Pose5> & source_poses,
  const cv::Mat & /*dot_normal*/, const Eigen::Vector3d & /*normal*/,
  Verdict & out) const
{
  const int planes = static_cast<int>(heights.size());

  // Featureless asphalt matches every plane equally well, so the winner
  // there is whichever hypothesis noise favoured. Variance over the match
  // window, against the squared contrast bar.
  const cv::Size box(settings_.window_x, settings_.window_y);
  cv::Mat mean;
  cv::Mat mean_sq;
  cv::boxFilter(reference32, mean, -1, box);
  cv::boxFilter(reference32.mul(reference32), mean_sq, -1, box);
  cv::Mat variance = mean_sq - mean.mul(mean);
  cv::Mat textured = variance >
    static_cast<float>(settings_.min_contrast * settings_.min_contrast);

  // What separates two hypotheses at a pixel is the part of the camera's
  // travel that runs across its line of sight, not the travel itself. Each
  // source's (x, y) step in the reference frame, then the component of the
  // largest step perpendicular to the pixel's unit ray.
  const double cr = std::cos(-reference_pose.yaw);
  const double sr = std::sin(-reference_pose.yaw);
  std::vector<Eigen::Vector2d> steps;
  steps.reserve(source_poses.size());
  for (const Pose5 & source : source_poses) {
    const double dx = source.x - reference_pose.x;
    const double dy = source.y - reference_pose.y;
    steps.emplace_back(cr * dx - sr * dy, sr * dx + cr * dy);
  }
  cv::Mat judgeable(height_, width_, CV_8U);
  for (int v = 0; v < height_; ++v) {
    const auto * rray = ray_base_.ptr<cv::Vec3d>(v);
    uint8_t * jd = judgeable.ptr<uint8_t>(v);
    for (int u = 0; u < width_; ++u) {
      const cv::Vec3d & r = rray[u];
      const double inv = 1.0 / std::sqrt(r[0] * r[0] + r[1] * r[1] + r[2] * r[2]);
      const double ux = r[0] * inv;
      const double uy = r[1] * inv;
      const double uz = r[2] * inv;
      double crosswise = 0.0;
      for (const Eigen::Vector2d & step : steps) {
        // The step has no z, but the residual after removing the along-ray
        // part does; all three components carry.
        const double along = ux * step.x() + uy * step.y();
        const double px = step.x() - along * ux;
        const double py = step.y() - along * uy;
        const double pz = -along * uz;
        crosswise = std::max(crosswise, std::sqrt(px * px + py * py + pz * pz));
      }
      jd[u] = crosswise >= settings_.min_crosswise ? 255 : 0;
    }
  }

  out.believed = cv::Mat::zeros(height_, width_, CV_8U);
  out.clear = cv::Mat::zeros(height_, width_, CV_8U);
  out.confidence.create(height_, width_, CV_32F);
  out.surface.create(height_, width_, CV_32F);
  const float low = static_cast<float>(settings_.obstacle_min_height);
  for (int v = 0; v < height_; ++v) {
    const int * bt = best.ptr<int>(v);
    const float * bc = best_cost.ptr<float>(v);
    const float * rc = road_cost.ptr<float>(v);
    const float * sc = second_cost.ptr<float>(v);
    const uint8_t * jd = judgeable.ptr<uint8_t>(v);
    const uint8_t * tx = textured.ptr<uint8_t>(v);
    uint8_t * bl = out.believed.ptr<uint8_t>(v);
    uint8_t * cl = out.clear.ptr<uint8_t>(v);
    float * cf = out.confidence.ptr<float>(v);
    float * sf = out.surface.ptr<float>(v);
    for (int u = 0; u < width_; ++u) {
      // The winning rung is the surface. The python --subplane parabola
      // reads the neighbouring rungs of the volume, which this interface
      // does not carry -- and it measured worse there (kerb verdicts 0.483
      // -> 0.454), so the rung stands.
      const float surface = static_cast<float>(heights[bt[u]]);
      sf[u] = surface;
      const bool finite = std::isfinite(bc[u]) && std::isfinite(rc[u]);
      // A winner at the top of the swept range is something taller than the
      // model pinned to the boundary, not a 2.5 m obstacle.
      const bool saturated = bt[u] >= planes - 1;
      bl[u] = (jd[u] && !saturated && tx[u] && finite &&
        bc[u] <= settings_.uniqueness * sc[u] &&
        surface >= low &&
        rc[u] - bc[u] > settings_.margin) ? 255 : 0;
      // How sure the pixel is, rather than whether it passed: how clearly
      // the winner beat the runner-up, times how good the match itself was.
      double decisive = 1.0 - bc[u] / std::max(sc[u], 1e-6f);
      decisive = std::isnan(decisive) ? 0.0 : std::clamp(decisive, 0.0, 1.0);
      const double quality = std::clamp(
        1.0 - bc[u] / std::max(settings_.vote_scale, 1e-6), 0.0, 1.0);
      cf[u] = static_cast<float>(decisive * quality);
      // Erasing needs a stricter test than warning: an absolute cost bar on
      // the road plane, a margin over the runner-up, and a winner at or
      // below the road rung. inf - inf reads NaN and fails, as it should.
      cl[u] = (jd[u] && tx[u] && finite &&
        bc[u] <= settings_.free_uniqueness * sc[u] &&
        rc[u] <= settings_.free_cost_max &&
        sc[u] - rc[u] >= settings_.free_margin &&
        bt[u] <= road) ? 255 : 0;
    }
  }

  // An obstacle is a surface, so it comes as a patch of pixels that agree;
  // a lone pixel that beat the road is what a per-corner method cannot
  // reject.
  if (settings_.min_blob > 0) {
    cv::Mat labels;
    cv::Mat stats;
    cv::Mat centroids;
    cv::connectedComponentsWithStats(out.believed, labels, stats, centroids, 8);
    for (int v = 0; v < height_; ++v) {
      const int * lb = labels.ptr<int>(v);
      uint8_t * bl = out.believed.ptr<uint8_t>(v);
      for (int u = 0; u < width_; ++u) {
        if (bl[u] && stats.at<int>(lb[u], cv::CC_STAT_AREA) < settings_.min_blob) {
          bl[u] = 0;
        }
      }
    }
  }
}

}  // namespace monoscale_sweep
