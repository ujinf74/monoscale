// Occupied votes, the margin-tiered free carve, the two-slab column test, and
// the publish chain. Ported from plane_sweep.py's mark_occupied / carve_road /
// slab_carve body and the prob/slab/erode publish functions, at the adopted
// operating point (subplane off for placement, no ray-votes, no vertical
// family, no shadow projection -- all measured to lose).

#include "monoscale_sweep/sweep.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include <opencv2/imgproc.hpp>

namespace monoscale_sweep
{

namespace
{

// Subcell occupied vote: split each point's weight across the four cells it
// falls between, so a surface near a boundary registers between the cells
// rather than a whole cell out. Matches mark_occupied(subcell=True).
void mark_occupied_subcell(
  CameraGrid & grid, const SweepSettings & s,
  const std::vector<double> & wx, const std::vector<double> & wy,
  const std::vector<double> & weight)
{
  const int W = s.grid_width;
  const int H = s.grid_height;
  cv::Mat votes = cv::Mat::zeros(H, W, CV_32F);
  for (size_t i = 0; i < wx.size(); ++i) {
    const double cx = (wx[i] - s.origin_x) / s.resolution - 0.5;
    const double cy = (wy[i] - s.origin_y) / s.resolution - 0.5;
    const int left = static_cast<int>(std::floor(cx));
    const int bottom = static_cast<int>(std::floor(cy));
    const float fx = static_cast<float>(cx - left);
    const float fy = static_cast<float>(cy - bottom);
    const float held = static_cast<float>(weight[i]);
    const int cols[2] = {left, left + 1};
    const int rows[2] = {bottom, bottom + 1};
    const float sx[2] = {1.0f - fx, fx};
    const float sy[2] = {1.0f - fy, fy};
    for (int a = 0; a < 2; ++a) {
      for (int b = 0; b < 2; ++b) {
        const float share = held * sx[a] * sy[b];
        if (share <= 0.0f) {continue;}
        const int c = cols[a];
        const int r = rows[b];
        if (c < 0 || c >= W || r < 0 || r >= H) {continue;}
        votes.at<float>(r, c) += share;
      }
    }
  }
  // Vote cap: what one keyframe may say about one cell, so the grid counts
  // observations and not pixels.
  for (int r = 0; r < H; ++r) {
    const float * v = votes.ptr<float>(r);
    float * lo = grid.log_odds.ptr<float>(r);
    uint8_t * ob = grid.observed.ptr<uint8_t>(r);
    for (int c = 0; c < W; ++c) {
      float w = v[c];
      if (w <= 0.0f) {continue;}
      if (s.vote_cap > 0.0) {w = std::min(w, static_cast<float>(s.vote_cap));}
      float gain = static_cast<float>(s.occupied_update) * w;
      // A cleared cell needs more than one frame's word to become occupied.
      if (s.free_to_occupied != 1.0 && lo[c] < 0.0f) {
        gain *= static_cast<float>(s.free_to_occupied);
      }
      lo[c] = std::min(4.0f, lo[c] + gain);
      ob[c] = 1;
    }
  }
}

// One ray carved: every cell it crosses drops by free_update, once per ray,
// but a ray that has already met an occupied cell (belief past the occlusion
// bar) says nothing beyond it. Matches mark_free's carve_rays with
// respect_occupied and the per-ray dedup (bincount over unique cells).
void carve_ray(
  CameraGrid & grid, const SweepSettings & s, double ex, double ey,
  double wx, double wy, const cv::Mat & stop, cv::Mat & hits, double free_update)
{
  const int W = s.grid_width;
  const int H = s.grid_height;
  const double dx = wx - ex;
  const double dy = wy - ey;
  const double reach = std::hypot(dx, dy);
  if (reach < 1e-9) {return;}
  const double spacing = 0.05;
  const int steps = static_cast<int>(std::ceil(reach / spacing)) + 1;
  int last_c = -1;
  int last_r = -1;
  for (int t = 0; t < steps; ++t) {
    const double frac = std::min(1.0, (t * spacing) / reach);
    const double px = ex + frac * dx;
    const double py = ey + frac * dy;
    const int c = static_cast<int>(std::floor((px - s.origin_x) / s.resolution));
    const int r = static_cast<int>(std::floor((py - s.origin_y) / s.resolution));
    if (c < 0 || c >= W || r < 0 || r >= H) {continue;}
    if (c == last_c && r == last_r) {continue;}
    last_c = c;
    last_r = r;
    if (!stop.empty() && stop.at<uint8_t>(r, c)) {break;}
    hits.at<float>(r, c) += 1.0f;
  }
}

}  // namespace

void Sweep::integrate(
  const Pose5 & reference_pose, const std::vector<double> & heights, int road,
  const cv::Mat & best, const cv::Mat & best_cost, const cv::Mat & second_cost,
  const Verdict & verdict, CameraGrid & grid) const
{
  (void)best_cost;
  (void)second_cost;
  const SweepSettings & s = settings_;
  const Eigen::Matrix3d world = attitude(reference_pose);
  const Eigen::Vector3d & eye_base = lens_.translation_base_from_camera;
  const Eigen::Vector3d eye = world * eye_base;
  const double ex = reference_pose.x + eye.x();
  const double ey = reference_pose.y + eye.y();
  const double ez = eye.z();

  // --- Occupied placements: believed pixels, put where their ray meets their
  // own swept height, world frame. Soft-voted by confidence, far-weighted.
  std::vector<double> ox;
  std::vector<double> oy;
  std::vector<double> ow;
  const int stride = s.pixel_stride;
  for (int v = 0; v < height_; v += stride) {
    const uint8_t * bel = verdict.believed.ptr<uint8_t>(v);
    const float * srf = verdict.surface.ptr<float>(v);
    const float * conf = verdict.confidence.ptr<float>(v);
    const auto * rray = ray_base_.ptr<cv::Vec3d>(v);
    for (int u = 0; u < width_; u += stride) {
      if (!bel[u]) {continue;}
      const cv::Vec3d & rb = rray[u];
      const Eigen::Vector3d ray_world = world * Eigen::Vector3d(rb[0], rb[1], rb[2]);
      const double height = srf[u];
      const double lam = (height - ez) / ray_world.z();
      if (!std::isfinite(lam) || lam <= 0.0) {continue;}
      const double px = eye.x() + lam * ray_world.x();
      const double py = eye.y() + lam * ray_world.y();
      const double reach = std::hypot(px - eye.x(), py - eye.y());
      if (reach > s.max_range) {continue;}
      double weight = conf[u];
      // Soft cap on range: a placement past obstacle_range still votes, at
      // far_weight; a real barrier is re-placed close every keyframe and
      // clears the bar on accumulated votes, a halo never does.
      if (s.obstacle_range > 0.0 && reach > s.obstacle_range) {
        weight *= s.far_weight;
      }
      if (weight <= 0.0) {continue;}
      ox.push_back(reference_pose.x + px);
      oy.push_back(reference_pose.y + py);
      ow.push_back(weight);
    }
  }
  if (!ox.empty()) {
    mark_occupied_subcell(grid, s, ox, oy, ow);
  }

  // The occlusion barrier: cells whose belief is past the bar block carving.
  cv::Mat stop;
  if (s.occlusion_aware > 0.0) {
    const double bar = std::log(s.occlusion_aware / (1.0 - s.occlusion_aware));
    stop = cv::Mat::zeros(s.grid_height, s.grid_width, CV_8U);
    for (int r = 0; r < s.grid_height; ++r) {
      const uint8_t * ob = grid.observed.ptr<uint8_t>(r);
      const float * lo = grid.log_odds.ptr<float>(r);
      uint8_t * st = stop.ptr<uint8_t>(r);
      for (int c = 0; c < s.grid_width; ++c) {
        st[c] = (ob[c] && lo[c] > bar) ? 1 : 0;
      }
    }
  }

  // --- Free carve: the clear pixels, their ground point, and the two-tier
  // margin weighting. Each pixel's road-plane intersection is the ray end;
  // the margin (second - road cost) sorts the confident half from the tail.
  std::vector<double> fx;
  std::vector<double> fy;
  std::vector<double> fz;   // the swept height, for the slab column test
  std::vector<double> fm;   // margin
  for (int v = 0; v < height_; v += stride) {
    const uint8_t * clr = verdict.clear.ptr<uint8_t>(v);
    const float * srf = verdict.surface.ptr<float>(v);
    const auto * rray = ray_base_.ptr<cv::Vec3d>(v);
    for (int u = 0; u < width_; u += stride) {
      if (!clr[u]) {continue;}
      const cv::Vec3d & rb = rray[u];
      const Eigen::Vector3d ray_world = world * Eigen::Vector3d(rb[0], rb[1], rb[2]);
      // Carve to the road-plane intersection (z=0), matching carve_road's
      // free_lam = -eye_z / ray_z.
      const double lam = -ez / ray_world.z();
      if (!std::isfinite(lam) || lam <= 0.0) {continue;}
      const double px = eye.x() + lam * ray_world.x();
      const double py = eye.y() + lam * ray_world.y();
      const double reach = std::hypot(px - eye.x(), py - eye.y());
      if (reach > s.max_range) {continue;}
      fx.push_back(reference_pose.x + px);
      fy.push_back(reference_pose.y + py);
      fz.push_back(srf[u]);
      fm.push_back(second_cost.at<float>(v, u) - best_cost.at<float>(v, u));
    }
  }

  auto carve_group = [&](const std::vector<size_t> & idx, double update) {
    if (idx.empty()) {return;}
    cv::Mat hits = cv::Mat::zeros(s.grid_height, s.grid_width, CV_32F);
    for (size_t k = 0; k < idx.size(); k += s.free_ray_stride) {
      carve_ray(grid, s, ex, ey, fx[idx[k]], fy[idx[k]], stop, hits, update);
    }
    for (int r = 0; r < s.grid_height; ++r) {
      float * h = hits.ptr<float>(r);
      float * lo = grid.log_odds.ptr<float>(r);
      uint8_t * ob = grid.observed.ptr<uint8_t>(r);
      for (int c = 0; c < s.grid_width; ++c) {
        float n = h[c];
        if (n <= 0.0f) {continue;}
        if (s.free_cap > 0.0) {n = std::min(n, static_cast<float>(s.free_cap));}
        lo[c] = std::max(static_cast<float>(s.log_odds_floor),
          lo[c] - static_cast<float>(update) * n);
        ob[c] = 1;
      }
    }
  };

  if (s.free_margin_weight > 0.0 && !fm.empty()) {
    // Split at the margin quantile: the confident head carves at boost, the
    // doubtful tail at weight, of the base free_update.
    std::vector<double> sorted = fm;
    std::nth_element(
      sorted.begin(),
      sorted.begin() + static_cast<long>(sorted.size() * s.free_margin_quantile),
      sorted.end());
    const double bar = sorted[static_cast<size_t>(sorted.size() * s.free_margin_quantile)];
    std::vector<size_t> hi;
    std::vector<size_t> lo;
    for (size_t i = 0; i < fm.size(); ++i) {
      (fm[i] >= bar ? hi : lo).push_back(i);
    }
    carve_group(hi, s.free_update * s.free_margin_boost);
    carve_group(lo, s.free_update * s.free_margin_weight);
  } else {
    std::vector<size_t> all(fx.size());
    for (size_t i = 0; i < all.size(); ++i) {all[i] = i;}
    carve_group(all, s.free_update);
  }

  // --- Two-slab free evidence: sample each carved ray, split by height at
  // slab_split, count clears per slab. The column test at publish keeps a car
  // from reading free without the car being detected.
  if (s.slab_carve) {
    for (size_t i = 0; i < fx.size(); i += s.free_ray_stride) {
      const double dx = fx[i] - ex;
      const double dy = fy[i] - ey;
      for (int t = 0; t < s.slab_samples; ++t) {
        const double frac = 0.05 + (0.98 - 0.05) * t / (s.slab_samples - 1);
        const double px = ex + frac * dx;
        const double py = ey + frac * dy;
        const double pz = ez * (1.0 - frac);
        if (pz <= 0.03) {continue;}
        const int c = static_cast<int>(std::floor((px - s.origin_x) / s.resolution));
        const int r = static_cast<int>(std::floor((py - s.origin_y) / s.resolution));
        if (c < 0 || c >= s.grid_width || r < 0 || r >= s.grid_height) {continue;}
        const int band = pz < s.slab_split ? 0 : 1;
        int16_t & cell = grid.slab_free[band].at<int16_t>(r, c);
        if (cell < 32000) {cell += 1;}
      }
    }
  }
  (void)heights;
  (void)road;
  (void)best;
}

cv::Mat publish(const SweepSettings & s, std::vector<CameraGrid *> grids)
{
  const int W = s.grid_width;
  const int H = s.grid_height;

  // Per-cell slab pass/fail, pooled over cameras.
  cv::Mat low_ok = cv::Mat::zeros(H, W, CV_8U);
  cv::Mat high_ok = cv::Mat::zeros(H, W, CV_8U);
  cv::Mat occupied_any = cv::Mat::zeros(H, W, CV_8U);
  if (s.slab_carve) {
    for (auto * g : grids) {
      for (int r = 0; r < H; ++r) {
        const int16_t * l = g->slab_free[0].ptr<int16_t>(r);
        const int16_t * h = g->slab_free[1].ptr<int16_t>(r);
        const float * lo = g->log_odds.ptr<float>(r);
        uint8_t * lok = low_ok.ptr<uint8_t>(r);
        uint8_t * hok = high_ok.ptr<uint8_t>(r);
        uint8_t * oc = occupied_any.ptr<uint8_t>(r);
        for (int c = 0; c < W; ++c) {
          if (l[c] >= s.slab_min) {lok[c] = 1;}
          if (h[c] >= s.slab_min) {hok[c] = 1;}
          if (lo[c] > 0.0f) {oc[c] = 1;}
        }
      }
    }
  }

  // neutralise_slab: reset the failing cells' free belief to neutral, gated to
  // a local radius around occupancy so open road (which rarely collects upper
  // slab sweeps) is left alone.
  cv::Mat failing;
  if (s.slab_carve) {
    failing = cv::Mat::zeros(H, W, CV_8U);
    for (int r = 0; r < H; ++r) {
      const uint8_t * lok = low_ok.ptr<uint8_t>(r);
      const uint8_t * hok = high_ok.ptr<uint8_t>(r);
      uint8_t * f = failing.ptr<uint8_t>(r);
      for (int c = 0; c < W; ++c) {f[c] = (lok[c] && hok[c]) ? 0 : 1;}
    }
    if (s.slab_local > 0.0) {
      cv::Mat notocc;
      cv::bitwise_not(occupied_any * 255, notocc);
      cv::Mat dist;
      cv::distanceTransform((occupied_any == 0), dist, cv::DIST_L2, 5);
      dist *= s.resolution;
      for (int r = 0; r < H; ++r) {
        uint8_t * f = failing.ptr<uint8_t>(r);
        const float * d = dist.ptr<float>(r);
        for (int c = 0; c < W; ++c) {
          if (d[c] > s.slab_local) {f[c] = 0;}
        }
      }
    }
    for (auto * g : grids) {
      for (int r = 0; r < H; ++r) {
        const uint8_t * f = failing.ptr<uint8_t>(r);
        float * lo = g->log_odds.ptr<float>(r);
        for (int c = 0; c < W; ++c) {
          if (f[c] && lo[c] < 0.0f) {lo[c] = 0.0f;}
        }
      }
    }
  }

  // message_values per camera, then union: occupied where any camera is
  // occupied, free where any is free.
  cv::Mat values(H, W, CV_8S, cv::Scalar(-1));
  cv::Mat occupied = cv::Mat::zeros(H, W, CV_8U);
  cv::Mat freed = cv::Mat::zeros(H, W, CV_8U);
  for (auto * g : grids) {
    for (int r = 0; r < H; ++r) {
      const uint8_t * ob = g->observed.ptr<uint8_t>(r);
      const float * lo = g->log_odds.ptr<float>(r);
      uint8_t * occ = occupied.ptr<uint8_t>(r);
      uint8_t * fr = freed.ptr<uint8_t>(r);
      for (int c = 0; c < W; ++c) {
        if (!ob[c]) {continue;}
        const double p = 1.0 / (1.0 + std::exp(-lo[c]));
        if (p < s.free_probability) {fr[c] = 1;} else if (p > s.occupied_probability) {
          occ[c] = 1;
        }
      }
    }
  }
  for (int r = 0; r < H; ++r) {
    int8_t * val = values.ptr<int8_t>(r);
    const uint8_t * occ = occupied.ptr<uint8_t>(r);
    const uint8_t * fr = freed.ptr<uint8_t>(r);
    for (int c = 0; c < W; ++c) {
      if (fr[c]) {val[c] = 0;}
      if (occ[c]) {val[c] = 100;}
    }
  }

  // slab_gate: free only where both slabs held clear (local to occupancy).
  if (s.slab_carve && !failing.empty()) {
    for (int r = 0; r < H; ++r) {
      int8_t * val = values.ptr<int8_t>(r);
      const uint8_t * f = failing.ptr<uint8_t>(r);
      for (int c = 0; c < W; ++c) {
        if (val[c] == 0 && f[c]) {val[c] = -1;}
      }
    }
  }

  // free_erode: shave the free rind that the loose carve walks onto the
  // pavement past the kerb. One cell takes most of the over-claim.
  if (s.free_erode > 0) {
    cv::Mat freemask = (values == 0);
    const int radius = s.free_erode;
    cv::Mat kernel = cv::getStructuringElement(
      cv::MORPH_ELLIPSE, cv::Size(2 * radius + 1, 2 * radius + 1));
    cv::Mat kept;
    cv::erode(freemask, kept, kernel);
    for (int r = 0; r < H; ++r) {
      int8_t * val = values.ptr<int8_t>(r);
      const uint8_t * fm = freemask.ptr<uint8_t>(r);
      const uint8_t * kp = kept.ptr<uint8_t>(r);
      for (int c = 0; c < W; ++c) {
        if (fm[c] && !kp[c]) {val[c] = 50;}
      }
    }
  }
  return values;
}

}  // namespace monoscale_sweep
