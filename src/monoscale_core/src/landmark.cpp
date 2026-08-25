#include "monoscale_core/landmark.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace monoscale
{

namespace
{

Eigen::Matrix3d turn_of(double yaw)
{
  const double c = std::cos(yaw);
  const double s = std::sin(yaw);
  Eigen::Matrix3d out = Eigen::Matrix3d::Identity();
  out(0, 0) = c; out(0, 1) = -s; out(1, 0) = s; out(1, 1) = c;
  return out;
}

double wrap(double angle)
{
  return std::remainder(angle, 2.0 * M_PI);
}

// Two directions across the ray, so a bearing residual is two numbers rather
// than three that cannot all be independent.
void frame_across(const Eigen::Vector3d & ray, Eigen::Vector3d & u, Eigen::Vector3d & v)
{
  const Eigen::Vector3d away =
    std::abs(ray.z()) < 0.9 ? Eigen::Vector3d::UnitZ() : Eigen::Vector3d::UnitX();
  u = ray.cross(away).normalized();
  v = ray.cross(u).normalized();
}

}  // namespace

LandmarkFilter::LandmarkFilter(const LandmarkSettings & settings)
: settings_(settings)
{
  covariance_ = Eigen::MatrixXd::Zero(3, 3);
  // The pose starts known: it defines the frame everything else is written in.
  landmarks_.resize(0);
}

void LandmarkFilter::predict(const Eigen::Vector2d & hop_body, double yaw_delta)
{
  const double yaw = pose_.z();
  const double c = std::cos(yaw);
  const double s = std::sin(yaw);
  pose_.x() += c * hop_body.x() - s * hop_body.y();
  pose_.y() += s * hop_body.x() + c * hop_body.y();
  pose_.z() = wrap(yaw + yaw_delta);

  // The step moves the pose and nothing else, so only the pose block gains
  // process noise -- but every cross term with a landmark has to be carried
  // through the same Jacobian, which for a planar step is the derivative of
  // the rotation by the heading.
  Eigen::Matrix3d jacobian = Eigen::Matrix3d::Identity();
  jacobian(0, 2) = -s * hop_body.x() - c * hop_body.y();
  jacobian(1, 2) = c * hop_body.x() - s * hop_body.y();

  const Eigen::Index n = size();
  covariance_.topLeftCorner<3, 3>() =
    jacobian * covariance_.topLeftCorner<3, 3>() * jacobian.transpose();
  if (n > 3) {
    covariance_.topRightCorner(3, n - 3) =
      jacobian * covariance_.topRightCorner(3, n - 3);
    covariance_.bottomLeftCorner(n - 3, 3) =
      covariance_.topRightCorner(3, n - 3).transpose();
  }
  Eigen::Matrix3d process = Eigen::Matrix3d::Zero();
  const double q = settings_.hop_process_noise_m * settings_.hop_process_noise_m;
  process(0, 0) = q;
  process(1, 1) = q;
  process(2, 2) = settings_.yaw_noise_rad * settings_.yaw_noise_rad;
  covariance_.topLeftCorner<3, 3>() += process;
}

void LandmarkFilter::world_of(
  const Sighting & sighting, Eigen::Vector3d & centre, Eigen::Vector3d & ray) const
{
  const Eigen::Matrix3d turn = turn_of(pose_.z());
  centre = Eigen::Vector3d(pose_.x(), pose_.y(), 0.0) + turn * sighting.mount;
  ray = turn * sighting.bearing;
}

double LandmarkFilter::contribution(const Entry & entry) const
{
  // How much of the pose's uncertainty this landmark is holding down. Not a
  // proxy for it -- the joint covariance is carried precisely so this can be
  // read off, and a landmark uncorrelated with the pose is earning nothing
  // however certain it is of its own position.
  const Eigen::Matrix3d own = covariance_.block<3, 3>(entry.at, entry.at);
  const Eigen::Matrix3d cross = covariance_.block<3, 3>(0, entry.at);
  Eigen::Matrix3d inverse;
  double determinant = 0.0;
  bool invertible = false;
  own.computeInverseAndDetWithCheck(inverse, determinant, invertible);
  if (!invertible || !(determinant > 0.0)) {
    return 0.0;
  }
  return (cross * inverse * cross.transpose()).trace();
}

std::optional<std::pair<int64_t, double>> LandmarkFilter::weakest() const
{
  std::optional<std::pair<int64_t, double>> worst;
  for (const auto & held : index_) {
    const double earning = contribution(held.second);
    if (!worst.has_value() || earning < worst->second) {
      worst = std::make_pair(held.first, earning);
    }
  }
  return worst;
}

bool LandmarkFilter::admit(
  int64_t identity, const Eigen::Vector3d & position, const Eigen::Matrix3d & block,
  const Eigen::Matrix3d & by_pose)
{
  if (index_.count(identity) > 0) {
    return false;
  }
  if (static_cast<int>(order_.size()) >= settings_.max_landmarks) {
    // Full. Either the newcomer is worth more than the worst seat, or it is
    // not; refusing on arrival order is what starved this filter.
    if (!settings_.evict_by_contribution || evictions_left_ <= 0) {
      return false;
    }
    // The newcomer's coupling is the pose error it inherits, which is what
    // `by_pose` carries: it knows about the pose exactly as much as it was
    // built from it.
    const Eigen::Matrix3d incoming = by_pose * covariance_.topLeftCorner<3, 3>();
    Eigen::Matrix3d inverse;
    double determinant = 0.0;
    bool invertible = false;
    (by_pose * covariance_.topLeftCorner<3, 3>() * by_pose.transpose() + block)
      .computeInverseAndDetWithCheck(inverse, determinant, invertible);
    if (!invertible || !(determinant > 0.0)) {
      return false;
    }
    const double earning = (incoming * inverse * incoming.transpose()).trace();
    const auto worst = weakest();
    if (!worst.has_value() || worst->second >= earning) {
      return false;
    }
    drop(worst->first);
    --evictions_left_;
    ++evicted_;
  }
  const Eigen::Index n = size();
  Eigen::MatrixXd grown = Eigen::MatrixXd::Zero(n + 3, n + 3);
  grown.topLeftCorner(n, n) = covariance_;
  // A new landmark is **not** independent of what is already here. Its position
  // was computed from the current pose, so it inherits that pose's error, and
  // through the pose it inherits the correlation with every landmark already
  // held. Writing zeros there tells the filter this is fresh information, and
  // it then counts the same error twice: the covariance collapses, the gate
  // closes, and the sightings stop being accepted -- measured, from 47 used per
  // frame down to 1 over eight hundred frames.
  //
  // This is the same mistake, one layer down, that the pooled-vote experiment
  // was measured to make: an anchor's error is correlated with the pose that
  // wrote it, and there has to be somewhere to say so. Here there is.
  const Eigen::MatrixXd cross = by_pose * covariance_.topRows(3);   // 3 x n
  grown.bottomRows<3>().leftCols(n) = cross;
  grown.topRows(n).rightCols<3>() = cross.transpose();
  grown.bottomRightCorner<3, 3>() =
    by_pose * covariance_.topLeftCorner<3, 3>() * by_pose.transpose() + block;
  covariance_ = grown;
  landmarks_.conservativeResize(landmarks_.size() + 3);
  landmarks_.tail<3>() = position;
  index_[identity] = Entry{n, seen_frame_, 0};
  order_.push_back(identity);
  ++initialised_;
  return true;
}

void LandmarkFilter::drop(int64_t identity)
{
  const auto found = index_.find(identity);
  if (found == index_.end()) {
    return;
  }
  const Eigen::Index at = found->second.at;
  const Eigen::Index n = size();
  const Eigen::Index tail = n - at - 3;
  if (tail > 0) {
    covariance_.block(at, 0, tail, n) = covariance_.block(at + 3, 0, tail, n).eval();
    covariance_.block(0, at, n, tail) = covariance_.block(0, at + 3, n, tail).eval();
    landmarks_.segment(at - 3, tail) = landmarks_.segment(at, tail).eval();
  }
  covariance_.conservativeResize(n - 3, n - 3);
  landmarks_.conservativeResize(landmarks_.size() - 3);
  order_.erase(std::find(order_.begin(), order_.end(), identity));
  index_.erase(found);
  for (auto & entry : index_) {
    if (entry.second.at > at) {
      entry.second.at -= 3;
    }
  }
}

int LandmarkFilter::observe(const std::vector<Sighting> & sightings, int64_t frame)
{
  seen_frame_ = frame;
  evictions_left_ = settings_.evict_per_frame;
  // Everything this frame saw, in one update.
  //
  // Sequentially is cheaper and wrong in a way that matters: each update moves
  // the point the next one linearises about, so the answer depends on the
  // order the features happened to arrive in. Batched and relinearised a few
  // times, it does not.
  struct Row
  {
    Eigen::Index at;                 // landmark's row in the state
    Eigen::Vector3d bearing;         // measured, body frame
    Eigen::Vector3d mount;
    Eigen::Vector3d across_u;
    Eigen::Vector3d across_v;
  };
  std::vector<Row> rows;
  rows.reserve(sightings.size());

  for (const auto & sighting : sightings) {
    // A seat in the state is for what lasts. The plane answers for this one
    // again next frame from one bearing, so keeping it costs the scarcest
    // resource here and buys a number that arrives free anyway.
    if (!settings_.ground_in_state && sighting.ground_range.has_value()) {
      continue;
    }
    const auto found = index_.find(sighting.identity);
    if (found != index_.end()) {
      found->second.seen = frame;
      ++found->second.sightings;
      Row row;
      row.at = found->second.at;
      row.bearing = sighting.bearing.normalized();
      row.mount = sighting.mount;
      frame_across(row.bearing, row.across_u, row.across_v);
      rows.push_back(row);
      continue;
    }

    // Not held yet. The plane can hand it a range now; without one it waits
    // until its own bearings have spread far enough to carry a position, which
    // is the same test applied at a different moment rather than a different
    // kind of feature.
    Eigen::Vector3d centre;
    Eigen::Vector3d ray;
    world_of(sighting, centre, ray);
    if (sighting.ground_range.has_value() && settings_.range_from_plane) {
      const double range = *sighting.ground_range;
      if (range > 0.1) {
        // Along the ray the plane sets it; across, the bearing does. The along
        // term is the plane geometry: a bearing error at range R over a camera
        // height h moves the intersection by (R^2 + h^2)/h.
        const double height = std::max(std::abs(sighting.mount.z()), 1e-3);
        const double along = (range * range + height * height) / height *
          settings_.bearing_noise_rad;
        const double across = range * settings_.bearing_noise_rad;
        Eigen::Vector3d u;
        Eigen::Vector3d v;
        frame_across(ray, u, v);
        const Eigen::Matrix3d block =
          along * along * ray * ray.transpose() +
          across * across * (u * u.transpose() + v * v.transpose());
        // L = p + R(yaw) (mount + range * bearing), so the pose enters through
        // its position directly and through the heading that turns the offset.
        const Eigen::Vector3d offset = sighting.mount + range * sighting.bearing;
        Eigen::Matrix3d by_pose = Eigen::Matrix3d::Zero();
        by_pose(0, 0) = 1.0;
        by_pose(1, 1) = 1.0;
        const double c = std::cos(pose_.z());
        const double s = std::sin(pose_.z());
        by_pose(0, 2) = -s * offset.x() - c * offset.y();
        by_pose(1, 2) = c * offset.x() - s * offset.y();
        admit(sighting.identity, centre + range * ray, block, by_pose);
        continue;
      }
    }
    auto & waiting = pending_[sighting.identity];
    waiting.seen = frame;
    waiting.centres.push_back(centre);
    waiting.bearings.push_back(ray);
    if (waiting.centres.size() > 24) {
      waiting.centres.erase(waiting.centres.begin());
      waiting.bearings.erase(waiting.bearings.begin());
    }
    if (static_cast<int>(waiting.centres.size()) < settings_.initialise_min_views) {
      continue;
    }
    double widest = 1.0;
    for (size_t a = 0; a < waiting.bearings.size(); ++a) {
      for (size_t b = a + 1; b < waiting.bearings.size(); ++b) {
        widest = std::min(widest, waiting.bearings[a].dot(waiting.bearings[b]));
      }
    }
    if (widest > std::cos(settings_.initialise_parallax_rad)) {
      continue;
    }
    Eigen::Matrix3d normal = Eigen::Matrix3d::Zero();
    Eigen::Vector3d moment = Eigen::Vector3d::Zero();
    for (size_t n = 0; n < waiting.centres.size(); ++n) {
      const Eigen::Matrix3d part =
        Eigen::Matrix3d::Identity() - waiting.bearings[n] * waiting.bearings[n].transpose();
      normal += part;
      moment += part * waiting.centres[n];
    }
    Eigen::Matrix3d inverse;
    double determinant = 0.0;
    bool invertible = false;
    normal.computeInverseAndDetWithCheck(inverse, determinant, invertible);
    if (!invertible || determinant < 1e-6) {
      continue;
    }
    const Eigen::Vector3d point = inverse * moment;
    if (!point.allFinite() || (point - centre).dot(ray) <= 0.0) {
      continue;
    }
    // The triangulation's own covariance, scaled by what one bearing is worth.
    const Eigen::Matrix3d block =
      inverse * (settings_.bearing_noise_rad * settings_.bearing_noise_rad *
      (point - centre).squaredNorm());
    // Triangulated from several past poses rather than this one, so this is
    // the standard approximation: charge it to the pose it is admitted under.
    const Eigen::Vector3d offset = point - Eigen::Vector3d(pose_.x(), pose_.y(), 0.0);
    Eigen::Matrix3d by_pose = Eigen::Matrix3d::Zero();
    by_pose(0, 0) = 1.0;
    by_pose(1, 1) = 1.0;
    by_pose(0, 2) = -offset.y();
    by_pose(1, 2) = offset.x();
    if (admit(sighting.identity, point, block, by_pose)) {
      pending_.erase(sighting.identity);
    }
  }

  if (rows.empty()) {
    return 0;
  }

  const double noise = settings_.bearing_noise_rad * settings_.bearing_noise_rad;
  const double reject = settings_.reject_chi_square;
  int used = 0;
  for (int pass = 0; pass < std::max(settings_.iterations, 1); ++pass) {
    const Eigen::Index n = size();
    const Eigen::Matrix3d turn = turn_of(pose_.z());
    Eigen::Matrix3d turn_rate = Eigen::Matrix3d::Zero();
    const double c = std::cos(pose_.z());
    const double s = std::sin(pose_.z());
    turn_rate(0, 0) = -s; turn_rate(0, 1) = -c;
    turn_rate(1, 0) = c;  turn_rate(1, 1) = -s;
    const Eigen::Vector2d here(pose_.x(), pose_.y());

    std::vector<Eigen::Index> keep;
    std::vector<Eigen::Matrix<double, 2, 6>> blocks;
    std::vector<Eigen::Vector2d> residuals;
    keep.reserve(rows.size());
    for (const auto & row : rows) {
      // `at` indexes the state, which the pose occupies the first three of.
      const Eigen::Vector3d world = landmarks_.segment<3>(row.at - 3);
      Eigen::Vector3d offset = world;
      offset.head<2>() -= here;
      // The landmark seen from the body, before the mount is taken off.
      const Eigen::Vector3d turned = turn.transpose() * offset;
      const Eigen::Vector3d direction = turned - row.mount;
      const double range = direction.norm();
      if (!(range > 0.1)) {
        continue;
      }
      const Eigen::Vector3d unit = direction / range;
      const Eigen::Vector2d residual(
        row.across_u.dot(unit), row.across_v.dot(unit));
      // d(unit)/d(direction), then the chain onto the pose and the landmark.
      const Eigen::Matrix3d slope =
        (Eigen::Matrix3d::Identity() - unit * unit.transpose()) / range;
      Eigen::Matrix<double, 3, 3> by_pose;
      by_pose.leftCols<2>() = -turn.transpose().leftCols<2>();
      by_pose.col(2) = turn_rate.transpose() * offset;
      Eigen::Matrix<double, 2, 6> block;
      Eigen::Matrix<double, 2, 3> across;
      across.row(0) = row.across_u.transpose();
      across.row(1) = row.across_v.transpose();
      block.leftCols<3>() = across * slope * by_pose;
      block.rightCols<3>() = across * slope * turn.transpose();
      // Gated on what the innovation is *expected* to be, not on the
      // measurement noise alone. A pose that has just been carried forward on
      // a guess is allowed a large residual; a settled one is not. Gating on
      // the noise by itself rejects every sighting whenever the pose is
      // uncertain, which is exactly when the sightings are needed.
      Eigen::Matrix<double, 2, 6> spread;
      spread.leftCols<3>() =
        block.leftCols<3>() * covariance_.topLeftCorner<3, 3>() +
        block.rightCols<3>() * covariance_.block<3, 3>(row.at, 0);
      spread.rightCols<3>() =
        block.leftCols<3>() * covariance_.block<3, 3>(0, row.at) +
        block.rightCols<3>() * covariance_.block<3, 3>(row.at, row.at);
      Eigen::Matrix2d innovation_here =
        spread.leftCols<3>() * block.leftCols<3>().transpose() +
        spread.rightCols<3>() * block.rightCols<3>().transpose();
      innovation_here.diagonal().array() += noise;
      const double distance =
        residual.transpose() * innovation_here.inverse() * residual;
      if (!(distance < reject)) {
        continue;
      }
      keep.push_back(row.at);
      blocks.push_back(block);
      residuals.push_back(residual);
    }
    if (keep.empty()) {
      return 0;
    }
    used = static_cast<int>(keep.size());

    // Stacked, with only the six columns each row actually touches.
    const Eigen::Index m = 2 * static_cast<Eigen::Index>(keep.size());
    Eigen::MatrixXd carried(m, n);
    carried.setZero();
    Eigen::VectorXd error(m);
    for (size_t i = 0; i < keep.size(); ++i) {
      const Eigen::Index at = static_cast<Eigen::Index>(2 * i);
      carried.block<2, 3>(at, 0) = blocks[i].leftCols<3>();
      carried.block<2, 3>(at, keep[i]) = blocks[i].rightCols<3>();
      error.segment<2>(at) = -residuals[i];
    }
    Eigen::MatrixXd carried_p = carried * covariance_;
    Eigen::MatrixXd innovation = carried_p * carried.transpose();
    innovation.diagonal().array() += noise;
    const Eigen::MatrixXd gain =
      innovation.ldlt().solve(carried_p).transpose();
    const Eigen::VectorXd step = gain * error;
    pose_ += step.head<3>();
    pose_.z() = wrap(pose_.z());
    if (n > 3) {
      landmarks_ += step.tail(n - 3);
    }
    covariance_ -= gain * carried_p;
    covariance_ = 0.5 * (covariance_ + covariance_.transpose()).eval();
  }
  updated_ += used;
  rejected_ += static_cast<int64_t>(rows.size()) - used;
  return used;
}

void LandmarkFilter::retire(int64_t frame)
{
  std::vector<int64_t> gone;
  for (const auto & entry : index_) {
    if (frame - entry.second.seen > settings_.retire_unseen_frames) {
      gone.push_back(entry.first);
    }
  }
  for (const int64_t identity : gone) {
    drop(identity);
  }
  for (auto it = pending_.begin(); it != pending_.end(); ) {
    it = frame - it->second.seen > settings_.retire_unseen_frames
      ? pending_.erase(it) : std::next(it);
  }
}

}  // namespace monoscale
