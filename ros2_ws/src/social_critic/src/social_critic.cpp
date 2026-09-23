#include "social_critic/social_critic.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>

#include "tf2/utils.h"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"

namespace mppi::critics
{

void SocialCritic::initialize()
{
  auto node = parent_.lock();
  logger_ = node->get_logger();

  auto getParam = parameters_handler_->getParamGetter(name_);
  getParam(enabled_, "enabled", true);
  getParam(social_distance_, "social_distance", 0.94f);
  getParam(weight_, "cost_weight", 40.0f);
  getParam(critical_distance_, "critical_distance", 0.35f);
  getParam(collision_cost_, "collision_cost", 10000.0f);
  getParam(cost_power_, "cost_power", 1);
  getParam(use_prediction_, "use_prediction", true);
  getParam(prediction_weight_, "prediction_weight", 0.5f);
  getParam(time_aware_, "time_aware", true);
  getParam(max_prediction_time_, "max_prediction_time", 3.0f);
  getParam(trajectory_point_step_, "trajectory_point_step", 4);
  getParam(track_timeout_, "track_timeout", 0.5);
  getParam(max_coast_time_, "max_coast_time", 2.5);
  getParam(coast_weight_, "coast_weight", 0.7f);
  getParam(min_coast_speed_, "min_coast_speed", 0.15f);
  getParam(max_coast_speed_, "max_coast_speed", 2.5f);
  getParam(person_frame_, "person_frame", std::string("map"));
  getParam(topic_, "topic", std::string("/predicted_person_positions"));

  // Reuse the costmap's TF buffer rather than starting a second
  // listener inside controller_server.
  tf_buffer_ = costmap_ros_->getTfBuffer();

  sub_ = node->create_subscription<std_msgs::msg::String>(
    topic_, rclcpp::QoS(10),
    std::bind(&SocialCritic::positionsCallback, this, std::placeholders::_1));

  RCLCPP_INFO(
    logger_,
    "SocialCritic: social_distance=%.2f m, weight=%.1f, critical=%.2f m, "
    "point_step=%d, time_aware=%s (horizon %.1f s), topic=%s",
    social_distance_, weight_, critical_distance_,
    trajectory_point_step_, time_aware_ ? "on" : "off",
    max_prediction_time_, topic_.c_str());
}

// Message format produced by human_kf_predictor and consumed by
// predicted_person_cloud_node — comma separated, field indices matched
// to that node so both stay in sync:
//   [0] track_id  [2] cur_x  [3] cur_y  [4] vx  [5] vy
//   [6] pred_x    [7] pred_y [9] rotation_gated flag
void SocialCritic::positionsCallback(const std_msgs::msg::String::SharedPtr msg)
{
  std::vector<std::string> parts;
  std::stringstream ss(msg->data);
  std::string item;
  while (std::getline(ss, item, ',')) {
    parts.push_back(item);
  }
  if (parts.size() < 9) {
    return;
  }

  auto node = parent_.lock();
  if (!node) {
    return;
  }

  PersonState p;
  try {
    p.track_id = static_cast<int>(std::stod(parts[0]));
    p.x = std::stod(parts[2]);
    p.y = std::stod(parts[3]);
    p.vx = std::stod(parts[4]);
    p.vy = std::stod(parts[5]);
    p.speed = std::hypot(p.vx, p.vy);
    p.pred_x = std::stod(parts[6]);
    p.pred_y = std::stod(parts[7]);
  } catch (const std::exception &) {
    return;
  }
  p.rotation_gated = (parts.size() > 9 && parts[9] == "1");
  p.coastable = (p.track_id >= 0);
  p.last_seen = node->now();

  std::lock_guard<std::mutex> lock(people_mutex_);

  auto it = std::find_if(
    people_.begin(), people_.end(),
    [&p](const PersonState & q) { return q.track_id == p.track_id; });

  if (it != people_.end()) {
    // Preserve the last good velocity across a frame the tracker could
    // not associate, so re-acquisition does not reset the motion model.
    if (!p.coastable && it->coastable) {
      p.vx = it->vx;
      p.vy = it->vy;
      p.speed = it->speed;
    }
    *it = p;
  } else {
    people_.push_back(p);
  }

  // An unassociated detection that sits on top of an identified track is
  // the same person seen twice. Left in place it becomes a second target
  // and biases the penalty toward whichever copy is more wrong.
  if (p.coastable) {
    people_.erase(
      std::remove_if(
        people_.begin(), people_.end(),
        [&p](const PersonState & q) {
          return q.track_id < 0 &&
          std::hypot(q.x - p.x, q.y - p.y) < 1.0;
        }),
      people_.end());
  }
}

std::vector<Target> SocialCritic::collectTargets()
{
  std::vector<Target> out;

  std::vector<PersonState> snapshot;
  {
    std::lock_guard<std::mutex> lock(people_mutex_);
    snapshot = people_;
  }
  if (snapshot.empty()) {
    return out;
  }

  auto node = parent_.lock();
  if (!node) {
    return out;
  }
  const rclcpp::Time now = node->now();

  const std::string costmap_frame = costmap_ros_->getGlobalFrameID();

  // Person positions are published in map; the local costmap (and hence
  // the MPPI trajectories) live in odom. Skipping this transform is the
  // classic way to get a critic that "works" until AMCL applies its
  // first correction.
  geometry_msgs::msg::TransformStamped tf;
  const bool need_tf = (costmap_frame != person_frame_);
  if (need_tf) {
    try {
      tf = tf_buffer_->lookupTransform(
        costmap_frame, person_frame_, tf2::TimePointZero,
        tf2::durationFromSec(0.05));
    } catch (const tf2::TransformException & ex) {
      RCLCPP_WARN_THROTTLE(
        logger_, *node->get_clock(), 2000,
        "SocialCritic: %s -> %s unavailable (%s); skipping this cycle",
        person_frame_.c_str(), costmap_frame.c_str(), ex.what());
      return out;
    }
  }

  const double tx = need_tf ? tf.transform.translation.x : 0.0;
  const double ty = need_tf ? tf.transform.translation.y : 0.0;
  const double yaw = need_tf ? tf2::getYaw(tf.transform.rotation) : 0.0;
  const double c = std::cos(yaw);
  const double s = std::sin(yaw);

  auto toCostmap = [&](double mx, double my, double mvx, double mvy, float scale) {
      Target t;
      t.x = static_cast<float>(tx + c * mx - s * my);
      t.y = static_cast<float>(ty + s * mx + c * my);
      // Velocity is a free vector: rotate, do not translate.
      t.vx = static_cast<float>(c * mvx - s * mvy);
      t.vy = static_cast<float>(s * mvx + c * mvy);
      t.weight_scale = scale;
      return t;
    };

  for (const auto & p : snapshot) {
    const double age = (now - p.last_seen).seconds();

    if (age <= track_timeout_) {
      // Fresh observation: penalise the reported position, carrying the
      // velocity so score() can propagate it along the rollout.
      out.push_back(toCostmap(p.x, p.y, p.vx, p.vy, 1.0f));

      // The KF's 1 s prediction is only worth adding as a separate
      // target when the critic is NOT propagating targets itself.
      // Otherwise it is the same information counted twice, at a
      // horizon the propagation already covers.
      if (!time_aware_ && use_prediction_ && !p.rotation_gated) {
        out.push_back(
          toCostmap(p.pred_x, p.pred_y, p.vx, p.vy, prediction_weight_));
      }
      continue;
    }

    // Everything below is extrapolation. A stale unidentified detection
    // or an implausible velocity produces a phantom that drifts away
    // from the real person — measured once at min_distance 0.027 m,
    // where the robot cleared the phantom and drove through the person.
    // Drop rather than coast in those cases.
    if (!p.coastable || age > max_coast_time_) {
      continue;
    }
    if (p.speed > max_coast_speed_) {
      continue;
    }

    // Coasting. The last real observation is `age` seconds old, so
    // advance it along the last known velocity. Below min_coast_speed_
    // the direction estimate is dominated by jitter, so hold position
    // rather than extrapolate into a wrong heading.
    double cx = p.x;
    double cy = p.y;
    if (p.speed >= min_coast_speed_) {
      cx += p.vx * age;
      cy += p.vy * age;
    }

    // Confidence decays linearly from the end of track_timeout_ to
    // max_coast_time_, so a long-coasted target still repels but never
    // outvotes a live observation.
    const double coast_span = std::max(1e-3, max_coast_time_ - track_timeout_);
    const double decay = 1.0 - (age - track_timeout_) / coast_span;
    const float scale = coast_weight_ * static_cast<float>(std::max(0.0, decay));

    out.push_back(toCostmap(cx, cy, p.vx, p.vy, scale));
  }

  return out;
}

void SocialCritic::score(CriticData & data)
{
  if (!enabled_) {
    return;
  }

  auto node = parent_.lock();
  if (!node) {
    return;
  }

  const auto targets = collectTargets();
  if (targets.empty()) {
    RCLCPP_WARN_THROTTLE(
      logger_, *node->get_clock(), 2000,
      "SocialCritic: no person data this cycle — critic is inert");
    return;
  }

  // Works for both the xtensor and Eigen backends of nav2_mppi_controller:
  // costs.size() is the batch, and total elements / batch is the horizon.
  const size_t batch = static_cast<size_t>(data.costs.size());
  if (batch == 0) {
    return;
  }
  const size_t time_steps =
    static_cast<size_t>(data.trajectories.x.size()) / batch;

  const int step = std::max(1, trajectory_point_step_);
  const float span = social_distance_ - critical_distance_;

  // DIAGNOSTIC: the critic enforces clearance from the ESTIMATED person
  // position, while analyse_avoidance.py measures clearance from the
  // ground truth. Any systematic perception offset subtracts directly
  // from the achieved min_distance. Compare the position logged here
  // against /person_ground_truth for the same timestamp. A scale below
  // 1.0 on the first target means it is being coasted, not observed.
  //
  // t= is node->now(), i.e. SIM time under use_sim_time. The bracketed
  // stamp rclcpp prints is wall clock and cannot be matched against a
  // bag recorded in sim time, which is why it is repeated here.
  RCLCPP_INFO_THROTTLE(
    logger_, *node->get_clock(), 1000,
    "SocialCritic: t=%.3f %zu target(s) in %s, first=(%.2f, %.2f) scale=%.2f",
    node->now().seconds(),
    targets.size(), costmap_ros_->getGlobalFrameID().c_str(),
    targets[0].x, targets[0].y, targets[0].weight_scale);

  float max_added = 0.0f;
  float min_added = std::numeric_limits<float>::max();
  double sum_added = 0.0;
  size_t n_penalised = 0;

  for (size_t i = 0; i < batch; ++i) {
    float traj_cost = 0.0f;

    for (size_t j = 0; j < time_steps; j += static_cast<size_t>(step)) {
      const float rx = data.trajectories.x(i, j);
      const float ry = data.trajectories.y(i, j);

      // Time at which the robot reaches this trajectory point. Clamped
      // because constant velocity stops being credible after a few
      // seconds — past the clamp the target simply stops advancing.
      const float t_ahead = time_aware_
        ? std::min(static_cast<float>(j) * data.model_dt, max_prediction_time_)
        : 0.0f;

      for (const auto & t : targets) {
        const float px = t.x + t.vx * t_ahead;
        const float py = t.y + t.vy * t_ahead;

        const float dx = rx - px;
        const float dy = ry - py;
        const float dist = std::sqrt(dx * dx + dy * dy);

        if (dist >= social_distance_) {
          continue;
        }

        if (dist <= critical_distance_) {
          traj_cost += collision_cost_ * t.weight_scale;
          continue;
        }

        // Normalised penetration depth: 0 at social_distance_, 1 at
        // critical_distance_. Unlike the costmap gradient this is
        // guaranteed non-zero right up to the boundary, which is the
        // whole point of the critic.
        const float depth = (social_distance_ - dist) / span;
        const float shaped = (cost_power_ == 1) ? depth : depth * depth;
        traj_cost += weight_ * t.weight_scale * shaped;
      }
    }

    data.costs(i) += traj_cost;

    if (traj_cost > 0.0f) {
      ++n_penalised;
      max_added = std::max(max_added, traj_cost);
      min_added = std::min(min_added, traj_cost);
      sum_added += traj_cost;
    }
  }

  // If n_penalised is ~0, no sampled trajectory ever entered the social
  // zone — the critic is loaded but has nothing to act on, and weight
  // tuning is pointless. If it is ~batch, every candidate is penalised,
  // so the penalty carries no discriminating information and the robot
  // falls back on whatever the other critics prefer.
  // What matters is not how MANY trajectories are penalised but whether
  // their costs DIFFER. n = batch with a wide spread is healthy: every
  // candidate is in the zone, but some are clearly better. n = batch
  // with min ~= max is the degenerate case — a constant offset, which
  // MPPI's softmax weighting cancels exactly, leaving the critic with
  // no influence at all. Read `spread`, not `penalised`.
  if (n_penalised > 0) {
    const float mean_added = static_cast<float>(sum_added / n_penalised);
    const float spread = (max_added > 1e-6f)
      ? (max_added - min_added) / max_added : 0.0f;
    RCLCPP_INFO_THROTTLE(
      logger_, *node->get_clock(), 1000,
      "SocialCritic: t=%.3f penalised %zu/%zu, cost min/mean/max "
      "%.1f/%.1f/%.1f, spread %.2f",
      node->now().seconds(), n_penalised, batch,
      min_added, mean_added, max_added, spread);
  } else {
    RCLCPP_INFO_THROTTLE(
      logger_, *node->get_clock(), 1000,
      "SocialCritic: t=%.3f penalised 0/%zu", node->now().seconds(), batch);
  }
}

}  // namespace mppi::critics

#include "pluginlib/class_list_macros.hpp"
PLUGINLIB_EXPORT_CLASS(
  mppi::critics::SocialCritic,
  mppi::critics::CriticFunction)