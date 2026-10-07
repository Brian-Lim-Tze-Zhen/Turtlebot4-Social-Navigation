#include "social_critic/social_critic.hpp"

#include <array>
#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>

#include "sensor_msgs/point_cloud2_iterator.hpp"
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
  getParam(group_aware_, "group_aware", false);
  getParam(narrow_social_distance_, "narrow_social_distance", 0.60f);
  getParam(group_topic_, "group_topic", std::string("/social_groups"));
  getParam(group_timeout_, "group_timeout", 4.0);
  getParam(member_match_radius_, "member_match_radius", 0.5);
  getParam(narrow_buffer_threshold_, "narrow_buffer_threshold", 0.39);
  getParam(pass_side_weight_, "pass_side_weight", 0.0f);
  getParam(pass_side_margin_, "pass_side_margin", 0.9f);
  getParam(pass_side_range_, "pass_side_range", 2.0f);
  getParam(pass_side_min_closing_, "pass_side_min_closing", 0.5f);
  getParam(no_retreat_weight_, "no_retreat_weight", 0.0f);
  getParam(lane_lateral_trust_, "lane_lateral_trust", 0.15f);
  getParam(pass_side_max_offset_, "pass_side_max_offset", 0.0f);
  getParam(lane_on_first_sight_s_, "lane_on_first_sight_s", 0.0f);
  getParam(pass_side_auto_, "pass_side_auto", false);
  getParam(lane_carry_radius_, "lane_carry_radius", 0.0f);
  getParam(lane_carry_along_, "lane_carry_along", 2.0f);
  getParam(side_decision_s_, "side_decision_s", 1.0f);
  getParam(side_switch_offset_, "side_switch_offset", 0.2f);
  getParam(side_commit_offset_, "side_commit_offset", 0.0f);
  getParam(side_ambiguous_offset_, "side_ambiguous_offset", 0.0f);
  getParam(publish_markers_, "publish_markers", false);
  getParam(publish_lane_block_, "publish_lane_block", false);
  getParam(lane_block_topic_, "lane_block_topic", std::string("/social_critic/lane_block"));
  getParam(lane_block_width_, "lane_block_width", 1.5f);
  getParam(lane_nearest_only_, "lane_nearest_only", false);
  getParam(lane_nearest_hysteresis_, "lane_nearest_hysteresis", 0.5f);
  getParam(lane_jump_reset_, "lane_jump_reset", 0.0f);
  getParam(occlusion_slow_weight_, "occlusion_slow_weight", 0.0f);
  getParam(occlusion_slow_speed_, "occlusion_slow_speed", 0.10f);
  getParam(occlusion_slow_gap_, "occlusion_slow_gap", 1.0f);
  getParam(occlusion_slow_after_s_, "occlusion_slow_after_s", 1.0f);
  getParam(occlusion_slow_horizon_s_, "occlusion_slow_horizon_s", 1.5f);
  getParam(lane_block_overlap_, "lane_block_overlap", 0.30f);
  getParam(lane_block_robot_gap_, "lane_block_robot_gap", 1.0f);
  getParam(marker_topic_, "marker_topic", std::string("/social_critic/lane_markers"));
  getParam(lane_first_sight_half_angle_, "lane_first_sight_half_angle", 0.52f);
  getParam(lane_first_sight_range_, "lane_first_sight_range", 9.0f);
  getParam(lane_first_sight_confirm_s_, "lane_first_sight_confirm_s", 0.0f);
  getParam(lane_first_sight_min_move_, "lane_first_sight_min_move", 0.25f);
  getParam(lane_ignore_beyond_goal_, "lane_ignore_beyond_goal", false);
  getParam(lane_beyond_goal_margin_, "lane_beyond_goal_margin", 0.5f);
  getParam(lane_skip_group_members_, "lane_skip_group_members", false);
  getParam(lane_group_match_radius_, "lane_group_match_radius", 0.0);
  getParam(lane_group_hold_s_, "lane_group_hold_s", 0.0);
  getParam(lane_approach_window_s_, "lane_approach_window_s", 0.0f);
  getParam(lane_approach_min_move_, "lane_approach_min_move", 0.5f);
  getParam(lane_approach_consistency_, "lane_approach_consistency", 0.6f);

  // Reuse the costmap's TF buffer rather than starting a second
  // listener inside controller_server.
  tf_buffer_ = costmap_ros_->getTfBuffer();

  sub_ = node->create_subscription<std_msgs::msg::String>(
    topic_, rclcpp::QoS(10),
    std::bind(&SocialCritic::positionsCallback, this, std::placeholders::_1));

  if (publish_markers_) {
    // A plain (non-lifecycle) publisher: the critic has no activate hook, and
    // a lifecycle publisher would stay silent.
    auto node_params = node->get_node_parameters_interface();
    auto node_topics = node->get_node_topics_interface();
    marker_pub_ = rclcpp::create_publisher<visualization_msgs::msg::MarkerArray>(
      node_params, node_topics, marker_topic_, rclcpp::QoS(1));
  }

  if (publish_lane_block_) {
    auto node_params = node->get_node_parameters_interface();
    auto node_topics = node->get_node_topics_interface();
    block_pub_ = rclcpp::create_publisher<sensor_msgs::msg::PointCloud2>(
      node_params, node_topics, lane_block_topic_, rclcpp::QoS(1));
  }

  if (group_aware_) {
    group_sub_ = node->create_subscription<std_msgs::msg::String>(
      group_topic_, rclcpp::QoS(10),
      std::bind(&SocialCritic::groupsCallback, this, std::placeholders::_1));
  }

  RCLCPP_INFO(
    logger_,
    "SocialCritic: social_distance=%.2f m, weight=%.1f, critical=%.2f m, "
    "point_step=%d, time_aware=%s (horizon %.1f s), topic=%s",
    social_distance_, weight_, critical_distance_,
    trajectory_point_step_, time_aware_ ? "on" : "off",
    max_prediction_time_, topic_.c_str());
  RCLCPP_INFO(
    logger_,
    "SocialCritic: group_aware=%s (narrow_social_distance=%.2f m, "
    "group_topic=%s, match=%.2f m, narrow if buffer < %.2f)",
    group_aware_ ? "on" : "off", narrow_social_distance_,
    group_topic_.c_str(), member_match_radius_, narrow_buffer_threshold_);
  RCLCPP_INFO(
    logger_,
    "SocialCritic: keep-right pass side %s (weight=%.1f, margin=%.2f m, "
    "range=%.2f m, min closing %.2f m/s, far edge %.2f m, lane on first "
    "sight %.1f s), no-retreat weight=%.1f",
    pass_side_weight_ > 0.0f ? "on" : "off", pass_side_weight_,
    pass_side_margin_, pass_side_range_, pass_side_min_closing_,
    pass_side_max_offset_, lane_on_first_sight_s_, no_retreat_weight_);
  RCLCPP_INFO(
    logger_,
    "SocialCritic: first sight needs %.2f m of approach within %.1f s (0 s = "
    "lane at once), people beyond the goal %s",
    lane_first_sight_min_move_, lane_first_sight_confirm_s_,
    lane_ignore_beyond_goal_ ? "ignored" : "not ignored");
  RCLCPP_INFO(
    logger_, "SocialCritic: group members get no lane unless closing: %s",
    (lane_skip_group_members_ && group_aware_) ? "on" : "off");
  RCLCPP_INFO(
    logger_, "SocialCritic: group skip match radius %.2f m (0 = match radius of the group), hold %.1f s",
    lane_group_match_radius_, lane_group_hold_s_);
  RCLCPP_INFO(
    logger_, "SocialCritic: lane needs a sustained approach: %s (%.2f m within %.2f s, %.0f %% of the steps towards the robot)",
    lane_approach_window_s_ > 0.0f ? "on" : "off", lane_approach_min_move_, lane_approach_window_s_,
    100.0f * lane_approach_consistency_);
}

// /social_groups, from social_group_detector_node_lidarhold_sim.py:
//   [0] group_id  [1] type  ...  [9] "x;y|x;y" member map positions
//   [10] effective buffer (narrow if < narrow_buffer_threshold_)
void SocialCritic::groupsCallback(const std_msgs::msg::String::SharedPtr msg)
{
  std::vector<std::string> parts;
  std::stringstream ss(msg->data);
  std::string item;
  while (std::getline(ss, item, ',')) {
    parts.push_back(item);
  }
  if (parts.size() < 11) {
    return;
  }

  auto node = parent_.lock();
  if (!node) {
    return;
  }

  GroupState g;
  try {
    std::stringstream ms(parts[9]);
    std::string member;
    while (std::getline(ms, member, '|')) {
      const auto sep = member.find(';');
      if (sep == std::string::npos) {
        return;
      }
      g.members.emplace_back(
        std::stod(member.substr(0, sep)), std::stod(member.substr(sep + 1)));
    }
    g.narrow = std::stod(parts[10]) < narrow_buffer_threshold_;
  } catch (const std::exception &) {
    return;
  }
  if (g.members.size() != 2) {
    return;
  }
  g.last_seen = node->now();

  std::lock_guard<std::mutex> lock(groups_mutex_);
  groups_[parts[0]] = g;
}

bool SocialCritic::isNarrowGroupMember(
  double x, double y, const rclcpp::Time & now)
{
  std::lock_guard<std::mutex> lock(groups_mutex_);
  for (auto it = groups_.begin(); it != groups_.end(); ) {
    if ((now - it->second.last_seen).seconds() > group_timeout_) {
      it = groups_.erase(it);
      continue;
    }
    if (it->second.narrow) {
      for (const auto & m : it->second.members) {
        if (std::hypot(x - m.first, y - m.second) <= member_match_radius_) {
          return true;
        }
      }
    }
    ++it;
  }
  return false;
}

bool SocialCritic::isGroupMember(
  double x, double y, const rclcpp::Time & now, double radius)
{
  std::lock_guard<std::mutex> lock(groups_mutex_);
  for (const auto & kv : groups_) {
    if ((now - kv.second.last_seen).seconds() > group_timeout_) {
      continue;
    }
    for (const auto & m : kv.second.members) {
      if (std::hypot(x - m.first, y - m.second) <= radius) {
        return true;
      }
    }
  }
  return false;
}

bool SocialCritic::approachConfirmed(int track_id, float ux, float uy) const
{
  auto it = approach_hist_.find(track_id);
  if (it == approach_hist_.end() || it->second.size() < 3) {
    return false;
  }
  const auto & h = it->second;
  if (h.back().t - h.front().t < 0.8 * lane_approach_window_s_) {
    return false;   // not watched long enough
  }
  // (ux, uy) points from the person to the robot.
  const float moved = (h.back().x - h.front().x) * ux + (h.back().y - h.front().y) * uy;
  if (moved < lane_approach_min_move_) {
    return false;
  }
  int towards = 0, steps = 0;
  for (size_t k = 1; k < h.size(); ++k) {
    const float s = (h[k].x - h[k - 1].x) * ux + (h[k].y - h[k - 1].y) * uy;
    if (std::fabs(s) < 1e-4f) {
      continue;   // repeated message, no new measurement
    }
    ++steps;
    if (s > 0.0f) {
      ++towards;
    }
  }
  return steps > 0 && static_cast<float>(towards) >= lane_approach_consistency_ * static_cast<float>(steps);
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

  auto toCostmap = [&](double mx, double my, double mvx, double mvy, float scale,
      float sd) {
      Target t;
      t.x = static_cast<float>(tx + c * mx - s * my);
      t.y = static_cast<float>(ty + s * mx + c * my);
      // Velocity is a free vector: rotate, do not translate.
      t.vx = static_cast<float>(c * mvx - s * mvy);
      t.vy = static_cast<float>(s * mvx + c * mvy);
      t.weight_scale = scale;
      t.social_distance = sd;
      return t;
    };

  for (const auto & p : snapshot) {
    const double age = (now - p.last_seen).seconds();

    // Matched on the person's last observed map position (same frame and
    // same pipeline as the group's member positions).
    const float sd = (group_aware_ && isNarrowGroupMember(p.x, p.y, now))
      ? narrow_social_distance_ : social_distance_;
    bool in_group = false;
    if (lane_skip_group_members_ && group_aware_) {
      const double rad = lane_group_match_radius_ > 0.0 ? lane_group_match_radius_ : member_match_radius_;
      const double now_s = now.seconds();
      in_group = isGroupMember(p.x, p.y, now, rad);
      if (in_group && lane_group_hold_s_ > 0.0) {
        group_member_until_[p.track_id] = now_s + lane_group_hold_s_;
      } else if (!in_group && lane_group_hold_s_ > 0.0) {
        auto h = group_member_until_.find(p.track_id);
        if (h != group_member_until_.end() && now_s <= h->second) {
          in_group = true;   // was a member a moment ago
        }
      }
    }

    if (age <= track_timeout_) {
      // Fresh observation: penalise the reported position, carrying the
      // velocity so score() can propagate it along the rollout.
      out.push_back(toCostmap(p.x, p.y, p.vx, p.vy, 1.0f, sd));
      out.back().track_id = p.track_id;
      out.back().in_group = in_group;

      // The KF's 1 s prediction is only worth adding as a separate
      // target when the critic is NOT propagating targets itself.
      // Otherwise it is the same information counted twice, at a
      // horizon the propagation already covers.
      if (!time_aware_ && use_prediction_ && !p.rotation_gated) {
        out.push_back(
          toCostmap(p.pred_x, p.pred_y, p.vx, p.vy, prediction_weight_, sd));
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

    out.push_back(toCostmap(cx, cy, p.vx, p.vy, scale, sd));
    out.back().track_id = p.track_id;
    out.back().coasted = true;
    out.back().in_group = in_group;
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

  auto targets = collectTargets();
  if (targets.empty()) {
    // The costmap layer keeps the last cloud it received, so the block has
    // to be cleared explicitly once there is nobody to avoid.
    if (block_pub_ && (block_tick_++ % 4) == 0) {
      publishLaneBlock(targets, node->now(), 0.0f, 0.0f);
    }
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

  // Keep-right geometry per target, from the robot's current pose (the
  // state is in the costmap frame, like the targets). The person-robot
  // line is used rather than the person's heading: over several metres it
  // is far steadier than the KF heading.
  const float robot_x0 = static_cast<float>(data.state.pose.pose.position.x);
  const float robot_y0 = static_cast<float>(data.state.pose.pose.position.y);
  bool slow_active = false;   // see occlusion_slow_weight_
  if (pass_side_weight_ > 0.0f || no_retreat_weight_ > 0.0f) {
    const float rx0 = static_cast<float>(data.state.pose.pose.position.x);
    const float ry0 = static_cast<float>(data.state.pose.pose.position.y);
    for (auto & t : targets) {
      const float ex = rx0 - t.x;
      const float ey = ry0 - t.y;
      const float d = std::sqrt(ex * ex + ey * ey);
      if (d < 1e-3f) {
        t.closing = false;
        continue;
      }
      t.ux = ex / d;
      t.uy = ey / d;
      t.closing = (t.vx * t.ux + t.vy * t.uy) > pass_side_min_closing_;
      if (lane_approach_window_s_ > 0.0f && t.track_id >= 0) {
        // Sustained approach (see lane_approach_window_s_): history of the
        // measured positions of this track, then the test.
        const double hs = node->now().seconds();
        auto & h = approach_hist_[t.track_id];
        if (!t.coasted) {
          h.push_back({hs, t.x, t.y});
        }
        while (!h.empty() && hs - h.front().t > lane_approach_window_s_) {
          h.pop_front();
        }
        t.closing = t.closing && approachConfirmed(t.track_id, t.ux, t.uy);
      }
      if (lane_approach_window_s_ > 0.0f) {
        const double hs = node->now().seconds();
        for (auto ah = approach_hist_.begin(); ah != approach_hist_.end(); ) {
          if (ah->second.empty() || hs - ah->second.back().t > lane_approach_window_s_ + 2.0) {
            ah = approach_hist_.erase(ah);
          } else {
            ++ah;
          }
        }
      }
      // lane_skip_group_members_: the speed of a standing member is jitter
      // (its position estimate jumps by up to 0.3 m and the KF reports
      // 0.3-0.4 m/s for a moment), so a group member is never "closing".
      if (t.in_group) {
        t.closing = false;
      }
    }

    // The estimates as received, before a walker is put on its lane below:
    // a walker that loses the lane rule to a nearer one (lane_nearest_only_)
    // is scored at its raw position again.
    std::vector<std::array<float, 4>> raw;
    raw.reserve(targets.size());
    for (const auto & t : targets) {
      raw.push_back({t.x, t.y, t.vx, t.vy});
    }

    // Freeze the lane axis per track the first time it closes on the robot,
    // or on first sight if it appears ahead (see lane_on_first_sight_s_).
    const double now_s = node->now().seconds();
    const float robot_yaw0 =
      static_cast<float>(tf2::getYaw(data.state.pose.pose.orientation));
    // The robot's direction of travel: along the start of the global path
    // (steadier than the heading, which wobbles a few degrees - at 7 m that
    // is several decimetres sideways). Falls back to the heading.
    float trav_x = std::cos(robot_yaw0);
    float trav_y = std::sin(robot_yaw0);
    {
      const size_t np = static_cast<size_t>(data.path.x.size());
      for (size_t k = 1; k < np; ++k) {
        const float qx = data.path.x(k) - data.path.x(0);
        const float qy = data.path.y(k) - data.path.y(0);
        const float ql = std::sqrt(qx * qx + qy * qy);
        if (ql >= 1.0f) {
          trav_x = qx / ql;
          trav_y = qy / ql;
          break;
        }
      }
    }
    auto newLane = [&](const Target & t, bool provisional) {
        LaneAxis a;
        a.ux = t.ux;
        a.uy = t.uy;
        a.ax = robot_x0;
        a.ay = robot_y0;
        a.stamp = now_s;
        a.first_seen = now_s;
        a.provisional = provisional;
        a.pdx = trav_x;
        a.pdy = trav_y;
        a.frozen_at = now_s;
        return a;
      };
    for (auto & t : targets) {
      if (t.track_id < 0) {
        continue;
      }
      if (t.in_group) {
        // lane_skip_group_members_: a member of a standing group gets no
        // lane and no side decision, whatever the KF velocity says.
        auto g = lane_axes_.find(t.track_id);
        if (g != lane_axes_.end()) {
          lane_axes_.erase(g);
        }
        continue;
      }
      auto it = lane_axes_.find(t.track_id);
      if (it != lane_axes_.end() && lane_jump_reset_ > 0.0f && it->second.has_last &&
        !t.coasted)
      {
        // The same id on a different person (see lane_jump_reset_): forget
        // the old lane, so this one is handled as a newly seen walker.
        const float jx = t.x - it->second.last_x;
        const float jy = t.y - it->second.last_y;
        const float jump = std::sqrt(jx * jx + jy * jy);
        if (jump > lane_jump_reset_ && it->second.first_seen >= 0.0) {
          // This person was hidden behind the previous one and is first seen
          // close. Assume it walks PARALLEL to the previous lane: keep that
          // direction and put the line through where the person is now. The
          // usual lane (from the robot toward the person) and the side read
          // off the bent global path were both wrong at 3 m: the robot was
          // told to cross in front of the walker
          // (headon_slow15_stag_y-0.5_trial2: 0.248 m, 2.3 s spinning).
          LaneAxis a = it->second;
          a.ax = t.x;
          a.ay = t.y;
          a.stamp = now_s;
          a.first_seen = now_s;
          a.frozen_at = now_s;
          a.provisional = true;
          a.offset_sum = 0.0;
          a.offset_n = 0;
          a.ambiguous = false;
          // q > 0: the robot is on this walker's right (its own LEFT).
          const float q_robot = (robot_x0 - a.ax) * a.uy - (robot_y0 - a.ay) * a.ux;
          a.side_frozen = std::fabs(q_robot) > 0.06f;
          if (a.side_frozen) {
            a.side = q_robot > 0.0f ? -1 : 1;
          }
          it->second = a;
          RCLCPP_INFO(
            logger_,
            "SocialCritic: track %d jumped %.2f m - a different person; parallel "
            "lane through it, robot is %.2f m to its %s, keeps %s",
            t.track_id, jump, std::fabs(q_robot), q_robot > 0.0f ? "right" : "left",
            a.side_frozen ? (a.side > 0 ? "RIGHT" : "LEFT") : "(undecided)");
        } else if (jump > lane_jump_reset_) {
          lane_axes_.erase(it);
          it = lane_axes_.end();
        }
      }
      if (it == lane_axes_.end() && lane_carry_radius_ > 0.0f && !t.coasted) {
        // A new id: is it the walker of a lane whose own track has gone
        // quiet? (see lane_carry_radius_)
        int from = -1;
        LaneAxis carried;
        for (const auto & kv : lane_axes_) {
          const LaneAxis & a = kv.second;
          if (a.first_seen < 0.0) {
            continue;
          }
          bool live = false;
          bool along_ok = true;
          for (const auto & o : targets) {
            if (o.track_id != kv.first) {
              continue;
            }
            if (!o.coasted) {
              live = true;
            } else {
              const float ds = (t.x - o.x) * a.ux + (t.y - o.y) * a.uy;
              along_ok = std::fabs(ds) < lane_carry_along_;
            }
          }
          if (live || !along_ok) {
            continue;
          }
          const float q = (t.x - a.ax) * a.uy - (t.y - a.ay) * a.ux;
          if (std::fabs(q) > lane_carry_radius_) {
            continue;
          }
          from = kv.first;
          carried = a;
          break;
        }
        if (from >= 0) {
          // The new track starts with zero velocity in the KF, so keep the
          // lane alive provisionally until the approach is confirmed again.
          carried.stamp = now_s;
          carried.first_seen = now_s;
          carried.provisional = true;
          it = lane_axes_.emplace(t.track_id, carried).first;
          superseded_[from] = t.track_id;
          RCLCPP_INFO(
            logger_,
            "SocialCritic: track %d inherits the lane of track %d (re-acquired "
            "walker), robot keeps %s", t.track_id, from,
            carried.side > 0 ? "RIGHT" : "LEFT");
        }
      }
      if (it == lane_axes_.end()) {
        // (ux, uy) points from the person to the robot, so the bearing of
        // the person from the robot is the opposite direction.
        const float ex = t.x - robot_x0;
        const float ey = t.y - robot_y0;
        const float d = std::sqrt(ex * ex + ey * ey);
        float off = std::atan2(ey, ex) - robot_yaw0;
        off = std::atan2(std::sin(off), std::cos(off));
        bool ahead = lane_on_first_sight_s_ > 0.0f &&
          d < lane_first_sight_range_ &&
          std::fabs(off) < lane_first_sight_half_angle_;
        if (ahead && lane_ignore_beyond_goal_) {
          const float gx = static_cast<float>(data.goal.position.x) - robot_x0;
          const float gy = static_cast<float>(data.goal.position.y) - robot_y0;
          if (d > std::sqrt(gx * gx + gy * gy) + lane_beyond_goal_margin_) {
            ahead = false;   // beyond the goal: the robot never gets there
          }
        }
        if (ahead && !t.closing && lane_first_sight_confirm_s_ > 0.0f) {
          // Watch it first (see lane_first_sight_confirm_s_): no lane yet.
          LaneAxis seen = newLane(t, false);
          seen.first_seen = -1.0;
          seen.pending = true;
          seen.pend_x = t.x;
          seen.pend_y = t.y;
          seen.pend_t = now_s;
          lane_axes_.emplace(t.track_id, seen);
          continue;
        }
        if (!t.closing && !ahead) {
          // Remember that this track has been seen, so it is not treated as
          // "new" later: an entry with no lane until it closes.
          if (lane_on_first_sight_s_ > 0.0f) {
            LaneAxis seen = newLane(t, false);
            seen.first_seen = -1.0;   // marks "no provisional lane"
            lane_axes_.emplace(t.track_id, seen);
          }
          continue;
        }
        it = lane_axes_.emplace(t.track_id, newLane(t, !t.closing)).first;
      } else if (it->second.first_seen < 0.0 && t.closing) {
        // Seen earlier without a lane (or lane dropped) and now closing:
        // freeze the lane afresh from where the robot is now.
        it->second = newLane(t, false);
      } else if (it->second.pending &&
        now_s - it->second.pend_t >= lane_first_sight_confirm_s_)
      {
        // (t.ux, t.uy) points from the person to the robot.
        const float moved = (t.x - it->second.pend_x) * t.ux +
          (t.y - it->second.pend_y) * t.uy;
        const bool sustained = lane_approach_window_s_ <= 0.0f ||
          approachConfirmed(t.track_id, t.ux, t.uy);
        if (moved >= lane_first_sight_min_move_ && sustained) {
          it->second = newLane(t, true);   // the first-sight lane, a little late
        } else {
          it->second.pending = false;      // standing: no lane unless it closes
        }
      }
      it->second.stamp = now_s;
      it->second.last_x = t.x;   // still the raw estimate here
      it->second.last_y = t.y;
      it->second.has_last = true;
      if (it->second.first_seen < 0.0) {
        continue;
      }
      t.lx = it->second.ux;
      t.ly = it->second.uy;
      t.ax = it->second.ax;
      t.ay = it->second.ay;

      // Pass side (see pass_side_auto_). The offset is taken from the RAW
      // estimate, before it is put on the lane below. Positive = the walker
      // is to the robot's left of its line of travel.
      if (pass_side_auto_ && !it->second.side_frozen) {
        LaneAxis & a = it->second;
        a.offset_sum += -a.pdy * (t.x - a.ax) + a.pdx * (t.y - a.ay);
        a.offset_n += 1;
        const double mean = a.offset_sum / static_cast<double>(a.offset_n);
        a.side = (mean < -static_cast<double>(side_switch_offset_)) ? -1 : 1;
        a.ambiguous = side_ambiguous_offset_ > 0.0f && a.side > 0 &&
          mean < -static_cast<double>(side_ambiguous_offset_);
        // Where the robot already is relative to the lane: q > 0 is the
        // walker's right, i.e. the robot's LEFT.
        const float q_robot =
          (robot_x0 - a.ax) * a.uy - (robot_y0 - a.ay) * a.ux;
        const bool committed = side_commit_offset_ > 0.0f &&
          std::fabs(q_robot) > side_commit_offset_;
        if (committed) {
          a.side = q_robot > 0.0f ? -1 : 1;
        }
        if (committed || now_s - a.frozen_at >= side_decision_s_) {
          a.side_frozen = true;
          RCLCPP_INFO(
            logger_,
            "SocialCritic: track %d pass side fixed: robot keeps %s (walker's "
            "mean offset %+.2f m over %d samples%s%s)",
            t.track_id, a.side > 0 ? "RIGHT" : "LEFT", mean, a.offset_n,
            committed ? ", robot already committed" : "",
            a.ambiguous ? ", ambiguous: no planner block" : "");
        }
      }
      t.side = static_cast<float>(it->second.side);
      // No planner block for an ambiguous side, nor while the lane is only
      // provisional: a single false detection far down the corridor got a
      // lane on first sight, its block covered the goal, the planner failed
      // and the goal was aborted (bag headon_hwreq_wall_keepright_trial1).
      // The block waits until the KF has confirmed an approach.
      t.block_ok = !it->second.ambiguous && !it->second.provisional;
      // Still coming down the lane toward where the robot was.
      const float v_along = t.vx * t.lx + t.vy * t.ly;
      const bool approaching = v_along > pass_side_min_closing_;
      if (approaching) {
        it->second.provisional = false;
      } else if (it->second.provisional &&
        now_s - it->second.first_seen > lane_on_first_sight_s_)
      {
        // Never confirmed as approaching: drop the provisional lane.
        it->second.provisional = false;
        it->second.first_seen = -1.0;
      }
      t.lane = approaching || it->second.provisional;
      if (t.lane) {
        // Put the walker on its lane (see lane_lateral_trust_).
        const float s = (t.x - t.ax) * t.lx + (t.y - t.ay) * t.ly;
        const float l = std::clamp(
          (t.x - t.ax) * t.ly - (t.y - t.ay) * t.lx,
          -lane_lateral_trust_, lane_lateral_trust_);
        t.x = t.ax + s * t.lx + l * t.ly;
        t.y = t.ay + s * t.ly - l * t.lx;
        t.vx = v_along * t.lx;
        t.vy = v_along * t.ly;
      }
    }
    for (auto it = lane_axes_.begin(); it != lane_axes_.end(); ) {
      if (now_s - it->second.stamp > lane_timeout_ || now_s < it->second.stamp) {
        it = lane_axes_.erase(it);
      } else {
        ++it;
      }
    }

    // A walker whose lane was handed to a new track must not be scored twice:
    // drop the old extrapolated target for as long as it is still around.
    for (auto it = superseded_.begin(); it != superseded_.end(); ) {
      bool present = false;
      for (auto & o : targets) {
        if (o.track_id == it->first) {
          present = true;
          if (o.coasted) {
            o.weight_scale = 0.0f;
            o.lane = false;
            o.closing = false;
          }
        }
      }
      if (present) {
        ++it;
      } else {
        it = superseded_.erase(it);
      }
    }

    // One encounter at a time (see lane_nearest_only_): the lane rule, the
    // no-retreat term and the planner block apply to the nearest walker that
    // is still ahead. The others keep only the social-distance cost until
    // the nearest one has passed.
    if (lane_nearest_only_) {
      int best = -1;
      int cur = -1;
      float best_gap = std::numeric_limits<float>::max();
      float cur_gap = 0.0f;
      for (size_t k = 0; k < targets.size(); ++k) {
        const Target & t = targets[k];
        if (!t.lane || t.weight_scale <= 0.0f) {
          continue;
        }
        // Along the lane, from the walker to the robot: > 0 is still ahead.
        const float gap = (robot_x0 - t.x) * t.lx + (robot_y0 - t.y) * t.ly;
        if (gap < -pass_side_behind_) {
          continue;
        }
        if (t.track_id == primary_track_) {
          cur = static_cast<int>(k);
          cur_gap = gap;
        }
        if (gap < best_gap) {
          best = static_cast<int>(k);
          best_gap = gap;
        }
      }
      // Hold on to the current one unless another is clearly nearer.
      if (cur >= 0 && best != cur && cur_gap - best_gap < lane_nearest_hysteresis_) {
        best = cur;
        best_gap = cur_gap;
      }
      const int best_id = best >= 0 ? targets[best].track_id : -1;
      if (best_id != primary_track_) {
        if (best_id >= 0) {
          RCLCPP_INFO(
            logger_,
            "SocialCritic: lane rule now follows track %d (%.2f m ahead), robot keeps %s",
            best_id, best_gap, targets[best].side > 0 ? "RIGHT" : "LEFT");
        }
        primary_track_ = best_id;
      }
      for (size_t k = 0; k < targets.size(); ++k) {
        Target & t = targets[k];
        if (static_cast<int>(k) == best || !t.lane) {
          continue;
        }
        const float gap = (robot_x0 - t.x) * t.lx + (robot_y0 - t.y) * t.ly;
        if (gap < -pass_side_behind_) {
          continue;   // already passed: the rule no longer reaches it
        }
        t.lane = false;
        t.closing = false;
        t.block_ok = false;
        t.x = raw[k][0];
        t.y = raw[k][1];
        t.vx = raw[k][2];
        t.vy = raw[k][3];
      }
    }

    // Slow down while a walker goes by (see occlusion_slow_weight_).
    if (occlusion_slow_weight_ > 0.0f) {
      bool passing = false;
      bool other_ahead = false;
      for (const auto & t : targets) {
        if (!t.lane || t.weight_scale <= 0.0f) {
          continue;
        }
        const float gap = (robot_x0 - t.x) * t.lx + (robot_y0 - t.y) * t.ly;
        if (gap > -0.3f && gap < occlusion_slow_gap_) {
          passing = true;
        } else if (gap >= occlusion_slow_gap_) {
          other_ahead = true;
        }
      }
      if (now_s + 5.0 < slow_until_) {
        slow_until_ = -1.0;   // clock went back (new sim run)
      }
      if (passing) {
        slow_until_ = now_s + occlusion_slow_after_s_;
      }
      // A walker already tracked further ahead needs the robot's full speed
      // to get to its strip; what it hides is the next pass's concern.
      slow_active = now_s < slow_until_ && !other_ahead;
      if (slow_active != slow_was_active_) {
        RCLCPP_INFO(
          logger_, "SocialCritic: occlusion slow-down %s",
          slow_active ? "ON (a walker is going by, the view behind it is blocked)" : "off");
        slow_was_active_ = slow_active;
      }
    }
  }

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
  size_t n_narrow = 0;
  for (const auto & t : targets) {
    if (t.social_distance < social_distance_) {
      ++n_narrow;
    }
  }
  RCLCPP_INFO_THROTTLE(
    logger_, *node->get_clock(), 1000,
    "SocialCritic: t=%.3f %zu target(s) in %s, first=(%.2f, %.2f) scale=%.2f "
    "narrow-relaxed=%zu",
    node->now().seconds(),
    targets.size(), costmap_ros_->getGlobalFrameID().c_str(),
    targets[0].x, targets[0].y, targets[0].weight_scale, n_narrow);

  if (block_pub_ && (block_tick_++ % 4) == 0) {
    publishLaneBlock(targets, node->now(), robot_x0, robot_y0);
  }

  if (marker_pub_ && (marker_tick_++ % 4) == 0) {
    publishMarkers(targets, node->now(), robot_x0, robot_y0);
  }

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

        if (pass_side_weight_ > 0.0f && t.lane) {
          // Along the frozen lane: + is ahead of the walker (not yet passed).
          const float along = dx * t.lx + dy * t.ly;
          if (along > -pass_side_behind_ && along < pass_side_range_) {
            // Walker's right-hand side, facing down the lane: (ly, -lx).
            // lateral > 0 is the wrong side; the point must be at least
            // pass_side_margin_ on the other side to be free of this term.
            // Measured from the fixed lane line, not from the estimate.
            // side = +1: the robot must end up on the walker's left (its own
            // right); side = -1 mirrors the strip to the other side.
            const float lateral =
              t.side * ((rx - t.ax) * t.ly - (ry - t.ay) * t.lx);
            if (lateral > -pass_side_margin_) {
              traj_cost += pass_side_weight_ * t.weight_scale *
                (lateral + pass_side_margin_);
            } else if (pass_side_max_offset_ > 0.0f &&
              lateral < -pass_side_max_offset_)
            {
              // Past the far edge of the strip (see pass_side_max_offset_).
              traj_cost += pass_side_weight_ * t.weight_scale *
                (-pass_side_max_offset_ - lateral);
            }
          }
        }

        if (dist >= t.social_distance) {
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
        const float span = std::max(1e-3f, t.social_distance - critical_distance_);
        const float depth = (t.social_distance - dist) / span;
        const float shaped = (cost_power_ == 1) ? depth : depth * depth;
        traj_cost += weight_ * t.weight_scale * shaped;
      }
    }

    if (no_retreat_weight_ > 0.0f && time_steps > 0) {
      // (ux, uy) points from the walker to the robot, so a step with a
      // positive projection is a step away from the walker. Summed step by
      // step, not end point minus start: a rollout that curls back and then
      // drives on has no net retreat, and with the end-point version the
      // robot still curled back-right until it faced down the corridor
      // (bags headon_hwreq_r2_trial1..3, the last one at weight 300).
      for (const auto & t : targets) {
        // Only while the walker is still AHEAD of the robot. Once it has
        // passed, (ux, uy) points forward, and this term charged the robot
        // for driving on toward its goal: it stopped and turned round
        // behind the walker instead (bags headon_hwreq_r6_trial1..3 and
        // r7_trial1; the "lane" flag stays true after the pass because the
        // walker is still moving down its lane).
        const bool ahead = t.lane
          ? ((robot_x0 - t.x) * t.lx + (robot_y0 - t.y) * t.ly) > 0.3f
          : t.closing;
        if (!ahead) {
          continue;
        }
        float retreat = 0.0f;
        float px_prev = robot_x0;
        float py_prev = robot_y0;
        for (size_t j = 0; j < time_steps; j += static_cast<size_t>(step)) {
          const float qx = data.trajectories.x(i, j);
          const float qy = data.trajectories.y(i, j);
          const float d = (qx - px_prev) * t.ux + (qy - py_prev) * t.uy;
          if (d > 0.0f) {
            retreat += d;
          }
          px_prev = qx;
          py_prev = qy;
        }
        traj_cost += no_retreat_weight_ * t.weight_scale * retreat;
      }
    }

    if (slow_active && time_steps > 0) {
      // Distance driven beyond occlusion_slow_speed_ over the first
      // occlusion_slow_horizon_s_ of the rollout. Turning is not limited.
      const float cap = occlusion_slow_speed_ * data.model_dt * static_cast<float>(step);
      float excess = 0.0f;
      float px_prev = robot_x0;
      float py_prev = robot_y0;
      for (size_t j = 0; j < time_steps; j += static_cast<size_t>(step)) {
        if (static_cast<float>(j) * data.model_dt > occlusion_slow_horizon_s_) {
          break;
        }
        const float qx = data.trajectories.x(i, j);
        const float qy = data.trajectories.y(i, j);
        const float d = std::sqrt(
          (qx - px_prev) * (qx - px_prev) + (qy - py_prev) * (qy - py_prev));
        if (j > 0 && d > cap) {
          excess += d - cap;
        }
        px_prev = qx;
        py_prev = qy;
      }
      traj_cost += occlusion_slow_weight_ * excess;
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

void SocialCritic::publishLaneBlock(
  const std::vector<Target> & targets, const rclcpp::Time & stamp,
  float robot_x, float robot_y)
{
  std::vector<std::pair<float, float>> pts;
  for (const auto & t : targets) {
    if (!t.lane || !t.block_ok) {
      continue;
    }
    // Along the lane, s grows from the walker toward the robot.
    const float s_walker = (t.x - t.ax) * t.lx + (t.y - t.ay) * t.ly;
    const float s_robot = (robot_x - t.ax) * t.lx + (robot_y - t.ay) * t.ly;
    const float s0 = s_walker - 0.3f;
    const float s1 = s_robot - lane_block_robot_gap_;
    if (s1 <= s0) {
      continue;   // walker too close, or already past the robot
    }
    // Sideways, q > 0 is the walker's right. The robot must end up at
    // q < 0 for side +1 (q > 0 for side -1), so the block covers the wrong
    // side plus a little of the allowed side.
    for (float s = s0; s <= s1; s += lane_block_spacing_) {
      for (float w = -lane_block_overlap_; w <= lane_block_width_;
        w += lane_block_spacing_)
      {
        const float q = t.side * w;
        pts.emplace_back(
          t.ax + s * t.lx + q * t.ly,
          t.ay + s * t.ly - q * t.lx);
      }
    }
  }

  sensor_msgs::msg::PointCloud2 cloud;
  cloud.header.frame_id = costmap_ros_->getGlobalFrameID();
  cloud.header.stamp = stamp;
  sensor_msgs::PointCloud2Modifier mod(cloud);
  mod.setPointCloud2FieldsByString(1, "xyz");
  mod.resize(pts.size());
  sensor_msgs::PointCloud2Iterator<float> ix(cloud, "x");
  sensor_msgs::PointCloud2Iterator<float> iy(cloud, "y");
  sensor_msgs::PointCloud2Iterator<float> iz(cloud, "z");
  for (const auto & p : pts) {
    *ix = p.first;
    *iy = p.second;
    *iz = 0.3f;
    ++ix;
    ++iy;
    ++iz;
  }
  block_pub_->publish(cloud);
}

void SocialCritic::publishMarkers(
  const std::vector<Target> & targets, const rclcpp::Time & stamp,
  float robot_x, float robot_y)
{
  visualization_msgs::msg::MarkerArray arr;
  visualization_msgs::msg::Marker clear;
  clear.header.frame_id = costmap_ros_->getGlobalFrameID();
  clear.header.stamp = stamp;
  clear.action = visualization_msgs::msg::Marker::DELETEALL;
  arr.markers.push_back(clear);

  auto line = [&](const std::string & ns, int id, float r, float g, float b, float w) {
      visualization_msgs::msg::Marker m;
      m.header = clear.header;
      m.ns = ns;
      m.id = id;
      m.type = visualization_msgs::msg::Marker::LINE_STRIP;
      m.action = visualization_msgs::msg::Marker::ADD;
      m.pose.orientation.w = 1.0;
      m.scale.x = w;
      m.color.r = r;
      m.color.g = g;
      m.color.b = b;
      m.color.a = 0.9f;
      return m;
    };
  auto point = [](float x, float y) {
      geometry_msgs::msg::Point p;
      p.x = x;
      p.y = y;
      p.z = 0.05;
      return p;
    };

  int id = 0;
  for (const auto & t : targets) {
    if (!t.lane) {
      continue;
    }
    // Stop drawing once the walker is behind the robot: the rule no longer
    // applies there, and the markers used to linger for several seconds.
    if (((robot_x - t.x) * t.lx + (robot_y - t.y) * t.ly) < -pass_side_behind_) {
      continue;
    }
    // Along the lane: s = 0 at the anchor (robot at freeze), negative
    // toward the walker. Lateral offset q is along (ly, -lx): the walker's
    // right is +, so the robot's side is negative.
    const float s_walker = (t.x - t.ax) * t.lx + (t.y - t.ay) * t.ly;
    const float s0 = s_walker - pass_side_behind_;
    const float s1 = s_walker + pass_side_range_;
    auto at = [&](float s, float q) {
        return point(t.ax + s * t.lx + q * t.ly, t.ay + s * t.ly - q * t.lx);
      };

    // The frozen lane (blue) over the stretch where the rule applies.
    auto lane = line("lane", id, 0.2f, 0.4f, 1.0f, 0.04f);
    lane.points.push_back(at(s0, 0.0f));
    lane.points.push_back(at(std::min(s1, 1.0f), 0.0f));
    arr.markers.push_back(lane);

    // Target strip (green): near edge at pass_side_margin_, far edge at
    // pass_side_max_offset_ when set.
    auto near_edge = line("strip_near", id, 0.1f, 0.9f, 0.2f, 0.03f);
    near_edge.points.push_back(at(s0, -t.side * pass_side_margin_));
    near_edge.points.push_back(at(std::min(s1, 1.0f), -t.side * pass_side_margin_));
    arr.markers.push_back(near_edge);
    if (pass_side_max_offset_ > 0.0f) {
      auto far_edge = line("strip_far", id, 0.1f, 0.6f, 0.2f, 0.03f);
      far_edge.points.push_back(at(s0, -t.side * pass_side_max_offset_));
      far_edge.points.push_back(
        at(std::min(s1, 1.0f), -t.side * pass_side_max_offset_));
      arr.markers.push_back(far_edge);
    }

    // Where the critic takes the walker to be, now and at each second of
    // the rollout (red, fading), with the social distance around "now".
    visualization_msgs::msg::Marker pred;
    pred.header = clear.header;
    pred.ns = "predicted";
    pred.id = id;
    pred.type = visualization_msgs::msg::Marker::SPHERE_LIST;
    pred.action = visualization_msgs::msg::Marker::ADD;
    pred.pose.orientation.w = 1.0;
    pred.scale.x = pred.scale.y = pred.scale.z = 0.18;
    for (float tt = 0.0f; tt <= max_prediction_time_ + 1e-3f; tt += 1.0f) {
      pred.points.push_back(point(t.x + t.vx * tt, t.y + t.vy * tt));
      std_msgs::msg::ColorRGBA c;
      c.r = 1.0f;
      c.g = 0.1f;
      c.b = 0.1f;
      c.a = std::max(0.25f, 1.0f - 0.12f * tt);
      pred.colors.push_back(c);
    }
    arr.markers.push_back(pred);

    auto ring = line("social_distance", id, 1.0f, 0.6f, 0.0f, 0.02f);
    for (int k = 0; k <= 36; ++k) {
      const float a = static_cast<float>(k) * 0.17453293f;
      ring.points.push_back(
        point(
          t.x + t.social_distance * std::cos(a),
          t.y + t.social_distance * std::sin(a)));
    }
    arr.markers.push_back(ring);
    ++id;
  }
  marker_pub_->publish(arr);
}

}  // namespace mppi::critics

#include "pluginlib/class_list_macros.hpp"
PLUGINLIB_EXPORT_CLASS(
  mppi::critics::SocialCritic,
  mppi::critics::CriticFunction)