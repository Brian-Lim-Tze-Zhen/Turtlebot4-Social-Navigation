// SocialCritic — MPPI critic that scores trajectories directly against
// tracked person positions, bypassing the costmap distance field.
//
// WHY THIS EXISTS
// ---------------
// ObstaclesCritic reads the local costmap's inflation gradient. With
// inflation_radius = 1.0 m the cost has already decayed to near zero by
// the time a trajectory point is ~0.94 m from the person's marked cells,
// which is exactly the clearance we want to enforce. Raising
// repulsion_weight multiplies a value that is already ~0, so it cannot
// move the decision boundary outward. Measured: four ablations
// (inflation 1.3, repulsion 5.0, disk radius 0.55, path weights 2.0) all
// scored at or below the 0.582 +/- 0.025 baseline.
//
// This critic instead takes person positions straight from
// /predicted_person_positions and applies its own distance penalty, so
// the clearance boundary is an explicit parameter rather than an
// emergent property of costmap resolution and inflation decay.

#ifndef SOCIAL_CRITIC__SOCIAL_CRITIC_HPP_
#define SOCIAL_CRITIC__SOCIAL_CRITIC_HPP_

#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "nav2_mppi_controller/critic_function.hpp"
#include "nav2_mppi_controller/models/state.hpp"
#include "nav2_mppi_controller/tools/utils.hpp"

#include "std_msgs/msg/string.hpp"
#include "tf2_ros/buffer.h"
#include "visualization_msgs/msg/marker_array.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"

namespace mppi::critics
{

struct PersonState
{
  int track_id{0};
  double x{0.0};          // current position, map frame
  double y{0.0};
  double pred_x{0.0};     // KF 1 s prediction, map frame
  double pred_y{0.0};
  double vx{0.0};         // KF velocity, map frame — used to coast the
  double vy{0.0};         // track while the person is outside camera FOV
  double speed{0.0};
  bool rotation_gated{false};
  // A track with id -1 is a detection the tracker could not associate.
  // Its position is usable while fresh, but it carries no reliable
  // velocity, so extrapolating it invents a phantom obstacle that
  // drifts away from any real person. Only identified tracks coast.
  bool coastable{false};
  rclcpp::Time last_seen;
};

// A single point the critic penalises proximity to, already expressed in
// the costmap frame. weight_scale folds together the current-vs-predicted
// distinction and the confidence decay applied to coasted tracks.
struct Target
{
  float x{0.0f};
  float y{0.0f};
  float vx{0.0f};
  float vy{0.0f};
  float weight_scale{1.0f};
  // Per-target clearance. social_distance_ normally; narrow_social_distance_
  // for members of a confirmed NARROW group when group_aware_ is on.
  float social_distance{0.94f};
  // Keep-right geometry, filled in score(): unit vector from the person to
  // the robot, and whether the person is closing on the robot.
  float ux{1.0f};
  float uy{0.0f};
  bool closing{false};
  // Track this target came from (-1 for the extra KF-prediction target),
  // and the walker's lane axis frozen when it first closed on the robot.
  int track_id{-1};
  float lx{1.0f};
  float ly{0.0f};
  float ax{0.0f};      // a point on the lane: where the robot was at freeze
  float ay{0.0f};
  bool lane{false};
  float side{1.0f};    // +1 pass on the walker's left (robot keeps right), -1 the other way
  bool coasted{false}; // extrapolated from an old observation, not seen this cycle
  bool block_ok{true}; // false: do not publish the planner block for this lane
};

// THESIS ADDITION (ablation F, narrow case). One group from /social_groups,
// member positions in the map frame (field 9), narrow flag from field 10.
struct GroupState
{
  std::vector<std::pair<double, double>> members;
  bool narrow{false};
  rclcpp::Time last_seen;
};

class SocialCritic : public CriticFunction
{
public:
  void initialize() override;
  void score(CriticData & data) override;

private:
  void positionsCallback(const std_msgs::msg::String::SharedPtr msg);
  void groupsCallback(const std_msgs::msg::String::SharedPtr msg);

  // True if (x, y) in the map frame is within member_match_radius_ of a
  // member of a fresh narrow group.
  bool isNarrowGroupMember(double x, double y, const rclcpp::Time & now);

  // Returns everything worth penalising, in the costmap frame: current
  // positions, KF predictions, and coasted extrapolations of tracks that
  // have gone silent but not yet exceeded max_coast_time_.
  std::vector<Target> collectTargets();

  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr sub_;
  std::shared_ptr<tf2_ros::Buffer> tf_buffer_;

  std::mutex people_mutex_;
  std::vector<PersonState> people_;

  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr group_sub_;
  std::mutex groups_mutex_;
  std::unordered_map<std::string, GroupState> groups_;

  // --- parameters ---
  // Centre-to-centre distance at which the penalty reaches zero. This is
  // the knob the whole critic exists for: set it to
  //   desired surface clearance + robot footprint radius
  // e.g. 0.75 + 0.189 = 0.94 for the TurtleBot4 octagonal footprint.
  float social_distance_{0.94f};

  // Penalty scale. Unlike ObstaclesCritic's repulsion_weight this
  // multiplies a term that is genuinely non-zero at social_distance_,
  // so it has real authority against PathAlign/PathFollow.
  float weight_{40.0f};

  // Distance at which cost saturates at collision_cost_ (hard core).
  float critical_distance_{0.35f};
  float collision_cost_{10000.0f};

  // 1 = linear ramp, 2 = quadratic (steeper close in, gentler far out).
  int cost_power_{1};

  // Also penalise the KF-predicted position, not just the current one.
  bool use_prediction_{true};
  float prediction_weight_{0.5f};   // relative to weight_

  // Compare trajectory point j against where the person will be at
  // j * model_dt, not where they are now.
  bool time_aware_{true};
  float max_prediction_time_{3.0f};

  // Evaluate every Nth trajectory point. 120 time_steps x 2000 batch is
  // 240k point-person distance checks per cycle at step 1; step 4 keeps
  // the 20 Hz control loop comfortable.
  int trajectory_point_step_{4};

  double track_timeout_{0.5};       // seconds of silence before coasting

  // COASTING. The OAK-D loses the person laterally at roughly 3-3.5 m
  // during a head-on pass — measured: last usable reading at 3.5 m,
  // while min_distance occurs near 0.5 m. Without coasting the critic
  // has no target across the entire terminal approach, which is exactly
  // the interval social_distance_ is meant to govern.
  //
  // The person walks a straight line at constant speed, so a constant
  // velocity extrapolation over ~2 s carries far less error than having
  // no estimate at all. Coasted targets are down-weighted since their
  // confidence decays with time since the last real observation.
  double max_coast_time_{2.5};      // seconds; drop the track past this
  float coast_weight_{0.7};         // multiplier applied to coasted targets
  float min_coast_speed_{0.15};     // below this, coast in place instead
  float max_coast_speed_{2.5};      // above this the KF estimate is junk
  std::string person_frame_{"map"};
  std::string topic_{"/predicted_person_positions"};

  // --- GROUP AWARENESS (ablation F narrow case) ---
  // The global-costmap social zone prices the gap between the two members
  // of a NARROW group as passable (o-space 35). With social_distance_ 0.94
  // and a ~1.4 m gap, this critic penalised every trajectory through that
  // gap, so MPPI stalled at the entrance (pilot conv_F_narrow_pilot01:
  // robot stopped 0.94 m from both members, Failed to make progress).
  // When enabled, members of a fresh narrow group get narrow_social_distance_
  // instead, so the critic defers to the zone's narrow decision.
  // Default OFF: conditions A-E are unchanged unless their YAML sets it.
  bool group_aware_{false};
  float narrow_social_distance_{0.60f};
  std::string group_topic_{"/social_groups"};
  double group_timeout_{4.0};          // s; matches the zone node GROUP_TIMEOUT
  double member_match_radius_{0.5};    // m; person <-> group member association
  // MUST match the zone node: narrow = buffer < 0.4 - 0.01
  double narrow_buffer_threshold_{0.39};

  // Keep-right pass side for an approaching walker. The social term is
  // symmetric, so in a head-on encounter MPPI picked a side from tracking
  // noise, committed late and sometimes against the global plan
  // (sim bags headon_avoid_v1..v7). With pass_side_weight_ > 0, rollout
  // points within pass_side_range_ of the walker's predicted position that
  // are not at least pass_side_margin_ to the walker's LEFT (the robot's
  // right) are penalised in proportion to how far they are on the wrong
  // side. Default 0 = off, so every existing config is unchanged.
  // The side is measured against the walker's LANE: the line through the
  // walker along the walker->robot direction, frozen per track when the
  // walker first closes on the robot. A line recomputed every cycle always
  // passes through the robot, so it cannot say which side the robot is on;
  // a penalty limited to points near the walker was cheapest for a robot
  // that stopped and waited (it spun on the spot for ~2.5 s in bag
  // headon_avoid_v11_trial1). Against a fixed lane, stopping or turning on
  // the spot gains nothing: only leaving the lane to the right does, and a
  // gentle early veer is enough. pass_side_range_ is how far ahead of the
  // walker, along the lane, the rule applies.
  float pass_side_weight_{0.0f};
  float pass_side_margin_{0.9f};
  float pass_side_range_{2.0f};
  float pass_side_behind_{0.5f};        // m behind the walker still covered
  // Far edge of the target strip. The rule above only says "at least
  // pass_side_margin_ to the right", so a steep crossing overshot to within
  // ~0.12 m of the wall, where the robot crawled and turned for several
  // seconds (bags headon_avoid_confirm_trial2, headon_avoid_nospin_trial3).
  // With a value > 0, rollout points further right of the lane than this
  // are penalised too, so the controller flattens its approach before the
  // wall. 0 = no far edge (previous behaviour).
  float pass_side_max_offset_{0.0f};
  // The lane is anchored at the ROBOT's position when it was frozen, not at
  // the walker's estimate: the camera-ray position is a LiDAR return on the
  // body surface nearest the robot and wandered 0.2-0.4 m sideways as the
  // robot moved aside. Measured against the estimate, the required offset
  // went beyond the wall and the robot backed along it and turned round
  // (bag headon_avoid_v17_trial3). A line fixed through the robot's own
  // starting point does not move with that error: near the robot, an
  // estimate off by 0.25 m at 10 m shifts it by only ~0.04 m.
  struct LaneAxis
  {
    float ux{1.0f};
    float uy{0.0f};
    float ax{0.0f};
    float ay{0.0f};
    double stamp{0.0};
    double first_seen{0.0};
    bool provisional{false};   // frozen on first sight, approach not confirmed yet
    // Pass-side decision (pass_side_auto_): the robot's direction of travel
    // when the lane was frozen, and the walker's sideways offset from that
    // line, averaged while the decision is still open.
    float pdx{1.0f};
    float pdy{0.0f};
    double frozen_at{0.0};
    double offset_sum{0.0};
    int offset_n{0};
    int side{1};
    bool side_frozen{false};
    bool ambiguous{false};   // walker may be on the robot's right: no planner block
  };
  std::unordered_map<int, LaneAxis> lane_axes_;
  // While a walker is in its lane, the critic takes the walker to be ON the
  // lane line: the estimate's sideways offset is clamped to this much and
  // its sideways velocity dropped. The camera-ray estimate swings 0.2-0.4 m
  // toward the robot while the robot is turning (bearing error from the
  // camera/scan time offset), which put the perceived walker within
  // social_distance of the only free strip by the wall; the robot then
  // crept backwards and rotated there (bag headon_avoid_v18_trial1). The
  // person cloud in the local costmap still marks the raw estimate, so a
  // walker who really steps sideways remains a hard obstacle.
  float lane_lateral_trust_{0.15f};

  // Lane on first sight. At the real robot's detection range (~7 m) the
  // walker is ~4.7 s away, and waiting for the KF to report an approach
  // costs ~1 s of that: in sim at hardware range the robot only began to
  // move aside at 3.7 m and reached 0.56-0.86 m (bags
  // headon_avoid_hwrange_*). With a value > 0, a NEW track that is ahead of
  // the robot (within lane_first_sight_half_angle_ of its heading and
  // lane_first_sight_range_) gets its lane frozen immediately and is
  // treated as approaching for this many seconds. If the KF has not
  // confirmed an approach by then the lane is dropped, and frozen afresh if
  // the person starts closing later. 0 = off.
  float lane_on_first_sight_s_{0.0f};

  // Choosing the side. Keep-right is wrong for a walker who is coming down
  // the robot's RIGHT-hand side: the robot would cut across them or be
  // squeezed against the wall. With pass_side_auto_ on, the walker's
  // sideways offset from the robot's line of travel is averaged over the
  // first side_decision_s_ after the lane is frozen; if the walker is more
  // than side_switch_offset_ to the robot's right the robot passes on the
  // LEFT instead. The side is then fixed for the encounter. Averaging is
  // needed because a single estimate at ~7 m is off by up to +/-0.5 m
  // sideways. Default off = always keep right.
  // Carrying the lane over to a re-acquired track. On the real robot the
  // track is often lost mid-approach and comes back with a NEW id (7 of 11
  // bags on 25 Sep). A new id would get a new lane frozen from where the
  // robot is NOW - already off to the side - and the rule would then ask for
  // another 0.8 m beyond that, toward the wall. With lane_carry_radius_ > 0,
  // a new track that lies within that distance of the lane of a track no
  // longer being observed (and within lane_carry_along_ of where that track
  // is expected along it) inherits that lane, its side and its anchor, and
  // the old extrapolated target is dropped so the walker is not counted
  // twice. 0 = off.
  float lane_carry_radius_{0.0f};
  float lane_carry_along_{2.0f};
  std::unordered_map<int, int> superseded_;   // old track id -> new track id

  bool pass_side_auto_{false};
  float side_decision_s_{1.0f};
  float side_switch_offset_{0.2f};
  // The side is also fixed as soon as the robot itself is this far off the
  // lane, on the side it has moved to. Changing sides after the robot has
  // started is worse than either side: a centred walker read as -0.22 m at
  // the end of the 1 s window flipped the decision to LEFT with the robot
  // already 0.11 m to the right, and it passed at 0.46 m (bag
  // headon_hwreq_block_corr_trial2). 0 = no such commit.
  float side_commit_offset_{0.0f};
  // Three zones. The sideways estimate at ~7 m is only good to a few
  // decimetres (a centred walker read -0.22 .. +0.50 m over 13 runs; one at
  // -0.40 m read -0.40 and -0.19), so a walker slightly to the right cannot
  // be told from a centred one. Only a walker CLEARLY on the right, beyond
  // side_switch_offset_, switches the robot to the left. Between
  // side_ambiguous_offset_ and side_switch_offset_ the robot keeps right
  // but the planner block is withheld: with the block, a walker really on
  // the right and the blocked left half closed the corridor and the goal
  // was aborted with the robot stopped in the walker's line (bag
  // headon_hwreq_b2_y-0.4_trial1). 0 = no ambiguous zone.
  float side_ambiguous_offset_{0.0f};
  float lane_first_sight_half_angle_{0.52f};   // rad (~30 deg)
  float lane_first_sight_range_{9.0f};         // m
  double lane_timeout_{3.0};            // s without the track before forgetting
  float pass_side_min_closing_{0.5f};   // m/s toward the robot

  // No retreat from an approaching walker. The proximity term is lowest
  // for a while if the robot simply moves away along the person-robot
  // line, so MPPI reversed, and with reversing disabled turned round on
  // the spot to drive off (sim bags headon_avoid_v10_trial2/3) instead of
  // stepping aside. With no_retreat_weight_ > 0, a rollout is charged for
  // every step it takes away from a closing walker along that line (summed
  // step by step); sideways motion is free. Default 0 = off.
  float no_retreat_weight_{0.0f};

  // RViz view of what the lane rule is doing (publish_markers, default off):
  // the frozen lane, the target strip to its right, and the walker's
  // predicted positions over the rollout. Display only; no effect on cost.
  bool publish_markers_{false};
  std::string marker_topic_{"/social_critic/lane_markers"};
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;
  int marker_tick_{0};
  void publishMarkers(
    const std::vector<Target> & targets, const rclcpp::Time & stamp,
    float robot_x, float robot_y);

  // Lane block for the GLOBAL planner (publish_lane_block_, default off).
  // The planner only sees the walker as a small disk and has no side rule,
  // so with a centred walker it picked a side by chance and pointed the
  // path the other way from the controller for the first metres (in one
  // open-space run for 2 m, which cost the 0.8 m target: bag
  // headon_hwreq_wide_trial1). The critic owns the lane and the side, so it
  // publishes the region the robot must NOT use - the lane itself and the
  // wrong side of it, from the walker back to lane_block_robot_gap_ short of
  // the robot - as a point cloud for the global costmap ONLY. The plan then
  // takes the same side as the controller from first sight. It must not be
  // fed to the local costmap: a lethal strip there trapped the robot
  // (headon_avoid_v5/v6).
  bool publish_lane_block_{false};
  std::string lane_block_topic_{"/social_critic/lane_block"};
  float lane_block_width_{1.5f};       // m onto the wrong side of the lane
  float lane_block_overlap_{0.30f};    // m onto the allowed side (covers the lane)
  float lane_block_robot_gap_{1.0f};   // m of clear lane left in front of the robot
  float lane_block_spacing_{0.05f};
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr block_pub_;
  int block_tick_{0};
  void publishLaneBlock(
    const std::vector<Target> & targets, const rclcpp::Time & stamp,
    float robot_x, float robot_y);

  rclcpp::Logger logger_{rclcpp::get_logger("SocialCritic")};
};

}  // namespace mppi::critics

#endif  // SOCIAL_CRITIC__SOCIAL_CRITIC_HPP_