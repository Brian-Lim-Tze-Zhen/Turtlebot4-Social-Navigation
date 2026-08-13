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
#include <vector>

#include "nav2_mppi_controller/critic_function.hpp"
#include "nav2_mppi_controller/models/state.hpp"
#include "nav2_mppi_controller/tools/utils.hpp"

#include "std_msgs/msg/string.hpp"
#include "tf2_ros/buffer.h"

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
};

class SocialCritic : public CriticFunction
{
public:
  void initialize() override;
  void score(CriticData & data) override;

private:
  void positionsCallback(const std_msgs::msg::String::SharedPtr msg);

  // Returns everything worth penalising, in the costmap frame: current
  // positions, KF predictions, and coasted extrapolations of tracks that
  // have gone silent but not yet exceeded max_coast_time_.
  std::vector<Target> collectTargets();

  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr sub_;
  std::shared_ptr<tf2_ros::Buffer> tf_buffer_;

  std::mutex people_mutex_;
  std::vector<PersonState> people_;

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

  rclcpp::Logger logger_{rclcpp::get_logger("SocialCritic")};
};

}  // namespace mppi::critics

#endif  // SOCIAL_CRITIC__SOCIAL_CRITIC_HPP_