/**
 * @file pid_controller.cpp
 * @brief PID controller implementation
 */

#include "pid_controller.h"
#include <algorithm>
#include <cmath>

namespace aircraft_controls {

PIDController::PIDController(const PIDConfig& config)
    : config_(config),
      error_(0.0f),
      error_prev_(0.0f),
      integral_(0.0f),
      derivative_(0.0f),
      derivative_filtered_(0.0f),
      output_(0.0f) {}

PIDController::PIDController()
    : PIDController(PIDConfig()) {}

float PIDController::compute(float setpoint, float measurement, float dt) {
    // Error calculation
    error_ = setpoint - measurement;

    // Proportional term
    float p_term = config_.gains.kp * error_;

    // Integral term with anti-windup
    integral_ += error_ * dt;
    integral_ = clamp(integral_, config_.integral_min, config_.integral_max);
    float i_term = config_.gains.ki * integral_;

    // Derivative term with filtering
    if (dt > 1e-6f) {  // Avoid division by zero
        derivative_ = (error_ - error_prev_) / dt;
    } else {
        derivative_ = 0.0f;
    }

    // Low-pass filter on derivative to reduce noise
    // filtered_deriv = alpha * new_deriv + (1-alpha) * old_filtered_deriv
    derivative_filtered_ = config_.derivative_filter_alpha * derivative_ +
                          (1.0f - config_.derivative_filter_alpha) * derivative_filtered_;

    float d_term = config_.gains.kd * derivative_filtered_;

    // Total output
    output_ = p_term + i_term + d_term;

    // Output saturation
    output_ = clamp(output_, config_.output_min, config_.output_max);

    // Save error for next iteration
    error_prev_ = error_;

    return output_;
}

void PIDController::reset() {
    error_ = 0.0f;
    error_prev_ = 0.0f;
    integral_ = 0.0f;
    derivative_ = 0.0f;
    derivative_filtered_ = 0.0f;
    output_ = 0.0f;
}

void PIDController::setGains(const PIDGains& gains) {
    config_.gains = gains;
}

PIDGains PIDController::getGains() const {
    return config_.gains;
}

float PIDController::clamp(float value, float min_val, float max_val) const {
    return std::max(min_val, std::min(value, max_val));
}

// ============================================================================
// MultiAxisPIDController Implementation
// ============================================================================

MultiAxisPIDController::MultiAxisPIDController(
    const PIDConfig& roll_config,
    const PIDConfig& pitch_config,
    const PIDConfig& yaw_config
)
    : roll_pid_(roll_config),
      pitch_pid_(pitch_config),
      yaw_pid_(yaw_config),
      last_output_() {}

MultiAxisPIDController::MultiAxisPIDController()
    : MultiAxisPIDController(PIDConfig(), PIDConfig(), PIDConfig()) {}

ControlOutput MultiAxisPIDController::compute(
    const Vector3& setpoint,
    const Vector3& measurement,
    float dt
) {
    // Compute each axis independently
    last_output_.roll = roll_pid_.compute(setpoint.x, measurement.x, dt);
    last_output_.pitch = pitch_pid_.compute(setpoint.y, measurement.y, dt);
    last_output_.yaw = yaw_pid_.compute(setpoint.z, measurement.z, dt);

    return last_output_;
}

void MultiAxisPIDController::reset() {
    roll_pid_.reset();
    pitch_pid_.reset();
    yaw_pid_.reset();
    last_output_ = ControlOutput();
}

void MultiAxisPIDController::setGains(int axis, const PIDGains& gains) {
    switch (axis) {
        case 0: roll_pid_.setGains(gains); break;
        case 1: pitch_pid_.setGains(gains); break;
        case 2: yaw_pid_.setGains(gains); break;
        default: break;  // Invalid axis, do nothing
    }
}

PIDGains MultiAxisPIDController::getGains(int axis) const {
    switch (axis) {
        case 0: return roll_pid_.getGains();
        case 1: return pitch_pid_.getGains();
        case 2: return yaw_pid_.getGains();
        default: return PIDGains();  // Invalid axis
    }
}

Vector3 MultiAxisPIDController::getError() const {
    return Vector3(
        roll_pid_.getError(),
        pitch_pid_.getError(),
        yaw_pid_.getError()
    );
}

Vector3 MultiAxisPIDController::getIntegral() const {
    return Vector3(
        roll_pid_.getIntegral(),
        pitch_pid_.getIntegral(),
        yaw_pid_.getIntegral()
    );
}

} // namespace aircraft_controls
