/**
 * @file pid_controller.h
 * @brief PID controller implementation for aircraft control
 *
 * High-performance C++ PID controller based on dRehmFlight architecture.
 * Designed for 100-500 Hz update rates on embedded systems.
 */

#ifndef PID_CONTROLLER_H
#define PID_CONTROLLER_H

#include <cstdint>

namespace aircraft_controls {

/**
 * @brief PID controller gains
 */
struct PIDGains {
    float kp;  ///< Proportional gain
    float ki;  ///< Integral gain
    float kd;  ///< Derivative gain

    PIDGains() : kp(0.0f), ki(0.0f), kd(0.0f) {}
    PIDGains(float p, float i, float d) : kp(p), ki(i), kd(d) {}
};

/**
 * @brief PID controller configuration
 */
struct PIDConfig {
    PIDGains gains;
    float output_min;  ///< Minimum output value (saturation)
    float output_max;  ///< Maximum output value (saturation)
    float integral_min;  ///< Integral wind-up limit (negative)
    float integral_max;  ///< Integral wind-up limit (positive)
    float derivative_filter_alpha;  ///< Low-pass filter coefficient for derivative (0-1)

    PIDConfig()
        : gains(0.0f, 0.0f, 0.0f),
          output_min(-1.0f),
          output_max(1.0f),
          integral_min(-10.0f),
          integral_max(10.0f),
          derivative_filter_alpha(0.1f) {}
};

/**
 * @brief High-performance PID controller
 *
 * Features:
 * - Anti-windup (integral clamping)
 * - Derivative filtering (low-pass)
 * - Output saturation
 * - Reset capability
 * - Optimized for real-time performance
 */
class PIDController {
public:
    /**
     * @brief Construct a new PID Controller
     * @param config PID configuration
     */
    explicit PIDController(const PIDConfig& config);

    /**
     * @brief Default constructor
     */
    PIDController();

    /**
     * @brief Compute PID output
     * @param setpoint Desired value
     * @param measurement Current measured value
     * @param dt Time step in seconds
     * @return Control output (saturated to output_min/max)
     */
    float compute(float setpoint, float measurement, float dt);

    /**
     * @brief Reset controller state (zeros integral and derivative)
     */
    void reset();

    /**
     * @brief Set PID gains
     * @param gains New PID gains
     */
    void setGains(const PIDGains& gains);

    /**
     * @brief Get current PID gains
     * @return Current gains
     */
    PIDGains getGains() const;

    /**
     * @brief Get current error
     * @return Current error (setpoint - measurement)
     */
    float getError() const { return error_; }

    /**
     * @brief Get integral term
     * @return Current integral accumulation
     */
    float getIntegral() const { return integral_; }

    /**
     * @brief Get derivative term
     * @return Current derivative (filtered)
     */
    float getDerivative() const { return derivative_filtered_; }

    /**
     * @brief Get last output
     * @return Last computed output
     */
    float getOutput() const { return output_; }

private:
    PIDConfig config_;

    // Controller state
    float error_;
    float error_prev_;
    float integral_;
    float derivative_;
    float derivative_filtered_;
    float output_;

    // Clamp value to range
    float clamp(float value, float min_val, float max_val) const;
};

/**
 * @brief 3-axis output for multi-axis PID controller
 */
struct ControlOutput {
    float roll;   ///< Roll control output
    float pitch;  ///< Pitch control output
    float yaw;    ///< Yaw control output

    ControlOutput() : roll(0.0f), pitch(0.0f), yaw(0.0f) {}
    ControlOutput(float r, float p, float y) : roll(r), pitch(p), yaw(y) {}
};

/**
 * @brief 3-axis vector for setpoint/measurement
 */
struct Vector3 {
    float x;  ///< X component (roll)
    float y;  ///< Y component (pitch)
    float z;  ///< Z component (yaw)

    Vector3() : x(0.0f), y(0.0f), z(0.0f) {}
    Vector3(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}
};

/**
 * @brief Multi-axis PID controller for 3-axis attitude control
 *
 * Controls roll, pitch, and yaw axes independently using separate PID controllers.
 * Designed for cascaded control architectures (e.g., angle mode → rate mode).
 */
class MultiAxisPIDController {
public:
    /**
     * @brief Construct multi-axis PID controller
     * @param roll_config Roll axis PID configuration
     * @param pitch_config Pitch axis PID configuration
     * @param yaw_config Yaw axis PID configuration
     */
    MultiAxisPIDController(
        const PIDConfig& roll_config,
        const PIDConfig& pitch_config,
        const PIDConfig& yaw_config
    );

    /**
     * @brief Default constructor
     */
    MultiAxisPIDController();

    /**
     * @brief Compute PID output for all 3 axes
     * @param setpoint Desired 3-axis values [roll, pitch, yaw]
     * @param measurement Current measured 3-axis values [roll, pitch, yaw]
     * @param dt Time step in seconds
     * @return Control output for all 3 axes
     */
    ControlOutput compute(const Vector3& setpoint, const Vector3& measurement, float dt);

    /**
     * @brief Reset all axis controllers
     */
    void reset();

    /**
     * @brief Set gains for specific axis
     * @param axis Axis index (0=roll, 1=pitch, 2=yaw)
     * @param gains New gains for that axis
     */
    void setGains(int axis, const PIDGains& gains);

    /**
     * @brief Get gains for specific axis
     * @param axis Axis index (0=roll, 1=pitch, 2=yaw)
     * @return Current gains for that axis
     */
    PIDGains getGains(int axis) const;

    /**
     * @brief Get current error for all axes
     * @return Error vector [roll_err, pitch_err, yaw_err]
     */
    Vector3 getError() const;

    /**
     * @brief Get integral terms for all axes
     * @return Integral vector [roll_int, pitch_int, yaw_int]
     */
    Vector3 getIntegral() const;

    /**
     * @brief Get last output for all axes
     * @return Last computed output
     */
    ControlOutput getOutput() const { return last_output_; }

private:
    PIDController roll_pid_;    ///< Roll axis PID
    PIDController pitch_pid_;   ///< Pitch axis PID
    PIDController yaw_pid_;     ///< Yaw axis PID
    ControlOutput last_output_; ///< Last computed output
};

} // namespace aircraft_controls

#endif // PID_CONTROLLER_H
