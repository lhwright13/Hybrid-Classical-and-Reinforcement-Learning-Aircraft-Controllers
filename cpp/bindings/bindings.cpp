/**
 * @file bindings.cpp
 * @brief Pybind11 bindings for aircraft controls C++ library
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "pid_controller.h"

namespace py = pybind11;
using namespace aircraft_controls;

PYBIND11_MODULE(aircraft_controls_bindings, m) {
    m.doc() = "Python bindings for aircraft controls C++ library";

    // PIDGains struct
    py::class_<PIDGains>(m, "PIDGains", "PID controller gains")
        .def(py::init<>(), "Default constructor (all gains = 0)")
        .def(py::init<float, float, float>(),
             py::arg("kp"), py::arg("ki"), py::arg("kd"),
             "Constructor with gains")
        .def_readwrite("kp", &PIDGains::kp, "Proportional gain")
        .def_readwrite("ki", &PIDGains::ki, "Integral gain")
        .def_readwrite("kd", &PIDGains::kd, "Derivative gain")
        .def("__repr__", [](const PIDGains& g) {
            return "PIDGains(kp=" + std::to_string(g.kp) +
                   ", ki=" + std::to_string(g.ki) +
                   ", kd=" + std::to_string(g.kd) + ")";
        });

    // PIDConfig struct
    py::class_<PIDConfig>(m, "PIDConfig", "PID controller configuration")
        .def(py::init<>(), "Default constructor")
        .def_readwrite("gains", &PIDConfig::gains, "PID gains")
        .def_readwrite("output_min", &PIDConfig::output_min, "Minimum output value")
        .def_readwrite("output_max", &PIDConfig::output_max, "Maximum output value")
        .def_readwrite("integral_min", &PIDConfig::integral_min, "Integral wind-up limit (min)")
        .def_readwrite("integral_max", &PIDConfig::integral_max, "Integral wind-up limit (max)")
        .def_readwrite("derivative_filter_alpha", &PIDConfig::derivative_filter_alpha,
                      "Derivative low-pass filter coefficient (0-1)")
        .def("__repr__", [](const PIDConfig& c) {
            return "PIDConfig(kp=" + std::to_string(c.gains.kp) +
                   ", ki=" + std::to_string(c.gains.ki) +
                   ", kd=" + std::to_string(c.gains.kd) +
                   ", output_range=[" + std::to_string(c.output_min) +
                   ", " + std::to_string(c.output_max) + "])";
        });

    // PIDController class
    py::class_<PIDController>(m, "PIDController", "High-performance PID controller")
        .def(py::init<>(), "Default constructor")
        .def(py::init<const PIDConfig&>(),
             py::arg("config"),
             "Constructor with configuration")
        .def("compute", &PIDController::compute,
             py::arg("setpoint"),
             py::arg("measurement"),
             py::arg("dt"),
             "Compute PID output\n\n"
             "Args:\n"
             "    setpoint: Desired value\n"
             "    measurement: Current measured value\n"
             "    dt: Time step in seconds\n\n"
             "Returns:\n"
             "    Control output (saturated to output_min/max)")
        .def("reset", &PIDController::reset,
             "Reset controller state (zeros integral and derivative)")
        .def("set_gains", &PIDController::setGains,
             py::arg("gains"),
             "Set PID gains")
        .def("get_gains", &PIDController::getGains,
             "Get current PID gains")
        .def("get_error", &PIDController::getError,
             "Get current error (setpoint - measurement)")
        .def("get_integral", &PIDController::getIntegral,
             "Get integral term")
        .def("get_derivative", &PIDController::getDerivative,
             "Get derivative term (filtered)")
        .def("get_output", &PIDController::getOutput,
             "Get last computed output")
        .def("__repr__", [](const PIDController& pid) {
            auto gains = pid.getGains();
            return "PIDController(kp=" + std::to_string(gains.kp) +
                   ", ki=" + std::to_string(gains.ki) +
                   ", kd=" + std::to_string(gains.kd) + ")";
        });

    // Vector3 struct
    py::class_<Vector3>(m, "Vector3", "3-axis vector for setpoint/measurement")
        .def(py::init<>(), "Default constructor (all zeros)")
        .def(py::init<float, float, float>(),
             py::arg("x"), py::arg("y"), py::arg("z"),
             "Constructor with components")
        .def_readwrite("x", &Vector3::x, "X component (roll)")
        .def_readwrite("y", &Vector3::y, "Y component (pitch)")
        .def_readwrite("z", &Vector3::z, "Z component (yaw)")
        .def("__repr__", [](const Vector3& v) {
            return "Vector3(x=" + std::to_string(v.x) +
                   ", y=" + std::to_string(v.y) +
                   ", z=" + std::to_string(v.z) + ")";
        });

    // ControlOutput struct
    py::class_<ControlOutput>(m, "ControlOutput", "3-axis control output")
        .def(py::init<>(), "Default constructor (all zeros)")
        .def(py::init<float, float, float>(),
             py::arg("roll"), py::arg("pitch"), py::arg("yaw"),
             "Constructor with components")
        .def_readwrite("roll", &ControlOutput::roll, "Roll control output")
        .def_readwrite("pitch", &ControlOutput::pitch, "Pitch control output")
        .def_readwrite("yaw", &ControlOutput::yaw, "Yaw control output")
        .def("__repr__", [](const ControlOutput& o) {
            return "ControlOutput(roll=" + std::to_string(o.roll) +
                   ", pitch=" + std::to_string(o.pitch) +
                   ", yaw=" + std::to_string(o.yaw) + ")";
        });

    // MultiAxisPIDController class
    py::class_<MultiAxisPIDController>(m, "MultiAxisPIDController",
                                       "Multi-axis PID controller for 3-axis attitude control")
        .def(py::init<>(), "Default constructor")
        .def(py::init<const PIDConfig&, const PIDConfig&, const PIDConfig&>(),
             py::arg("roll_config"),
             py::arg("pitch_config"),
             py::arg("yaw_config"),
             "Constructor with axis configurations")
        .def("compute", &MultiAxisPIDController::compute,
             py::arg("setpoint"),
             py::arg("measurement"),
             py::arg("dt"),
             "Compute PID output for all 3 axes\n\n"
             "Args:\n"
             "    setpoint: Desired 3-axis values (Vector3)\n"
             "    measurement: Current measured 3-axis values (Vector3)\n"
             "    dt: Time step in seconds\n\n"
             "Returns:\n"
             "    ControlOutput for all 3 axes")
        .def("reset", &MultiAxisPIDController::reset,
             "Reset all axis controllers")
        .def("set_gains", &MultiAxisPIDController::setGains,
             py::arg("axis"),
             py::arg("gains"),
             "Set gains for specific axis (0=roll, 1=pitch, 2=yaw)")
        .def("get_gains", &MultiAxisPIDController::getGains,
             py::arg("axis"),
             "Get gains for specific axis (0=roll, 1=pitch, 2=yaw)")
        .def("get_error", &MultiAxisPIDController::getError,
             "Get current error for all axes")
        .def("get_integral", &MultiAxisPIDController::getIntegral,
             "Get integral terms for all axes")
        .def("get_output", &MultiAxisPIDController::getOutput,
             "Get last computed output for all axes")
        .def("__repr__", [](const MultiAxisPIDController& pid) {
            auto roll_gains = pid.getGains(0);
            auto pitch_gains = pid.getGains(1);
            auto yaw_gains = pid.getGains(2);
            return "MultiAxisPIDController(roll_kp=" + std::to_string(roll_gains.kp) +
                   ", pitch_kp=" + std::to_string(pitch_gains.kp) +
                   ", yaw_kp=" + std::to_string(yaw_gains.kp) + ")";
        });

    // Module version
    m.attr("__version__") = "1.1.0";  // Bumped version for multi-axis support
}
