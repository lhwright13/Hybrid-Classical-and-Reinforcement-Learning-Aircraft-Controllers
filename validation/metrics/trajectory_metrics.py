"""Trajectory comparison metrics for validation."""

import numpy as np
import pandas as pd
from typing import Dict, Any
from scipy import stats


def compute_rmse(data1: np.ndarray, data2: np.ndarray) -> float:
    """Compute Root Mean Square Error.

    Args:
        data1: First data array
        data2: Second data array

    Returns:
        RMSE value
    """
    return np.sqrt(np.mean((data1 - data2) ** 2))


def compute_nrmse(data1: np.ndarray, data2: np.ndarray) -> float:
    """Compute Normalized RMSE (percentage of range).

    Args:
        data1: First data array (reference)
        data2: Second data array

    Returns:
        NRMSE as percentage
    """
    rmse = compute_rmse(data1, data2)
    data_range = np.max(data1) - np.min(data1)

    if data_range == 0:
        return 0.0

    return (rmse / data_range) * 100.0


def compute_correlation(data1: np.ndarray, data2: np.ndarray) -> float:
    """Compute Pearson correlation coefficient.

    Args:
        data1: First data array
        data2: Second data array

    Returns:
        Correlation coefficient (-1 to 1)
    """
    if len(data1) < 2:
        return 0.0

    r, _ = stats.pearsonr(data1, data2)
    return r


def compute_max_error(data1: np.ndarray, data2: np.ndarray) -> float:
    """Compute maximum absolute error.

    Args:
        data1: First data array
        data2: Second data array

    Returns:
        Maximum absolute error
    """
    return np.max(np.abs(data1 - data2))


def compare_trajectories(df_simplified: pd.DataFrame, df_jsbsim: pd.DataFrame) -> Dict[str, Any]:
    """Compare two trajectory DataFrames and compute all metrics.

    Args:
        df_simplified: Trajectory from simplified 6-DOF
        df_jsbsim: Trajectory from JSBSim

    Returns:
        Dictionary with all comparison metrics
    """
    # Ensure same length (interpolate if needed)
    if len(df_simplified) != len(df_jsbsim):
        # Simple approach: use minimum length
        min_len = min(len(df_simplified), len(df_jsbsim))
        df_simplified = df_simplified.iloc[:min_len]
        df_jsbsim = df_jsbsim.iloc[:min_len]

    metrics = {}

    # Position metrics (NED frame)
    for axis, name in zip(['north', 'east', 'down'], ['North', 'East', 'Down']):
        metrics[f'position_{axis}_rmse'] = compute_rmse(
            df_simplified[axis].values,
            df_jsbsim[axis].values
        )
        metrics[f'position_{axis}_correlation'] = compute_correlation(
            df_simplified[axis].values,
            df_jsbsim[axis].values
        )
        metrics[f'position_{axis}_max_error'] = compute_max_error(
            df_simplified[axis].values,
            df_jsbsim[axis].values
        )

    # Overall position RMSE (3D Euclidean distance)
    pos_simple = df_simplified[['north', 'east', 'down']].values
    pos_jsbsim = df_jsbsim[['north', 'east', 'down']].values
    pos_errors = np.linalg.norm(pos_simple - pos_jsbsim, axis=1)
    metrics['position_3d_rmse'] = np.sqrt(np.mean(pos_errors ** 2))
    metrics['position_3d_max_error'] = np.max(pos_errors)

    # Altitude (more intuitive than down)
    metrics['altitude_rmse'] = compute_rmse(
        df_simplified['altitude'].values,
        df_jsbsim['altitude'].values
    )
    metrics['altitude_correlation'] = compute_correlation(
        df_simplified['altitude'].values,
        df_jsbsim['altitude'].values
    )

    # Velocity metrics
    for axis in ['u', 'v', 'w']:
        metrics[f'velocity_{axis}_rmse'] = compute_rmse(
            df_simplified[axis].values,
            df_jsbsim[axis].values
        )

    metrics['airspeed_rmse'] = compute_rmse(
        df_simplified['airspeed'].values,
        df_jsbsim['airspeed'].values
    )

    # Attitude metrics (convert to degrees for interpretability)
    for axis, name in zip(['roll', 'pitch', 'yaw'], ['Roll', 'Pitch', 'Yaw']):
        simple_deg = np.degrees(df_simplified[axis].values)
        jsbsim_deg = np.degrees(df_jsbsim[axis].values)

        # Handle yaw wraparound (-180 to 180)
        if axis == 'yaw':
            simple_deg = np.unwrap(simple_deg, period=360)
            jsbsim_deg = np.unwrap(jsbsim_deg, period=360)

        metrics[f'attitude_{axis}_rmse_deg'] = compute_rmse(simple_deg, jsbsim_deg)
        metrics[f'attitude_{axis}_correlation'] = compute_correlation(simple_deg, jsbsim_deg)
        metrics[f'attitude_{axis}_max_error_deg'] = compute_max_error(simple_deg, jsbsim_deg)

    # Angular rate metrics (convert to deg/s)
    for axis in ['p', 'q', 'r']:
        simple_dps = np.degrees(df_simplified[axis].values)
        jsbsim_dps = np.degrees(df_jsbsim[axis].values)

        metrics[f'rate_{axis}_rmse_dps'] = compute_rmse(simple_dps, jsbsim_dps)
        metrics[f'rate_{axis}_correlation'] = compute_correlation(simple_dps, jsbsim_dps)

    # Summary statistics
    metrics['mean_position_correlation'] = np.mean([
        metrics['position_north_correlation'],
        metrics['position_east_correlation'],
        metrics['position_down_correlation']
    ])

    metrics['mean_attitude_correlation'] = np.mean([
        metrics['attitude_roll_correlation'],
        metrics['attitude_pitch_correlation'],
        metrics['attitude_yaw_correlation']
    ])

    metrics['overall_correlation'] = np.mean([
        metrics['mean_position_correlation'],
        metrics['mean_attitude_correlation']
    ])

    return metrics


def format_metrics_summary(metrics: Dict[str, Any]) -> str:
    """Format metrics as human-readable summary.

    Args:
        metrics: Metrics dictionary from compare_trajectories()

    Returns:
        Formatted string summary
    """
    lines = []
    lines.append("=" * 60)
    lines.append("TRAJECTORY COMPARISON METRICS")
    lines.append("=" * 60)

    lines.append("\nPosition Errors:")
    lines.append(f"  3D RMSE:      {metrics['position_3d_rmse']:8.3f} m")
    lines.append(f"  3D Max Error: {metrics['position_3d_max_error']:8.3f} m")
    lines.append(f"  Altitude RMSE: {metrics['altitude_rmse']:7.3f} m")
    lines.append(f"  Mean Correlation: {metrics['mean_position_correlation']:5.3f}")

    lines.append("\nAttitude Errors:")
    lines.append(f"  Roll RMSE:    {metrics['attitude_roll_rmse_deg']:8.3f} deg")
    lines.append(f"  Pitch RMSE:   {metrics['attitude_pitch_rmse_deg']:8.3f} deg")
    lines.append(f"  Yaw RMSE:     {metrics['attitude_yaw_rmse_deg']:8.3f} deg")
    lines.append(f"  Mean Correlation: {metrics['mean_attitude_correlation']:5.3f}")

    lines.append("\nOverall:")
    lines.append(f"  Overall Correlation: {metrics['overall_correlation']:5.3f}")

    lines.append("=" * 60)

    return "\n".join(lines)
