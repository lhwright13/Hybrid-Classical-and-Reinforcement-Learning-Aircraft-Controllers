"""3D visualization of multi-aircraft fleet using PyVista.

This module provides real-time 3D visualization of multiple aircraft
with trajectory traces, formation lines, and camera controls.
"""

import numpy as np
import pyvista as pv
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

from controllers.types import AircraftState


class FleetVisualizer3D:
    """3D visualization of multi-aircraft fleet.

    Features:
    - Real-time aircraft rendering with distinct colors
    - Trajectory traces per aircraft
    - Formation lines between aircraft
    - Multiple camera modes (follow, fleet center, free orbit)
    - Ground plane with altitude reference
    - Performance: 30 FPS with 10+ aircraft

    Example:
        >>> viz = FleetVisualizer3D(aircraft_ids=["001", "002", "003"])
        >>> for i in range(1000):
        ...     for aircraft_id in ["001", "002", "003"]:
        ...         state = aircraft[aircraft_id].step(0.01)
        ...         viz.update(aircraft_id, state)
        ...     viz.render()
    """

    # Color palette (same as plotter for consistency)
    COLORS = [
        (31, 119, 180),   # Blue
        (255, 127, 14),   # Orange
        (44, 160, 44),    # Green
        (214, 39, 40),    # Red
        (148, 103, 189),  # Purple
        (140, 86, 75),    # Brown
        (227, 119, 194),  # Pink
        (127, 127, 127),  # Gray
        (188, 189, 34),   # Olive
        (23, 190, 207),   # Cyan
    ]

    def __init__(
        self,
        aircraft_ids: List[str],
        trajectory_length: int = 500,
        show_formation_lines: bool = True,
        window_size: Tuple[int, int] = (1024, 768)
    ):
        """Initialize 3D fleet visualizer.

        Args:
            aircraft_ids: List of aircraft IDs
            trajectory_length: Number of points in trajectory trace
            show_formation_lines: Show lines between aircraft
            window_size: Window size (width, height)
        """
        self.aircraft_ids = list(aircraft_ids)
        self.trajectory_length = trajectory_length
        self.show_formation_lines = show_formation_lines

        # Color assignment
        self.colors = {
            aid: np.array(self.COLORS[i % len(self.COLORS)]) / 255.0
            for i, aid in enumerate(aircraft_ids)
        }

        # Trajectory buffers
        self.trajectories: Dict[str, List[np.ndarray]] = defaultdict(list)

        # Current positions
        self.positions: Dict[str, np.ndarray] = {}

        # Current attitudes (for orientation)
        self.attitudes: Dict[str, np.ndarray] = {}

        # Create plotter
        self.plotter = pv.Plotter(window_size=window_size)
        self.plotter.set_background('lightblue')

        # Initialize scene
        self._setup_scene()

        # Actor references
        self.aircraft_actors: Dict[str, pv.Actor] = {}
        self.trajectory_actors: Dict[str, pv.Actor] = {}
        self.label_actors: Dict[str, pv.Actor] = {}

    def _setup_scene(self) -> None:
        """Set up the 3D scene with ground plane and lights."""
        # Ground plane (10 km x 10 km grid)
        grid_size = 5000  # meters
        grid = pv.Plane(
            center=(0, 0, 0),
            direction=(0, 0, 1),
            i_size=grid_size * 2,
            j_size=grid_size * 2,
            i_resolution=20,
            j_resolution=20
        )
        self.plotter.add_mesh(grid, color='lightgray', opacity=0.3, show_edges=True)

        # Axes
        self.plotter.add_axes(
            xlabel='East (m)',
            ylabel='North (m)',
            zlabel='Altitude (m)',
            line_width=3
        )

        # Lighting
        self.plotter.add_light(pv.Light(position=(1000, 1000, 1000), intensity=0.6))

        # Camera setup (initial: isometric view)
        self.plotter.camera.position = (1000, 1000, 500)
        self.plotter.camera.focal_point = (0, 0, 100)
        self.plotter.camera.up = (0, 0, 1)

    def _create_aircraft_model(self, scale: float = 5.0) -> pv.PolyData:
        """Create simple aircraft model (arrow shape).

        Args:
            scale: Model scale factor

        Returns:
            Aircraft mesh
        """
        # Create arrow pointing in +X direction (forward)
        arrow = pv.Arrow(
            start=(0, 0, 0),
            direction=(1, 0, 0),
            tip_length=0.3,
            tip_radius=0.15,
            shaft_radius=0.05,
            scale=scale
        )
        return arrow

    def register_aircraft(self, aircraft_id: str) -> None:
        """Register new aircraft for visualization.

        Args:
            aircraft_id: Aircraft identifier
        """
        if aircraft_id not in self.aircraft_ids:
            self.aircraft_ids.append(aircraft_id)
            self.colors[aircraft_id] = np.array(
                self.COLORS[len(self.aircraft_ids) % len(self.COLORS)]
            ) / 255.0

    def update(self, aircraft_id: str, state: AircraftState) -> None:
        """Update aircraft state.

        Args:
            aircraft_id: Aircraft identifier
            state: Aircraft state
        """
        # Auto-register if needed
        if aircraft_id not in self.aircraft_ids:
            self.register_aircraft(aircraft_id)

        # Store position (East, North, Up)
        # NED -> ENU for PyVista (Z-up convention)
        position = np.array([
            state.position[1],  # East
            state.position[0],  # North
            -state.position[2]  # Up (negate down to get up)
        ])
        self.positions[aircraft_id] = position

        # Store attitude
        self.attitudes[aircraft_id] = state.attitude.copy()

        # Update trajectory
        self.trajectories[aircraft_id].append(position.copy())
        if len(self.trajectories[aircraft_id]) > self.trajectory_length:
            self.trajectories[aircraft_id].pop(0)

    def render(self, camera_mode: str = 'fleet_center') -> None:
        """Render the current scene.

        Args:
            camera_mode: Camera mode ('fleet_center', 'follow', 'free')
        """
        # Remove old actors
        for actor in list(self.aircraft_actors.values()):
            self.plotter.remove_actor(actor)
        for actor in list(self.trajectory_actors.values()):
            self.plotter.remove_actor(actor)
        for actor in list(self.label_actors.values()):
            self.plotter.remove_actor(actor)

        self.aircraft_actors.clear()
        self.trajectory_actors.clear()
        self.label_actors.clear()

        # Render each aircraft
        for aircraft_id in self.aircraft_ids:
            if aircraft_id not in self.positions:
                continue

            position = self.positions[aircraft_id]
            attitude = self.attitudes.get(aircraft_id, np.zeros(3))
            color = self.colors[aircraft_id]

            # Create and position aircraft model
            aircraft = self._create_aircraft_model(scale=10.0)

            # Rotation matrix from body frame to world frame (simplified)
            # For now, just apply yaw rotation (around Z-axis)
            yaw = attitude[2]
            aircraft = aircraft.rotate_z(np.degrees(yaw), point=(0, 0, 0))

            # Translate to current position
            aircraft = aircraft.translate(position)

            # Add to scene
            actor = self.plotter.add_mesh(
                aircraft,
                color=color,
                opacity=0.9,
                smooth_shading=True
            )
            self.aircraft_actors[aircraft_id] = actor

            # Add label
            label_pos = position + np.array([0, 0, 15])  # Above aircraft
            label_actor = self.plotter.add_point_labels(
                [label_pos],
                [aircraft_id],
                point_size=0,
                font_size=12,
                text_color=color,
                bold=True
            )
            self.label_actors[aircraft_id] = label_actor

            # Render trajectory
            if len(self.trajectories[aircraft_id]) > 1:
                traj_points = np.array(self.trajectories[aircraft_id])
                traj_line = pv.Spline(traj_points, n_points=len(traj_points))
                traj_actor = self.plotter.add_mesh(
                    traj_line,
                    color=color,
                    line_width=2,
                    opacity=0.6
                )
                self.trajectory_actors[aircraft_id] = traj_actor

        # Formation lines
        if self.show_formation_lines and len(self.positions) > 1:
            positions_list = list(self.positions.values())
            for i in range(len(positions_list)):
                for j in range(i + 1, len(positions_list)):
                    line = pv.Line(positions_list[i], positions_list[j])
                    self.plotter.add_mesh(
                        line,
                        color='white',
                        line_width=1,
                        opacity=0.3
                    )

        # Update camera
        if camera_mode == 'fleet_center' and self.positions:
            # Center on fleet centroid
            centroid = np.mean(list(self.positions.values()), axis=0)
            self.plotter.camera.focal_point = centroid

            # Position camera at appropriate distance
            max_spread = 0
            if len(self.positions) > 1:
                positions_array = np.array(list(self.positions.values()))
                max_spread = np.max(np.linalg.norm(
                    positions_array - centroid, axis=1
                ))

            distance = max(500, max_spread * 3)
            camera_pos = centroid + np.array([distance, distance, distance * 0.5])
            self.plotter.camera.position = camera_pos

        elif camera_mode == 'follow' and self.positions:
            # Follow first aircraft
            first_id = self.aircraft_ids[0]
            if first_id in self.positions:
                pos = self.positions[first_id]
                self.plotter.camera.focal_point = pos
                # Position camera behind and above
                attitude = self.attitudes.get(first_id, np.zeros(3))
                yaw = attitude[2]
                offset = np.array([
                    -50 * np.cos(yaw),
                    -50 * np.sin(yaw),
                    30
                ])
                self.plotter.camera.position = pos + offset

        # Render frame
        self.plotter.render()

    def show(self, interactive: bool = True) -> None:
        """Show visualization window.

        Args:
            interactive: Enable interactive mode (user can rotate camera)
        """
        if interactive:
            self.plotter.show(interactive=True, auto_close=False)
        else:
            self.plotter.show(auto_close=False)

    def close(self) -> None:
        """Close the visualization window."""
        self.plotter.close()

    def screenshot(self, filename: str) -> None:
        """Save screenshot to file.

        Args:
            filename: Output filename (e.g., 'fleet_view.png')
        """
        self.plotter.screenshot(filename)
        print(f"Saved screenshot to {filename}")

    def set_camera_mode(self, mode: str, aircraft_id: Optional[str] = None) -> None:
        """Set camera tracking mode.

        Args:
            mode: 'fleet_center', 'follow', or 'free'
            aircraft_id: Aircraft to follow (for 'follow' mode)
        """
        if mode == 'follow' and aircraft_id:
            # Update which aircraft to follow
            if aircraft_id in self.aircraft_ids:
                # Move to front of list
                self.aircraft_ids.remove(aircraft_id)
                self.aircraft_ids.insert(0, aircraft_id)

    def get_active_aircraft_count(self) -> int:
        """Get number of active aircraft in visualization."""
        return len(self.positions)


class SimpleFleetVisualizer3D:
    """Simplified 3D visualizer using matplotlib (fallback if PyVista unavailable).

    Uses matplotlib 3D plotting as a lightweight alternative.
    """

    def __init__(self, aircraft_ids: List[str]):
        """Initialize simple 3D visualizer.

        Args:
            aircraft_ids: List of aircraft IDs
        """
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        self.aircraft_ids = aircraft_ids
        self.positions: Dict[str, np.ndarray] = {}
        self.trajectories: Dict[str, List[np.ndarray]] = defaultdict(list)

        # Create figure
        self.fig = plt.figure(figsize=(12, 9))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_xlabel('East (m)')
        self.ax.set_ylabel('North (m)')
        self.ax.set_zlabel('Altitude (m)')
        self.ax.set_title('Multi-Aircraft 3D View')

        # Color palette
        self.colors = {
            aid: f'C{i}'
            for i, aid in enumerate(aircraft_ids)
        }

    def update(self, aircraft_id: str, state: AircraftState) -> None:
        """Update aircraft position."""
        position = np.array([
            state.position[1],  # East
            state.position[0],  # North
            -state.position[2]  # Up
        ])
        self.positions[aircraft_id] = position
        self.trajectories[aircraft_id].append(position.copy())

        # Limit trajectory length
        if len(self.trajectories[aircraft_id]) > 500:
            self.trajectories[aircraft_id].pop(0)

    def render(self) -> None:
        """Render the scene."""
        self.ax.clear()
        self.ax.set_xlabel('East (m)')
        self.ax.set_ylabel('North (m)')
        self.ax.set_zlabel('Altitude (m)')

        # Plot each aircraft
        for aircraft_id in self.aircraft_ids:
            if aircraft_id not in self.positions:
                continue

            pos = self.positions[aircraft_id]
            color = self.colors[aircraft_id]

            # Current position
            self.ax.scatter(
                pos[0], pos[1], pos[2],
                c=color, marker='o', s=100, label=aircraft_id
            )

            # Trajectory
            if len(self.trajectories[aircraft_id]) > 1:
                traj = np.array(self.trajectories[aircraft_id])
                self.ax.plot(
                    traj[:, 0], traj[:, 1], traj[:, 2],
                    c=color, linewidth=2, alpha=0.6
                )

        self.ax.legend()
        plt.draw()
        plt.pause(0.001)

    def show(self) -> None:
        """Show plot."""
        import matplotlib.pyplot as plt
        plt.show()

    def close(self) -> None:
        """Close plot."""
        import matplotlib.pyplot as plt
        plt.close(self.fig)
