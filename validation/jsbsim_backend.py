"""JSBSim backend implementing AircraftInterface for physics validation.

This backend wraps JSBSim to provide high-fidelity flight dynamics for
comparison against the simplified 6-DOF model.
"""

import os
import numpy as np
from typing import Optional, Dict, Any
from pathlib import Path

import jsbsim

from interfaces.aircraft import AircraftInterface
from controllers.types import AircraftState, ControlSurfaces


class JSBSimBackend(AircraftInterface):
    """JSBSim flight dynamics backend for validation.

    This class wraps JSBSim's FGFDMExec to provide high-fidelity physics
    simulation compatible with our AircraftInterface.

    Attributes:
        fdm: JSBSim FGFDMExec instance
        aircraft_name: Name of aircraft model (e.g., 'c172x')
        dt_physics: JSBSim integration timestep (default 1/120 Hz)
        initial_conditions: Dict of initial state parameters
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize JSBSim backend.

        Args:
            config: Configuration dictionary with keys:
                - aircraft: Aircraft model name (default 'c172x')
                - dt_physics: JSBSim timestep in seconds (default 1/120)
                - jsbsim_root: Path to JSBSim data (default auto-detect)
                - initial_lat: Initial latitude in degrees (default 37.0)
                - initial_lon: Initial longitude in degrees (default -122.0)
                - initial_altitude: Initial altitude MSL in meters (default 100.0)
        """
        self.config = config or {}

        # Aircraft selection (c172p is a good default - small aircraft, well-documented)
        self.aircraft_name = self.config.get('aircraft', 'c172p')

        # Physics timestep
        self.dt_physics = self.config.get('dt_physics', 1.0/120.0)

        # JSBSim root directory (contains aircraft/, engines/, etc.)
        jsbsim_root = self.config.get('jsbsim_root', None)
        if jsbsim_root is None:
            # Try to find JSBSim data directory
            jsbsim_root = self._find_jsbsim_root()

        # Initialize JSBSim FDM
        self.fdm = jsbsim.FGFDMExec(jsbsim_root)
        self.fdm.set_debug_level(0)  # Quiet output

        # If using custom rc_plane, set up custom paths
        if self.aircraft_name == 'rc_plane':
            # Use custom aircraft directory
            custom_aircraft_dir = Path(__file__).parent / 'aircraft_models'
            if custom_aircraft_dir.exists():
                self.fdm.set_aircraft_path(str(custom_aircraft_dir))
                # Also set engine path for custom engine files
                engine_path = custom_aircraft_dir / 'engine'
                if engine_path.exists():
                    self.fdm.set_engine_path(str(engine_path))

        # Load aircraft model
        if not self.fdm.load_model(self.aircraft_name):
            raise RuntimeError(f"Failed to load JSBSim aircraft model: {self.aircraft_name}")

        # Set simulation timestep
        self.fdm.set_dt(self.dt_physics)

        # Initialize default initial conditions
        self.initial_lat = self.config.get('initial_lat', 37.0)  # deg
        self.initial_lon = self.config.get('initial_lon', -122.0)  # deg
        self.initial_altitude_msl = self.config.get('initial_altitude', 100.0)  # meters MSL

        # NED origin for local coordinate conversion
        self.origin_lat = self.initial_lat
        self.origin_lon = self.initial_lon
        self.origin_alt = self.initial_altitude_msl

        # Current state cache
        self._current_state: Optional[AircraftState] = None

        # Initialize to default state
        self.reset()

    def _find_jsbsim_root(self) -> str:
        """Attempt to find JSBSim data directory.

        Returns:
            Path to JSBSim root directory
        """
        # First, try to find JSBSim package installation
        try:
            import jsbsim
            import site

            # Check site-packages for jsbsim directory
            for site_dir in site.getsitepackages():
                jsbsim_path = os.path.join(site_dir, 'jsbsim')
                if os.path.exists(os.path.join(jsbsim_path, 'aircraft')):
                    return jsbsim_path
        except Exception:
            pass

        # Check common system locations
        possible_paths = [
            '/usr/local/share/JSBSim',
            '/usr/share/JSBSim',
            str(Path.home() / 'JSBSim'),
            '.',  # Current directory
        ]

        for path in possible_paths:
            if os.path.exists(os.path.join(path, 'aircraft')):
                return path

        # Default to current directory (JSBSim will use its internal models)
        return '.'

    def reset(self, initial_state: Optional[AircraftState] = None) -> AircraftState:
        """Reset simulation to initial conditions.

        Args:
            initial_state: Optional initial state. If None, uses default IC.

        Returns:
            Initial aircraft state
        """
        # Reset FDM
        self.fdm.reset_to_initial_conditions(0)

        if initial_state is not None:
            # Set custom initial conditions from AircraftState
            self._set_state_from_aircraft_state(initial_state)
        else:
            # Use default initial conditions via properties
            # Position (geodetic)
            self.fdm['ic/lat-gc-deg'] = self.initial_lat
            self.fdm['ic/long-gc-deg'] = self.initial_lon
            self.fdm['ic/h-sl-ft'] = self.initial_altitude_msl * 3.28084  # m to ft

            # Attitude
            self.fdm['ic/phi-deg'] = 0.0  # Roll
            self.fdm['ic/theta-deg'] = 3.0  # Pitch (3° nose up for level cruise)
            self.fdm['ic/psi-true-deg'] = 0.0  # Yaw/heading

            # Velocities (body frame) - use safe cruise speed above stall
            # C172P stall speed ~48 knots, cruise ~100 knots, use 60 knots = 30 m/s
            self.fdm['ic/u-fps'] = 30.0 * 3.28084  # 30 m/s forward velocity (safe above stall)
            self.fdm['ic/v-fps'] = 0.0
            self.fdm['ic/w-fps'] = 0.0

            # Set initial throttle for level flight (~60% for cruise)
            self.fdm['ic/throttle-pos'] = 0.6

            # Rates
            self.fdm['ic/p-rad_sec'] = 0.0
            self.fdm['ic/q-rad_sec'] = 0.0
            self.fdm['ic/r-rad_sec'] = 0.0

            # Apply initial conditions
            self.fdm.run_ic()

        # Update state cache
        self._current_state = self.get_state()

        return self._current_state

    def _set_state_from_aircraft_state(self, state: AircraftState):
        """Set JSBSim state from AircraftState (NED coordinates).

        Args:
            state: Aircraft state in NED coordinates
        """
        # Convert NED position to geodetic (lat/lon/alt)
        lat, lon, alt_msl = self._ned_to_geodetic(
            state.position[0], state.position[1], state.position[2]
        )

        # Position
        self.fdm['ic/lat-gc-deg'] = lat
        self.fdm['ic/long-gc-deg'] = lon
        self.fdm['ic/h-sl-ft'] = alt_msl * 3.28084  # m to ft

        # Attitude (radians to degrees)
        self.fdm['ic/phi-deg'] = np.degrees(state.attitude[0])
        self.fdm['ic/theta-deg'] = np.degrees(state.attitude[1])
        self.fdm['ic/psi-true-deg'] = np.degrees(state.attitude[2])

        # Body velocities (m/s to ft/s)
        self.fdm['ic/u-fps'] = state.velocity[0] * 3.28084
        self.fdm['ic/v-fps'] = state.velocity[1] * 3.28084
        self.fdm['ic/w-fps'] = state.velocity[2] * 3.28084

        # Angular rates (already in rad/s)
        self.fdm['ic/p-rad_sec'] = state.angular_rate[0]
        self.fdm['ic/q-rad_sec'] = state.angular_rate[1]
        self.fdm['ic/r-rad_sec'] = state.angular_rate[2]

        # Apply initial conditions
        self.fdm.run_ic()

    def _ned_to_geodetic(self, north: float, east: float, down: float) -> tuple:
        """Convert NED coordinates to geodetic (lat, lon, alt MSL).

        Simplified flat-earth approximation for validation purposes.

        Args:
            north: North position in meters
            east: East position in meters
            down: Down position in meters (negative of altitude AGL)

        Returns:
            (latitude_deg, longitude_deg, altitude_msl_m)
        """
        # Earth radius
        R_earth = 6371000.0  # meters

        # Convert NED displacement to lat/lon delta
        dlat = north / R_earth
        dlon = east / (R_earth * np.cos(np.radians(self.origin_lat)))

        lat = self.origin_lat + np.degrees(dlat)
        lon = self.origin_lon + np.degrees(dlon)
        alt_msl = self.origin_alt - down  # down is negative of altitude

        return lat, lon, alt_msl

    def _geodetic_to_ned(self, lat: float, lon: float, alt_msl: float) -> np.ndarray:
        """Convert geodetic to NED coordinates relative to origin.

        Args:
            lat: Latitude in degrees
            lon: Longitude in degrees
            alt_msl: Altitude MSL in meters

        Returns:
            NED position [N, E, D] in meters
        """
        R_earth = 6371000.0

        dlat = np.radians(lat - self.origin_lat)
        dlon = np.radians(lon - self.origin_lon)

        north = dlat * R_earth
        east = dlon * R_earth * np.cos(np.radians(self.origin_lat))
        down = -(alt_msl - self.origin_alt)

        return np.array([north, east, down])

    def step(self, dt: float) -> AircraftState:
        """Advance simulation by dt seconds.

        Args:
            dt: Time step in seconds

        Returns:
            Updated aircraft state
        """
        # JSBSim uses fixed internal timestep, run multiple steps if needed
        num_steps = max(1, int(dt / self.dt_physics))

        for _ in range(num_steps):
            self.fdm.run()

        # Update state cache
        self._current_state = self.get_state()

        return self._current_state

    def set_controls(self, surfaces: ControlSurfaces) -> None:
        """Set control surface deflections.

        Args:
            surfaces: Control surface commands (normalized -1 to 1, throttle 0 to 1)
        """
        # Map normalized controls to JSBSim FCS commands
        # JSBSim uses -1 to 1 for most surfaces, 0 to 1 for throttle

        self.fdm['fcs/aileron-cmd-norm'] = surfaces.aileron
        self.fdm['fcs/elevator-cmd-norm'] = surfaces.elevator
        self.fdm['fcs/rudder-cmd-norm'] = surfaces.rudder
        self.fdm['fcs/throttle-cmd-norm'] = surfaces.throttle

    def get_state(self) -> AircraftState:
        """Get current aircraft state.

        Returns:
            Current aircraft state in NED frame
        """
        # Get geodetic position
        lat = self.fdm['position/lat-gc-deg']
        lon = self.fdm['position/long-gc-deg']
        alt_msl_ft = self.fdm['position/h-sl-ft']
        alt_msl_m = alt_msl_ft / 3.28084

        # Convert to NED
        position = self._geodetic_to_ned(lat, lon, alt_msl_m)

        # Velocity (body frame, ft/s to m/s)
        velocity = np.array([
            self.fdm['velocities/u-fps'] / 3.28084,
            self.fdm['velocities/v-fps'] / 3.28084,
            self.fdm['velocities/w-fps'] / 3.28084,
        ])

        # Attitude (degrees to radians)
        attitude = np.array([
            np.radians(self.fdm['attitude/phi-deg']),
            np.radians(self.fdm['attitude/theta-deg']),
            np.radians(self.fdm['attitude/psi-deg']),
        ])

        # Angular rates (already in rad/s)
        angular_rate = np.array([
            self.fdm['velocities/p-rad_sec'],
            self.fdm['velocities/q-rad_sec'],
            self.fdm['velocities/r-rad_sec'],
        ])

        # Derived quantities
        airspeed = self.fdm['velocities/vt-fps'] / 3.28084  # ft/s to m/s
        altitude = -position[2]  # Altitude AGL (negative of down)
        ground_speed = self.fdm['velocities/vg-fps'] / 3.28084
        heading = attitude[2]  # Yaw angle

        # Simulation time
        time = self.fdm['simulation/sim-time-sec']

        return AircraftState(
            time=time,
            position=position,
            velocity=velocity,
            attitude=attitude,
            angular_rate=angular_rate,
            airspeed=airspeed,
            altitude=altitude,
            ground_speed=ground_speed,
            heading=heading
        )

    def get_backend_type(self) -> str:
        """Get backend type identifier.

        Returns:
            Backend type string
        """
        return "jsbsim"

    def get_dt_nominal(self) -> float:
        """Get nominal timestep for this backend.

        Returns:
            Nominal timestep in seconds
        """
        return self.dt_physics

    def get_info(self) -> Dict[str, Any]:
        """Get backend information and configuration.

        Returns:
            Dictionary with backend details
        """
        return {
            'backend_type': 'jsbsim',
            'aircraft': self.aircraft_name,
            'dt_physics': self.dt_physics,
            'jsbsim_version': jsbsim.__version__,
            'initial_position': {
                'latitude': self.initial_lat,
                'longitude': self.initial_lon,
                'altitude_msl': self.initial_altitude_msl,
            }
        }

    def __repr__(self) -> str:
        """String representation."""
        return f"JSBSimBackend(aircraft={self.aircraft_name}, dt={self.dt_physics}s)"
