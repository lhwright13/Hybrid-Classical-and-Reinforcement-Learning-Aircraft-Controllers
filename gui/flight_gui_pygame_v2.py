"""Flight Control GUI using Pygame - Enhanced Version.

Features:
- Drag-and-drop virtual joystick for pitch/roll
- Throttle and rudder sliders (all modes)
- Artificial horizon with ROLL ROTATION
- 3D aircraft visualization (wireframe + OBJ support)
- Text input for HSA/Waypoint modes
- Lat/Lon display
- Debug panel for independent axis testing
- Telemetry display
- Mode selection (all 5 control levels)
"""

import pygame
import pygame.gfxdraw
import sys
import math
import numpy as np
import threading
import time
from dataclasses import dataclass
from typing import Optional, Tuple, List
from pathlib import Path

from gui.simulation_worker import SimulationWorker, FlightCommand
from controllers.types import ControlMode


# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRAY = (128, 128, 128)
DARK_GRAY = (26, 26, 26)
MED_GRAY = (51, 51, 51)
LIGHT_GRAY = (100, 100, 100)
CYAN = (0, 255, 255)
RED = (255, 68, 68)
GREEN = (0, 170, 0)
BLUE = (0, 61, 122)
BROWN = (92, 64, 51)
YELLOW = (255, 255, 0)

# Command limits (degrees)
MAX_ROLL_RATE_DEG = 60.0    # deg/s - max roll rate command
MAX_PITCH_RATE_DEG = 30.0   # deg/s - max pitch rate command
MAX_YAW_RATE_DEG = 10.0     # deg/s - max yaw rate command
MAX_ROLL_ANGLE_DEG = 30.0   # deg - max roll angle command
MAX_PITCH_ANGLE_DEG = 20.0  # deg - max pitch angle command
MAX_YAW_ANGLE_DEG = 180.0   # deg - max yaw/heading command

# GUI update rates
COMMAND_LOOP_HZ = 30.0      # Hz - command send rate
DISPLAY_FPS = 60            # frames per second for rendering

# Display constants
PITCH_SCALE = 3.5           # pixels per degree for artificial horizon
M_PER_DEG_LAT = 111319.9    # meters per degree latitude (at equator)


class Button:
    """Simple button widget."""

    def __init__(self, x, y, width, height, text, color=GREEN, text_color=WHITE):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.color = color
        self.text_color = text_color
        self.hovered = False

    def draw(self, surface, font):
        color = tuple(min(255, c + 50) for c in self.color) if self.hovered else self.color
        pygame.draw.rect(surface, color, self.rect, border_radius=5)
        pygame.draw.rect(surface, WHITE, self.rect, 2, border_radius=5)

        text_surf = font.render(self.text, True, self.text_color)
        text_rect = text_surf.get_rect(center=self.rect.center)
        surface.blit(text_surf, text_rect)

    def handle_event(self, event):
        if event.type == pygame.MOUSEMOTION:
            self.hovered = self.rect.collidepoint(event.pos)
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                return True
        return False


class Slider:
    """Slider widget."""

    def __init__(self, x, y, width, height, min_val, max_val, initial_val, label=""):
        self.rect = pygame.Rect(x, y, width, height)
        self.min_val = min_val
        self.max_val = max_val
        self.value = initial_val
        self.label = label
        self.dragging = False
        self.handle_radius = 10

    def draw(self, surface, font):
        # Track
        pygame.draw.rect(surface, MED_GRAY, self.rect, border_radius=5)
        pygame.draw.rect(surface, CYAN, self.rect, 2, border_radius=5)

        # Handle position
        ratio = (self.value - self.min_val) / (self.max_val - self.min_val)
        handle_x = self.rect.x + int(ratio * self.rect.width)
        handle_y = self.rect.centery

        # Handle
        pygame.gfxdraw.filled_circle(surface, handle_x, handle_y, self.handle_radius, RED)
        pygame.gfxdraw.aacircle(surface, handle_x, handle_y, self.handle_radius, WHITE)

        # Label
        if self.label:
            text = font.render(f"{self.label}: {self.value:.2f}", True, WHITE)
            surface.blit(text, (self.rect.x, self.rect.y - 25))

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            ratio = (self.value - self.min_val) / (self.max_val - self.min_val)
            handle_x = self.rect.x + int(ratio * self.rect.width)
            handle_y = self.rect.centery
            dist = math.sqrt((event.pos[0] - handle_x)**2 + (event.pos[1] - handle_y)**2)
            if dist < self.handle_radius * 2:
                self.dragging = True
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            self.dragging = False
        elif event.type == pygame.MOUSEMOTION and self.dragging:
            # Update value
            ratio = (event.pos[0] - self.rect.x) / self.rect.width
            ratio = max(0, min(1, ratio))
            self.value = self.min_val + ratio * (self.max_val - self.min_val)


class TextInput:
    """Text input widget for numbers."""

    def __init__(self, x, y, width, height, label="", initial_value="0"):
        self.rect = pygame.Rect(x, y, width, height)
        self.label = label
        self.text = str(initial_value)
        self.active = False

    def draw(self, surface, font):
        # Label
        if self.label:
            label_surf = font.render(self.label, True, WHITE)
            surface.blit(label_surf, (self.rect.x, self.rect.y - 25))

        # Input box
        color = CYAN if self.active else MED_GRAY
        pygame.draw.rect(surface, BLACK, self.rect)
        pygame.draw.rect(surface, color, self.rect, 2)

        # Text
        text_surf = font.render(self.text, True, WHITE)
        surface.blit(text_surf, (self.rect.x + 5, self.rect.y + 5))

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            self.active = self.rect.collidepoint(event.pos)
        elif event.type == pygame.KEYDOWN and self.active:
            if event.key == pygame.K_BACKSPACE:
                self.text = self.text[:-1]
            elif event.key == pygame.K_RETURN:
                self.active = False
            elif event.unicode.isdigit() or event.unicode in ['.', '-']:
                self.text += event.unicode

    def get_value(self):
        try:
            return float(self.text) if self.text else 0.0
        except ValueError:
            return 0.0


class Joystick:
    """Virtual joystick widget."""

    def __init__(self, x, y, radius):
        self.center_x = x
        self.center_y = y
        self.radius = radius
        self.stick_x = 0.0  # -1 to +1
        self.stick_y = 0.0  # -1 to +1
        self.dragging = False

    def draw(self, surface, font):
        # Outer circle
        pygame.gfxdraw.aacircle(surface, self.center_x, self.center_y, self.radius, CYAN)
        pygame.gfxdraw.circle(surface, self.center_x, self.center_y, self.radius, MED_GRAY)

        # Deadzone
        deadzone_radius = int(self.radius * 0.1)
        pygame.gfxdraw.aacircle(surface, self.center_x, self.center_y, deadzone_radius, GRAY)

        # Crosshairs
        pygame.draw.line(surface, CYAN, (self.center_x - self.radius, self.center_y),
                        (self.center_x + self.radius, self.center_y), 1)
        pygame.draw.line(surface, CYAN, (self.center_x, self.center_y - self.radius),
                        (self.center_x, self.center_y + self.radius), 1)

        # Labels
        label_font = pygame.font.Font(None, 20)
        pitch_up = label_font.render("Pitch +", True, WHITE)
        pitch_down = label_font.render("Pitch -", True, WHITE)
        roll_left = label_font.render("Roll -", True, WHITE)
        roll_right = label_font.render("Roll +", True, WHITE)

        surface.blit(pitch_up, (self.center_x - pitch_up.get_width()//2, self.center_y - self.radius - 25))
        surface.blit(pitch_down, (self.center_x - pitch_down.get_width()//2, self.center_y + self.radius + 10))
        surface.blit(roll_left, (self.center_x - self.radius - roll_left.get_width() - 10, self.center_y - 10))
        surface.blit(roll_right, (self.center_x + self.radius + 10, self.center_y - 10))

        # Stick position
        stick_pixel_x = self.center_x + int(self.stick_x * self.radius)
        stick_pixel_y = self.center_y - int(self.stick_y * self.radius)  # Invert Y

        pygame.gfxdraw.filled_circle(surface, stick_pixel_x, stick_pixel_y, 15, RED)
        pygame.gfxdraw.aacircle(surface, stick_pixel_x, stick_pixel_y, 15, WHITE)

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            dist = math.sqrt((event.pos[0] - self.center_x)**2 + (event.pos[1] - self.center_y)**2)
            if dist < self.radius:
                self.dragging = True
                self._update_position(event.pos)
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            self.dragging = False
        elif event.type == pygame.MOUSEMOTION and self.dragging:
            self._update_position(event.pos)

    def _update_position(self, pos):
        dx = pos[0] - self.center_x
        dy = self.center_y - pos[1]  # Invert Y

        # Clamp to circle
        distance = math.sqrt(dx**2 + dy**2)
        if distance > self.radius:
            dx = dx / distance * self.radius
            dy = dy / distance * self.radius

        self.stick_x = dx / self.radius
        self.stick_y = dy / self.radius


class OBJLoader:
    """Simple OBJ file loader for 3D aircraft models."""

    @staticmethod
    def load(filepath: str) -> Tuple[np.ndarray, List]:
        """Load vertices and faces from .obj file.

        Returns:
            vertices: Nx3 array of vertex positions
            faces: List of face vertex indices
        """
        vertices = []
        faces = []

        try:
            with open(filepath, 'r') as f:
                for line in f:
                    if line.startswith('v '):  # Vertex
                        parts = line.split()
                        vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
                    elif line.startswith('f '):  # Face
                        parts = line.split()
                        # Handle faces (convert to vertex indices, 1-indexed)
                        face_indices = [int(p.split('/')[0]) - 1 for p in parts[1:]]
                        faces.append(face_indices)

            return np.array(vertices), faces
        except Exception as e:
            print(f"Error loading OBJ file: {e}")
            return None, None


class Aircraft3DView:
    """3D aircraft visualization with OBJ support."""

    def __init__(self, x, y, width, height, model_path: Optional[str] = None):
        self.rect = pygame.Rect(x, y, width, height)
        self.center_x = x + width // 2
        self.center_y = y + height // 2
        self.scale = 30  # Pixels per unit

        # Load model or create default
        if model_path and Path(model_path).exists():
            self.vertices, self.faces = OBJLoader.load(model_path)
            if self.vertices is None:
                self._create_default_model()
        else:
            self._create_default_model()

    def _create_default_model(self):
        """Create simple wireframe aircraft."""
        # Define vertices (X=forward, Y=right, Z=up in aircraft frame)
        self.vertices = np.array([
            # Fuselage
            [2, 0, 0],    # Nose
            [-2, 0, 0],   # Tail
            # Wings
            [0, -2, 0],   # Left wing tip
            [0, 2, 0],    # Right wing tip
            [0, 0, 0],    # Wing center
            # Vertical stabilizer
            [-2, 0, 1],   # Tail top
        ])

        # Define faces (lines to draw)
        self.faces = [
            [0, 4],  # Nose to center
            [4, 1],  # Center to tail
            [2, 4],  # Left wing
            [4, 3],  # Right wing
            [1, 5],  # Vertical stabilizer
        ]

    def draw(self, surface, font, roll_deg, pitch_deg, yaw_deg):
        """Draw 3D aircraft with current orientation."""
        # Convert to radians
        roll = np.radians(roll_deg)
        pitch = np.radians(pitch_deg)
        yaw = np.radians(yaw_deg)

        # Rotation matrices
        # Roll (around X-axis)
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll), np.cos(roll)]
        ])

        # Pitch (around Y-axis)
        Ry = np.array([
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)]
        ])

        # Yaw (around Z-axis)
        Rz = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1]
        ])

        # Combined rotation (yaw * pitch * roll)
        R = Rz @ Ry @ Rx

        # Rotate vertices
        rotated = (R @ self.vertices.T).T

        # Project to 2D (perspective projection)
        projected = []
        for v in rotated:
            # Simple orthographic projection for now
            x_proj = int(self.center_x + v[1] * self.scale)  # Y -> screen X
            y_proj = int(self.center_y - v[2] * self.scale)  # Z -> screen Y (inverted)
            projected.append((x_proj, y_proj))

        # Draw faces/edges
        for face in self.faces:
            if len(face) == 2:  # Line
                pygame.draw.line(surface, CYAN, projected[face[0]], projected[face[1]], 2)
            else:  # Polygon (draw edges)
                for i in range(len(face)):
                    start = projected[face[i]]
                    end = projected[face[(i + 1) % len(face)]]
                    pygame.draw.line(surface, CYAN, start, end, 2)

        # Draw axes (optional)
        axis_length = 40
        axes = [
            ([axis_length, 0, 0], RED, "X"),    # Forward (red)
            ([0, axis_length, 0], GREEN, "Y"),  # Right (green)
            ([0, 0, axis_length], YELLOW, "Z"), # Up (yellow)
        ]

        for axis_vec, color, label in axes:
            rotated_axis = R @ np.array(axis_vec)
            x_end = int(self.center_x + rotated_axis[1] * self.scale)
            y_end = int(self.center_y - rotated_axis[2] * self.scale)
            pygame.draw.line(surface, color, (self.center_x, self.center_y), (x_end, y_end), 2)
            text = font.render(label, True, color)
            surface.blit(text, (x_end, y_end))


class ArtificialHorizon:
    """Artificial horizon widget with roll rotation and pitch indication."""

    def __init__(self, x, y, width, height):
        self.rect = pygame.Rect(x, y, width, height)
        self.center_x = x + width // 2
        self.center_y = y + height // 2
        self.pitch_scale = PITCH_SCALE

    def draw(self, surface, font, roll_deg, pitch_deg):
        """Draw professional artificial horizon with roll and pitch."""
        # Draw background frame
        pygame.draw.rect(surface, (30, 30, 30), self.rect)
        pygame.draw.rect(surface, WHITE, self.rect, 3)

        # Create large temporary surface for rotation (must be big enough after rotation)
        temp_size = int(math.sqrt(self.rect.width**2 + self.rect.height**2) * 2)
        temp_surf = pygame.Surface((temp_size, temp_size))
        temp_center = temp_size // 2

        # Calculate pitch offset (positive pitch = nose up = horizon moves down)
        pitch_offset = int(-pitch_deg * self.pitch_scale)

        # Draw sky (upper half)
        sky_rect = pygame.Rect(0, 0, temp_size, temp_center + pitch_offset)
        pygame.draw.rect(temp_surf, (0, 120, 255), sky_rect)

        # Draw ground (lower half)
        ground_rect = pygame.Rect(0, temp_center + pitch_offset, temp_size, temp_size)
        pygame.draw.rect(temp_surf, (139, 90, 43), ground_rect)

        # Draw horizon line (thick and bright)
        pygame.draw.line(temp_surf, WHITE,
                        (0, temp_center + pitch_offset),
                        (temp_size, temp_center + pitch_offset), 5)

        # Draw pitch ladder
        for pitch_line in range(-90, 91, 10):
            if pitch_line == 0:
                continue  # Skip zero (that's the horizon line)

            y_pos = temp_center + pitch_offset + int(-pitch_line * self.pitch_scale)

            # Draw pitch lines
            if pitch_line % 10 == 0:
                line_width = 80 if pitch_line % 20 == 0 else 50
                line_thickness = 3 if pitch_line % 20 == 0 else 2

                # Draw line
                pygame.draw.line(temp_surf, WHITE,
                               (temp_center - line_width, y_pos),
                               (temp_center + line_width, y_pos), line_thickness)

                # Draw angle labels on both sides
                label_text = f"{abs(pitch_line)}"
                label = font.render(label_text, True, WHITE)

                # Left side
                temp_surf.blit(label, (temp_center - line_width - 35, y_pos - 10))
                # Right side
                temp_surf.blit(label, (temp_center + line_width + 10, y_pos - 10))

        # Rotate the entire horizon by roll angle
        # Note: pygame rotation is counter-clockwise, but we want clockwise for roll
        rotated_surf = pygame.transform.rotate(temp_surf, roll_deg)
        rotated_rect = rotated_surf.get_rect(center=(self.center_x, self.center_y))

        # Create a clipping region and blit the rotated surface
        clip_region = surface.subsurface(self.rect)
        clip_region.blit(rotated_surf,
                        (rotated_rect.x - self.rect.x,
                         rotated_rect.y - self.rect.y))

        # Draw roll scale at top (arc with tick marks)
        self._draw_roll_scale(surface, font, roll_deg)

        # Draw fixed aircraft reference symbol (yellow chevron)
        wing_y = self.center_y
        wing_length = 70
        wing_thickness = 6
        center_gap = 20

        # Left wing
        pygame.draw.line(surface, YELLOW,
                        (self.center_x - center_gap, wing_y),
                        (self.center_x - wing_length, wing_y), wing_thickness)
        # Right wing
        pygame.draw.line(surface, YELLOW,
                        (self.center_x + center_gap, wing_y),
                        (self.center_x + wing_length, wing_y), wing_thickness)
        # Center dot
        pygame.gfxdraw.filled_circle(surface, self.center_x, wing_y, 5, YELLOW)
        pygame.gfxdraw.aacircle(surface, self.center_x, wing_y, 5, YELLOW)

        # Draw pitch readout at bottom
        pitch_text = font.render(f"PITCH: {pitch_deg:+.1f}°", True, WHITE)
        pitch_bg = pygame.Rect(self.center_x - 70, self.rect.bottom - 35, 140, 28)
        pygame.draw.rect(surface, BLACK, pitch_bg)
        pygame.draw.rect(surface, WHITE, pitch_bg, 2)
        surface.blit(pitch_text, (self.center_x - pitch_text.get_width() // 2,
                                   self.rect.bottom - 32))

    def _draw_roll_scale(self, surface, font, roll_deg):
        """Draw roll scale arc at top of horizon."""
        # Draw arc with tick marks for roll angle
        arc_radius = 80
        arc_center_x = self.center_x
        arc_center_y = self.rect.y + 60

        # Draw major tick marks at 0, ±10, ±20, ±30, ±45, ±60
        for angle in [0, 10, 20, 30, 45, 60]:
            for sign in [1, -1]:
                tick_angle = sign * angle
                # Convert to radians (0° is up, positive is right/clockwise)
                rad = math.radians(90 - tick_angle)

                # Tick mark positions
                if angle in [0, 30, 60]:
                    tick_length = 20
                    tick_thickness = 3
                else:
                    tick_length = 12
                    tick_thickness = 2

                x_outer = arc_center_x + arc_radius * math.cos(rad)
                y_outer = arc_center_y - arc_radius * math.sin(rad)
                x_inner = arc_center_x + (arc_radius - tick_length) * math.cos(rad)
                y_inner = arc_center_y - (arc_radius - tick_length) * math.sin(rad)

                pygame.draw.line(surface, WHITE,
                               (int(x_inner), int(y_inner)),
                               (int(x_outer), int(y_outer)), tick_thickness)

        # Draw roll indicator triangle (moves with roll)
        indicator_angle = math.radians(90 - roll_deg)
        indicator_x = arc_center_x + arc_radius * math.cos(indicator_angle)
        indicator_y = arc_center_y - arc_radius * math.sin(indicator_angle)

        # Triangle points
        tri_size = 10
        triangle = [
            (int(indicator_x), int(indicator_y)),
            (int(indicator_x - tri_size), int(indicator_y - tri_size)),
            (int(indicator_x + tri_size), int(indicator_y - tri_size))
        ]
        pygame.draw.polygon(surface, YELLOW, triangle)
        pygame.draw.polygon(surface, WHITE, triangle, 2)

        # Draw roll readout at top
        roll_text = font.render(f"ROLL: {roll_deg:+.1f}°", True, WHITE)
        roll_bg = pygame.Rect(self.center_x - 70, self.rect.y + 5, 140, 28)
        pygame.draw.rect(surface, BLACK, roll_bg)
        pygame.draw.rect(surface, WHITE, roll_bg, 2)
        surface.blit(roll_text, (self.center_x - roll_text.get_width() // 2, self.rect.y + 8))


def ned_to_latlon(north_m, east_m, lat_origin=0.0, lon_origin=0.0):
    """Convert NED coordinates to lat/lon.

    Args:
        north_m: North offset in meters
        east_m: East offset in meters
        lat_origin: Origin latitude (degrees)
        lon_origin: Origin longitude (degrees)

    Returns:
        (latitude, longitude) in degrees
    """
    # Latitude is straightforward
    lat = lat_origin + (north_m / M_PER_DEG_LAT)

    # Longitude depends on latitude
    m_per_deg_lon = M_PER_DEG_LAT * np.cos(np.radians(lat))
    lon = lon_origin + (east_m / m_per_deg_lon) if m_per_deg_lon > 0 else lon_origin

    return lat, lon


class FlightControlGUI:
    """Main pygame GUI with all enhancements."""

    def __init__(self, aircraft_model_path: Optional[str] = None):
        pygame.init()

        # Window size
        self.width = 1600
        self.height = 900
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Flight Control Dashboard - Enhanced")

        # Fonts
        self.font_large = pygame.font.Font(None, 32)
        self.font_medium = pygame.font.Font(None, 24)
        self.font_small = pygame.font.Font(None, 20)

        # Simulation worker
        self.sim_worker = SimulationWorker()
        self.sim_worker.start()

        # State
        self.running = True
        self.clock = pygame.time.Clock()
        self.current_mode = ControlMode.SURFACE
        self.current_state = {}
        self.latest_state = None  # AircraftState object for external plotting
        self.show_debug = False  # Toggle with 'D' key
        self.current_command = None  # Track current command for display

        # Create widgets
        self._create_widgets(aircraft_model_path)

        # Start command thread
        self.command_thread = threading.Thread(target=self._command_loop, daemon=True)
        self.command_thread.start()

    def _create_widgets(self, aircraft_model_path):
        """Create UI widgets."""
        # Joystick
        self.joystick = Joystick(200, 400, 120)

        # Throttle slider
        self.throttle_slider = Slider(50, 620, 300, 20, 0, 1, 0.5, "Throttle")

        # Rudder slider (now shown for all manual modes!)
        self.rudder_slider = Slider(50, 700, 300, 20, -1, 1, 0, "Rudder")

        # Artificial horizon with roll rotation
        self.horizon = ArtificialHorizon(440, 350, 400, 400)

        # 3D aircraft view
        self.aircraft_3d = Aircraft3DView(440, 50, 400, 280, aircraft_model_path)

        # Mode buttons
        self.mode_buttons = []
        modes = [
            ("Surface", ControlMode.SURFACE),
            ("Rate", ControlMode.RATE),
            ("Attitude", ControlMode.ATTITUDE),
            ("HSA", ControlMode.HSA),
            ("Waypoint", ControlMode.WAYPOINT),
        ]
        for i, (name, mode) in enumerate(modes):
            btn = Button(20, 50 + i * 45, 100, 35, name, GREEN if mode == self.current_mode else MED_GRAY)
            btn.mode = mode
            self.mode_buttons.append(btn)

        # Reset button
        self.reset_button = Button(140, 50, 110, 35, "Reset", RED)

        # HSA text inputs
        self.hsa_heading = TextInput(50, 300, 130, 30, "Heading (°)", "0")
        self.hsa_speed = TextInput(50, 370, 130, 30, "Speed (m/s)", "25")
        self.hsa_altitude = TextInput(200, 370, 130, 30, "Altitude (m)", "100")
        self.hsa_send_button = Button(50, 430, 280, 40, "Send HSA Command", GREEN)

        # Waypoint text inputs
        self.wp_north = TextInput(50, 300, 130, 30, "North (m)", "0")
        self.wp_east = TextInput(200, 300, 130, 30, "East (m)", "0")
        self.wp_altitude = TextInput(50, 370, 130, 30, "Altitude (m)", "100")
        self.wp_send_button = Button(50, 430, 280, 40, "Send Waypoint", GREEN)

    def _command_loop(self):
        """Background command sending loop."""
        while self.running:
            if self.current_mode in [ControlMode.SURFACE, ControlMode.RATE, ControlMode.ATTITUDE]:
                if self.current_mode == ControlMode.SURFACE:
                    command = FlightCommand(
                        mode=self.current_mode,
                        elevator=-self.joystick.stick_y,  # Inverted: up = positive pitch
                        aileron=self.joystick.stick_x,
                        rudder=self.rudder_slider.value,
                        throttle=self.throttle_slider.value
                    )
                elif self.current_mode == ControlMode.RATE:
                    command = FlightCommand(
                        mode=self.current_mode,
                        roll_rate=self.joystick.stick_x * np.radians(MAX_ROLL_RATE_DEG),
                        pitch_rate=-self.joystick.stick_y * np.radians(MAX_PITCH_RATE_DEG),
                        yaw_rate=self.rudder_slider.value * np.radians(MAX_YAW_RATE_DEG),
                        throttle=self.throttle_slider.value
                    )
                elif self.current_mode == ControlMode.ATTITUDE:
                    command = FlightCommand(
                        mode=self.current_mode,
                        roll_angle=self.joystick.stick_x * np.radians(MAX_ROLL_ANGLE_DEG),
                        pitch_angle=-self.joystick.stick_y * np.radians(MAX_PITCH_ANGLE_DEG),
                        yaw_angle=self.rudder_slider.value * np.radians(MAX_YAW_ANGLE_DEG),
                        throttle=self.throttle_slider.value
                    )

                self.sim_worker.send_command(command)
                self.current_command = command  # Store for display

            time.sleep(1.0 / COMMAND_LOOP_HZ)

    def _handle_events(self):
        """Handle pygame events."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

            # Mouse wheel to adjust throttle
            if event.type == pygame.MOUSEWHEEL:
                # Adjust throttle by 5% per wheel click
                self.throttle_slider.value += event.y * 0.05
                self.throttle_slider.value = max(0.0, min(1.0, self.throttle_slider.value))

            # Toggle debug panel
            if event.type == pygame.KEYDOWN and event.key == pygame.K_d:
                self.show_debug = not self.show_debug

            # Cycle control modes with 'M' key
            if event.type == pygame.KEYDOWN and event.key == pygame.K_m:
                # Get list of all modes
                modes = [ControlMode.SURFACE, ControlMode.RATE, ControlMode.ATTITUDE,
                        ControlMode.HSA, ControlMode.WAYPOINT]
                # Find current mode index and cycle to next
                current_idx = modes.index(self.current_mode)
                next_idx = (current_idx + 1) % len(modes)
                self.current_mode = modes[next_idx]
                # Update button colors
                for btn in self.mode_buttons:
                    btn.color = GREEN if btn.mode == self.current_mode else MED_GRAY
                print(f"🔄 Switched to {self.current_mode.name} mode")

            # Toggle learned/PID controller (if available)
            if event.type == pygame.KEYDOWN and event.key == pygame.K_l:
                if hasattr(self.sim_worker, 'toggle_controller'):
                    self.sim_worker.toggle_controller()
                else:
                    print("Warning: Controller toggle not available (use SimulationWorkerWithLearned)")

            # Joystick
            if self.current_mode in [ControlMode.SURFACE, ControlMode.RATE, ControlMode.ATTITUDE]:
                self.joystick.handle_event(event)
                self.throttle_slider.handle_event(event)
                self.rudder_slider.handle_event(event)  # Always handle rudder!

            # HSA inputs
            if self.current_mode == ControlMode.HSA:
                self.hsa_heading.handle_event(event)
                self.hsa_speed.handle_event(event)
                self.hsa_altitude.handle_event(event)
                if self.hsa_send_button.handle_event(event):
                    self._send_hsa_command()

            # Waypoint inputs
            if self.current_mode == ControlMode.WAYPOINT:
                self.wp_north.handle_event(event)
                self.wp_east.handle_event(event)
                self.wp_altitude.handle_event(event)
                if self.wp_send_button.handle_event(event):
                    self._send_waypoint_command()

            # Mode buttons
            for btn in self.mode_buttons:
                if btn.handle_event(event):
                    self.current_mode = btn.mode
                    for b in self.mode_buttons:
                        b.color = GREEN if b.mode == self.current_mode else MED_GRAY

            # Reset button
            if self.reset_button.handle_event(event):
                self.sim_worker.reset()
                self.joystick.stick_x = 0
                self.joystick.stick_y = 0

    def _send_hsa_command(self):
        """Send HSA command from inputs."""
        command = FlightCommand(
            mode=ControlMode.HSA,
            heading=np.radians(self.hsa_heading.get_value()),
            speed=self.hsa_speed.get_value(),
            altitude=self.hsa_altitude.get_value(),
            throttle=0.5
        )
        self.sim_worker.send_command(command)

    def _send_waypoint_command(self):
        """Send waypoint command from inputs."""
        command = FlightCommand(
            mode=ControlMode.WAYPOINT,
            waypoint_north=self.wp_north.get_value(),
            waypoint_east=self.wp_east.get_value(),
            altitude=self.wp_altitude.get_value(),
            throttle=0.5
        )
        self.sim_worker.send_command(command)

    def _draw(self):
        """Draw the GUI."""
        self.screen.fill(DARK_GRAY)

        # Title
        title = self.font_large.render("Flight Control Dashboard - Enhanced", True, WHITE)
        self.screen.blit(title, (self.width // 2 - title.get_width() // 2, 10))

        # Left panel: Controls
        pygame.draw.rect(self.screen, BLACK, (10, 40, 390, 860), border_radius=10)
        pygame.draw.rect(self.screen, CYAN, (10, 40, 390, 860), 2, border_radius=10)

        # Mode label
        mode_label = self.font_medium.render("Control Mode", True, CYAN)
        self.screen.blit(mode_label, (20, 15))

        # Mode buttons
        for btn in self.mode_buttons:
            btn.draw(self.screen, self.font_small)
        self.reset_button.draw(self.screen, self.font_small)

        # Mode-specific controls
        if self.current_mode in [ControlMode.SURFACE, ControlMode.RATE, ControlMode.ATTITUDE]:
            # Joystick
            self.joystick.draw(self.screen, self.font_small)
            self.throttle_slider.draw(self.screen, self.font_small)
            self.rudder_slider.draw(self.screen, self.font_small)  # Show for all modes!

        elif self.current_mode == ControlMode.HSA:
            # HSA inputs
            self.hsa_heading.draw(self.screen, self.font_small)
            self.hsa_speed.draw(self.screen, self.font_small)
            self.hsa_altitude.draw(self.screen, self.font_small)
            self.hsa_send_button.draw(self.screen, self.font_small)

        elif self.current_mode == ControlMode.WAYPOINT:
            # Waypoint inputs
            self.wp_north.draw(self.screen, self.font_small)
            self.wp_east.draw(self.screen, self.font_small)
            self.wp_altitude.draw(self.screen, self.font_small)
            self.wp_send_button.draw(self.screen, self.font_small)

        # Center panel: 3D View + Horizon
        pygame.draw.rect(self.screen, BLACK, (420, 40, 440, 860), border_radius=10)
        pygame.draw.rect(self.screen, CYAN, (420, 40, 440, 860), 2, border_radius=10)

        # Get state
        roll = np.degrees(self.current_state.get('roll', 0))
        pitch = np.degrees(self.current_state.get('pitch', 0))
        yaw = np.degrees(self.current_state.get('yaw', 0))

        # 3D aircraft view
        view_label = self.font_medium.render("3D Aircraft View", True, CYAN)
        self.screen.blit(view_label, (560, 15))
        self.aircraft_3d.draw(self.screen, self.font_small, roll, pitch, yaw)

        # Artificial horizon
        horizon_label = self.font_medium.render("Artificial Horizon", True, CYAN)
        self.screen.blit(horizon_label, (560, 325))
        self.horizon.draw(self.screen, self.font_small, roll, pitch)

        # Primary instruments
        alt = self.current_state.get('altitude', 0)
        airspeed = self.current_state.get('airspeed', 0)
        heading = np.degrees(self.current_state.get('heading', 0))

        inst_y = 780
        alt_text = self.font_medium.render(f"Alt: {alt:.1f} m", True, CYAN)
        as_text = self.font_medium.render(f"AS: {airspeed:.1f} m/s", True, CYAN)
        hdg_text = self.font_medium.render(f"HDG: {heading:.0f}°", True, CYAN)

        self.screen.blit(alt_text, (440, inst_y))
        self.screen.blit(as_text, (600, inst_y))
        self.screen.blit(hdg_text, (770, inst_y))

        # Right panel: Telemetry
        pygame.draw.rect(self.screen, BLACK, (870, 40, 720, 860), border_radius=10)
        pygame.draw.rect(self.screen, CYAN, (870, 40, 720, 860), 2, border_radius=10)

        telem_label = self.font_medium.render("Telemetry", True, CYAN)
        self.screen.blit(telem_label, (1150, 15))

        # Telemetry data
        vs = self.current_state.get('vertical_speed', 0)
        gforce = self.current_state.get('g_force', 1.0)
        north = self.current_state.get('north', 0)
        east = self.current_state.get('east', 0)

        # Angular rates (in rad/s, convert to deg/s)
        roll_rate = self.current_state.get('roll_rate', 0)  # deg/s
        pitch_rate = self.current_state.get('pitch_rate', 0)  # deg/s
        yaw_rate = self.current_state.get('yaw_rate', 0)  # deg/s

        # Convert to Lat/Lon
        lat, lon = ned_to_latlon(north, east)

        elevator = self.current_state.get('elevator', 0)
        aileron = self.current_state.get('aileron', 0)
        rudder = self.current_state.get('rudder', 0)
        throttle = self.current_state.get('throttle', 0.5)

        # Get commanded values from state (already in degrees)
        cmd_roll_angle = self.current_state.get('cmd_roll_angle')
        cmd_pitch_angle = self.current_state.get('cmd_pitch_angle')
        cmd_yaw_angle = self.current_state.get('cmd_yaw_angle')
        cmd_roll_rate = self.current_state.get('cmd_roll_rate')
        cmd_pitch_rate = self.current_state.get('cmd_pitch_rate')
        cmd_yaw_rate = self.current_state.get('cmd_yaw_rate')

        telem_data = [
            f"Pitch: {pitch:+.1f}°",
            f"Yaw: {yaw:+.1f}°",
            f"V/S: {vs:+.1f} m/s",
            f"G-Force: {gforce:.2f}",
            "",
        ]

        # Add commanded angles if in Attitude mode
        if self.current_mode == ControlMode.ATTITUDE and cmd_roll_angle is not None:
            telem_data.extend([
                "Commanded Angles:",
                f"  Roll: {cmd_roll_angle:+.1f}°",
                f"  Pitch: {cmd_pitch_angle:+.1f}°",
                f"  Yaw: {cmd_yaw_angle:+.1f}°",
                "",
            ])

        telem_data.extend([
            "Angular Rates:",
            f"  Roll: {roll_rate:+.1f}°/s",
            f"  Pitch: {pitch_rate:+.1f}°/s",
            f"  Yaw: {yaw_rate:+.1f}°/s",
            "",
        ])

        # Add commanded rates if in Rate mode
        if self.current_mode == ControlMode.RATE and cmd_roll_rate is not None:
            telem_data.extend([
                "Commanded Rates:",
                f"  Roll: {cmd_roll_rate:+.1f}°/s",
                f"  Pitch: {cmd_pitch_rate:+.1f}°/s",
                f"  Yaw: {cmd_yaw_rate:+.1f}°/s",
                "",
            ])

        # Show rate controller type if in Rate mode
        if self.current_mode == ControlMode.RATE:
            rate_controller = self.current_state.get('rate_controller', 'PID')
            telem_data.extend([
                f"Rate Controller: {rate_controller}",
                "(Press 'L' to toggle)" if hasattr(self.sim_worker, 'toggle_controller') else "",
                "",
            ])

        telem_data.extend([
            "Position:",
            f"  North: {north:.1f} m",
            f"  East: {east:.1f} m",
            f"  Lat: {lat:.6f}°",
            f"  Lon: {lon:.6f}°",
            "",
            "Control Surfaces:",
            f"  Elevator: {elevator:+.2f}",
            f"  Aileron: {aileron:+.2f}",
            f"  Rudder: {rudder:+.2f}",
            f"  Throttle: {throttle:.2f}",
        ])

        y = 60
        for line in telem_data:
            if line.startswith("Position:") or line.startswith("Control") or line.startswith("Angular") or line.startswith("Commanded"):
                text = self.font_small.render(line, True, CYAN)
            else:
                text = self.font_small.render(line, True, WHITE)
            self.screen.blit(text, (890, y))
            y += 28

        # Debug panel (toggle with 'D' key)
        if self.show_debug:
            debug_y = 450
            debug_label = self.font_medium.render("Debug Panel (Press D to hide)", True, YELLOW)
            self.screen.blit(debug_label, (890, debug_y))
            # TODO: Add debug sliders for independent axis testing

        pygame.display.flip()

    def _update_state(self):
        """Update state from simulation."""
        state = self.sim_worker.get_state()
        if state:
            self.current_state = state
            # Extract raw AircraftState object for external plotting
            if 'state_object' in state:
                self.latest_state = state['state_object']

            # Write state to JSON file for external plotting
            try:
                import json
                plot_state = {
                    'time': state.get('time', 0.0),
                    'altitude': state.get('altitude', 0.0),
                    'airspeed': state.get('airspeed', 0.0),
                    'roll': state.get('roll', 0.0),
                    'pitch': state.get('pitch', 0.0),
                    'yaw': state.get('yaw', 0.0),
                    'roll_rate': state.get('roll_rate', 0.0),
                    'pitch_rate': state.get('pitch_rate', 0.0),
                    'yaw_rate': state.get('yaw_rate', 0.0),
                    # Commanded values (may be None)
                    'cmd_roll_angle': state.get('cmd_roll_angle'),
                    'cmd_pitch_angle': state.get('cmd_pitch_angle'),
                    'cmd_yaw_angle': state.get('cmd_yaw_angle'),
                    'cmd_roll_rate': state.get('cmd_roll_rate'),
                    'cmd_pitch_rate': state.get('cmd_pitch_rate'),
                    'cmd_yaw_rate': state.get('cmd_yaw_rate'),
                    'mode': state.get('mode', 'SURFACE')
                }
                with open('live_state.json', 'w') as f:
                    json.dump(plot_state, f)
            except:
                pass  # Don't let plotting break the GUI

    def run(self):
        """Main loop."""
        while self.running:
            self._handle_events()
            self._update_state()
            self._draw()
            self.clock.tick(DISPLAY_FPS)

        pygame.quit()
        sys.exit()


if __name__ == '__main__':
    print("=" * 70)
    print(" " * 15 + "Flight Control Dashboard - Enhanced")
    print("=" * 70)
    print()
    print("Features:")
    print("   • Drag-and-drop joystick")
    print("   • Artificial horizon with roll rotation")
    print("   • 3D aircraft visualization")
    print("   • Text inputs for HSA/Waypoint modes")
    print("   • Lat/Lon display")
    print("   • Press 'D' to toggle debug panel")
    print()
    print("Controls:")
    print("   • Drag joystick to control pitch/roll")
    print("   • Drag sliders for throttle and rudder")
    print("   • Mouse wheel to adjust throttle")
    print("   • Press 'M' to cycle control modes (Surface→Rate→Attitude→HSA→Waypoint)")
    print("   • Press 'L' to toggle RL/PID controllers (if available)")
    print("   • Press 'D' to toggle debug panel")
    print("   • Click mode buttons to change control level")
    print("   • Type in text boxes for HSA/Waypoint")
    print("   • Click Reset to restart aircraft")
    print()
    print("Close window to exit")
    print("=" * 70)
    print()

    # Optional: Load custom aircraft model
    # gui = FlightControlGUI(aircraft_model_path="path/to/model.obj")
    gui = FlightControlGUI()
    gui.run()
