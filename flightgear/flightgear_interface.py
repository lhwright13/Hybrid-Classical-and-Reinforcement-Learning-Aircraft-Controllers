"""FlightGear Interface - Bridge between Python controllers and FlightGear.

This module handles UDP communication with FlightGear for real-time visualization.
"""

import socket
import struct
import numpy as np
from dataclasses import dataclass
from typing import Optional
import time


@dataclass
class FlightGearState:
    """State received from FlightGear."""
    latitude: float  # degrees
    longitude: float  # degrees
    altitude_ft: float  # feet
    airspeed_kt: float  # knots
    groundspeed_kt: float  # knots
    vertical_speed_fps: float  # feet per second
    roll_deg: float  # degrees
    pitch_deg: float  # degrees
    heading_deg: float  # degrees
    roll_rate_degps: float  # degrees per second
    pitch_rate_degps: float  # degrees per second
    yaw_rate_degps: float  # degrees per second

    def to_python_state(self):
        """Convert FlightGear state to Python simulation state format.

        Returns:
            AircraftState-like object (you'll need to adapt this)
        """
        # Convert units
        altitude_m = self.altitude_ft * 0.3048
        airspeed_ms = self.airspeed_kt * 0.514444
        roll_rad = np.radians(self.roll_deg)
        pitch_rad = np.radians(self.pitch_deg)
        heading_rad = np.radians(self.heading_deg)
        p = np.radians(self.roll_rate_degps)
        q = np.radians(self.pitch_rate_degps)
        r = np.radians(self.yaw_rate_degps)

        return {
            'altitude': altitude_m,
            'airspeed': airspeed_ms,
            'roll': roll_rad,
            'pitch': pitch_rad,
            'heading': heading_rad,
            'yaw': heading_rad,  # FlightGear heading = yaw
            'p': p,
            'q': q,
            'r': r,
        }


class FlightGearInterface:
    """Interface to FlightGear simulator via UDP."""

    def __init__(self, fg_host='localhost', fg_in_port=5500, fg_out_port=5501):
        """Initialize FlightGear interface.

        Args:
            fg_host: FlightGear hostname
            fg_in_port: Port to send controls TO FlightGear
            fg_out_port: Port to receive state FROM FlightGear
        """
        self.fg_host = fg_host
        self.fg_in_port = fg_in_port
        self.fg_out_port = fg_out_port

        # Socket for sending controls to FlightGear
        self.send_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        # Socket for receiving state from FlightGear
        self.recv_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.recv_socket.bind(('', fg_out_port))
        self.recv_socket.settimeout(1.0)  # 1 second timeout

        self.last_state = None
        print(f"FlightGear interface initialized:")
        print(f"  Sending controls to {fg_host}:{fg_in_port}")
        print(f"  Receiving state on port {fg_out_port}")

    def send_controls(self, aileron, elevator, rudder, throttle):
        """Send control surfaces to FlightGear.

        Args:
            aileron: -1.0 to 1.0
            elevator: -1.0 to 1.0
            rudder: -1.0 to 1.0
            throttle: 0.0 to 1.0
        """
        # Format: aileron,elevator,rudder,throttle\n
        message = f"{aileron:.6f},{elevator:.6f},{rudder:.6f},{throttle:.6f}\n"
        self.send_socket.sendto(
            message.encode('ascii'),
            (self.fg_host, self.fg_in_port)
        )

    def receive_state(self) -> Optional[FlightGearState]:
        """Receive aircraft state from FlightGear.

        Returns:
            FlightGearState or None if no data available
        """
        try:
            data, addr = self.recv_socket.recvfrom(1024)
            # Parse CSV: lat,lon,alt,airspeed,groundspeed,vs,roll,pitch,heading,p,q,r
            values = data.decode('ascii').strip().split(',')
            if len(values) >= 12:
                state = FlightGearState(
                    latitude=float(values[0]),
                    longitude=float(values[1]),
                    altitude_ft=float(values[2]),
                    airspeed_kt=float(values[3]),
                    groundspeed_kt=float(values[4]),
                    vertical_speed_fps=float(values[5]),
                    roll_deg=float(values[6]),
                    pitch_deg=float(values[7]),
                    heading_deg=float(values[8]),
                    roll_rate_degps=float(values[9]),
                    pitch_rate_degps=float(values[10]),
                    yaw_rate_degps=float(values[11]),
                )
                self.last_state = state
                return state
        except socket.timeout:
            pass
        except Exception as e:
            print(f"Error receiving state: {e}")

        return self.last_state

    def close(self):
        """Close sockets."""
        self.send_socket.close()
        self.recv_socket.close()


def test_interface():
    """Test FlightGear interface with simple control inputs."""
    interface = FlightGearInterface()

    print("\nTesting FlightGear interface...")
    print("Make sure FlightGear is running with the protocol configured!")
    print("\nPress Ctrl+C to stop\n")

    try:
        t = 0
        dt = 0.01
        while True:
            # Receive state
            state = interface.receive_state()
            if state:
                print(f"\r[{t:6.2f}s] Alt: {state.altitude_ft:7.1f}ft  "
                      f"Speed: {state.airspeed_kt:5.1f}kt  "
                      f"Hdg: {state.heading_deg:6.1f}°  "
                      f"Roll: {state.roll_deg:5.1f}°", end='')

            # Send test controls (gentle sinusoidal aileron)
            aileron = 0.2 * np.sin(0.5 * t)
            elevator = 0.0
            rudder = 0.0
            throttle = 0.3

            interface.send_controls(aileron, elevator, rudder, throttle)

            time.sleep(dt)
            t += dt

    except KeyboardInterrupt:
        print("\n\nStopping...")
    finally:
        interface.close()


if __name__ == "__main__":
    test_interface()
