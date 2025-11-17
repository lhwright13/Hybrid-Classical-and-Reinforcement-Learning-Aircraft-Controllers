#!/usr/bin/env python3
"""
Simple FlightGear test - connect and send basic control commands
"""
import socket
import time

def test_flightgear_connection():
    """Test connection to FlightGear via property server"""

    print("Connecting to FlightGear property server on port 5401...")

    try:
        # Connect to FlightGear's property server (telnet interface)
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect(('localhost', 5401))
        sock.settimeout(0.5)

        print("Connected!")

        # Get current position
        print("Reading aircraft state...")
        commands = [
            "get /position/latitude-deg",
            "get /position/longitude-deg",
            "get /position/altitude-ft",
            "get /orientation/roll-deg",
            "get /orientation/pitch-deg",
            "get /orientation/heading-deg",
            "get /velocities/airspeed-kt",
        ]

        for cmd in commands:
            sock.send(f"{cmd}\r\n".encode())
            time.sleep(0.1)
            try:
                response = sock.recv(4096).decode('utf-8', errors='ignore')
                print(f"  {cmd}: {response.strip()}")
            except socket.timeout:
                print(f"  {cmd}: (timeout)")

        # Test control - gently roll the aircraft left and right
        print("\n" + "="*60)
        print("Testing control - sending gentle aileron commands...")
        print("Watch your FlightGear window - aircraft should roll gently")
        print("="*60 + "\n")

        for i in range(20):
            # Sinusoidal aileron command
            aileron = 0.1 * (1 if (i // 5) % 2 == 0 else -1)

            sock.send(f"set /controls/flight/aileron {aileron}\r\n".encode())

            # Read aileron value
            sock.send("get /orientation/roll-deg\r\n".encode())
            time.sleep(0.2)
            try:
                response = sock.recv(4096).decode('utf-8', errors='ignore')
                print(f"[{i:2d}] Aileron: {aileron:+.2f}, Roll: {response.strip()}")
            except socket.timeout:
                pass

        # Return to neutral
        sock.send("set /controls/flight/aileron 0.0\r\n".encode())

        print("\nTest complete! Connection working.")
        sock.close()

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_flightgear_connection()
