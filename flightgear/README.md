# FlightGear Integration

This directory contains integration code to visualize your Python flight controllers in real-time using FlightGear.

## Setup Complete! ✅

FlightGear has been installed and configured. You're ready to fly!

## Quick Start

### 1. Launch FlightGear

Open a terminal and run:

```bash
./flightgear/launch_flightgear.sh
```

This will start FlightGear with:
- Cessna 172P aircraft
- San Francisco International Airport (KSFO)
- UDP communication enabled (ports 5500/5501)
- Nice weather and visibility

### 2. Test the Connection

In a new terminal:

```bash
source venv/bin/activate  # if using venv
python flightgear/flightgear_interface.py
```

You should see:
- FlightGear aircraft responding to gentle roll commands
- Real-time state data printed to console
- Aircraft gently rolling left and right

Press `Ctrl+C` to stop.

### 3. Fly with Your Controllers (Coming Soon)

We'll create a script to connect your tuned controllers to FlightGear. This will let you:
- See your waypoint navigation in 3D
- Visualize HSA/attitude control in real-time
- Debug controller behavior visually

---

## How It Works

### Architecture

```
Python Controller  →  UDP (port 5500)  →  FlightGear (controls)
       ↑                                         ↓
       └─────────  UDP (port 5501)  ←───── (aircraft state)
```

### Data Flow

1. **Python → FlightGear (controls)**:
   - Format: `aileron,elevator,rudder,throttle\n`
   - Example: `0.1,-0.05,0.0,0.3\n`
   - Rate: 100 Hz

2. **FlightGear → Python (state)**:
   - Format: CSV with 12 fields
   - Fields: `lat,lon,alt,airspeed,groundspeed,vs,roll,pitch,heading,p,q,r`
   - Rate: 100 Hz

### Files

- `protocol.xml` - Defines data format for FlightGear
- `flightgear_interface.py` - Python UDP bridge
- `launch_flightgear.sh` - Convenience launcher
- `README.md` - This file

---

## Manual Launch (Advanced)

If you want to customize the launch, here's the basic command:

```bash
/Applications/FlightGear.app/Contents/MacOS/fgfs \
    --aircraft=c172p \
    --airport=KSFO \
    --generic=socket,in,100,localhost,5500,udp,protocol \
    --generic=socket,out,100,localhost,5501,udp,protocol
```

Key options:
- `--aircraft=<name>` - Aircraft model (try: c172p, a4f, dhc2, f16)
- `--airport=<code>` - Starting airport (ICAO code)
- `--runway=<number>` - Starting runway
- `--altitude=<feet>` - Starting altitude
- `--heading=<degrees>` - Starting heading
- `--timeofday=<time>` - Time of day (noon, dawn, dusk, midnight)
- `--disable-sound` - No audio (faster)
- `--disable-random-objects` - Fewer objects (faster)

---

## Troubleshooting

### FlightGear won't start

```bash
# Check if installed
ls /Applications/FlightGear.app

# If not found:
brew install --cask flightgear
```

### No data received in Python

1. Check FlightGear is running
2. Check console for errors
3. Verify ports aren't in use:
   ```bash
   lsof -i :5500
   lsof -i :5501
   ```
4. Try restarting both FlightGear and Python script

### Aircraft behaves strangely

1. FlightGear physics are different from your Python sim
2. You may need to:
   - Adjust control surface limits
   - Scale control inputs
   - Account for different aircraft characteristics

### Performance issues

FlightGear can be resource-intensive. To improve performance:

1. Lower graphics settings in FlightGear
2. Disable unnecessary features:
   - Use `--disable-random-objects`
   - Use `--disable-ai-traffic`
   - Use `--disable-sound`
3. Use a simpler aircraft
4. Reduce window size: `--geometry=800x600`

---

## Next Steps

### Create Controller Integration

You'll want to create a script that:

1. Launches your Python controllers (HSA, Waypoint, etc.)
2. Sends computed control surfaces to FlightGear
3. Uses FlightGear state OR continues using Python sim

Example structure:

```python
from flightgear.flightgear_interface import FlightGearInterface
from controllers.hsa_agent import HSAAgent
from controllers.types import ControlCommand, ControlMode

# Initialize
fg = FlightGearInterface()
controller = HSAAgent(config)

while True:
    # Get state from FlightGear
    fg_state = fg.receive_state()
    if fg_state:
        # Convert to Python format
        state = fg_state.to_python_state()

        # Compute controls
        command = ControlCommand(
            mode=ControlMode.HSA,
            heading=np.radians(280),
            speed=25.0,  # m/s ≈ 50 knots
            altitude=1000.0  # meters ≈ 3280 feet
        )
        surfaces = controller.compute_action(command, state, dt=0.01)

        # Send to FlightGear
        fg.send_controls(
            surfaces.aileron,
            surfaces.elevator,
            surfaces.rudder,
            surfaces.throttle
        )
```

### Visualize Waypoints

Add waypoint markers in FlightGear using telnet interface:

```python
import telnetlib

tn = telnetlib.Telnet('localhost', 5401)
tn.write(b"set /sim/model/marker[0]/latitude-deg 37.619\n")
tn.write(b"set /sim/model/marker[0]/longitude-deg -122.375\n")
```

---

## References

- [FlightGear Generic Protocol](https://wiki.flightgear.org/Generic_protocol)
- [FlightGear Command Line Options](https://wiki.flightgear.org/Command_line_options)
- [FlightGear Property Tree](https://wiki.flightgear.org/Property_tree)
- [FlightGear Aircraft](https://wiki.flightgear.org/Aircraft)

---

## Tips for Real-Time Testing

1. **Start simple**: Test individual control axes before full automation
2. **Use safety limits**: Add sanity checks to prevent extreme inputs
3. **Monitor state**: Print key variables to debug unexpected behavior
4. **Record data**: Log FlightGear state to compare with Python sim
5. **Be patient**: FlightGear startup takes 10-30 seconds

Enjoy flying! 🛩️
