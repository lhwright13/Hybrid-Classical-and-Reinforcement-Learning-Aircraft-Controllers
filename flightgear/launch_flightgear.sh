#!/bin/bash
# Launch FlightGear with Python controller integration

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROTOCOL_FILE="${SCRIPT_DIR}/protocol.xml"

# FlightGear application path (macOS)
FG_APP="/Applications/FlightGear.app/Contents/MacOS/fgfs"

# Check if FlightGear is installed
if [ ! -f "$FG_APP" ]; then
    echo "Error: FlightGear not found at $FG_APP"
    echo "Please install FlightGear first: brew install --cask flightgear"
    exit 1
fi

# Check if protocol file exists
if [ ! -f "$PROTOCOL_FILE" ]; then
    echo "Error: Protocol file not found at $PROTOCOL_FILE"
    exit 1
fi

echo "========================================"
echo "Launching FlightGear with Python Control"
echo "========================================"
echo ""
echo "Protocol file: $PROTOCOL_FILE"
echo "Listening for controls on port 5500"
echo "Sending state on port 5501"
echo ""
echo "Aircraft: Cessna 172P (c172p)"
echo "Location: San Francisco (KSFO)"
echo ""
echo "Press Ctrl+C to stop"
echo ""

# Launch FlightGear with:
# - Cessna 172P aircraft
# - San Francisco airport (KSFO)
# - Generic protocol for UDP communication
# - Telnet server for debugging (optional)
# - Window size 1280x720

"$FG_APP" \
    --aircraft=c172p \
    --airport=KSFO \
    --runway=28L \
    --altitude=3000 \
    --heading=280 \
    --timeofday=noon \
    --disable-random-objects \
    --disable-ai-models \
    --disable-ai-traffic \
    --prop:/sim/rendering/particles=0 \
    --generic=socket,in,100,localhost,5500,udp,protocol \
    --generic=socket,out,100,localhost,5501,udp,protocol \
    --telnet=5401 \
    --httpd=8080 \
    --geometry=1280x720 \
    --enable-hud \
    --fog-disable \
    --disable-sound \
    --timeofday=noon

# Alternative minimal launch (faster):
# "$FG_APP" \
#     --aircraft=c172p \
#     --airport=KSFO \
#     --generic=socket,in,100,localhost,5500,udp,protocol \
#     --generic=socket,out,100,localhost,5501,udp,protocol \
#     --disable-random-objects \
#     --disable-ai-traffic \
#     --disable-sound
