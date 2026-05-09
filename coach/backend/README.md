arduino-cli lib install "M5Core2"
need: esp32, M5Stack

audio needs to be 44100 Hz / 16-bit / mono

## MQTT broker (required)

The Core2 communicates entirely over WiFi via MQTT — no USB needed at runtime.

```bash
brew install mosquitto
brew services start mosquitto   # starts broker on localhost:1883
```

Find your Mac's LAN IP (Core2 can't use 127.0.0.1):

```bash
ipconfig getifaddr en0
```

Run the pipeline pointing at that IP:

```bash
python run.py --broker <your-mac-lan-ip>
```

The Core2 firmware must have the same LAN IP hardcoded as its MQTT broker address.