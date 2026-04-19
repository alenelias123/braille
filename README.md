# Braille Assistive Raspberry Pi Prototype

Python prototype for a deafblind assistive device with:
- Dual-button gesture controls (short/medium/long press)
- Vibration + LED Morse feedback
- Room description and human-presence detection using Gemini vision
- Sign-language capture flow
- SOS and caretaker polling hooks
- Ultrasonic obstacle/navigation guidance

## Repository Files

- `/home/runner/work/braille/braille/code.py` — main runtime
- `/home/runner/work/braille/braille/api_config.json` — API endpoint/key config
- `/home/runner/work/braille/braille/.env` — environment overrides

## Requirements

- Raspberry Pi (GPIO-capable), Python 3.9+
- Optional camera support:
  - `fswebcam` (used first if available)
  - `picamera2` (fallback)
- Network access for Gemini and app-service APIs

Install optional camera utility on Debian/Raspberry Pi OS:

```bash
sudo apt update
sudo apt install -y fswebcam
```

## Configuration

Set values in `/home/runner/work/braille/braille/.env`:

- `GEMINI_API_KEY=...`
- `CARETAKER_API_URL=...` (optional)
- `CARETAKER_API_TOKEN=...` (optional)
- `SOS_API_URL=...` (optional)
- `SOS_API_TOKEN=...` (optional)

Optional Gemini compatibility flags:
- `GEMINI_USE_QUERY_API_KEY=0` (default; uses `x-goog-api-key` header)
- `GEMINI_MODEL_FALLBACKS=gemini-2.0-flash,gemini-1.5-flash-latest,gemini-1.5-flash`

## Run

```bash
cd /home/runner/work/braille/braille
python3 code.py
```

## Button Controls

- **Button 1 short**: Toggle navigation mode ON/OFF
- **Button 1 medium**: Check incoming caretaker message
- **Button 1 long**: Sign-language mode
- **Button 2 short**: Room description mode
- **Button 2 medium**: Human presence detection
- **Button 2 long**: SOS mode

## GPIO Pin Connections (BCM + Physical Pin)

> All grounds must be common (Pi GND + all module GNDs).

### Inputs

| Function | BCM | Physical pin | Wiring |
|---|---:|---:|---|
| Button 1 | 17 | 11 | One side to GPIO17, other side to GND (internal pull-up used) |
| Button 2 | 27 | 13 | One side to GPIO27, other side to GND (internal pull-up used) |
| Ultrasonic ECHO | 6 | 31 | Sensor ECHO to GPIO6 (use level shifting to 3.3V-safe input) |

### Outputs

| Function | BCM | Physical pin | Wiring |
|---|---:|---:|---|
| Vibration motor control | 18 | 12 | GPIO18 to motor driver/transistor input (not direct motor drive) |
| Heartbeat LED | 22 | 15 | GPIO22 -> resistor -> LED -> GND |
| Processing LED | 23 | 16 | GPIO23 -> resistor -> LED -> GND |
| Alert LED | 24 | 18 | GPIO24 -> resistor -> LED -> GND |
| Obstacle-mode LED | 25 | 22 | GPIO25 -> resistor -> LED -> GND |
| Ultrasonic TRIG | 5 | 29 | Sensor TRIG to GPIO5 |

### Notes

- `NAV_GUIDANCE_LED_PIN` defaults to `LED3_PIN` (BCM 24), so navigation pulses share Alert LED unless overridden.
- If your ultrasonic module is 5V logic (e.g., HC-SR04), protect GPIO ECHO with a divider/level shifter.

## API Config Notes

`api_config.json` supports env references such as `${GEMINI_API_KEY}` and defaults to Gemini generateContent endpoints.

If one model/endpoint is denied, the runtime automatically tries fallback models/API-base variants before entering cooldown and local fallback mode.
