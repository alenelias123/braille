#!/usr/bin/env python3
"""Assistive system for deafblind users on Raspberry Pi.

Features:
- Button 1 short press: toggle navigation mode (ultrasonic guidance).
- Button 1 medium press: fetch incoming caretaker message and play Morse.
- Button 1 long press: sign-language capture, AI correction, Braille output.
- Button 2 short press: room description mode (vision -> Morse).
- Button 2 medium press: human presence detection (YES/NO -> Morse).
- Button 2 long press: SOS alert with haptic confirmation.
- Ultrasonic obstacle awareness and LED status indicators.

The script is designed to keep working even when some hardware or APIs are
unavailable. It logs failures, uses configurable retries/backoff, and keeps the main
loop responsive by sending Morse output through a background worker.
"""

from __future__ import annotations

import base64
import json
import os
import queue
import re
import shutil
import signal
import ssl
import subprocess
import tempfile
import threading
import time
import unicodedata
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from urllib import error as urllib_error
from urllib import parse as urllib_parse
from urllib import request as urllib_request


def load_dotenv_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _resolve_env_refs(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _resolve_env_refs(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve_env_refs(v) for v in value]
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("${") and stripped.endswith("}") and len(stripped) > 3:
            env_name = stripped[2:-1].strip()
            return os.getenv(env_name, "")
    return value


def load_api_config(path: Path) -> Dict[str, Any]:
    defaults: Dict[str, Any] = {
        "gemini_flash": {
            "endpoint": "https://generativelanguage.googleapis.com/v1beta/models/gemini-flash-latest:generateContent",
            "api_key": "${GEMINI_API_KEY}",
            "timeout_seconds": 6,
            "retries": 1,
            "max_output_tokens": 64,
        },
        "gemini_vision": {
            "endpoint": "https://generativelanguage.googleapis.com/v1beta/models/gemini-flash-latest:generateContent",
            "api_key": "${GEMINI_API_KEY}",
            "timeout_seconds": 8,
            "retries": 1,
            "max_output_tokens": 80,
        },
        "sign_language": {
            "endpoint": "https://generativelanguage.googleapis.com/v1beta/models/gemini-flash-latest:generateContent",
            "api_key": "${GEMINI_API_KEY}",
            "timeout_seconds": 8,
            "retries": 1,
            "max_output_tokens": 96,
        },
        "caretaker": {
            "endpoint": "${CARETAKER_API_URL}",
            "token": "${CARETAKER_API_TOKEN}",
            "timeout_seconds": 10,
        },
        "sos": {
            "endpoint": "${SOS_API_URL}",
            "token": "${SOS_API_TOKEN}",
            "timeout_seconds": 10,
        },
    }

    loaded: Dict[str, Any] = {}
    if path.exists():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(loaded, dict):
                loaded = {}
        except Exception:
            loaded = {}
    merged = _deep_merge(defaults, loaded)
    return _resolve_env_refs(merged)


PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv_file(PROJECT_ROOT / ".env")
API_CONFIG_PATH = Path(os.getenv("API_CONFIG_PATH", str(PROJECT_ROOT / "api_config.json")))
API_CONFIG = load_api_config(API_CONFIG_PATH)

try:
    import certifi  # type: ignore
except Exception:
    certifi = None  # type: ignore


try:
    import RPi.GPIO as _rpi_gpio  # type: ignore
except Exception:
    _rpi_gpio = None

try:
    from gpiozero import Button as _GpioZeroButton  # type: ignore
    from gpiozero import Device as _GpioZeroDevice  # type: ignore
    from gpiozero import DigitalInputDevice as _GpioZeroDigitalInputDevice  # type: ignore
    from gpiozero import DigitalOutputDevice as _GpioZeroDigitalOutputDevice  # type: ignore
    from gpiozero import DistanceSensor as _GpioZeroDistanceSensor  # type: ignore
except Exception:
    _GpioZeroButton = None
    _GpioZeroDevice = None
    _GpioZeroDigitalInputDevice = None
    _GpioZeroDigitalOutputDevice = None
    _GpioZeroDistanceSensor = None

try:
    from gpiozero.pins.lgpio import LGPIOFactory as _LGPIOFactory  # type: ignore
except Exception:
    _LGPIOFactory = None


class _GPIOBackend:
    BCM = "BCM"
    OUT = "OUT"
    IN = "IN"
    PUD_UP = "PUD_UP"
    HIGH = 1
    LOW = 0

    def __init__(self) -> None:
        self.mode = "dummy"
        self._outputs: Dict[int, Any] = {}
        self._inputs: Dict[int, Any] = {}
        self._distance_sensor: Any = None

        if _rpi_gpio is not None:
            self.mode = "rpi"
            self._gpio = _rpi_gpio
            return

        if _GpioZeroButton is not None and _GpioZeroDigitalOutputDevice is not None and _GpioZeroDistanceSensor is not None:
            self.mode = "gpiozero"
            if _GpioZeroDevice is not None and _LGPIOFactory is not None:
                try:
                    _GpioZeroDevice.pin_factory = _LGPIOFactory()
                except Exception:
                    pass
            return

        self._gpio = None

    def setmode(self, *_args: Any, **_kwargs: Any) -> None:
        if self.mode == "rpi":
            self._gpio.setmode(*_args, **_kwargs)

    def setup(self, pin: int, mode: Any, pull_up_down: Any = None) -> None:
        if self.mode == "rpi":
            if pull_up_down is None:
                self._gpio.setup(pin, mode)
            else:
                self._gpio.setup(pin, mode, pull_up_down=pull_up_down)
            return

        if self.mode != "gpiozero":
            return

        if pin in {TRIG_PIN, ECHO_PIN}:
            return

        if mode == self.OUT:
            if pin not in self._outputs:
                self._outputs[pin] = _GpioZeroDigitalOutputDevice(pin, initial_value=False)
        elif mode == self.IN:
            if pin in {BUTTON_PIN, BUTTON2_PIN} and pin not in self._inputs:
                self._inputs[pin] = _GpioZeroButton(pin, pull_up=True, bounce_time=BUTTON_DEBOUNCE_SECONDS)
            elif pin == ECHO_PIN and pin not in self._inputs and _GpioZeroDigitalInputDevice is not None:
                self._inputs[pin] = _GpioZeroDigitalInputDevice(pin, pull_up=False)

    def output(self, pin: int, value: Any) -> None:
        if self.mode == "rpi":
            self._gpio.output(pin, value)
            return

        if self.mode != "gpiozero":
            return

        device = self._outputs.get(pin)
        if device is None:
            device = _GpioZeroDigitalOutputDevice(pin, initial_value=False)
            self._outputs[pin] = device
        device.value = 1 if value else 0

    def input(self, pin: int) -> int:
        if self.mode == "rpi":
            return self._gpio.input(pin)

        if self.mode != "gpiozero":
            return self.HIGH

        device = self._inputs.get(pin)
        if device is None:
            if pin in {BUTTON_PIN, BUTTON2_PIN}:
                device = _GpioZeroButton(pin, pull_up=True, bounce_time=BUTTON_DEBOUNCE_SECONDS)
            elif pin == ECHO_PIN and _GpioZeroDigitalInputDevice is not None:
                device = _GpioZeroDigitalInputDevice(pin, pull_up=False)
            if device is not None:
                self._inputs[pin] = device

        if device is None:
            return self.HIGH
        if hasattr(device, "is_pressed"):
            return self.LOW if device.is_pressed else self.HIGH
        if hasattr(device, "value"):
            return self.LOW if not bool(device.value) else self.HIGH
        return self.HIGH

    def read_distance_cm(self) -> Optional[float]:
        if self.mode == "rpi":
            try:
                self._gpio.output(TRIG_PIN, self._gpio.LOW)
                time.sleep(0.0002)
                self._gpio.output(TRIG_PIN, self._gpio.HIGH)
                time.sleep(0.00001)
                self._gpio.output(TRIG_PIN, self._gpio.LOW)

                start_time = time.monotonic()
                timeout = 0.03
                while self._gpio.input(ECHO_PIN) == self._gpio.LOW:
                    if time.monotonic() - start_time > timeout:
                        return None
                    time.sleep(0.0001)

                pulse_start = time.monotonic()
                while self._gpio.input(ECHO_PIN) == self._gpio.HIGH:
                    if time.monotonic() - pulse_start > timeout:
                        return None
                    time.sleep(0.0001)

                pulse_end = time.monotonic()
                pulse_duration = pulse_end - pulse_start
                return round((pulse_duration * 34300.0) / 2.0, 2)
            except Exception:
                return None

        if self.mode != "gpiozero" or _GpioZeroDistanceSensor is None:
            return None

        try:
            if self._distance_sensor is None:
                self._distance_sensor = _GpioZeroDistanceSensor(
                    echo=ECHO_PIN,
                    trigger=TRIG_PIN,
                    max_distance=2.5,
                    threshold_distance=1.0,
                )
            distance_m = self._distance_sensor.distance * 2.5
            return round(distance_m * 100.0, 2)
        except Exception:
            return None

    def cleanup(self) -> None:
        if self.mode == "rpi":
            self._gpio.cleanup()
            return

        if self.mode == "gpiozero":
            for device in self._outputs.values():
                try:
                    device.off()
                    device.close()
                except Exception:
                    pass
            for device in self._inputs.values():
                try:
                    device.close()
                except Exception:
                    pass
            if self._distance_sensor is not None:
                try:
                    self._distance_sensor.close()
                except Exception:
                    pass


GPIO = _GPIOBackend()  # type: ignore


try:
    from picamera2 import Picamera2  # type: ignore
except Exception:
    Picamera2 = None  # type: ignore


APP_NAME = os.getenv("ASSISTIVE_APP_NAME", "assistive-system")
DEVICE_ID = os.getenv("DEVICE_ID", "raspberry-pi")

# Raspberry Pi 5 wiring layout, using BCM numbering.
# All modules should share one common ground on any Pi GND pin.
BUTTON_PIN = int(os.getenv("BUTTON_PIN", "17"))   # Button 1, physical pin 11
BUTTON2_PIN = int(os.getenv("BUTTON2_PIN", "27"))  # Button 2, physical pin 13
VIBRATION_PIN = int(os.getenv("VIBRATION_PIN", "18"))  # Motor driver input, physical pin 12
LED1_PIN = int(os.getenv("LED1_PIN", "22"))  # Heartbeat LED, physical pin 15
LED2_PIN = int(os.getenv("LED2_PIN", "23"))  # Processing LED, physical pin 16
LED3_PIN = int(os.getenv("LED3_PIN", "24"))  # Alert LED, physical pin 18
TRIG_PIN = int(os.getenv("TRIG_PIN", "5"))  # HC-SR04 TRIG, physical pin 29
ECHO_PIN = int(os.getenv("ECHO_PIN", "6"))  # HC-SR04 ECHO, physical pin 31

BUTTON_DEBOUNCE_SECONDS = float(os.getenv("BUTTON_DEBOUNCE_SECONDS", "0.05"))
BUTTON_POLL_SECONDS = float(os.getenv("BUTTON_POLL_SECONDS", "0.01"))
SHORT_PRESS_SECONDS = float(os.getenv("SHORT_PRESS_SECONDS", "0.60"))
LONG_PRESS_SECONDS = float(os.getenv("LONG_PRESS_SECONDS", "1.80"))
MEDIUM_PRESS_SECONDS = float(os.getenv("MEDIUM_PRESS_SECONDS", "1.10"))
DOUBLE_PRESS_WINDOW_SECONDS = float(os.getenv("DOUBLE_PRESS_WINDOW_SECONDS", "0.60"))

DISTANCE_POLL_SECONDS = float(os.getenv("DISTANCE_POLL_SECONDS", "0.20"))
OBSTACLE_THRESHOLD_CM = float(os.getenv("OBSTACLE_THRESHOLD_CM", "100"))
OBSTACLE_ALERT_COOLDOWN_SECONDS = float(os.getenv("OBSTACLE_ALERT_COOLDOWN_SECONDS", "2.50"))
ENABLE_OBSTACLE_AWARENESS = os.getenv("ENABLE_OBSTACLE_AWARENESS", "1").strip().lower() in {"1", "true", "yes", "on"}
OBSTACLE_MODE_LED_PIN = int(os.getenv("OBSTACLE_MODE_LED_PIN", "25"))  # Obstacle mode LED, physical pin 22
ENABLE_NAV_MORSE_GUIDANCE = os.getenv("ENABLE_NAV_MORSE_GUIDANCE", "1").strip().lower() in {"1", "true", "yes", "on"}
NAV_GUIDANCE_MIN_INTERVAL_SECONDS = float(os.getenv("NAV_GUIDANCE_MIN_INTERVAL_SECONDS", "3.0"))
NAV_GUIDANCE_LED_PIN = int(os.getenv("NAV_GUIDANCE_LED_PIN", str(LED3_PIN)))

MORSE_DOT_SECONDS = float(os.getenv("MORSE_DOT_SECONDS", "0.20"))
MORSE_DASH_SECONDS = float(os.getenv("MORSE_DASH_SECONDS", "0.60"))
MORSE_ELEMENT_GAP = float(os.getenv("MORSE_ELEMENT_GAP", "0.15"))
MORSE_LETTER_GAP = float(os.getenv("MORSE_LETTER_GAP", "0.45"))
MORSE_WORD_GAP = float(os.getenv("MORSE_WORD_GAP", "0.90"))
STOP_EVENT_CHECK_INTERVAL_SECONDS = float(os.getenv("STOP_EVENT_CHECK_INTERVAL_SECONDS", "0.02"))

MAX_MORSE_CHARS = int(os.getenv("MAX_MORSE_CHARS", "18"))
CARETAKER_POLL_SECONDS = float(os.getenv("CARETAKER_POLL_SECONDS", "10"))
REQUEST_TIMEOUT_SECONDS = float(os.getenv("REQUEST_TIMEOUT_SECONDS", "10"))
REQUEST_RETRIES = max(0, int(os.getenv("REQUEST_RETRIES", "1")))
REQUEST_RETRY_BASE_DELAY_SECONDS = float(os.getenv("REQUEST_RETRY_BASE_DELAY_SECONDS", "0.35"))
REQUEST_RETRY_MAX_DELAY_SECONDS = float(os.getenv("REQUEST_RETRY_MAX_DELAY_SECONDS", "3.0"))
ALLOW_INSECURE_SSL = os.getenv("GEMINI_ALLOW_INSECURE_SSL", "0").strip().lower() in {"1", "true", "yes", "on"}
GEMINI_CA_BUNDLE = os.getenv("GEMINI_CA_BUNDLE", "").strip()
GEMINI_USE_QUERY_API_KEY = os.getenv("GEMINI_USE_QUERY_API_KEY", "0").strip().lower() in {"1", "true", "yes", "on"}

GEMINI_FLASH_CONFIG = API_CONFIG.get("gemini_flash", {}) if isinstance(API_CONFIG, dict) else {}
GEMINI_VISION_CONFIG = API_CONFIG.get("gemini_vision", {}) if isinstance(API_CONFIG, dict) else {}
SIGN_LANGUAGE_CONFIG = API_CONFIG.get("sign_language", {}) if isinstance(API_CONFIG, dict) else {}
CARETAKER_CONFIG = API_CONFIG.get("caretaker", {}) if isinstance(API_CONFIG, dict) else {}
SOS_CONFIG = API_CONFIG.get("sos", {}) if isinstance(API_CONFIG, dict) else {}

GEMINI_FLASH_ENDPOINT = str(GEMINI_FLASH_CONFIG.get("endpoint", "")).strip()
GEMINI_FLASH_API_KEY = str(GEMINI_FLASH_CONFIG.get("api_key", "")).strip()
GEMINI_FLASH_TIMEOUT_SECONDS = float(GEMINI_FLASH_CONFIG.get("timeout_seconds", 6))
GEMINI_FLASH_RETRIES = max(0, int(GEMINI_FLASH_CONFIG.get("retries", 1)))
GEMINI_FLASH_MAX_OUTPUT_TOKENS = int(GEMINI_FLASH_CONFIG.get("max_output_tokens", 64))

GEMINI_VISION_ENDPOINT = str(GEMINI_VISION_CONFIG.get("endpoint", "")).strip()
GEMINI_VISION_API_KEY = str(GEMINI_VISION_CONFIG.get("api_key", GEMINI_FLASH_API_KEY)).strip()
GEMINI_VISION_TIMEOUT_SECONDS = float(GEMINI_VISION_CONFIG.get("timeout_seconds", 8))
GEMINI_VISION_RETRIES = max(0, int(GEMINI_VISION_CONFIG.get("retries", 1)))
GEMINI_VISION_MAX_OUTPUT_TOKENS = int(GEMINI_VISION_CONFIG.get("max_output_tokens", 80))

SIGN_LANGUAGE_ENDPOINT = str(SIGN_LANGUAGE_CONFIG.get("endpoint", GEMINI_VISION_ENDPOINT)).strip()
SIGN_LANGUAGE_API_KEY = str(SIGN_LANGUAGE_CONFIG.get("api_key", GEMINI_VISION_API_KEY)).strip()
SIGN_LANGUAGE_TIMEOUT_SECONDS = float(SIGN_LANGUAGE_CONFIG.get("timeout_seconds", GEMINI_VISION_TIMEOUT_SECONDS))
SIGN_LANGUAGE_RETRIES = max(0, int(SIGN_LANGUAGE_CONFIG.get("retries", GEMINI_VISION_RETRIES)))
SIGN_LANGUAGE_MAX_OUTPUT_TOKENS = int(SIGN_LANGUAGE_CONFIG.get("max_output_tokens", 96))

GEMINI_TIMEOUT_SECONDS = GEMINI_VISION_TIMEOUT_SECONDS
GEMINI_RETRIES = GEMINI_VISION_RETRIES
GEMINI_RATE_LIMIT_COOLDOWN_SECONDS = float(os.getenv("GEMINI_RATE_LIMIT_COOLDOWN_SECONDS", "60"))
GEMINI_COOLDOWN_LOG_INTERVAL_SECONDS = float(os.getenv("GEMINI_COOLDOWN_LOG_INTERVAL_SECONDS", "8"))
GEMINI_AUTH_COOLDOWN_SECONDS = float(os.getenv("GEMINI_AUTH_COOLDOWN_SECONDS", "900"))

IMAGE_DIRECTORY = Path(os.getenv("IMAGE_DIRECTORY", str(Path(tempfile.gettempdir()) / "assistive_images")))

CARETAKER_API_URL = str(CARETAKER_CONFIG.get("endpoint", os.getenv("CARETAKER_API_URL", os.getenv("CAREGIVER_API_URL", "")))).strip()
CARETAKER_API_TOKEN = str(CARETAKER_CONFIG.get("token", os.getenv("CARETAKER_API_TOKEN", os.getenv("CAREGIVER_API_TOKEN", "")))).strip()
CARETAKER_API_TIMEOUT_SECONDS = float(CARETAKER_CONFIG.get("timeout_seconds", REQUEST_TIMEOUT_SECONDS))
ENABLE_CARETAKER_POLL = os.getenv("ENABLE_CARETAKER_POLL", "0").strip().lower() in {"1", "true", "yes", "on"}
SOS_API_URL = str(SOS_CONFIG.get("endpoint", os.getenv("SOS_API_URL", ""))).strip()
SOS_API_TOKEN = str(SOS_CONFIG.get("token", os.getenv("SOS_API_TOKEN", ""))).strip()
SOS_API_TIMEOUT_SECONDS = float(SOS_CONFIG.get("timeout_seconds", REQUEST_TIMEOUT_SECONDS))
ENABLE_APP_SERVICE_APIS = os.getenv("ENABLE_APP_SERVICE_APIS", "0").strip().lower() in {"1", "true", "yes", "on"}

GEMINI_API_KEY = GEMINI_FLASH_API_KEY or GEMINI_VISION_API_KEY or os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-flash-latest").strip()
GEMINI_FLASH_MODEL = os.getenv("GEMINI_FLASH_MODEL", GEMINI_DEFAULT_MODEL).strip()
GEMINI_VISION_MODEL = os.getenv("GEMINI_VISION_MODEL", GEMINI_DEFAULT_MODEL).strip()
GEMINI_API_BASE = os.getenv("GEMINI_API_BASE", "https://generativelanguage.googleapis.com/v1beta").strip()
GEMINI_MODEL_FALLBACKS = [
    item.strip()
    for item in os.getenv("GEMINI_MODEL_FALLBACKS", "gemini-2.0-flash,gemini-1.5-flash-latest,gemini-1.5-flash").split(",")
    if item.strip()
]

NAVIGATION_SAFE_DISTANCE_CM = float(os.getenv("NAVIGATION_SAFE_DISTANCE_CM", "150"))
NAVIGATION_MIN_PULSE_INTERVAL_SECONDS = float(os.getenv("NAVIGATION_MIN_PULSE_INTERVAL_SECONDS", "0.18"))
NAVIGATION_MAX_PULSE_INTERVAL_SECONDS = float(os.getenv("NAVIGATION_MAX_PULSE_INTERVAL_SECONDS", "0.95"))
NAVIGATION_MODE_ENABLED = False

LED_HEARTBEAT_STATE = False
LED_PROCESSING_STATE = False
LED_ALERT_STATE = False
LED_OBSTACLE_MODE_STATE = False

PERSON_ALERT_BLINK_INTERVAL_SECONDS = float(os.getenv("PERSON_ALERT_BLINK_INTERVAL_SECONDS", "0.40"))
PERSON_ALERT_ACTIVE = False
PERSON_ALERT_LED_ON = False
PERSON_ALERT_LAST_TOGGLE_AT = 0.0
_GEMINI_BACKOFF_UNTIL = 0.0
_LAST_GEMINI_BACKOFF_LOG_AT = 0.0


MORSE_CODE: Dict[str, str] = {
    "A": ".-",
    "B": "-...",
    "C": "-.-.",
    "D": "-..",
    "E": ".",
    "F": "..-.",
    "G": "--.",
    "H": "....",
    "I": "..",
    "J": ".---",
    "K": "-.-",
    "L": ".-..",
    "M": "--",
    "N": "-.",
    "O": "---",
    "P": ".--.",
    "Q": "--.-",
    "R": ".-.",
    "S": "...",
    "T": "-",
    "U": "..-",
    "V": "...-",
    "W": ".--",
    "X": "-..-",
    "Y": "-.--",
    "Z": "--..",
    "0": "-----",
    "1": ".----",
    "2": "..---",
    "3": "...--",
    "4": "....-",
    "5": ".....",
    "6": "-....",
    "7": "--...",
    "8": "---..",
    "9": "----.",
    ".": ".-.-.-",
    ",": "--..--",
    "?": "..--..",
    "!": "-.-.--",
    "/": "-..-.",
    "-": "-....-",
    "(": "-.--.",
    ")": "-.--.-",
    "&": ".-...",
    ":": "---...",
    ";": "-.-.-.",
    "=": "-...-",
    "+": ".-.-.",
    "_": "..--.-",
    '"': ".-..-.",
    "$": "...-..-",
    "@": ".--.-.",
}

MODE_PREFIXES = {
    "sos": "...",
    "detection": "--",
    "room_scan": ".-",
}


def sanitize_log_message(message: str) -> str:
    """Redact common secret patterns from log messages before printing."""
    sanitized = str(message)
    sanitized = re.sub(r"([?&](?:key|api_key|token|access_token)=)[^&\s]+", r"\1***", sanitized, flags=re.IGNORECASE)
    sanitized = re.sub(r"(Bearer\s+)[A-Za-z0-9._\-]+", r"\1***", sanitized, flags=re.IGNORECASE)
    sanitized = re.sub(r"(GEMINI_API_KEY=)[^\s]+", r"\1***", sanitized, flags=re.IGNORECASE)
    sanitized = re.sub(r"AIza[0-9A-Za-z\-_]{20,}", "***", sanitized)
    return sanitized


def log(message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {sanitize_log_message(message)}", flush=True)


def normalize_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    ascii_text = re.sub(r"\s+", " ", ascii_text).strip()
    return ascii_text


def shorten_for_morse(text: str, max_chars: int = MAX_MORSE_CHARS) -> str:
    cleaned = normalize_text(text).upper()
    cleaned = re.sub(r"[^A-Z0-9 ?!.,/()&:+=_\-@;$\"]", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if not cleaned:
        return ""

    words = cleaned.split()
    selected: List[str] = []
    for word in words:
        candidate = " ".join(selected + [word])
        if len(candidate) > max_chars and selected:
            break
        selected.append(word)
        if len(" ".join(selected)) >= max_chars:
            break

    shortened = " ".join(selected).strip()
    if not shortened:
        shortened = cleaned[:max_chars].strip()
    return shortened[:max_chars]


def extract_text_from_payload(payload: Any) -> str:
    if payload is None:
        return ""
    if isinstance(payload, str):
        return payload.strip()
    if isinstance(payload, list):
        for item in payload:
            text = extract_text_from_payload(item)
            if text:
                return text
        return ""
    if isinstance(payload, dict):
        for key in ("text", "message", "body", "content", "result", "description", "output"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        for key in ("messages", "data", "choices", "parts", "candidates", "content"):
            value = payload.get(key)
            if value:
                text = extract_text_from_payload(value)
                if text:
                    return text
        return ""
    return str(payload).strip()


def parse_retry_after_seconds(value: Optional[str]) -> Optional[float]:
    if not value:
        return None
    try:
        retry_seconds = float(value.strip())
    except Exception:
        return None
    return max(0.0, retry_seconds)


def should_retry_request_error(exc: Exception) -> bool:
    if isinstance(exc, urllib_error.HTTPError):
        return exc.code in {408, 425, 429, 500, 502, 503, 504}
    if isinstance(exc, urllib_error.URLError):
        return True
    if isinstance(exc, TimeoutError):
        return True
    return "timed out" in str(exc).lower()


def retry_delay_seconds(exc: Exception, attempt: int) -> float:
    if isinstance(exc, urllib_error.HTTPError):
        retry_after = parse_retry_after_seconds(exc.headers.get("Retry-After"))
        if retry_after is not None:
            return min(REQUEST_RETRY_MAX_DELAY_SECONDS, retry_after)
    exponential = REQUEST_RETRY_BASE_DELAY_SECONDS * (2 ** max(0, attempt - 1))
    return min(REQUEST_RETRY_MAX_DELAY_SECONDS, exponential)


def is_ssl_verification_error(exc: Exception) -> bool:
    if isinstance(exc, ssl.SSLCertVerificationError):
        return True
    if isinstance(exc, urllib_error.URLError) and isinstance(exc.reason, ssl.SSLCertVerificationError):
        return True
    return "certificate verify failed" in str(exc).lower()


def collect_ca_bundle_candidates(cli_override: Optional[str] = None) -> List[str]:
    candidates: List[str] = []

    if cli_override and cli_override.strip():
        candidates.append(cli_override.strip())
    if GEMINI_CA_BUNDLE:
        candidates.append(GEMINI_CA_BUNDLE)

    for env_name in ("SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE"):
        bundle_path = os.getenv(env_name, "").strip()
        if bundle_path:
            candidates.append(bundle_path)

    try:
        default_paths = ssl.get_default_verify_paths()
        if default_paths.cafile:
            candidates.append(default_paths.cafile)
    except Exception:
        pass

    if certifi is not None:
        try:
            candidates.append(certifi.where())
        except Exception:
            pass

    candidates.extend(
        [
            r"/etc/ssl/certs/ca-certificates.crt",
            r"/etc/pki/tls/certs/ca-bundle.crt",
            r"/etc/ssl/cert.pem",
            r"C:\Program Files\Git\usr\ssl\certs\ca-bundle.crt",
            r"C:\Program Files\Git\mingw64\ssl\certs\ca-bundle.crt",
            r"C:\msys64\ucrt64\etc\ssl\cert.pem",
            r"C:\msys64\mingw64\etc\ssl\cert.pem",
            r"C:\msys64\usr\ssl\cert.pem",
        ]
    )

    unique_candidates: List[str] = []
    seen: set[str] = set()
    for path in candidates:
        normalized = str(Path(path).expanduser())
        if normalized not in seen:
            seen.add(normalized)
            unique_candidates.append(normalized)
    return unique_candidates


def build_ssl_context(cli_ca_bundle: Optional[str] = None) -> ssl.SSLContext:
    context = ssl.create_default_context()
    for candidate in collect_ca_bundle_candidates(cli_ca_bundle):
        candidate_path = Path(candidate).expanduser()
        if not candidate_path.exists() or not candidate_path.is_file():
            continue
        try:
            context.load_verify_locations(cafile=str(candidate_path))
            return context
        except Exception:
            continue
    return context


def request_json(
    method: str,
    url: str,
    payload: Optional[Dict[str, Any]] = None,
    headers: Optional[Dict[str, str]] = None,
    timeout: float = REQUEST_TIMEOUT_SECONDS,
    retries: int = REQUEST_RETRIES,
    log_failures: bool = True,
    ca_bundle: Optional[str] = None,
) -> Any:
    redacted_url = redact_url_for_logs(url)
    request_headers = {
        "Content-Type": "application/json",
        "User-Agent": f"{APP_NAME}/1.0",
    }
    if headers:
        request_headers.update(headers)

    body = None
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
    ssl_context = build_ssl_context(ca_bundle)

    last_error: Optional[Exception] = None
    attempt_count = max(1, retries + 1)
    for attempt in range(1, attempt_count + 1):
        request = urllib_request.Request(url, data=body, headers=request_headers, method=method.upper())
        try:
            with urllib_request.urlopen(request, timeout=timeout, context=ssl_context) as response:
                response_text = response.read().decode("utf-8", errors="replace")
                if not response_text.strip():
                    return {}
                return json.loads(response_text)
        except Exception as exc:
            failure: Exception = exc
            if ALLOW_INSECURE_SSL and is_ssl_verification_error(exc):
                log("SSL verification failed; retrying once with insecure SSL mode enabled.")
                try:
                    insecure_context = ssl._create_unverified_context()
                    with urllib_request.urlopen(request, timeout=timeout, context=insecure_context) as response:
                        response_text = response.read().decode("utf-8", errors="replace")
                        if not response_text.strip():
                            return {}
                        return json.loads(response_text)
                except Exception as insecure_exc:
                    failure = insecure_exc
                    if log_failures:
                        log(f"Insecure SSL retry failed for {url}: {insecure_exc}")

            last_error = failure
            if log_failures:
                log(f"Request attempt {attempt}/{attempt_count} failed for {redacted_url}: {failure}")
            if attempt < attempt_count and should_retry_request_error(failure):
                time.sleep(retry_delay_seconds(failure, attempt))
                continue
            break

    if last_error is not None:
        raise last_error
    raise RuntimeError(f"Request failed for {url}")


def activate_gemini_rate_limit(retry_after_seconds: Optional[float] = None) -> None:
    global _GEMINI_BACKOFF_UNTIL
    cooldown = retry_after_seconds if retry_after_seconds is not None else GEMINI_RATE_LIMIT_COOLDOWN_SECONDS
    cooldown = max(1.0, cooldown)
    _GEMINI_BACKOFF_UNTIL = max(_GEMINI_BACKOFF_UNTIL, time.monotonic() + cooldown)


def activate_gemini_auth_cooldown() -> None:
    global _GEMINI_BACKOFF_UNTIL
    cooldown = max(60.0, GEMINI_AUTH_COOLDOWN_SECONDS)
    _GEMINI_BACKOFF_UNTIL = max(_GEMINI_BACKOFF_UNTIL, time.monotonic() + cooldown)
    log(
        "Gemini auth/access error detected (401/403). "
        f"Pausing Gemini calls for {cooldown:.0f}s and using local fallbacks."
    )


def gemini_available() -> bool:
    global _LAST_GEMINI_BACKOFF_LOG_AT
    now = time.monotonic()
    remaining = _GEMINI_BACKOFF_UNTIL - now
    if remaining <= 0:
        return True
    if now - _LAST_GEMINI_BACKOFF_LOG_AT >= GEMINI_COOLDOWN_LOG_INTERVAL_SECONDS:
        log(f"Gemini cooldown active for {remaining:.0f}s; using local fallback.")
        _LAST_GEMINI_BACKOFF_LOG_AT = now
    return False


def gemini_cooldown_active() -> bool:
    return (time.monotonic() < _GEMINI_BACKOFF_UNTIL)


def redact_url_for_logs(url: str) -> str:
    """Redact sensitive query parameter values (for example API keys/tokens) in URLs."""
    try:
        parsed = urllib_parse.urlparse(url)
        if not parsed.query:
            return url
        sensitive_keys = {"key", "api_key", "token", "access_token"}
        query_pairs = urllib_parse.parse_qsl(parsed.query, keep_blank_values=True)
        redacted_pairs = [
            (name, "***" if name.lower() in sensitive_keys else value)
            for name, value in query_pairs
        ]
        redacted_query = urllib_parse.urlencode(redacted_pairs, doseq=True)
        return urllib_parse.urlunparse(parsed._replace(query=redacted_query))
    except Exception:
        return url


def extract_http_error_details(exc: urllib_error.HTTPError) -> str:
    """Return HTTPError text plus a truncated response body, when available."""
    detail = str(exc)
    try:
        body = exc.read().decode("utf-8", errors="replace").strip()
        if body:
            detail = f"{detail} | {body[:500]}"
    except Exception:
        pass
    return detail


def clamp_text_for_prompt(text: str, max_chars: int = 20) -> str:
    cleaned = normalize_text(text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned[:max_chars]


def text_to_morse_local(text: str) -> str:
    sanitized = shorten_for_morse(text, max_chars=MAX_MORSE_CHARS).upper()
    pieces: List[str] = []
    for character in sanitized:
        if character == " ":
            if pieces and pieces[-1] != " ":
                pieces.append(" ")
            continue
        code = MORSE_CODE.get(character)
        if code:
            pieces.append(code)
            pieces.append(" ")
    morse = "".join(pieces).strip()
    morse = re.sub(r"\s+", " ", morse)
    return morse


def sanitize_morse_output(text: str) -> str:
    cleaned = normalize_text(text)
    cleaned = cleaned.replace("•", ".").replace("—", "-").replace("–", "-")
    cleaned = re.sub(r"[^.\- ]+", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned[:MAX_MORSE_CHARS * 4]


def _build_gemini_generate_content_url(endpoint: str, api_key: str, fallback_model: str) -> str:
    """Build Gemini URL in the same format as api_test.py.

    Final format:
      {api_base}/models/{urlencoded_model}:generateContent?key={urlencoded_key}
    """
    model = fallback_model.strip()
    api_base = GEMINI_API_BASE.rstrip("/")

    if endpoint:
        parsed = urllib_parse.urlparse(endpoint)
        path = (parsed.path or "").rstrip("/")
        if parsed.scheme and parsed.netloc:
            models_marker = "/models/"
            if models_marker in path and path.endswith(":generateContent"):
                base_path, model_part = path.split(models_marker, 1)
                model_name = model_part.rsplit(":generateContent", 1)[0].strip()
                if model_name:
                    model = urllib_parse.unquote(model_name)
                api_base = f"{parsed.scheme}://{parsed.netloc}{base_path}".rstrip("/")
            else:
                api_base = f"{parsed.scheme}://{parsed.netloc}{path}".rstrip("/")

    base_url = f"{api_base}/models/{urllib_parse.quote(model, safe='')}:generateContent"
    if GEMINI_USE_QUERY_API_KEY:
        return f"{base_url}?key={urllib_parse.quote(api_key, safe='')}"
    return base_url


def _gemini_base_candidates(primary_base: str) -> List[str]:
    """Return deduplicated Gemini API base candidates, including v1/v1beta fallbacks."""
    candidates: List[str] = []

    def _add(base: str) -> None:
        normalized = base.rstrip("/")
        if normalized and normalized not in candidates:
            candidates.append(normalized)

    _add(primary_base)
    _add(GEMINI_API_BASE)
    for base in list(candidates):
        if "/v1beta" in base:
            _add(base.replace("/v1beta", "/v1"))
        elif base.endswith("/v1"):
            _add(f"{base}beta")
    return candidates


def _gemini_model_candidates(primary_model: str) -> List[str]:
    """Return deduplicated model candidates ordered from preferred to fallback options."""
    candidates: List[str] = []

    def _add(model_name: str) -> None:
        normalized = model_name.strip()
        if normalized and normalized not in candidates:
            candidates.append(normalized)

    _add(primary_model)
    _add(GEMINI_DEFAULT_MODEL)
    _add(GEMINI_FLASH_MODEL)
    _add(GEMINI_VISION_MODEL)
    for fallback_model in GEMINI_MODEL_FALLBACKS:
        _add(fallback_model)
    return candidates


def build_gemini_candidate_urls(endpoint: str, api_key: str, fallback_model: str) -> List[str]:
    """Create ordered endpoint/model candidates to improve compatibility across Gemini API versions."""
    parsed = urllib_parse.urlparse(endpoint.strip()) if endpoint.strip() else None
    extracted_model = fallback_model.strip()
    extracted_base = GEMINI_API_BASE.rstrip("/")

    if parsed and parsed.scheme and parsed.netloc:
        path = (parsed.path or "").rstrip("/")
        models_marker = "/models/"
        if models_marker in path and path.endswith(":generateContent"):
            base_path, model_part = path.split(models_marker, 1)
            model_name = model_part.rsplit(":generateContent", 1)[0].strip()
            if model_name:
                extracted_model = urllib_parse.unquote(model_name)
            extracted_base = f"{parsed.scheme}://{parsed.netloc}{base_path}".rstrip("/")
        else:
            extracted_base = f"{parsed.scheme}://{parsed.netloc}{path}".rstrip("/")

    urls: List[str] = []
    for base_candidate in _gemini_base_candidates(extracted_base):
        for model_candidate in _gemini_model_candidates(extracted_model):
            url = _build_gemini_generate_content_url(
                f"{base_candidate}/models/{urllib_parse.quote(model_candidate, safe='')}:generateContent",
                api_key,
                model_candidate,
            )
            if url not in urls:
                urls.append(url)
    return urls


def call_gemini_flash(text: str) -> Optional[str]:
    cleaned_text = clamp_text_for_prompt(text, max_chars=MAX_MORSE_CHARS)
    if not GEMINI_FLASH_API_KEY or not GEMINI_FLASH_ENDPOINT:
        log("Gemini Flash API config missing; using local Morse conversion.")
        return sanitize_morse_output(text_to_morse_local(cleaned_text)) or None
    if not gemini_available():
        return sanitize_morse_output(text_to_morse_local(cleaned_text)) or None

    prompt = f"Convert this text into Morse code using only dots and dashes: {cleaned_text}"
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": prompt}],
            }
        ],
        "generationConfig": {
            "temperature": 0.0,
            "maxOutputTokens": GEMINI_FLASH_MAX_OUTPUT_TOKENS,
        },
    }

    gemini_headers = {"x-goog-api-key": GEMINI_FLASH_API_KEY} if not GEMINI_USE_QUERY_API_KEY else None
    auth_error_seen = False
    try:
        for endpoint in build_gemini_candidate_urls(
            GEMINI_FLASH_ENDPOINT,
            GEMINI_FLASH_API_KEY,
            GEMINI_FLASH_MODEL,
        ):
            try:
                response_json = request_json(
                    "POST",
                    endpoint,
                    payload=payload,
                    headers=gemini_headers,
                    timeout=GEMINI_FLASH_TIMEOUT_SECONDS,
                    retries=GEMINI_FLASH_RETRIES,
                )
                result_text = extract_text_from_payload(response_json)
                morse_text = sanitize_morse_output(result_text)
                if morse_text:
                    return morse_text
            except urllib_error.HTTPError as exc:
                auth_error_seen = auth_error_seen or exc.code in {401, 403}
                if exc.code == 429:
                    activate_gemini_rate_limit(parse_retry_after_seconds(exc.headers.get("Retry-After")))
                    break
                if exc.code not in {400, 401, 403, 404}:
                    raise
                continue
    except urllib_error.HTTPError as exc:
        if exc.code == 429:
            activate_gemini_rate_limit(parse_retry_after_seconds(exc.headers.get("Retry-After")))
        elif exc.code in {401, 403}:
            activate_gemini_auth_cooldown()
        log(f"Gemini Flash Morse conversion failed: {extract_http_error_details(exc)}")
    except Exception as exc:
        if is_ssl_verification_error(exc):
            log(
                "Gemini TLS verification failed. Set GEMINI_CA_BUNDLE or SSL_CERT_FILE "
                "to a valid CA bundle path."
            )
        log(f"Gemini Flash Morse conversion failed: {exc}")
    if auth_error_seen:
        activate_gemini_auth_cooldown()
        log("Gemini Flash Morse conversion failed after trying fallback models/endpoints.")

    return sanitize_morse_output(text_to_morse_local(cleaned_text)) or None


def call_gemini_vision(
    image_path: Path,
    prompt: str,
    *,
    endpoint: Optional[str] = None,
    api_key: Optional[str] = None,
    timeout_seconds: Optional[float] = None,
    retries: Optional[int] = None,
    max_output_tokens: Optional[int] = None,
) -> Optional[str]:
    selected_endpoint = (endpoint or GEMINI_VISION_ENDPOINT).strip()
    selected_api_key = (api_key or GEMINI_VISION_API_KEY).strip()
    selected_timeout = timeout_seconds if timeout_seconds is not None else GEMINI_VISION_TIMEOUT_SECONDS
    selected_retries = retries if retries is not None else GEMINI_VISION_RETRIES
    selected_max_tokens = max_output_tokens if max_output_tokens is not None else GEMINI_VISION_MAX_OUTPUT_TOKENS

    if not selected_endpoint or not selected_api_key:
        log("Gemini Vision API config missing.")
        return None
    if not gemini_available():
        return None

    try:
        image_base64 = encode_base64(image_path)
        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"text": prompt},
                        {
                            "inline_data": {
                                "mime_type": guess_mime_type(image_path),
                                "data": image_base64,
                            }
                        },
                    ],
                }
            ],
            "generationConfig": {
                "temperature": 0.0,
                "maxOutputTokens": int(selected_max_tokens),
            },
        }
        gemini_headers = {"x-goog-api-key": selected_api_key} if not GEMINI_USE_QUERY_API_KEY else None
        auth_error_seen = False
        for request_url in build_gemini_candidate_urls(
            selected_endpoint,
            selected_api_key,
            GEMINI_VISION_MODEL,
        ):
            try:
                response_json = request_json(
                    "POST",
                    request_url,
                    payload=payload,
                    headers=gemini_headers,
                    timeout=float(selected_timeout),
                    retries=max(0, int(selected_retries)),
                )
                response_text = extract_text_from_payload(response_json)
                if response_text:
                    return response_text
            except urllib_error.HTTPError as exc:
                auth_error_seen = auth_error_seen or exc.code in {401, 403}
                if exc.code == 429:
                    activate_gemini_rate_limit(parse_retry_after_seconds(exc.headers.get("Retry-After")))
                    break
                if exc.code not in {400, 401, 403, 404}:
                    raise
                continue
        if auth_error_seen:
            activate_gemini_auth_cooldown()
            log("Gemini vision call failed after trying fallback models/endpoints.")
        return None
    except urllib_error.HTTPError as exc:
        if exc.code == 429:
            activate_gemini_rate_limit(parse_retry_after_seconds(exc.headers.get("Retry-After")))
        elif exc.code in {401, 403}:
            activate_gemini_auth_cooldown()
        log(f"Gemini vision call failed: {extract_http_error_details(exc)}")
        return None
    except Exception as exc:
        if is_ssl_verification_error(exc):
            log(
                "Gemini TLS verification failed. Set GEMINI_CA_BUNDLE or SSL_CERT_FILE "
                "to a valid CA bundle path."
            )
        log(f"Gemini vision call failed: {exc}")
        return None


def call_gemini_morse(text: str) -> Optional[str]:
    return call_gemini_flash(text)


def call_gemini_api(image_path: Path, prompt: str, model: str = GEMINI_VISION_MODEL) -> Optional[str]:
    _ = model
    return call_gemini_vision(image_path, prompt)


def measure_distance() -> Optional[float]:
    try:
        return GPIO.read_distance_cm()
    except Exception as exc:
        log(f"Distance measurement failed: {exc}")
        return None


def control_leds(*, heartbeat: Optional[bool] = None, processing: Optional[bool] = None, alert: Optional[bool] = None) -> None:
    global LED_HEARTBEAT_STATE, LED_PROCESSING_STATE, LED_ALERT_STATE

    if heartbeat is not None:
        LED_HEARTBEAT_STATE = heartbeat
    if processing is not None:
        LED_PROCESSING_STATE = processing
    if alert is not None:
        LED_ALERT_STATE = alert

    try:
        GPIO.output(LED1_PIN, GPIO.HIGH if LED_HEARTBEAT_STATE else GPIO.LOW)
        GPIO.output(LED2_PIN, GPIO.HIGH if LED_PROCESSING_STATE else GPIO.LOW)
        GPIO.output(LED3_PIN, GPIO.HIGH if LED_ALERT_STATE else GPIO.LOW)
    except Exception as exc:
        log(f"LED control failed: {exc}")


def set_obstacle_mode_led(enabled: bool) -> None:
    global LED_OBSTACLE_MODE_STATE
    LED_OBSTACLE_MODE_STATE = enabled
    try:
        GPIO.output(OBSTACLE_MODE_LED_PIN, GPIO.HIGH if enabled else GPIO.LOW)
    except Exception as exc:
        log(f"Obstacle mode LED control failed: {exc}")


def set_person_alert_mode(active: bool) -> None:
    global PERSON_ALERT_ACTIVE, PERSON_ALERT_LED_ON, PERSON_ALERT_LAST_TOGGLE_AT
    PERSON_ALERT_ACTIVE = active
    PERSON_ALERT_LED_ON = False
    PERSON_ALERT_LAST_TOGGLE_AT = 0.0
    if not active:
        control_leds(alert=False)


def update_person_alert_led(now: float) -> bool:
    global PERSON_ALERT_LED_ON, PERSON_ALERT_LAST_TOGGLE_AT
    if not PERSON_ALERT_ACTIVE:
        return False
    if PERSON_ALERT_LAST_TOGGLE_AT == 0.0 or (now - PERSON_ALERT_LAST_TOGGLE_AT) >= PERSON_ALERT_BLINK_INTERVAL_SECONDS:
        PERSON_ALERT_LED_ON = not PERSON_ALERT_LED_ON
        PERSON_ALERT_LAST_TOGGLE_AT = now
    return PERSON_ALERT_LED_ON


def navigation_band_from_distance(distance_cm: float) -> str:
    """Maps measured distance to navigation feedback bands."""
    if distance_cm > 150:
        return "CLEAR"
    if 80 <= distance_cm <= 150:
        return "FAR"
    if 30 <= distance_cm < 80:
        return "NEAR"
    return "STOP"


def send_navigation_guidance(morse_worker: "MorseWorker", distance_cm: float) -> None:
    """Converts distance band to Morse and pushes it to haptic+LED output."""
    band = navigation_band_from_distance(distance_cm)
    morse_pattern = text_to_morse_local(band)
    if morse_pattern:
        log(f"Navigation guidance: {band} ({distance_cm:.1f} cm)")
        play_haptic(morse_worker, morse_pattern)


def encode_base64(image_path: Path) -> str:
    with image_path.open("rb") as image_file:
        encoded_bytes = base64.b64encode(image_file.read())
    return encoded_bytes.decode("ascii")


def encode_image_base64(image_path: Path) -> str:
    return encode_base64(image_path)


def guess_mime_type(path: Path) -> str:
    extension = path.suffix.lower()
    if extension in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if extension == ".png":
        return "image/png"
    if extension == ".webp":
        return "image/webp"
    return "application/octet-stream"


def capture_image(prefix: str) -> Optional[Path]:
    IMAGE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    image_path = IMAGE_DIRECTORY / f"{prefix}_{timestamp}.jpg"
    errors: List[str] = []

    if Picamera2 is not None:
        camera = None
        try:
            camera = Picamera2()
            try:
                config = camera.create_still_configuration(
                    main={"size": (1280, 720), "format": "RGB888"}
                )
            except Exception:
                # Some USB/UVC cameras expose limited formats; use backend default.
                config = camera.create_still_configuration()
            camera.configure(config)
            camera.start()
            time.sleep(0.8)
            camera.capture_file(str(image_path))
            if image_path.exists() and image_path.stat().st_size > 0:
                log(f"Captured image with Picamera2: {image_path}")
                return image_path
            errors.append("Picamera2 created an empty image")
        except Exception as exc:
            errors.append(f"Picamera2 failed: {exc}")
        finally:
            if camera is not None:
                try:
                    camera.stop()
                except Exception:
                    pass
                try:
                    camera.close()
                except Exception:
                    pass

    for tool_name in ("rpicam-still", "libcamera-still"):
        if not shutil.which(tool_name):
            continue
        command = [
            tool_name,
            "-n",
            "-t",
            "1000",
            "-o",
            str(image_path),
        ]
        try:
            subprocess.run(command, check=True, timeout=30, capture_output=True, text=True)
            if image_path.exists() and image_path.stat().st_size > 0:
                log(f"Captured image with {tool_name}: {image_path}")
                return image_path
            errors.append(f"{tool_name} created an empty image")
        except Exception as exc:
            errors.append(f"{tool_name} failed: {exc}")

    if shutil.which("fswebcam"):
        command = ["fswebcam", "-r", "640x480", "--no-banner", str(image_path)]
        try:
            subprocess.run(command, check=True, timeout=30, capture_output=True, text=True)
            if image_path.exists() and image_path.stat().st_size > 0:
                log(f"Captured image with fswebcam: {image_path}")
                return image_path
            errors.append("fswebcam created an empty image")
        except Exception as exc:
            errors.append(f"fswebcam failed: {exc}")

    if errors:
        log("Image capture failed: " + " | ".join(errors))
    else:
        log("Image capture failed: no camera backend available")
    return None


def sanitize_detection_output(response_text: str) -> str:
    upper_text = normalize_text(response_text).upper()
    if re.search(r"\bYES\b", upper_text):
        return "YES"
    if re.search(r"\bNO\b", upper_text):
        return "NO"
    if "TRUE" in upper_text or "PRESENT" in upper_text or "FOUND" in upper_text:
        return "YES"
    if "FALSE" in upper_text or "ABSENT" in upper_text or "NONE" in upper_text:
        return "NO"
    return "ERR"


def sanitize_room_output(response_text: str) -> str:
    normalized = normalize_text(response_text).upper()
    normalized = re.sub(r"[^A-Z0-9 ]+", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    if not normalized:
        return "ERR"

    words = normalized.split()
    selected: List[str] = []
    for word in words:
        if len(selected) >= 5:
            break
        candidate = " ".join(selected + [word])
        if len(candidate) > MAX_MORSE_CHARS and selected:
            break
        selected.append(word)

    result = " ".join(selected).strip()
    if not result:
        result = normalized[:MAX_MORSE_CHARS].strip()
    return result[:MAX_MORSE_CHARS] or "ERR"


def description_mentions_person(text: str) -> bool:
    upper_text = normalize_text(text).upper()
    return any(token in upper_text for token in ("PERSON", "HUMAN", "MAN", "WOMAN", "PEOPLE", "CHILD"))


def build_room_description(image_path: Path, *, prioritize_person: bool = False) -> str:
    if prioritize_person:
        prompt = (
            "Look carefully for people first. Reply in this format only: "
            "PERSON:YES|NO; DESC:<3 to 6 words about room>."
        )
    else:
        prompt = "Describe the room in 3-5 short words for navigation"

    response_text = call_gemini_api(image_path, prompt, model=GEMINI_VISION_MODEL)
    if not response_text:
        return "UNKNOWN AREA"
    return response_text


def parse_person_priority_response(text: str) -> tuple[Optional[bool], str]:
    normalized = normalize_text(text)
    upper_text = normalized.upper()
    person_present: Optional[bool] = None

    if "PERSON:YES" in upper_text:
        person_present = True
    elif "PERSON:NO" in upper_text:
        person_present = False

    description_text = normalized
    match = re.search(r"DESC\s*:\s*(.+)", normalized, flags=re.IGNORECASE)
    if match:
        description_text = match.group(1).strip()

    return person_present, sanitize_room_output(description_text)


def post_sos_alert() -> bool:
    if not ENABLE_APP_SERVICE_APIS:
        log("App-service APIs are disabled; SOS API call skipped.")
        return False
    if not SOS_API_URL:
        log("SOS_API_URL is not configured; alert will be logged only.")
        return False

    headers = {}
    if SOS_API_TOKEN:
        headers["Authorization"] = f"Bearer {SOS_API_TOKEN}"

    payload = {
        "device_id": DEVICE_ID,
        "event": "sos",
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }
    try:
        request_json("POST", SOS_API_URL, payload=payload, headers=headers, timeout=SOS_API_TIMEOUT_SECONDS, retries=1)
        log("SOS alert sent successfully.")
        return True
    except Exception as exc:
        log(f"SOS alert failed: {exc}")
        return False


def poll_caretaker_messages() -> List[str]:
    if not ENABLE_APP_SERVICE_APIS:
        return []
    if not CARETAKER_API_URL:
        return []

    headers = {}
    if CARETAKER_API_TOKEN:
        headers["Authorization"] = f"Bearer {CARETAKER_API_TOKEN}"

    try:
        payload = request_json("GET", CARETAKER_API_URL, headers=headers, timeout=CARETAKER_API_TIMEOUT_SECONDS, retries=1)
    except Exception as exc:
        log(f"Caretaker poll failed: {exc}")
        return []

    messages: List[str] = []
    if isinstance(payload, dict):
        raw_messages = payload.get("messages")
        if isinstance(raw_messages, list):
            for entry in raw_messages:
                message_text = extract_text_from_payload(entry)
                if message_text:
                    messages.append(message_text)
        else:
            message_text = extract_text_from_payload(payload)
            if message_text:
                messages.append(message_text)
    elif isinstance(payload, list):
        for entry in payload:
            message_text = extract_text_from_payload(entry)
            if message_text:
                messages.append(message_text)

    return messages


@dataclass
class MorseTask:
    pattern: str


@dataclass
class ButtonState:
    last_raw_state: bool = False
    stable_state: bool = False
    last_change_at: float = 0.0
    press_started_at: Optional[float] = None


@dataclass
class DoublePressState:
    waiting_second_press: bool = False
    first_press_at: float = 0.0


class MorseWorker:
    def __init__(self) -> None:
        self.queue: "queue.Queue[MorseTask]" = queue.Queue()
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.motor_lock = threading.Lock()

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        self.queue.put(MorseTask(""))
        self.thread.join(timeout=5.0)
        if self.thread.is_alive():
            log("Morse worker did not stop within timeout; forcing GPIO cleanup path.")

    def submit(self, pattern: str) -> None:
        prepared_pattern = sanitize_morse_output(pattern)
        if not prepared_pattern:
            return
        self.queue.put(MorseTask(pattern=prepared_pattern))

    def _run(self) -> None:
        while not self.stop_event.is_set():
            try:
                task = self.queue.get(timeout=0.2)
            except queue.Empty:
                continue

            try:
                if task.pattern:
                    self._play_task(task)
            except Exception as exc:
                log(f"Morse worker error: {exc}")
            finally:
                self.queue.task_done()

    def _play_task(self, task: MorseTask) -> None:
        with self.motor_lock:
            self._play_pattern(task.pattern)

    def _play_pattern(self, pattern: str) -> None:
        letters = [letter.strip() for letter in pattern.split() if letter.strip()]
        for letter_index, letter in enumerate(letters):
            if self.stop_event.is_set():
                return
            symbols = [symbol for symbol in letter if symbol in ".-"]
            for symbol_index, symbol in enumerate(symbols):
                if self.stop_event.is_set():
                    return
                self._pulse(MORSE_DOT_SECONDS if symbol == "." else MORSE_DASH_SECONDS)
                if symbol_index < len(symbols) - 1:
                    self._pause(MORSE_ELEMENT_GAP)
            if letter_index < len(letters) - 1:
                self._pause(MORSE_LETTER_GAP)

    def _pulse(self, duration: float) -> None:
        if self.stop_event.is_set():
            return
        try:
            GPIO.output(VIBRATION_PIN, GPIO.HIGH)
            GPIO.output(NAV_GUIDANCE_LED_PIN, GPIO.HIGH)
            self._pause(duration)
        finally:
            GPIO.output(VIBRATION_PIN, GPIO.LOW)
            GPIO.output(NAV_GUIDANCE_LED_PIN, GPIO.LOW)

    def _pause(self, duration: float) -> None:
        if duration <= 0:
            return
        deadline = time.monotonic() + duration
        while not self.stop_event.is_set():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            time.sleep(min(STOP_EVENT_CHECK_INTERVAL_SECONDS, remaining))


def setup_gpio() -> None:
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(VIBRATION_PIN, GPIO.OUT)
    GPIO.output(VIBRATION_PIN, GPIO.LOW)
    GPIO.setup(BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)
    GPIO.setup(BUTTON2_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)
    GPIO.setup(LED1_PIN, GPIO.OUT)
    GPIO.setup(LED2_PIN, GPIO.OUT)
    GPIO.setup(LED3_PIN, GPIO.OUT)
    GPIO.setup(OBSTACLE_MODE_LED_PIN, GPIO.OUT)
    GPIO.setup(NAV_GUIDANCE_LED_PIN, GPIO.OUT)
    GPIO.setup(TRIG_PIN, GPIO.OUT)
    GPIO.setup(ECHO_PIN, GPIO.IN)
    GPIO.output(TRIG_PIN, GPIO.LOW)
    GPIO.output(OBSTACLE_MODE_LED_PIN, GPIO.LOW)
    control_leds(heartbeat=False, processing=False, alert=False)


def cleanup_gpio() -> None:
    try:
        GPIO.output(VIBRATION_PIN, GPIO.LOW)
        GPIO.output(NAV_GUIDANCE_LED_PIN, GPIO.LOW)
        GPIO.output(OBSTACLE_MODE_LED_PIN, GPIO.LOW)
        GPIO.output(LED1_PIN, GPIO.LOW)
        GPIO.output(LED2_PIN, GPIO.LOW)
        GPIO.output(LED3_PIN, GPIO.LOW)
        GPIO.output(TRIG_PIN, GPIO.LOW)
    except Exception:
        pass
    try:
        GPIO.cleanup()
    except Exception:
        pass


def play_haptic(morse_worker: MorseWorker, pattern: str, prefix_key: Optional[str] = None) -> None:
    prefix_pattern = MODE_PREFIXES.get(prefix_key or "", "")
    combined = f"{prefix_pattern} {pattern}".strip() if prefix_pattern else pattern
    morse_worker.submit(combined)


def play_morse(morse_worker: MorseWorker, pattern: str, prefix_key: Optional[str] = None) -> None:
    """Primary Morse output helper required by the final system design."""
    play_haptic(morse_worker, pattern, prefix_key=prefix_key)


BRAILLE_MAP: Dict[str, str] = {
    "a": "⠁", "b": "⠃", "c": "⠉", "d": "⠙", "e": "⠑", "f": "⠋", "g": "⠛", "h": "⠓", "i": "⠊", "j": "⠚",
    "k": "⠅", "l": "⠇", "m": "⠍", "n": "⠝", "o": "⠕", "p": "⠏", "q": "⠟", "r": "⠗", "s": "⠎", "t": "⠞",
    "u": "⠥", "v": "⠧", "w": "⠺", "x": "⠭", "y": "⠽", "z": "⠵",
    "0": "⠚", "1": "⠁", "2": "⠃", "3": "⠉", "4": "⠙", "5": "⠑", "6": "⠋", "7": "⠛", "8": "⠓", "9": "⠊",
    " ": " ", ".": "⠲", ",": "⠂", "?": "⠦", "!": "⠖", "-": "⠤", ":": "⠒", ";": "⠆", "/": "⠌",
}
BRAILLE_NUMBER_PREFIX = "⠼"


def text_to_braille_script(text: str) -> str:
    cleaned = normalize_text(text).lower()
    output: List[str] = []
    for character in cleaned:
        if character.isdigit():
            output.append(BRAILLE_NUMBER_PREFIX + BRAILLE_MAP.get(character, ""))
            continue
        output.append(BRAILLE_MAP.get(character, "⠿"))
    return "".join(output).strip()


def navigation_pulse_interval_seconds(distance_cm: float) -> float:
    """Dynamic pulse frequency: closer object -> faster feedback."""
    if distance_cm > NAVIGATION_SAFE_DISTANCE_CM:
        return NAVIGATION_MAX_PULSE_INTERVAL_SECONDS
    minimum_distance = 10.0
    clamped = max(minimum_distance, min(NAVIGATION_SAFE_DISTANCE_CM, distance_cm))
    ratio = (clamped - minimum_distance) / (NAVIGATION_SAFE_DISTANCE_CM - minimum_distance)
    interval = NAVIGATION_MIN_PULSE_INTERVAL_SECONDS + (
        (NAVIGATION_MAX_PULSE_INTERVAL_SECONDS - NAVIGATION_MIN_PULSE_INTERVAL_SECONDS) * ratio
    )
    return max(NAVIGATION_MIN_PULSE_INTERVAL_SECONDS, min(NAVIGATION_MAX_PULSE_INTERVAL_SECONDS, interval))


def handle_room_description_mode(morse_worker: MorseWorker) -> None:
    log("Button 2 short press: room description mode")
    set_person_alert_mode(False)
    control_leds(processing=True)
    image_path = capture_image("room_desc")
    if image_path is None:
        play_morse(morse_worker, "...", prefix_key="room_scan")
        control_leds(processing=False)
        return

    try:
        prompt = "Describe the room in 3-5 short words for navigation"
        room_text = call_gemini_vision(image_path, prompt) or "UNKNOWN AREA"
        short_room_text = sanitize_room_output(room_text)
        morse_pattern = call_gemini_flash(short_room_text) or text_to_morse_local(short_room_text)
        play_morse(morse_worker, morse_pattern, prefix_key="room_scan")
    finally:
        control_leds(processing=False)


def handle_human_presence_mode(morse_worker: MorseWorker) -> None:
    log("Button 2 medium press: human presence detection")
    set_person_alert_mode(False)
    control_leds(processing=True)
    image_path = capture_image("human_presence")
    if image_path is None:
        play_morse(morse_worker, "...", prefix_key="detection")
        control_leds(processing=False)
        return

    try:
        prompt = "Answer with only YES or NO. Is a human present in this image?"
        response_text = call_gemini_vision(image_path, prompt)
        detection_result = sanitize_detection_output(response_text or "")
        if detection_result not in {"YES", "NO"}:
            detection_result = "NO"
        morse_pattern = call_gemini_flash(detection_result) or text_to_morse_local(detection_result)
        play_morse(morse_worker, morse_pattern, prefix_key="detection")
    finally:
        control_leds(processing=False)


def handle_sign_language_mode(morse_worker: MorseWorker) -> None:
    log("Button 1 long press: sign language mode")
    set_person_alert_mode(False)
    control_leds(processing=True)
    image_path = capture_image("sign_language")
    if image_path is None:
        play_morse(morse_worker, "...", prefix_key=None)
        control_leds(processing=False)
        return

    try:
        prompt = (
            "Recognize sign language from this image. "
            "Auto-correct the recognized text and return only the corrected plain text."
        )
        recognized_text = call_gemini_vision(
            image_path,
            prompt,
            endpoint=SIGN_LANGUAGE_ENDPOINT,
            api_key=SIGN_LANGUAGE_API_KEY,
            timeout_seconds=SIGN_LANGUAGE_TIMEOUT_SECONDS,
            retries=SIGN_LANGUAGE_RETRIES,
            max_output_tokens=SIGN_LANGUAGE_MAX_OUTPUT_TOKENS,
        )
        cleaned_text = normalize_text(recognized_text or "NO SIGN DETECTED")
        braille_text = text_to_braille_script(cleaned_text)
        log(f"Sign language text: {cleaned_text}")
        log(f"Braille output: {braille_text}")
        confirmation_morse = call_gemini_flash(cleaned_text) or text_to_morse_local(cleaned_text)
        play_morse(morse_worker, confirmation_morse)
    finally:
        control_leds(processing=False)


def handle_sos(morse_worker: MorseWorker) -> None:
    log("Button 2 long press: SOS mode")
    set_person_alert_mode(False)
    control_leds(processing=True)
    try:
        post_sos_alert()
        play_morse(morse_worker, "... --- ...", prefix_key="sos")
    finally:
        control_leds(processing=False)


def fetch_server_message() -> Optional[str]:
    messages = poll_caretaker_messages()
    if messages:
        return messages[0]
    return None


def handle_message_check(morse_worker: MorseWorker) -> None:
    log("Button 1 medium press: incoming message check")
    control_leds(processing=True)
    try:
        message = fetch_server_message()
        if not message:
            log("No message available from server.")
            return

        morse_pattern = call_gemini_flash(message)
        if not morse_pattern:
            morse_pattern = text_to_morse_local(message)
        play_morse(morse_worker, morse_pattern)
    finally:
        control_leds(processing=False)


def handle_caretaker_message(morse_worker: MorseWorker, message: str) -> None:
    short_message = shorten_for_morse(message)
    if not short_message:
        return
    log(f"Caretaker message received: {short_message}")
    morse_pattern = call_gemini_flash(short_message)
    if not morse_pattern:
        morse_pattern = text_to_morse_local(short_message)
    morse_worker.submit(morse_pattern)


def handle_detection(morse_worker: MorseWorker) -> None:
    """Backward-compatible wrapper."""
    handle_human_presence_mode(morse_worker)


def handle_room_scan(morse_worker: MorseWorker) -> None:
    """Backward-compatible wrapper."""
    handle_room_description_mode(morse_worker)


def handle_person_priority_room_scan(morse_worker: MorseWorker) -> None:
    """Backward-compatible wrapper retained for older integrations."""
    handle_room_description_mode(morse_worker)


def handle_button1(press_type: str, morse_worker: MorseWorker) -> None:
    global NAVIGATION_MODE_ENABLED
    if press_type == "short":
        NAVIGATION_MODE_ENABLED = not NAVIGATION_MODE_ENABLED
        set_obstacle_mode_led(NAVIGATION_MODE_ENABLED)
        status_word = "ON" if NAVIGATION_MODE_ENABLED else "OFF"
        log(f"Button 1 short press: navigation mode {status_word}")
        play_morse(morse_worker, ".-" if NAVIGATION_MODE_ENABLED else "-...")
        return
    if press_type == "medium":
        handle_message_check(morse_worker)
        return
    if press_type == "long":
        handle_sign_language_mode(morse_worker)


def handle_button2(press_type: str, morse_worker: MorseWorker) -> None:
    if press_type == "short":
        handle_room_description_mode(morse_worker)
        return
    if press_type == "medium":
        handle_human_presence_mode(morse_worker)
        return
    if press_type == "long":
        handle_sos(morse_worker)


def button_is_pressed() -> bool:
    return GPIO.input(BUTTON_PIN) == GPIO.LOW


def button2_is_pressed() -> bool:
    return GPIO.input(BUTTON2_PIN) == GPIO.LOW


def handle_button_input(
    now: float,
    pressed: bool,
    state: ButtonState,
    short_threshold: float,
    medium_threshold: float,
    long_threshold: float,
    short_action: Optional[Any] = None,
    medium_action: Optional[Any] = None,
    long_action: Optional[Any] = None,
) -> None:
    if pressed != state.last_raw_state:
        state.last_raw_state = pressed
        state.last_change_at = now

    if pressed != state.stable_state and (now - state.last_change_at) >= BUTTON_DEBOUNCE_SECONDS:
        state.stable_state = pressed
        if state.stable_state:
            state.press_started_at = now
        elif state.press_started_at is not None:
            press_duration = now - state.press_started_at
            state.press_started_at = None
            if press_duration < short_threshold:
                if short_action:
                    short_action()
            elif press_duration < medium_threshold:
                if medium_action:
                    medium_action()
            else:
                if press_duration < long_threshold:
                    if medium_action:
                        medium_action()
                    elif short_action:
                        short_action()
                elif long_action:
                    long_action()


def send_obstacle_warning(morse_worker: MorseWorker) -> None:
    control_leds(alert=True)
    play_haptic(morse_worker, "..", prefix_key=None)


def handle_obstacle_awareness_toggle(morse_worker: MorseWorker, currently_enabled: bool) -> bool:
    next_enabled = not currently_enabled
    status_text = "enabled" if next_enabled else "disabled"
    log(f"Button 2 long press: ultrasonic obstacle awareness {status_text}")
    set_obstacle_mode_led(next_enabled)

    if next_enabled:
        play_haptic(morse_worker, ".-", prefix_key=None)
    else:
        if not PERSON_ALERT_ACTIVE:
            control_leds(alert=False)
        play_haptic(morse_worker, "-...", prefix_key=None)
    return next_enabled


def main() -> None:
    global NAVIGATION_MODE_ENABLED
    setup_gpio()
    morse_worker = MorseWorker()
    morse_worker.start()

    stop_event = threading.Event()

    def _stop_handler(_signum: int, _frame: Any) -> None:
        stop_event.set()

    signal.signal(signal.SIGINT, _stop_handler)
    signal.signal(signal.SIGTERM, _stop_handler)

    log("System started.")
    log(f"GPIO backend: {GPIO.mode}")
    if not GEMINI_API_KEY:
        log("GEMINI_API_KEY is not configured; Gemini calls will use local fallbacks.")
    if not API_CONFIG_PATH.exists():
        log(f"Warning: API config file not found at {API_CONFIG_PATH}; using defaults/env.")

    last_caretaker_poll = 0.0
    last_distance_poll = 0.0
    last_heartbeat_toggle = 0.0
    heartbeat_on = False
    obstacle_awareness_enabled = NAVIGATION_MODE_ENABLED
    obstacle_active = False
    last_obstacle_alert_at = 0.0

    button1_state = ButtonState()
    button2_state = ButtonState()

    set_obstacle_mode_led(obstacle_awareness_enabled)
    log(f"Navigation mode: {'enabled' if obstacle_awareness_enabled else 'disabled'}")

    def _button1_short_press() -> None:
        nonlocal obstacle_awareness_enabled
        handle_button1("short", morse_worker)
        obstacle_awareness_enabled = NAVIGATION_MODE_ENABLED

    def _button1_medium_press() -> None:
        handle_button1("medium", morse_worker)

    def _button1_long_press() -> None:
        handle_button1("long", morse_worker)

    def _button2_short_press() -> None:
        handle_button2("short", morse_worker)

    def _button2_medium_press() -> None:
        handle_button2("medium", morse_worker)

    def _button2_long_press() -> None:
        handle_button2("long", morse_worker)

    try:
        while not stop_event.is_set():
            now = time.monotonic()

            if (now - last_heartbeat_toggle) >= 1.0:
                heartbeat_on = not heartbeat_on
                control_leds(heartbeat=heartbeat_on)
                last_heartbeat_toggle = now

            if ENABLE_CARETAKER_POLL and CARETAKER_API_URL and (now - last_caretaker_poll) >= CARETAKER_POLL_SECONDS:
                for caretaker_message in poll_caretaker_messages():
                    handle_caretaker_message(morse_worker, caretaker_message)
                last_caretaker_poll = now

            if obstacle_awareness_enabled and (now - last_distance_poll) >= DISTANCE_POLL_SECONDS:
                distance_cm = measure_distance()
                if distance_cm is not None and distance_cm <= NAVIGATION_SAFE_DISTANCE_CM:
                    obstacle_active = True
                    pulse_interval = navigation_pulse_interval_seconds(distance_cm)
                    if (now - last_obstacle_alert_at) >= pulse_interval:
                        if morse_worker.queue.qsize() <= 2:
                            play_morse(morse_worker, ".")
                        last_obstacle_alert_at = now
                else:
                    obstacle_active = False
                last_distance_poll = now
            elif not obstacle_awareness_enabled and obstacle_active:
                obstacle_active = False

            handle_button_input(
                now,
                button_is_pressed(),
                button1_state,
                SHORT_PRESS_SECONDS,
                MEDIUM_PRESS_SECONDS,
                LONG_PRESS_SECONDS,
                short_action=_button1_short_press,
                medium_action=_button1_medium_press,
                long_action=_button1_long_press,
            )

            handle_button_input(
                now,
                button2_is_pressed(),
                button2_state,
                SHORT_PRESS_SECONDS,
                MEDIUM_PRESS_SECONDS,
                LONG_PRESS_SECONDS,
                short_action=_button2_short_press,
                medium_action=_button2_medium_press,
                long_action=_button2_long_press,
            )

            person_alert_led = update_person_alert_led(now)
            should_light_alert = obstacle_active or person_alert_led
            control_leds(alert=should_light_alert)

            time.sleep(BUTTON_POLL_SECONDS)
    finally:
        stop_event.set()
        morse_worker.stop()
        cleanup_gpio()
        log("System stopped.")


if __name__ == "__main__":
    main()
