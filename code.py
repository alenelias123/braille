#!/usr/bin/env python3
"""Assistive system for deafblind users on Raspberry Pi.

Features:
- Button 1 short press: SOS alert.
- Button 1 medium press: human presence detection using local Gemma.
- Button 1 long press: room scan using Gemini vision.
- Button 2 press: fetch a server message and convert it to Morse with Gemini Flash.
- Ultrasonic obstacle awareness and LED status indicators.

The script is designed to keep working even when some hardware or APIs are
unavailable. It logs failures, retries network calls once, and keeps the main
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
import subprocess
import tempfile
import threading
import time
import unicodedata
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from urllib import parse as urllib_parse
from urllib import request as urllib_request


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

DISTANCE_POLL_SECONDS = float(os.getenv("DISTANCE_POLL_SECONDS", "0.20"))
OBSTACLE_THRESHOLD_CM = float(os.getenv("OBSTACLE_THRESHOLD_CM", "100"))
OBSTACLE_ALERT_COOLDOWN_SECONDS = float(os.getenv("OBSTACLE_ALERT_COOLDOWN_SECONDS", "2.50"))

MORSE_DOT_SECONDS = float(os.getenv("MORSE_DOT_SECONDS", "0.20"))
MORSE_DASH_SECONDS = float(os.getenv("MORSE_DASH_SECONDS", "0.60"))
MORSE_ELEMENT_GAP = float(os.getenv("MORSE_ELEMENT_GAP", "0.15"))
MORSE_LETTER_GAP = float(os.getenv("MORSE_LETTER_GAP", "0.45"))
MORSE_WORD_GAP = float(os.getenv("MORSE_WORD_GAP", "0.90"))

MAX_MORSE_CHARS = int(os.getenv("MAX_MORSE_CHARS", "18"))
CARETAKER_POLL_SECONDS = float(os.getenv("CARETAKER_POLL_SECONDS", "10"))
REQUEST_TIMEOUT_SECONDS = float(os.getenv("REQUEST_TIMEOUT_SECONDS", "10"))
REQUEST_RETRIES = 1

IMAGE_DIRECTORY = Path(os.getenv("IMAGE_DIRECTORY", str(Path(tempfile.gettempdir()) / "assistive_images")))

CARETAKER_API_URL = os.getenv("CAREGIVER_API_URL", "").strip()
CARETAKER_API_TOKEN = os.getenv("CAREGIVER_API_TOKEN", "").strip()
ENABLE_CARETAKER_POLL = os.getenv("ENABLE_CARETAKER_POLL", "0").strip().lower() in {"1", "true", "yes", "on"}
SOS_API_URL = os.getenv("SOS_API_URL", "").strip()
SOS_API_TOKEN = os.getenv("SOS_API_TOKEN", "").strip()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyDeer2H_t1KwQW9vbBZX1ufBDVqeGgGmKo").strip()
GEMINI_FLASH_MODEL = os.getenv("GEMINI_FLASH_MODEL", "gemini-2.0-flash").strip()
GEMINI_VISION_MODEL = os.getenv("GEMINI_VISION_MODEL", "gemini-2.5-pro").strip()
GEMINI_API_BASE = os.getenv("GEMINI_API_BASE", "https://generativelanguage.googleapis.com/v1beta").strip()

GEMMA_LOCAL_URL = os.getenv("GEMMA_LOCAL_URL", "http://localhost:11434/api/chat").strip()
GEMMA_MODEL = os.getenv("GEMMA_MODEL", "gemma3").strip()

LED_HEARTBEAT_STATE = False
LED_PROCESSING_STATE = False
LED_ALERT_STATE = False


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


def log(message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


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
        for key in ("messages", "data", "choices", "parts", "candidates"):
            value = payload.get(key)
            if value:
                text = extract_text_from_payload(value)
                if text:
                    return text
        return ""
    return str(payload).strip()


def request_json(
    method: str,
    url: str,
    payload: Optional[Dict[str, Any]] = None,
    headers: Optional[Dict[str, str]] = None,
    timeout: float = REQUEST_TIMEOUT_SECONDS,
    retries: int = REQUEST_RETRIES,
) -> Any:
    request_headers = {
        "Content-Type": "application/json",
        "User-Agent": f"{APP_NAME}/1.0",
    }
    if headers:
        request_headers.update(headers)

    body = None
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")

    last_error: Optional[Exception] = None
    attempt_count = retries + 1
    for attempt in range(1, attempt_count + 1):
        try:
            request = urllib_request.Request(url, data=body, headers=request_headers, method=method.upper())
            with urllib_request.urlopen(request, timeout=timeout) as response:
                response_text = response.read().decode("utf-8", errors="replace")
                if not response_text.strip():
                    return {}
                return json.loads(response_text)
        except Exception as exc:
            last_error = exc
            log(f"Request attempt {attempt}/{attempt_count} failed for {url}: {exc}")
            if attempt < attempt_count:
                time.sleep(0.25)

    if last_error is not None:
        raise last_error
    raise RuntimeError(f"Request failed for {url}")


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


def call_gemini_morse(text: str) -> Optional[str]:
    if not GEMINI_API_KEY:
        log("GEMINI_API_KEY is not configured for Morse conversion.")
        return sanitize_morse_output(text_to_morse_local(text)) or None

    cleaned_text = clamp_text_for_prompt(text, max_chars=20)
    prompt = f"Convert the following text into Morse code using dots and dashes only. Keep it short: {cleaned_text}"
    endpoint = (
        f"{GEMINI_API_BASE}/models/{urllib_parse.quote(GEMINI_FLASH_MODEL, safe='')}:generateContent"
        f"?key={urllib_parse.quote(GEMINI_API_KEY, safe='')}"
    )
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": prompt}],
            }
        ],
        "generationConfig": {
            "temperature": 0.0,
            "maxOutputTokens": 64,
        },
    }

    try:
        response_json = request_json("POST", endpoint, payload=payload, timeout=REQUEST_TIMEOUT_SECONDS)
        candidates = response_json.get("candidates", []) if isinstance(response_json, dict) else []
        for candidate in candidates:
            content = candidate.get("content", {}) if isinstance(candidate, dict) else {}
            parts = content.get("parts", []) if isinstance(content, dict) else []
            text_parts: List[str] = []
            for part in parts:
                if isinstance(part, dict):
                    value = part.get("text")
                    if isinstance(value, str) and value.strip():
                        text_parts.append(value.strip())
            morse_text = sanitize_morse_output(" ".join(text_parts))
            if morse_text:
                return morse_text
        fallback = sanitize_morse_output(extract_text_from_payload(response_json))
        if fallback:
            return fallback
    except Exception as exc:
        log(f"Gemini Flash Morse conversion failed: {exc}")

    return sanitize_morse_output(text_to_morse_local(cleaned_text)) or None


def call_gemini_api(image_path: Path, prompt: str, model: str = GEMINI_VISION_MODEL) -> Optional[str]:
    if not GEMINI_API_KEY:
        log("GEMINI_API_KEY is not configured.")
        return None

    try:
        image_base64 = encode_image_base64(image_path)
        endpoint = (
            f"{GEMINI_API_BASE}/models/{urllib_parse.quote(model, safe='')}:generateContent"
            f"?key={urllib_parse.quote(GEMINI_API_KEY, safe='')}"
        )
        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"text": prompt},
                        {
                            "inline_data": {
                                "mime_type": "image/jpeg",
                                "data": image_base64,
                            }
                        },
                    ],
                }
            ],
            "generationConfig": {
                "temperature": 0.2,
                "maxOutputTokens": 64,
            },
        }
        response_json = request_json("POST", endpoint, payload=payload, timeout=REQUEST_TIMEOUT_SECONDS)
        candidates = response_json.get("candidates", []) if isinstance(response_json, dict) else []
        for candidate in candidates:
            content = candidate.get("content", {}) if isinstance(candidate, dict) else {}
            parts = content.get("parts", []) if isinstance(content, dict) else []
            text_parts: List[str] = []
            for part in parts:
                if isinstance(part, dict):
                    text_value = part.get("text")
                    if isinstance(text_value, str) and text_value.strip():
                        text_parts.append(text_value.strip())
            if text_parts:
                return " ".join(text_parts).strip()
        fallback_text = extract_text_from_payload(response_json)
        return fallback_text or None
    except Exception as exc:
        log(f"Gemini vision call failed: {exc}")
        return None


def detect_human_gemma(image_path: Path) -> Optional[str]:
    image_base64 = encode_image_base64(image_path)
    prompt = "Answer with only YES or NO. Is a human present in this image?"
    payload = {
        "model": GEMMA_MODEL,
        "messages": [
            {
                "role": "user",
                "content": prompt,
                "images": [image_base64],
            }
        ],
        "stream": False,
    }

    try:
        response_json = request_json("POST", GEMMA_LOCAL_URL, payload=payload, timeout=REQUEST_TIMEOUT_SECONDS)
        response_text = extract_text_from_payload(response_json)
        response_text = normalize_text(response_text).upper()
        if "YES" in response_text:
            return "YES"
        if "NO" in response_text:
            return "NO"
        if any(token in response_text for token in ("TRUE", "PRESENT", "PERSON", "HUMAN")):
            return "YES"
        if any(token in response_text for token in ("FALSE", "ABSENT", "NONE")):
            return "NO"
        return None
    except Exception as exc:
        log(f"Gemma local detection failed: {exc}")
        return None


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


def encode_image_base64(image_path: Path) -> str:
    with image_path.open("rb") as image_file:
        encoded_bytes = base64.b64encode(image_file.read())
    return encoded_bytes.decode("ascii")


def capture_image(prefix: str) -> Optional[Path]:
    IMAGE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    image_path = IMAGE_DIRECTORY / f"{prefix}_{timestamp}.jpg"

    try:
        if Picamera2 is not None:
            camera = Picamera2()
            try:
                config = camera.create_still_configuration()
                camera.configure(config)
                camera.start()
                time.sleep(1.0)
                camera.capture_file(str(image_path))
                log(f"Captured image with Picamera2: {image_path}")
                return image_path
            finally:
                try:
                    camera.stop()
                except Exception:
                    pass
                try:
                    camera.close()
                except Exception:
                    pass

        if shutil.which("libcamera-still"):
            command = [
                "libcamera-still",
                "-n",
                "-t",
                "1000",
                "-o",
                str(image_path),
            ]
            subprocess.run(command, check=True, timeout=25, capture_output=True)
            log(f"Captured image with libcamera-still: {image_path}")
            return image_path

        log("No camera backend available.")
        return None
    except Exception as exc:
        log(f"Image capture failed: {exc}")
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


def post_sos_alert() -> bool:
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
        request_json("POST", SOS_API_URL, payload=payload, headers=headers, timeout=REQUEST_TIMEOUT_SECONDS)
        log("SOS alert sent successfully.")
        return True
    except Exception as exc:
        log(f"SOS alert failed: {exc}")
        return False


def poll_caretaker_messages() -> List[str]:
    if not CARETAKER_API_URL:
        return []

    headers = {}
    if CARETAKER_API_TOKEN:
        headers["Authorization"] = f"Bearer {CARETAKER_API_TOKEN}"

    try:
        payload = request_json("GET", CARETAKER_API_URL, headers=headers, timeout=REQUEST_TIMEOUT_SECONDS)
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
        self.thread.join(timeout=2.0)

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
            symbols = [symbol for symbol in letter if symbol in ".-"]
            for symbol_index, symbol in enumerate(symbols):
                self._pulse(MORSE_DOT_SECONDS if symbol == "." else MORSE_DASH_SECONDS)
                if symbol_index < len(symbols) - 1:
                    self._pause(MORSE_ELEMENT_GAP)
            if letter_index < len(letters) - 1:
                self._pause(MORSE_LETTER_GAP)

    def _pulse(self, duration: float) -> None:
        GPIO.output(VIBRATION_PIN, GPIO.HIGH)
        time.sleep(duration)
        GPIO.output(VIBRATION_PIN, GPIO.LOW)

    def _pause(self, duration: float) -> None:
        time.sleep(duration)


def setup_gpio() -> None:
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(VIBRATION_PIN, GPIO.OUT)
    GPIO.output(VIBRATION_PIN, GPIO.LOW)
    GPIO.setup(BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)
    GPIO.setup(BUTTON2_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)
    GPIO.setup(LED1_PIN, GPIO.OUT)
    GPIO.setup(LED2_PIN, GPIO.OUT)
    GPIO.setup(LED3_PIN, GPIO.OUT)
    GPIO.setup(TRIG_PIN, GPIO.OUT)
    GPIO.setup(ECHO_PIN, GPIO.IN)
    GPIO.output(TRIG_PIN, GPIO.LOW)
    control_leds(heartbeat=False, processing=False, alert=False)


def cleanup_gpio() -> None:
    try:
        GPIO.output(VIBRATION_PIN, GPIO.LOW)
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


def handle_sos(morse_worker: MorseWorker) -> None:
    log("Button 1 short press: SOS")
    control_leds(processing=True)
    try:
        post_sos_alert()
        play_haptic(morse_worker, "", prefix_key="sos")
    finally:
        control_leds(processing=False)


def handle_detection(morse_worker: MorseWorker) -> None:
    log("Button 1 medium press: human presence check")
    control_leds(processing=True)
    image_path = capture_image("detection")
    if image_path is None:
        play_haptic(morse_worker, "...", prefix_key="detection")
        control_leds(processing=False)
        return

    try:
        detection_result = detect_human_gemma(image_path)
        if not detection_result:
            detection_result = "NO"

        morse_pattern = call_gemini_morse(detection_result)
        if not morse_pattern:
            morse_pattern = text_to_morse_local(detection_result)
        play_haptic(morse_worker, morse_pattern, prefix_key="detection")
    finally:
        control_leds(processing=False)


def handle_room_scan(morse_worker: MorseWorker) -> None:
    log("Button 1 long press: room scan")
    control_leds(processing=True)
    image_path = capture_image("roomscan")
    if image_path is None:
        play_haptic(morse_worker, "...", prefix_key="room_scan")
        control_leds(processing=False)
        return

    try:
        prompt = "Describe the surroundings in 3 to 5 short words for navigation."
        response_text = call_gemini_api(image_path, prompt, model=GEMINI_VISION_MODEL)
        if not response_text:
            response_text = "UNKNOWN AREA"

        room_summary = sanitize_room_output(response_text)
        morse_pattern = call_gemini_morse(room_summary)
        if not morse_pattern:
            morse_pattern = text_to_morse_local(room_summary)
        play_haptic(morse_worker, morse_pattern, prefix_key="room_scan")
    finally:
        control_leds(processing=False)


def fetch_server_message() -> Optional[str]:
    messages = poll_caretaker_messages()
    if messages:
        return messages[0]
    return None


def handle_message_check(morse_worker: MorseWorker) -> None:
    log("Button 2 press: manual message check")
    control_leds(processing=True)
    try:
        message = fetch_server_message()
        if not message:
            log("No message available from server.")
            return

        morse_pattern = call_gemini_morse(message)
        if not morse_pattern:
            morse_pattern = text_to_morse_local(message)
        play_haptic(morse_worker, morse_pattern)
    finally:
        control_leds(processing=False)


def handle_caretaker_message(morse_worker: MorseWorker, message: str) -> None:
    short_message = shorten_for_morse(message)
    if not short_message:
        return
    log(f"Caretaker message received: {short_message}")
    morse_pattern = call_gemini_morse(short_message)
    if not morse_pattern:
        morse_pattern = text_to_morse_local(short_message)
    morse_worker.submit(morse_pattern)


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


def main() -> None:
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
        log("GEMINI_API_KEY is not configured; AI modes will only log failures.")

    last_caretaker_poll = 0.0
    last_distance_poll = 0.0
    last_heartbeat_toggle = 0.0
    heartbeat_on = False
    obstacle_active = False
    last_obstacle_alert_at = 0.0

    button1_state = ButtonState()
    button2_state = ButtonState()

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

            if (now - last_distance_poll) >= DISTANCE_POLL_SECONDS:
                distance_cm = measure_distance()
                if distance_cm is not None and distance_cm < OBSTACLE_THRESHOLD_CM:
                    obstacle_active = True
                    control_leds(alert=True)
                    if (now - last_obstacle_alert_at) >= OBSTACLE_ALERT_COOLDOWN_SECONDS:
                        send_obstacle_warning(morse_worker)
                        last_obstacle_alert_at = now
                else:
                    obstacle_active = False
                    control_leds(alert=False)
                last_distance_poll = now

            handle_button_input(
                now,
                button_is_pressed(),
                button1_state,
                SHORT_PRESS_SECONDS,
                MEDIUM_PRESS_SECONDS,
                LONG_PRESS_SECONDS,
                short_action=lambda: handle_sos(morse_worker),
                medium_action=lambda: handle_detection(morse_worker),
                long_action=lambda: handle_room_scan(morse_worker),
            )

            handle_button_input(
                now,
                button2_is_pressed(),
                button2_state,
                SHORT_PRESS_SECONDS,
                MEDIUM_PRESS_SECONDS,
                LONG_PRESS_SECONDS,
                short_action=lambda: handle_message_check(morse_worker),
            )

            if not obstacle_active and not LED_ALERT_STATE:
                control_leds(alert=False)

            time.sleep(BUTTON_POLL_SECONDS)
    finally:
        stop_event.set()
        morse_worker.stop()
        cleanup_gpio()
        log("System stopped.")


if __name__ == "__main__":
    main()
