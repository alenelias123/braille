#!/usr/bin/env python3
"""Standalone API tester for Gemini endpoint.

Examples:
  python3 api_test.py --mode text --prompt "Say hello in 5 words"
  python3 api_test.py --mode vision --use-sample-image --prompt "Describe the image"
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import ssl
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional
from urllib import error as urllib_error
from urllib import parse as urllib_parse
from urllib import request as urllib_request


DEFAULT_TIMEOUT_SECONDS = float(os.getenv("REQUEST_TIMEOUT_SECONDS", "10"))
DEFAULT_RETRIES = max(0, int(os.getenv("REQUEST_RETRIES", "1")))
ALLOW_INSECURE_SSL = os.getenv("GEMINI_ALLOW_INSECURE_SSL", "0").strip().lower() in {"1", "true", "yes", "on"}
CA_BUNDLE_OVERRIDE = os.getenv("GEMINI_CA_BUNDLE", "").strip()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AQ.Ab8RN6L5hO1mQdeb_vrzcfOPQPOC91BkEcc2YVkWM8AIjbcMCA").strip()
GEMINI_API_BASE = os.getenv("GEMINI_API_BASE", "https://generativelanguage.googleapis.com/v1beta").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-flash-latest").strip()

SAMPLE_PNG_BASE64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO2j5ZkAAAAASUVORK5CYII="
)

try:
    import certifi  # type: ignore
except Exception:
    certifi = None  # type: ignore


def log(message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def parse_retry_after_seconds(value: Optional[str]) -> Optional[float]:
    if not value:
        return None
    try:
        return max(0.0, float(value.strip()))
    except Exception:
        return None


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
            return min(5.0, retry_after)
    return min(5.0, 0.35 * (2 ** max(0, attempt - 1)))


def is_ssl_verification_error(exc: Exception) -> bool:
    if isinstance(exc, ssl.SSLCertVerificationError):
        return True
    if isinstance(exc, urllib_error.URLError) and isinstance(exc.reason, ssl.SSLCertVerificationError):
        return True
    return "certificate verify failed" in str(exc).lower()


def collect_ca_bundle_candidates(cli_override: Optional[str]) -> list[str]:
    candidates: list[str] = []

    if cli_override and cli_override.strip():
        candidates.append(cli_override.strip())
    if CA_BUNDLE_OVERRIDE:
        candidates.append(CA_BUNDLE_OVERRIDE)

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

    # Common Windows bundle locations (Python/MSYS/Git for Windows).
    candidates.extend(
        [
            r"C:\Program Files\Git\usr\ssl\certs\ca-bundle.crt",
            r"C:\Program Files\Git\mingw64\ssl\certs\ca-bundle.crt",
            r"C:\msys64\ucrt64\etc\ssl\cert.pem",
            r"C:\msys64\mingw64\etc\ssl\cert.pem",
            r"C:\msys64\usr\ssl\cert.pem",
        ]
    )

    unique_candidates: list[str] = []
    seen: set[str] = set()
    for path in candidates:
        normalized = str(Path(path).expanduser())
        if normalized not in seen:
            seen.add(normalized)
            unique_candidates.append(normalized)
    return unique_candidates


def build_ssl_context(cli_ca_bundle: Optional[str]) -> ssl.SSLContext:
    """Builds an SSL context with explicit CA bundle support.

    Priority:
    1) CLI and env configured CA bundle path
    2) Python default CA file, certifi, and known Windows CA bundle locations
    3) Python default trust store context
    """
    context = ssl.create_default_context()
    for candidate in collect_ca_bundle_candidates(cli_ca_bundle):
        candidate_path = Path(candidate)
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
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    retries: int = DEFAULT_RETRIES,
    ca_bundle: Optional[str] = None,
) -> Any:
    headers = {
        "Content-Type": "application/json",
        "User-Agent": "assistive-api-test/1.0",
    }
    body = json.dumps(payload).encode("utf-8") if payload is not None else None
    ssl_context = build_ssl_context(ca_bundle)

    last_error: Optional[Exception] = None
    attempt_count = max(1, retries + 1)
    for attempt in range(1, attempt_count + 1):
        try:
            request = urllib_request.Request(url, data=body, headers=headers, method=method.upper())
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
                    log(f"Insecure SSL retry failed: {insecure_exc}")

            last_error = failure
            log(f"Request attempt {attempt}/{attempt_count} failed: {failure}")
            if attempt < attempt_count and should_retry_request_error(failure):
                time.sleep(retry_delay_seconds(failure, attempt))
                continue
            break

    if last_error is not None:
        raise last_error
    raise RuntimeError(f"Request failed for URL: {url}")


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


def guess_mime_type(path: Path) -> str:
    extension = path.suffix.lower()
    if extension in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if extension == ".png":
        return "image/png"
    if extension == ".webp":
        return "image/webp"
    return "application/octet-stream"


def encode_image_base64(image_path: Path) -> str:
    with image_path.open("rb") as image_file:
        return base64.b64encode(image_file.read()).decode("ascii")


def create_sample_image() -> Path:
    sample_bytes = base64.b64decode(SAMPLE_PNG_BASE64)
    target = Path(tempfile.gettempdir()) / "assistive_api_test_sample.png"
    target.write_bytes(sample_bytes)
    return target


def resolve_image_path(image_path_arg: Optional[str], use_sample_image: bool) -> Optional[Path]:
    if image_path_arg:
        image_path = Path(image_path_arg).expanduser().resolve()
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")
        return image_path
    if use_sample_image:
        sample_path = create_sample_image()
        log(f"Using generated sample image: {sample_path}")
        return sample_path
    return None


def call_gemini(
    prompt: str,
    model: str,
    timeout: float,
    retries: int,
    image_path: Optional[Path],
    ca_bundle: Optional[str],
) -> Dict[str, Any]:
    endpoint = (
        f"{GEMINI_API_BASE}/models/{urllib_parse.quote(model, safe='')}:generateContent"
        f"?key={urllib_parse.quote(GEMINI_API_KEY, safe='')}"
    )

    parts: list[Dict[str, Any]] = [{"text": prompt}]
    if image_path is not None:
        parts.append(
            {
                "inline_data": {
                    "mime_type": guess_mime_type(image_path),
                    "data": encode_image_base64(image_path),
                }
            }
        )

    payload: Dict[str, Any] = {
        "contents": [{"role": "user", "parts": parts}],
        "generationConfig": {
            "temperature": 0.0,
            "maxOutputTokens": 128,
        },
    }
    return request_json("POST", endpoint, payload=payload, timeout=timeout, retries=retries, ca_bundle=ca_bundle)


def normalize_output_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def run_gemini(
    prompt: str,
    timeout: float,
    retries: int,
    image_path: Optional[Path],
    raw: bool,
    ca_bundle: Optional[str],
) -> int:
    if not GEMINI_API_KEY:
        log("GEMINI_API_KEY is not configured. Set it in environment before testing Gemini.")
        return 1

    log(f"Testing Gemini model: {GEMINI_MODEL}")
    if image_path:
        log(f"Gemini image input: {image_path}")

    try:
        response_json = call_gemini(
            prompt=prompt,
            model=GEMINI_MODEL,
            timeout=timeout,
            retries=retries,
            image_path=image_path,
            ca_bundle=ca_bundle,
        )
    except Exception as exc:
        if is_ssl_verification_error(exc):
            log(
                "TLS certificate verification failed. "
                "Try: --ca-bundle \"C:\\Program Files\\Git\\usr\\ssl\\certs\\ca-bundle.crt\""
            )
        log(f"Gemini test failed: {exc}")
        return 1

    text_output = normalize_output_text(extract_text_from_payload(response_json))
    print("\n=== Gemini Result ===")
    print(text_output or "(no text returned)")
    if raw:
        print("\n=== Gemini Raw JSON ===")
        print(json.dumps(response_json, indent=2, ensure_ascii=False))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test Gemini API with prompt or image.")
    parser.add_argument(
        "--mode",
        choices=["text", "vision"],
        default="text",
        help="Vision mode sends image + prompt; text mode sends only prompt.",
    )
    parser.add_argument("--prompt", default="Describe this briefly.", help="Prompt text to send.")
    parser.add_argument("--image", default=None, help="Path to image file.")
    parser.add_argument(
        "--use-sample-image",
        action="store_true",
        help="Generate and use a tiny sample PNG when mode=vision and --image is not set.",
    )
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_SECONDS, help="Request timeout in seconds.")
    parser.add_argument("--retries", type=int, default=DEFAULT_RETRIES, help="Retry count.")
    parser.add_argument("--ca-bundle", default=None, help="Path to CA bundle PEM/CRT file for TLS verification.")
    parser.add_argument(
        "--insecure",
        action="store_true",
        help="Disable TLS certificate verification for debugging only.",
    )
    parser.add_argument("--raw", action="store_true", help="Print raw JSON response.")
    return parser.parse_args()


def main() -> int:
    global ALLOW_INSECURE_SSL
    args = parse_args()
    retries = max(0, int(args.retries))
    if args.insecure:
        ALLOW_INSECURE_SSL = True
        log("Insecure SSL mode is enabled. Use only for testing.")

    image_path: Optional[Path] = None
    if args.mode == "vision":
        try:
            image_path = resolve_image_path(args.image, args.use_sample_image)
        except Exception as exc:
            log(str(exc))
            return 1
        if image_path is None:
            log("Vision mode requires --image or --use-sample-image.")
            return 1

    return run_gemini(
        prompt=args.prompt,
        timeout=args.timeout,
        retries=retries,
        image_path=image_path,
        raw=args.raw,
        ca_bundle=args.ca_bundle,
    )


if __name__ == "__main__":
    raise SystemExit(main())
