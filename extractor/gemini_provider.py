"""Gemini web-based VLM provider — uses the browser automation from mvp_1.

Reuses the Selenium infrastructure (Chrome + Gemini web) to get FREE vision
analysis.  The only new part vs mvp_1 is reading the TEXT response instead of
waiting for a generated image.

Two modes:
  - "persistent" (default, recommended): connect to an already-running Chrome
    via CDP (``python -m extractor.gemini_provider launch`` to start one).
  - "fresh": launch a fresh Chrome, log in, do the job, quit. Slower.

Architecture cribbed from ~/Desktop/mvp_1/headshot_pipeline/persistent_client.py
and ~/Desktop/mvp_1/gemini-image-gen-automation/gemini_automation/generator/image_generator.py.
"""
from __future__ import annotations

import base64
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

from .extract_prompt import SYSTEM_PROMPT, USER_INSTRUCTION, JSON_INSTRUCTION
from .providers import Provider, _loads_lenient

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CDP_PORT = 9222
GEMINI_URL = "https://gemini.google.com/app"
MVP1_ROOT = Path.home() / "Desktop" / "mvp_1"

# ---------------------------------------------------------------------------
# Chrome launcher (reuse mvp_1's approach)
# ---------------------------------------------------------------------------

def _find_chrome() -> str:
    candidates = [
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        "/Applications/Chromium.app/Contents/MacOS/Chromium",
    ]
    for p in candidates:
        if Path(p).exists():
            return p
    import shutil
    for name in ("google-chrome", "chromium", "chromium-browser"):
        found = shutil.which(name)
        if found:
            return found
    raise FileNotFoundError("Chrome not found")


def launch_chrome(port: int = CDP_PORT, profile_dir: Optional[str] = None):
    """Start Chrome with remote debugging. Call once, then reuse."""
    if profile_dir is None:
        profile_dir = str(Path(__file__).resolve().parent.parent / ".chrome_profile")
    Path(profile_dir).mkdir(parents=True, exist_ok=True)

    cmd = [
        _find_chrome(),
        f"--remote-debugging-port={port}",
        "--no-first-run",
        "--no-default-browser-check",
        f"--user-data-dir={profile_dir}",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(3)
    if proc.poll() is not None:
        raise RuntimeError(f"Chrome exited immediately (code {proc.returncode})")
    print(f"✓ Chrome running on CDP port {port} (PID {proc.pid})")
    return proc


# ---------------------------------------------------------------------------
# Gemini web driver
# ---------------------------------------------------------------------------

class _GeminiWebDriver:
    """Low-level Selenium bridge to Gemini web — connect, upload image, send
    prompt, read text response. No image-generation logic."""

    def __init__(self, port: int = CDP_PORT, timeout: int = 120):
        self.port = port
        self.timeout = timeout
        self.driver = None

    # --- connection --------------------------------------------------------

    def connect(self):
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
        opts = Options()
        opts.add_experimental_option("debuggerAddress", f"127.0.0.1:{self.port}")
        try:
            self.driver = webdriver.Chrome(options=opts)
        except Exception as e:
            raise ConnectionError(
                f"Cannot connect to Chrome CDP port {self.port}. "
                f"Run: python -m extractor.gemini_provider launch\n{e}"
            )
        print(f"✓ Connected to Chrome (CDP {self.port})")

    def ensure_gemini(self):
        if not self.driver:
            self.connect()
        url = self.driver.current_url or ""
        if "gemini.google.com" not in url:
            self.driver.get(GEMINI_URL)
            time.sleep(3)
        # check login
        if self._needs_login():
            print("⚠ Please log in to Gemini in the Chrome window (120s)...")
            self._wait_login(120)
            print("✓ Login detected")

    def _needs_login(self) -> bool:
        from selenium.webdriver.common.by import By
        from selenium.common.exceptions import NoSuchElementException
        for sel in ["//button[contains(text(),'Sign in')]", "//a[contains(text(),'Sign in')]"]:
            try:
                if self.driver.find_element(By.XPATH, sel).is_displayed():
                    return True
            except NoSuchElementException:
                pass
        return False

    def _wait_login(self, timeout: int = 120) -> bool:
        t0 = time.time()
        while time.time() - t0 < timeout:
            time.sleep(3)
            try:
                self._find_input()
                return True
            except Exception:
                continue
        return False

    # --- input helpers (from mvp_1 image_generator.py) --------------------

    def _find_input(self):
        from selenium.webdriver.common.by import By
        from selenium.webdriver.support.ui import WebDriverWait
        from selenium.webdriver.support import expected_conditions as EC
        from selenium.common.exceptions import TimeoutException, NoSuchElementException
        selectors = [
            (By.CSS_SELECTOR, "div[aria-label='Enter a prompt for Gemini']"),
            (By.CSS_SELECTOR, "div[aria-label*='Enter a prompt']"),
            (By.CSS_SELECTOR, "textarea[aria-label*='Enter a prompt']"),
            (By.CSS_SELECTOR, "textarea"),
            (By.CSS_SELECTOR, "div[contenteditable='true']"),
            (By.CSS_SELECTOR, "[role='textbox']"),
        ]
        for by, sel in selectors:
            try:
                el = WebDriverWait(self.driver, 3).until(
                    EC.presence_of_element_located((by, sel)))
                if el.is_displayed():
                    return el
            except (TimeoutException, NoSuchElementException):
                continue
        raise NoSuchElementException("Cannot find Gemini text input")

    def _upload_image(self, image_path: str):
        """Upload an image into Gemini's input (from mvp_1 paste_image)."""
        p = Path(image_path)
        with open(p, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode()
        ext = p.suffix.lower()
        mime = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                ".webp": "image/webp"}.get(ext, "image/png")
        text_input = self._find_input()
        script = """
        var base64 = arguments[0], mime = arguments[1], name = arguments[2], input = arguments[3];
        input.focus(); input.click();
        var bin = atob(base64), bytes = new Uint8Array(bin.length);
        for (var i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
        var blob = new Blob([bytes], {type: mime});
        var file = new File([blob], name, {type: mime});
        var dt = new DataTransfer(); dt.items.add(file);
        input.dispatchEvent(new ClipboardEvent('paste', {clipboardData: dt, bubbles: true, cancelable: true}));
        input.dispatchEvent(new Event('input', {bubbles: true}));
        """
        self.driver.execute_script(script, img_b64, mime, p.name, text_input)
        time.sleep(1.5)
        print(f"  📷 Uploaded: {p.name}")

    def _enter_text(self, text: str):
        inp = self._find_input()
        self.driver.execute_script("arguments[0].focus();", inp)
        inp.click()
        time.sleep(0.2)
        inp.clear()
        time.sleep(0.2)
        inp.send_keys(text)
        time.sleep(0.3)

    def _submit(self):
        from selenium.webdriver.common.by import By
        from selenium.webdriver.common.keys import Keys
        from selenium.common.exceptions import NoSuchElementException
        for sel in ["button[aria-label*='Send']", "button[aria-label*='Submit']"]:
            try:
                btn = self.driver.find_element(By.CSS_SELECTOR, sel)
                if btn.is_displayed() and btn.is_enabled():
                    btn.click()
                    return
            except NoSuchElementException:
                continue
        # fallback: Enter
        self._find_input().send_keys(Keys.RETURN)

    def _new_chat(self):
        from selenium.webdriver.common.by import By
        for sel in ["//a[@aria-label='New chat']", "//a[contains(@aria-label,'New chat')]",
                     "//a[@href='/app']"]:
            try:
                el = self.driver.find_element(By.XPATH, sel)
                if el.is_displayed():
                    el.click()
                    time.sleep(2)
                    return
            except Exception:
                continue
        self.driver.get(GEMINI_URL)
        time.sleep(2)

    # --- read TEXT response (the new part for i2e) ------------------------

    def _wait_for_text_response(self, max_wait: int = 180) -> str:
        """Wait for Gemini's text response to appear and extract it.

        Strategy: poll the DOM for response blocks. Gemini 2026 renders
        responses as <model-response> or div with specific markers. We look
        for the last large text block that wasn't there before.
        """
        from selenium.webdriver.common.by import By
        t0 = time.time()
        # give Gemini a moment to start
        time.sleep(4)

        while time.time() - t0 < max_wait:
            # Strategy 1: model-response containers
            try:
                responses = self.driver.find_elements(
                    By.CSS_SELECTOR, "model-response, message-content, "
                    "div.message-content, div.model-response-text")
                for r in reversed(responses):
                    txt = r.text.strip()
                    if len(txt) > 50:  # meaningful response
                        return txt
            except Exception:
                pass

            # Strategy 2: any large text in the conversation area
            try:
                # Gemini 2026: response is in divs with data attributes
                all_divs = self.driver.find_elements(
                    By.CSS_SELECTOR,
                    "div[data-message-id], div.conversation-container div")
                for d in reversed(all_divs):
                    txt = d.text.strip()
                    if len(txt) > 100:
                        return txt
            except Exception:
                pass

            # Strategy 3: grab all visible text from the page, find JSON
            try:
                page_text = self.driver.find_element(By.TAG_NAME, "body").text
                # look for our JSON structure
                start = page_text.rfind('{"elements"')
                if start != -1:
                    # find matching closing brace
                    depth = 0
                    for i in range(start, len(page_text)):
                        if page_text[i] == '{':
                            depth += 1
                        elif page_text[i] == '}':
                            depth -= 1
                            if depth == 0:
                                candidate = page_text[start:i+1]
                                try:
                                    json.loads(candidate)
                                    return candidate
                                except json.JSONDecodeError:
                                    break
            except Exception:
                pass

            time.sleep(2)

        raise TimeoutError(f"Gemini text response not found within {max_wait}s")

    # --- high-level -------------------------------------------------------

    def analyze(self, image_path: str, prompt: str) -> str:
        """Upload image, send prompt, return raw text response."""
        self.ensure_gemini()
        self._new_chat()
        time.sleep(1.5)
        self._upload_image(image_path)
        time.sleep(0.5)
        self._enter_text(prompt)
        self._submit()
        print("⏳ Waiting for Gemini response...")
        return self._wait_for_text_response(self.timeout)

    def disconnect(self):
        if self.driver:
            try:
                self.driver.quit()
            except Exception:
                pass
            self.driver = None
            print("✓ Disconnected from Chrome (browser stays open)")


# ---------------------------------------------------------------------------
# i2e Provider interface
# ---------------------------------------------------------------------------

class GeminiWebProvider:
    """i2e VLM provider that uses Gemini web (free) via browser automation.

    Setup:
      1. python -m extractor.gemini_provider launch   # start Chrome once
      2. Log in to Gemini in the Chrome window
      3. Use this provider in i2e pipeline

    No API key needed. Uses your personal Google account's Gemini access.
    """

    name = "gemini-web"
    model_version = "gemini:web-free"

    def __init__(self, port: int = CDP_PORT, timeout: int = 180):
        self._port = port
        self._timeout = timeout
        self._wd: Optional[_GeminiWebDriver] = None

    def _get_wd(self) -> _GeminiWebDriver:
        if self._wd is None:
            self._wd = _GeminiWebDriver(port=self._port, timeout=self._timeout)
        return self._wd

    def extract(self, image_path: str) -> list[dict]:
        prompt = SYSTEM_PROMPT + "\n\n" + JSON_INSTRUCTION + "\n\n" + USER_INSTRUCTION
        wd = self._get_wd()
        raw = wd.analyze(image_path, prompt)

        # parse elements from response
        elements = _loads_lenient(raw).get("elements", [])

        # bbox normalization: ensure fractions in [0,1] → pixels
        from PIL import Image
        W, H = Image.open(image_path).size
        for el in elements:
            b = el.get("bbox")
            if isinstance(b, dict):
                for k in ("x", "y", "w", "h"):
                    v = b.get(k, 0)
                    try:
                        v = float(v)
                    except (TypeError, ValueError):
                        v = 0.0
                    # if values look like fractions (<1.5), scale to pixels
                    # if already pixels (>1.5), keep as-is
                    if k in ("x", "w") and v <= 1.5:
                        v = v * W
                    elif k in ("y", "h") and v <= 1.5:
                        v = v * H
                    b[k] = max(0.0, v)
        return elements


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    import argparse
    p = argparse.ArgumentParser(description="i2e Gemini web provider utilities")
    sub = p.add_subparsers(dest="cmd")

    sub.add_parser("launch", help="Launch Chrome with CDP for Gemini")
    sub.add_parser("status", help="Check if Chrome CDP is running")

    ext = sub.add_parser("extract", help="Extract elements from an image")
    ext.add_argument("image", help="Path to image file")
    ext.add_argument("--port", type=int, default=CDP_PORT)
    ext.add_argument("--timeout", type=int, default=180)

    args = p.parse_args()
    if not args.cmd:
        p.print_help()
        return

    if args.cmd == "launch":
        launch_chrome(args.port if hasattr(args, 'port') else CDP_PORT)

    elif args.cmd == "status":
        try:
            wd = _GeminiWebDriver(port=CDP_PORT)
            wd.connect()
            wd.disconnect()
            print(f"✓ Chrome CDP running on port {CDP_PORT}")
        except Exception:
            print(f"✗ Chrome NOT detected on port {CDP_PORT}")

    elif args.cmd == "extract":
        prov = GeminiWebProvider(port=args.port, timeout=args.timeout)
        els = prov.extract(args.image)
        print(json.dumps(els, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
