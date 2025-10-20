#!/usr/bin/env python3
import os, base64, re, json, requests, zipfile, math, time, threading, sys, glob
from io import BytesIO
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont  # pip install pillow

# Optional, robust background removal (open-source models via rembg)
try:
    from rembg import remove, new_session  # pip install rembg  (or rembg[gpu])
    HAVE_REMBG = True
except Exception:
    print("rembg not found, using poor fallback...")
    print("pip install rembg  (or rembg[gpu])")
    HAVE_REMBG = False
    remove = new_session = None

def remove_bg_highres(img: Image.Image) -> Image.Image:
    """
    Remove background while preserving original resolution.
    Uses rembg to get mask, then resizes mask to original size and applies it if needed.
    """
    import numpy as np
    from rembg import remove

    # Run rembg in RGBA mode
    result = remove(img)

    # Ensure same size as original
    if result.size != img.size:
        result = result.resize(img.size, Image.LANCZOS)
    return result

# ============================================================================
# MULTI-IMAGE INPUT SUPPORT
# ============================================================================
# Find input files with pattern: input0.ext, input1.ext, input2.ext, etc.
SUPPORTED_EXTENSIONS = ['png', 'jpg', 'jpeg', 'webp', 'bmp', 'gif', 'tiff', 'tif']
INPUT_FILE_PATHS = []

print("="*60)
print("Searching for input images...")
print("="*60)

# Look for input0.ext, input1.ext, input2.ext, etc.
index = 0
while True:
    found = False
    for ext in SUPPORTED_EXTENSIONS:
        for pattern in [f"input{index}.{ext}", f"input{index}.{ext.upper()}"]:
            matches = glob.glob(pattern)
            if matches:
                INPUT_FILE_PATHS.append(matches[0])
                found = True
                break
        if found:
            break
    
    # Stop searching if we didn't find the current index
    # (assumes sequential numbering)
    if not found:
        break
    index += 1

# Validate that we found at least one input file
if not INPUT_FILE_PATHS:
    print(f"\n[Error] No input file found!")
    print(f"\nPlease provide input file(s) with the following naming pattern:")
    print(f"  - Single image: 'input0.{{ext}}'")
    print(f"  - Multiple images: 'input0.{{ext}}', 'input1.{{ext}}', 'input2.{{ext}}', etc.")
    print(f"\nSupported extensions: {', '.join(SUPPORTED_EXTENSIONS)}")
    raise SystemExit("No input file found")

print(f"\nFound {len(INPUT_FILE_PATHS)} input image(s):")
for i, path in enumerate(INPUT_FILE_PATHS):
    file_size = os.path.getsize(path) / 1024  # Size in KB
    with Image.open(path) as img:
        print(f"  [{i}] {path} - {img.size[0]}x{img.size[1]} pixels ({file_size:.1f} KB)")

print("="*60 + "\n")
# ============================================================================

EMOTIONS = {
    "admiration": "warm admiration directed towards the viewer",
    "amusement": "playful amusement directed towards the viewer",
    "anger": "intense anger directed towards the viewer",
    "annoyance": "subtle annoyance directed towards the viewer",
    "approval": "a look of approval directed towards the viewer",
    "caring": "attitude of care directed towards the viewer",
    "confusion": "look of confusion directed towards the viewer",
    "curiosity": "an expression of curiosity directed towards the viewer",
    "desire-0": "clear smiling interest directed towards the viewer",
    "desire-1": "clear smiling interest and flirtation directed towards the viewer",
    "desire-2": "strong flirtation and intense attraction directed towards the viewer",
    "disappointment": "disappointment directed at the viewer",
    "disapproval": "a look of disapproval directed at the viewer",
    "disgust": "a face of disgust directed at the viewer",
    "embarrassment": "an expression of personal embarrassment",
    "excitement": "an expression of clear excitement",
    "fear": "an expression of clear fear",
    "gratitude": "an expression of clear gratitude towards the viewer",
    "grief": "an expression and pose of grief",
    "joy": "an expression and pose of joy",
    "love": "a look of love directed towards the viewer",
    "nervousness": "an expression of nervousness in pose and expression",
    "neutral": "a neutral calm expression",
    "optimism": "an expression carrying a clear indication of optimism for the future",
    "pride": "an expression of pride looking at the viewer",
    "realization": "an expression of sudden realization",
    "relief": "an expression of profound relief",
    "remorse": "an expression of profound remorse directed towards the viewer",
    "sadness": "an expression of deep sadness",
    "surprise": "an expression of sudden surprise",
}

# API Configuration
API_PROVIDER = None  # Will be set in main()
API_URL = None  # Will be set based on provider
API_MODEL = None  # Will be set based on provider

# Default models
OPENROUTER_MODEL = os.getenv("OPENROUTER_IMAGE_MODEL", "google/gemini-2.5-flash-image-preview")
GOOGLE_MODEL = os.getenv("GOOGLE_IMAGE_MODEL", "gemini-2.5-flash-image-preview")
CUSTOM_MODEL = None  # Will be loaded from config or prompted

# API Endpoints
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# AI parameters (only those explicitly set in config.ini will be used)
AI_PARAMS = {}

# rembg config
REMBG_MODEL = os.getenv("REMBG_MODEL", "isnet-anime")  # "isnet-general-use" (default) or "isnet-anime"
REMBG_ALPHA_MATTING=True
REMBG_FG_THR=180    # lower than 220, so dark outlines aren't misclassified as background
REMBG_BG_THR=30     # slightly higher than 20, helps push light flat backgrounds to transparent
REMBG_ERODE=1       # small erosion, avoids eating away thin black lines
REMBG_BASE=2048     # Increased from 1024 for better quality on larger images
REMBG_POST_MASK=True
BACKGROUND_TYPE = 'grey'

OUT_DIR = Path("expressions")
OUT_DIR.mkdir(exist_ok=True)

# Global list to track failed URL downloads
FAILED_DOWNLOADS = []

# --- API Key Management ---
class KeyManager:
    def __init__(self, keys, api_provider):
        self.keys = [k for k in keys if k]  # Filter out empty keys
        self.current_index = 0
        self.exhausted_keys = set()
        self.api_provider = api_provider
    
    def get_current_key(self):
        if not self.keys:
            return None
        if self.current_index >= len(self.keys):
            return None
        return self.keys[self.current_index]
    
    def rotate_key(self, reason=""):
        """Rotate to the next available key"""
        if not self.keys:
            return None
        
        # Mark current key as exhausted
        self.exhausted_keys.add(self.current_index)
        
        # Find next non-exhausted key
        for i in range(len(self.keys)):
            next_index = (self.current_index + 1 + i) % len(self.keys)
            if next_index not in self.exhausted_keys:
                self.current_index = next_index
                print(f"  > Switched to {self.api_provider} API key #{self.current_index + 1} {reason}")
                return self.keys[self.current_index]
        
        # All keys exhausted, reset and try again
        print(f"  > All {self.api_provider} API keys exhausted, resetting rotation...")
        self.exhausted_keys.clear()
        self.current_index = 0
        return self.keys[0] if self.keys else None
    
    def get_key_count(self):
        return len(self.keys)
    
    def get_key_number(self):
        return self.current_index + 1
    
    def has_more_keys(self):
        """Check if there are any non-exhausted keys available"""
        return len(self.exhausted_keys) < len(self.keys)

# Global key manager (will be initialized in main())
KEY_MANAGER = None

# --- Helpers ---
def parse_config_value(value_str):
    """Parse a configuration value from string to appropriate type"""
    value_str = value_str.strip()
    
    # Handle None/null
    if value_str.lower() in ['none', 'null', '']:
        return None
    
    # Handle boolean
    if value_str.lower() in ['true', 'yes', 'on']:
        return True
    if value_str.lower() in ['false', 'no', 'off']:
        return False
    
    # Try to parse as number
    try:
        # Try integer first
        if '.' not in value_str:
            return int(value_str)
        # Then float
        return float(value_str)
    except ValueError:
        pass
    
    # Return as string (remove quotes if present)
    if (value_str.startswith('"') and value_str.endswith('"')) or \
       (value_str.startswith("'") and value_str.endswith("'")):
        return value_str[1:-1]
    
    return value_str

def load_config_and_keys(filepath="config.ini", provider="openrouter"):
    """Load configuration and API keys from a text file"""
    config = {}
    keys = []
    
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        in_config_section = False
        in_provider_section = False
        
        for line in lines:
            line = line.strip()
            
            # Skip empty lines and comments
            if not line or line.startswith('#'):
                continue
            
            # Check for section headers
            if line.lower() == '[config]':
                in_config_section = True
                in_provider_section = False
                continue
            elif line.lower() == f'[{provider}]':
                in_config_section = False
                in_provider_section = True
                continue
            elif line.startswith('[') and line.endswith(']'):
                in_config_section = False
                in_provider_section = False
                continue
            
            # Parse config section
            if in_config_section:
                if '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = parse_config_value(value)
                    config[key] = value
            
            # Parse provider section
            elif in_provider_section:
                keys.append(line)
            
            # If no sections defined, treat non-config lines as keys (backward compatibility)
            elif not any('[' in l and ']' in l for l in lines):
                keys.append(line)
    
    except FileNotFoundError:
        pass
    
    return config, keys

def extract_url_from_content(content):
    """Extract image URL from message content using various patterns"""
    if not content or not isinstance(content, str):
        return None
    
    # Pattern 1: HTML img tag with src attribute
    img_pattern = r'<img[^>]*src=["\']([^"\']+)["\'][^>]*>'
    matches = re.findall(img_pattern, content, re.IGNORECASE)
    if matches:
        return matches[0]
    
    # Pattern 2: Markdown image syntax ![alt](url)
    md_pattern = r'!\[.*?\]\(([^)]+)\)'
    matches = re.findall(md_pattern, content)
    if matches:
        return matches[0]
    
    # Pattern 3: Raw URL (common image extensions)
    url_pattern = r'(https?://[^\s]+\.(?:png|jpg|jpeg|gif|webp|bmp|tiff?)\b[^\s]*)'
    matches = re.findall(url_pattern, content, re.IGNORECASE)
    if matches:
        return matches[0]
    
    # Pattern 4: Any URL that looks like it might be an image service
    # (githubusercontent, cloudinary, imgur, etc.)
    service_pattern = r'(https?://[^\s]*(?:githubusercontent|cloudinary|imgur|imgbb|postimg|tinypic|photobucket|flickr|500px|unsplash|pexels)[^\s]+)'
    matches = re.findall(service_pattern, content, re.IGNORECASE)
    if matches:
        return matches[0]
    
    return None

def download_image_from_url(url):
    """Download image from URL and return as bytes"""
    try:
        print(f"  > Downloading image from URL: {url}")
        response = requests.get(url, timeout=30, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        response.raise_for_status()
        
        # Verify it's actually an image
        content_type = response.headers.get('content-type', '').lower()
        if 'image' not in content_type and not url.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.webp')):
            print(f"[Warning] URL does not appear to be an image (content-type: {content_type})")
        
        # Convert to PNG bytes for consistency
        with Image.open(BytesIO(response.content)) as im:
            # Ensure we're working with a compatible mode
            if im.mode not in ['RGB', 'RGBA', 'L', 'LA']:
                if 'transparency' in im.info:
                    im = im.convert('RGBA')
                else:
                    im = im.convert('RGB')
            
            out = BytesIO()
            im.save(out, format="PNG")
            return out.getvalue()
    except Exception as e:
        print(f"[Error] Failed to download image from URL: {e}")
        return None

def image_to_data_url(path: str) -> str:
    """Convert image file to data URL, supporting multiple formats"""
    try:
        with Image.open(path) as im:
            # Convert RGBA to RGB if saving as JPEG
            if im.mode == 'RGBA':
                # Create a white background
                background = Image.new('RGB', im.size, (255, 255, 255))
                background.paste(im, mask=im.split()[3] if len(im.split()) > 3 else None)
                im = background
            elif im.mode not in ['RGB', 'L']:
                # Convert other modes to RGB
                im = im.convert('RGB')
            
            # Determine format and MIME type
            original_format = (im.format or "PNG").upper()
            
            # Use PNG for formats that might have transparency or special features
            if original_format in ['PNG', 'WEBP', 'GIF', 'TIFF', 'BMP']:
                fmt = 'PNG'
                mime = 'image/png'
            else:
                # Use JPEG for photos
                fmt = 'JPEG'
                mime = 'image/jpeg'
            
            # Save to buffer
            buf = BytesIO()
            if fmt == 'JPEG':
                im.save(buf, format=fmt, quality=95, optimize=True)
            else:
                im.save(buf, format=fmt)
            
            # Encode to base64
            b64 = base64.b64encode(buf.getvalue()).decode("ascii")
            return f"data:{mime};base64,{b64}"
    except Exception as e:
        print(f"[Error] Failed to load image from {path}: {e}")
        raise

# ============================================================================
# CREATE DATA URLs FOR ALL INPUT IMAGES
# ============================================================================
DATA_URLS = []
print("Converting input images to data URLs...")
for i, input_path in enumerate(INPUT_FILE_PATHS):
    try:
        data_url = image_to_data_url(input_path)
        DATA_URLS.append(data_url)
        print(f"  [{i}] {input_path} -> converted")
    except Exception as e:
        print(f"[Error] Failed to process {input_path}: {e}")
        raise SystemExit(f"Failed to process input file: {input_path}")

print(f"\nSuccessfully loaded {len(DATA_URLS)} input image(s)")
print("="*60 + "\n")
# ============================================================================

def decode_data_url_to_png_bytes(data_url: str) -> bytes:
    """Decode data URL to PNG bytes, handling various input formats"""
    m = re.match(r"^data:([^;]+);base64,(.+)$", data_url)
    if not m:
        raise ValueError("Unexpected image URL format")
    mime, b64 = m.group(1), m.group(2)
    raw = base64.b64decode(b64)
    
    # Always convert to PNG for consistency
    with Image.open(BytesIO(raw)) as im:
        # Ensure we're working with a compatible mode
        if im.mode not in ['RGB', 'RGBA', 'L', 'LA']:
            if 'transparency' in im.info:
                im = im.convert('RGBA')
            else:
                im = im.convert('RGB')
        
        out = BytesIO()
        im.save(out, format="PNG")
        return out.getvalue()

# Input with timeout helper
def input_with_timeout(prompt, timeout_seconds=20):
    """Get user input with a timeout. Returns None if timeout occurs."""
    result = [None]
    
    def get_input():
        try:
            result[0] = input(prompt)
        except:
            pass
    
    thread = threading.Thread(target=get_input)
    thread.daemon = True
    thread.start()
    thread.join(timeout_seconds)
    
    if thread.is_alive():
        print(f"\n[Timeout] No response in {timeout_seconds} seconds, skipping...")
        return None
    return result[0]

def interruptible_wait(seconds, message="Waiting...", allow_interrupt=True):
    """
    Wait for specified seconds with ability to interrupt by pressing a single key.
    Returns:
        None: wait completed normally
        's': user pressed 's' (switch keys)
        'r': user pressed 'r' (retry with same key)
    """
    if not allow_interrupt:
        time.sleep(seconds)
        return None
    
    print(f"  > {message}")
    print(f"     (Press 's' to skip and switch keys, 'r' to skip with same key, or wait {seconds} seconds)")
    
    import sys
    
    # Platform-specific key detection
    if sys.platform == 'win32':
        import msvcrt
        start_time = time.time()
        while time.time() - start_time < seconds:
            if msvcrt.kbhit():
                key = msvcrt.getch().decode('utf-8', errors='ignore').lower()
                if key == 's':
                    print("  > User pressed 's' - switching keys...")
                    return 's'
                elif key == 'r':
                    print("  > User pressed 'r' - retrying with same key...")
                    return 'r'
            time.sleep(0.1)
    else:
        import select
        import tty
        import termios
        
        old_settings = termios.tcgetattr(sys.stdin)
        try:
            tty.setcbreak(sys.stdin.fileno())
            start_time = time.time()
            while time.time() - start_time < seconds:
                if select.select([sys.stdin], [], [], 0.1)[0]:
                    key = sys.stdin.read(1).lower()
                    if key == 's':
                        print("  > User pressed 's' - switching keys...")
                        return 's'
                    elif key == 'r':
                        print("  > User pressed 'r' - retrying with same key...")
                        return 'r'
        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
    
    return None

# Fallback white-knockout (kept as a safety net)
def _knock_out_white_bg_fallback(im: Image.Image, tol: int = 24) -> Image.Image:
    im = im.convert("RGBA")
    px = im.load(); w, h = im.size; t0 = 255 - tol
    for y in range(h):
        for x in range(w):
            r,g,b,a = px[x,y]
            if a == 0: continue
            wht = (r+g+b)//3
            if wht > t0:
                new_a = 0 if wht >= 255 else max(0, min(255, int(255*(1-(wht-t0)/float(tol)))))
                px[x,y] = (r,g,b,new_a)
    return im

# rembg session init (once)
_REMBG_SESSION = None
def _get_rembg_session():
    global _REMBG_SESSION
    if not HAVE_REMBG: return None
    if _REMBG_SESSION is None:
        _REMBG_SESSION = new_session(REMBG_MODEL)
    return _REMBG_SESSION

def remove_background(png_bytes: bytes) -> bytes:
    """
    Prefer rembg (ISNet/U²-Net) for high-quality matting.
    Falls back to simple white-threshold method if rembg is unavailable/errors.
    """

    if HAVE_REMBG:
        try:
            session = _get_rembg_session()
            out = remove(
                png_bytes,
                session=session,
                alpha_matting=REMBG_ALPHA_MATTING,
                alpha_matting_foreground_threshold=REMBG_FG_THR,
                alpha_matting_background_threshold=REMBG_BG_THR,
                alpha_matting_erode_structure_size=REMBG_ERODE,
                alpha_matting_base_size=REMBG_BASE,
                post_process_mask=REMBG_POST_MASK,
            )
            return out  # already PNG bytes with RGBA
        except Exception as e:
            print(f"[warn] rembg failed ({e}); using white-knockout fallback.")

    with Image.open(BytesIO(png_bytes)) as im:
        im = _knock_out_white_bg_fallback(im, tol=28)
        buf = BytesIO(); im.save(buf, format="PNG"); return buf.getvalue()

def is_rate_limit_error(error_msg, status_code=None):
    """Check if error is a rate limit error (429 or contains rate limit message)"""
    if status_code == 429:
        return True
    
    error_str = str(error_msg).lower()
    rate_limit_patterns = [
        "too many requests",
        "rate limit",
        "quota exceeded",
        "429",
        "throttl"
    ]
    return any(pattern in error_str for pattern in rate_limit_patterns)

def is_insufficient_credits_error(error_msg, response_json=None):
    """Check if error is an insufficient credits error"""
    error_str = str(error_msg).lower()
    credit_patterns = [
        "insufficient credits",
        "insufficient funds",
        "not enough credits",
        "credit balance",
        "out of credits",
        "no credits",
        "payment required"
    ]
    
    # Check in error message string
    if any(pattern in error_str for pattern in credit_patterns):
        return True
    
    # Check in JSON response if available
    if response_json:
        try:
            if isinstance(response_json, dict):
                error_msg = str(response_json.get("error", {}).get("message", "")).lower()
                if any(pattern in error_msg for pattern in credit_patterns):
                    return True
        except:
            pass
    
    return False

def is_prohibited_content_error(error_msg, response_json=None):
    """Check if error is a prohibited content error"""
    error_str = str(error_msg).lower()
    prohibited_patterns = [
        "prohibited_content",
        "prohibited content",
        "content policy",
        "safety filter",
        "unsafe content",
        "violates our policy",
        "blocked",
        "not allowed"
    ]
    
    # Check in error message string
    if any(pattern in error_str for pattern in prohibited_patterns):
        return True
    
    # Check in JSON response if available
    if response_json:
        try:
            # Check various possible error field locations
            if isinstance(response_json, dict):
                # OpenRouter style
                if response_json.get("error", {}).get("code") == "PROHIBITED_CONTENT":
                    return True
                # Alternative error formats
                error_msg = str(response_json.get("error", {}).get("message", "")).lower()
                if any(pattern in error_msg for pattern in prohibited_patterns):
                    return True
        except:
            pass
    
    return False

def print_error_details(response=None, exception=None, context=""):
    """Print detailed error information"""
    print("\n" + "="*60)
    print(f"[ERROR DETAILS] {context}")
    print("="*60)
    
    if response is not None:
        print(f"Status Code: {response.status_code}")
        print(f"Response Headers: {dict(response.headers)}")
        print(f"Response Body:")
        try:
            # Try to parse as JSON for pretty printing
            resp_json = response.json()
            print(json.dumps(resp_json, indent=2))
        except:
            # If not JSON, print raw text
            print(response.text)
    
    if exception is not None:
        print(f"Exception Type: {type(exception).__name__}")
        print(f"Exception Message: {str(exception)}")
    
    print("="*60 + "\n")

def generate_for_emotion(key: str, desc: str, remove_bg: bool, custom_tweak: str = ""):
    """
    Generate (or reuse cached) image for an emotion key.
    - If background removal is enabled:
      - If expressions/orig__{key}.png exists, skip API call and reuse
      - Otherwise call API, cache as orig__{key}.png
      - Apply background removal and save as {key}.png
    - If background removal is disabled:
      - If expressions/{key}.png exists, return existing path
      - Otherwise call API and save directly as {key}.png (no orig__ file)
    """
    global KEY_MANAGER
    
    final_paths = []
    
    # Different cache behavior based on background removal setting
    if remove_bg:
        # With background removal: use orig__ caching system
        orig_path = OUT_DIR / f"orig__{key}.png"
        final_path = OUT_DIR / f"{key}.png"
        
        # If cached original exists, reuse it
        if orig_path.exists():
            png_bytes = orig_path.read_bytes()
        else:
            # Generate new image
            png_bytes = _call_api_for_emotion(key, desc, custom_tweak)
            if png_bytes is None:
                return []
            # Cache the original
            orig_path.write_bytes(png_bytes)
        
        # Apply background removal
        processed_bytes = remove_background(png_bytes)
        final_path.write_bytes(processed_bytes)
        final_paths.append(str(final_path))
    else:
        # Without background removal: direct save only
        final_path = OUT_DIR / f"{key}.png"
        
        # If final image exists, return it without regenerating
        if final_path.exists():
            return [str(final_path)]
        
        # Generate new image
        png_bytes = _call_api_for_emotion(key, desc, custom_tweak)
        if png_bytes is None:
            return []
        
        # Save directly as final image (no processing, no orig__ file)
        final_path.write_bytes(png_bytes)
        final_paths.append(str(final_path))
    
    return final_paths

def _call_api_for_emotion(key: str, desc: str, custom_tweak: str = ""):
    """Helper function to call the API and return image bytes"""
    global KEY_MANAGER, API_PROVIDER, API_URL, API_MODEL, AI_PARAMS, FAILED_DOWNLOADS, DATA_URLS
    
    retry_count = 0
    max_retries = 5  # Max number of user-initiated retries
    current_custom_tweak = custom_tweak  # Can be modified if user chooses to tweak
    previous_key_number = KEY_MANAGER.get_key_number()  # Track key changes
    
    # Increase max_tries when we have multiple keys available
    base_max_tries = 3
    max_tries = base_max_tries + (KEY_MANAGER.get_key_count() * 2)  # More attempts with more keys
    
    while retry_count <= max_retries:
        if BACKGROUND_TYPE == 'white':
            bg_desc = "an entirely white background"
        else:
            bg_desc = "a neutral middle grey coloured background"
        
        # ========================================================================
        # MULTI-IMAGE PROMPT ADAPTATION
        # ========================================================================
        if len(DATA_URLS) > 1:
            base_prompt = (
                f"Please convert {'these images' if len(DATA_URLS) > 1 else 'this image'} in the same aspect ratio and style as the original, "
                "change the default pose, be creative with the poses, "
                "common visual tropes are acceptable, convey the emotions clearly with body language and expressions, "
                "do not make multiple variations in one image, "
                f"always with {bg_desc} "
                "so that it demonstrates {desc}. "
                f"Note: You are provided with {len(DATA_URLS)} reference image(s) - use them all as reference for style, character, and composition."
            )
        else:
            base_prompt = (
                "Please convert this image in the same aspect ratio and style as the original, "
                "change the default pose, be creative with the poses, "
                "common visual tropes are acceptable, convey the emotions clearly with body language and expressions, "
                "do not make multiple variations in one image, "
                f"always with {bg_desc} "
                "so that it demonstrates {desc}."
            )
        # ========================================================================
        
        final_prompt = base_prompt.format(desc=desc)
        if current_custom_tweak:
            final_prompt += f" Additional instruction: {current_custom_tweak}"

        # Output the exact prompt being sent
        if retry_count > 0:
            print(f"  > Retry attempt {retry_count}")
        print(f"  > Sending prompt to AI ({len(DATA_URLS)} image(s)):\n    \"{final_prompt}\"")

        # API call with retry logic
        response = None
        api_succeeded = False
        
        for attempt in range(max_tries):
            # Break if we've exhausted all keys multiple times
            if attempt > 0 and attempt % KEY_MANAGER.get_key_count() == 0 and not KEY_MANAGER.has_more_keys():
                print("[Info] All keys have been tried multiple times.")
                break
                
            try:
                current_key = KEY_MANAGER.get_current_key()
                if not current_key:
                    print("[Error] No API keys available.")
                    return None
                
                current_key_number = KEY_MANAGER.get_key_number()
                print(f"  > Calling {API_PROVIDER} API with key #{current_key_number}... (Attempt {attempt + 1}/{max_tries})")
                
                # ================================================================
                # BUILD PAYLOAD WITH MULTIPLE IMAGES
                # ================================================================
                if API_PROVIDER == "openrouter" or API_PROVIDER == "custom":
                    headers = {"Authorization": f"Bearer {current_key}", "Content-Type": "application/json"}
                    
                    # Build content with text first, then all images
                    content = [{"type": "text", "text": final_prompt}]
                    for data_url in DATA_URLS:
                        content.append({"type": "image_url", "image_url": {"url": data_url}})
                    
                    payload = {
                        "model": API_MODEL,
                        "messages": [{
                            "role": "user",
                            "content": content,
                        }],
                    }
                    # Add AI parameters from config (only those explicitly set)
                    for param, value in AI_PARAMS.items():
                        if value is not None:  # Only add non-None parameters
                            payload[param] = value
                    # Add modalities for OpenRouter
                    if API_PROVIDER == "openrouter":
                        payload["modalities"] = ["image", "text"]
                    
                    response = requests.post(API_URL, headers=headers, data=json.dumps(payload), timeout=300)
                
                elif API_PROVIDER == "google":
                    # Google Gemini REST call: use x-goog-api-key header and generateContent format
                    headers = {"x-goog-api-key": current_key, "Content-Type": "application/json"}
                    
                    # Build parts with all images first, then text prompt
                    parts = []
                    for data_url in DATA_URLS:
                        # Extract mime and b64 from DATA_URL
                        m = re.match(r"^data:([^;]+);base64,(.+)$", data_url)
                        if not m:
                            print("[Error] Could not parse DATA_URL for inline image data.")
                            return None
                        mime = m.group(1)
                        img_b64 = m.group(2)
                        parts.append({"inline_data": {"mime_type": mime, "data": img_b64}})
                    
                    # Add text prompt at the end
                    parts.append({"text": final_prompt})
                    
                    payload = {
                        "contents": [
                            {"parts": parts, "role": "user"}
                        ],
                        # Request image output. Use response_modalities to indicate we want images.
                        "generation_config": {
                            # Don't specify response_mime_type for image generation
                            # Gemini will return images in the response parts
                            "response_modalities": ["IMAGE"],
                            "candidate_count": 1
                        }
                    }
                    
                    # Merge in AI_PARAMS if any (map keys that make sense into generation_config)
                    if AI_PARAMS:
                        # Add any simple numeric / sampling params into generation_config
                        gen_conf = payload["generation_config"]
                        for k, v in AI_PARAMS.items():
                            # only add basic generation config items we expect
                            if v is not None:
                                gen_conf[k] = v
                        payload["generation_config"] = gen_conf
                    
                    response = requests.post(API_URL, headers=headers, data=json.dumps(payload), timeout=300)
                # ================================================================
                
                else:
                    print(f"[Error] Unknown API provider: {API_PROVIDER}")
                    return None
                
                # Parse response for error checking
                resp_json = None
                try:
                    resp_json = response.json()
                except:
                    pass
                
                # Check for insufficient credits error
                if is_insufficient_credits_error(response.text, resp_json):
                    print(f"\n[Insufficient Credits] on {API_PROVIDER} key #{current_key_number}")
                    new_key = KEY_MANAGER.rotate_key("(due to insufficient credits)")
                    if new_key:
                        continue  # Try next key immediately
                    else:
                        print(f"[Error] All {API_PROVIDER} API keys have insufficient credits.")
                        return None
                
                # Check for prohibited content error
                if is_prohibited_content_error(response.text, resp_json):
                    print("\n" + "!"*60)
                    print("[CONTENT POLICY VIOLATION DETECTED]")
                    print("!"*60)
                    print_error_details(response=response, context=f"Content policy violation for '{key}'")
                    
                    print(f"\nThe prompt for '{key}' was rejected due to content policy.")
                    print("Options:")
                    print("  [t] Tweak the prompt and retry")
                    print("  [r] Retry with the same prompt")
                    print("  [s] Skip this emotion")
                    
                    user_choice = input_with_timeout("Choose an option [t/r/s] (30 sec timeout): ", timeout_seconds=30)
                    
                    if user_choice and user_choice.lower().strip() == 't':
                        print("\nCurrent prompt additional instruction:")
                        print(f"  {current_custom_tweak if current_custom_tweak else '(none)'}")
                        new_tweak = input("Enter new/modified instruction (or press Enter to clear): ").strip()
                        current_custom_tweak = new_tweak
                        retry_count += 1
                        break  # Break inner loop to retry with new prompt
                    elif user_choice and user_choice.lower().strip() == 'r':
                        retry_count += 1
                        if retry_count > max_retries:
                            print(f"[Error] Maximum retries ({max_retries}) reached. Skipping '{key}'.")
                            return None
                        break  # Break inner loop to retry with same prompt
                    else:
                        print(f"  > Skipping '{key}'.")
                        return None
                
                # Check for rate limiting or other errors
                if response.status_code != 200:
                    print_error_details(response=response, context=f"API call failed with status {response.status_code}")
                    
                    if is_rate_limit_error(response.text, response.status_code):
                        print(f"[Rate Limit Detected] on {API_PROVIDER} key #{current_key_number}")
                        
                        # Always try to rotate to a new key when rate limited
                        new_key = KEY_MANAGER.rotate_key("(due to rate limit)")
                        if new_key:
                            # Shorter delay when switching keys
                            print(f"  > Retrying with different API key in 5 seconds...")
                            time.sleep(5)
                            previous_key_number = KEY_MANAGER.get_key_number()
                            continue  # Continue with next attempt
                        else:
                            print(f"[Error] No more {API_PROVIDER} API keys available to rotate.")
                            if attempt < max_tries - 1:
                                # Use interruptible wait here
                                action = interruptible_wait(30, "Waiting before retry...")
                                if action == 's':
                                    print("  > Switching to next key...")
                                    KEY_MANAGER.exhausted_keys.clear()  # Reset exhausted keys
                                    KEY_MANAGER.rotate_key("(user requested)")
                                    previous_key_number = KEY_MANAGER.get_key_number()
                                elif action == 'r':
                                    print("  > Retrying with same key...")
                                continue
                            return None
                    
                    # For non-rate-limit errors, retry after delay
                    if attempt < max_tries - 1:
                        # Check if we're using the same key
                        if current_key_number == previous_key_number:
                            # Use interruptible wait for same key retry
                            action = interruptible_wait(30, "Retrying in 30 seconds...")
                            if action == 's':
                                print("  > Switching to next key...")
                                new_key = KEY_MANAGER.rotate_key("(user requested)")
                                if new_key:
                                    previous_key_number = KEY_MANAGER.get_key_number()
                            elif action == 'r':
                                print("  > Retrying immediately with same key...")
                            # For both 's', 'r', and None, we continue to retry
                        else:
                            print(f"  > Retrying with different key in 5 seconds...")
                            time.sleep(5)
                            previous_key_number = current_key_number
                        continue
                
                response.raise_for_status()  # Raise an exception for bad status codes
                print("  > API call successful.")
                api_succeeded = True
                break  # Exit loop on success
                
            except requests.exceptions.RequestException as e:
                print_error_details(exception=e, response=response if response else None, 
                                  context=f"Request failed on attempt {attempt + 1}")
                
                # Check if it's a rate limit error
                if response and is_rate_limit_error(str(e)):
                    print(f"[Rate Limit Detected] on {API_PROVIDER} key #{current_key_number}")
                    
                    # Always try to rotate when rate limited
                    new_key = KEY_MANAGER.rotate_key("(due to rate limit)")
                    if new_key:
                        print(f"  > Retrying with different API key in 5 seconds...")
                        time.sleep(5)
                        previous_key_number = KEY_MANAGER.get_key_number()
                        continue
                    else:
                        print(f"[Error] No more {API_PROVIDER} API keys available to rotate.")
                        if attempt < max_tries - 1:
                            # Use interruptible wait here
                            action = interruptible_wait(30, "Waiting before retry...")
                            if action == 's':
                                print("  > Switching to next key...")
                                KEY_MANAGER.exhausted_keys.clear()  # Reset for retry
                                KEY_MANAGER.rotate_key("(user requested)")
                                previous_key_number = KEY_MANAGER.get_key_number()
                            elif action == 'r':
                                print("  > Retrying with same key...")
                            continue
                
                if attempt < max_tries - 1:
                    # Check if we're using the same key
                    if current_key_number == previous_key_number:
                        # Use interruptible wait for same key retry
                        action = interruptible_wait(30, "Retrying in 30 seconds...")
                        if action == 's':
                            print("  > Switching to next key...")
                            new_key = KEY_MANAGER.rotate_key("(user requested)")
                            if new_key:
                                previous_key_number = KEY_MANAGER.get_key_number()
                        elif action == 'r':
                            print("  > Retrying immediately with same key...")
                        # For both 's', 'r', and None, we continue to retry
                    else:
                        print(f"  > Retrying with different key in 5 seconds...")
                        time.sleep(5)
                        previous_key_number = current_key_number
                else:
                    print(f"[Error] All {max_tries} API attempts failed.")
                    return None

        # Check if we need to continue the outer retry loop
        if retry_count > 0 and retry_count <= max_retries and not api_succeeded:
            continue  # Continue outer loop for prohibited content retries

        if not response or not api_succeeded:
            return None

        try:
            resp = response.json()
        except json.JSONDecodeError as e:
            print_error_details(response=response, exception=e, context="Failed to parse JSON response")
            return None
        
        images = []
        png_bytes = None
        
        # OpenRouter-style response handling (existing logic)
        try:
            # Many non-Google providers return choices[...] structure
            if "choices" in resp:
                message = resp["choices"][0].get("message", {})
                images = message.get("images") or []
                
                # If no images, check if there's a URL in the content
                if not images and message.get("content"):
                    content = message["content"]
                    
                    # First check if content is a base64 image
                    if isinstance(content, str) and (content.startswith("data:image") or len(content) > 1000):
                        images = [{"url": content}]
                    # Otherwise try to extract URL from content
                    else:
                        extracted_url = extract_url_from_content(content)
                        if extracted_url:
                            print(f"  > Found image URL in message content: {extracted_url}")
                            # Download the image
                            png_bytes = download_image_from_url(extracted_url)
                            if png_bytes:
                                # Skip normal image processing since we already have bytes
                                pass
                            else:
                                # Store failed download for later display
                                FAILED_DOWNLOADS.append({"key": key, "url": extracted_url})
                                print(f"  > Failed to download - URL saved for manual download")
                                
            # Google Gemini style: 'candidates' with 'content' parts
            elif "candidates" in resp:
                candidates = resp.get("candidates", [])
                if candidates:
                    # parts list
                    parts = candidates[0].get("content", {}).get("parts", []) or candidates[0].get("content", {}).get("parts", [])
                    for p in parts:
                        # inline_data in many examples (REST uses inline_data), some samples use inlineData (camelCase)
                        inline = p.get("inline_data") or p.get("inlineData")
                        if inline:
                            # mime can be mime_type or mimeType depending on casing
                            mime = inline.get("mime_type") or inline.get("mimeType") or inline.get("mime") or "image/png"
                            data = inline.get("data")
                            if data:
                                # Create a data URL string
                                data_url = f"data:{mime};base64,{data}"
                                images = [{"url": data_url}]
                                break
                        # Some samples may give file_uri references or image_url; try to handle common possibilities
                        if "image" in p:
                            img = p.get("image")
                            # if img inline contains base64
                            if isinstance(img, str) and img.startswith("data:"):
                                images = [{"url": img}]
                                break
                            if isinstance(img, dict):
                                # might contain inline_data
                                inline2 = img.get("inline_data") or img.get("inlineData")
                                if inline2 and inline2.get("data"):
                                    mime = inline2.get("mime_type") or inline2.get("mimeType") or "image/png"
                                    data = inline2.get("data")
                                    images = [{"url": f"data:{mime};base64,{data}"}]
                                    break
                        if "image_url" in p:
                            url = p.get("image_url")
                            # image_url sometimes is an object or string
                            if isinstance(url, str):
                                images = [{"url": url}]
                                break
                            elif isinstance(url, dict):
                                urlstr = url.get("url")
                                if urlstr:
                                    images = [{"url": urlstr}]
                                    break
        except (KeyError, IndexError, Exception) as e:
            print_error_details(exception=e, context="Failed to extract images from response")
            print("Full response structure:")
            print(json.dumps(resp, indent=2))
        
        # If we already have png_bytes from URL extraction, use it
        if png_bytes:
            return png_bytes
        
        # Handle no images returned
        if not images:
            # Check if this was a URL extraction failure (image generated but download failed)
            url_extraction_failed = any(item['key'] == key for item in FAILED_DOWNLOADS)
            
            if url_extraction_failed:
                # Image was generated but download failed, no need to retry API
                print(f"\n[Info] Image URL was found but download failed for '{key}'")
                print(f"  > URL saved for manual download. Skipping API retry...")
                print(f"  > Waiting 10 seconds before continuing to next emotion...")
                time.sleep(10)
                return None
            
            # Otherwise, proceed with normal retry logic for truly missing images
            print("\n[Warning] No images returned for '{}'".format(key))
            print("Full API Response:")
            print(json.dumps(resp, indent=2))
            
            # Auto-retry logic with max 3 auto-retries
            auto_retry_count = 0
            max_auto_retries = 3
            
            while auto_retry_count < max_auto_retries:
                # Ask user if they want to retry (with 20 second timeout, defaults to retry)
                user_response = input_with_timeout(
                    f"  > Do you want to retry? (y/n) [20 sec timeout, defaults to retry, auto-retry {auto_retry_count + 1}/{max_auto_retries}]: ",
                    timeout_seconds=20
                )
                
                # Default to retry if timeout or yes
                if user_response is None or user_response.lower().strip() in ('', 'y', 'yes'):
                    auto_retry_count += 1
                    retry_count += 1
                    if retry_count > max_retries:
                        print(f"[Error] Maximum retries ({max_retries}) reached.")
                        return None
                    print(f"  > Auto-retrying ({auto_retry_count}/{max_auto_retries})...")
                    break  # Break to retry the entire generation
                elif user_response.lower().strip() in ('n', 'no'):
                    print(f"  > Skipping '{key}'.")
                    return None
                else:
                    # Invalid input, ask again
                    continue
            
            # If we've exhausted auto-retries
            if auto_retry_count >= max_auto_retries:
                print(f"  > Maximum auto-retries ({max_auto_retries}) reached. Skipping '{key}'.")
                return None
            
            # Continue to retry
            continue

        # If we got images, break out of retry loop
        break

    # Decode the first returned image
    data_url = images[0].get("image_url", {}).get("url") or images[0].get("url") or images[0].get("b64_json")
    if data_url and not data_url.startswith("data:"):
        data_url = f"data:image/png;base64,{data_url}"
    
    try:
        png_bytes = decode_data_url_to_png_bytes(data_url)
    except Exception as e:
        print_error_details(exception=e, context="Failed to decode image data")
        return None
    
    return png_bytes


def _load_font(size: int) -> ImageFont.ImageFont:
    for p in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/Library/Fonts/Arial.ttf",
        "C:\\Windows\\Fonts\\arial.ttf",
    ]:
        if Path(p).exists():
            try: return ImageFont.truetype(p, size=size)
            except Exception: pass
    return ImageFont.load_default()

def _fit_and_pad(im: Image.Image, target_w: int, target_h: int, bg=(255, 255, 255)) -> Image.Image:
    w, h = im.size
    scale = min(target_w / w, target_h / h)
    new_w = max(1, int(w * scale)); new_h = max(1, int(h * scale))
    im_resized = im.resize((new_w, new_h), Image.LANCZOS)
    canvas = Image.new("RGB", (target_w, target_h), bg)
    off_x = (target_w - new_w) // 2; off_y = (target_h - new_h) // 2
    if im_resized.mode != "RGBA": im_resized = im_resized.convert("RGBA")
    canvas.paste(im_resized, (off_x, off_y), mask=im_resized)
    return canvas

def build_example_grid(emotion_keys: list[str], tile_img_h=512, caption_h=48, gap=24, bg=(255, 255, 255)) -> Path:
    n = len(emotion_keys)
    cols = math.ceil(math.sqrt(n)); rows = math.ceil(n / cols)
    tile_w = tile_img_h; tile_h = tile_img_h + caption_h
    grid_w = cols * tile_w + (cols + 1) * gap
    grid_h = rows * tile_h + (rows + 1) * gap

    grid = Image.new("RGB", (grid_w, grid_h), bg)
    draw = ImageDraw.Draw(grid); font = _load_font(size=24)

    for idx, key in enumerate(emotion_keys):
        img_path = OUT_DIR / f"{key}.png"
        with Image.open(img_path) as im:
            img_area = _fit_and_pad(im, tile_w, tile_img_h, bg=bg)
        r = idx // cols; c = idx % cols
        x0 = gap + c * (tile_w + gap); y0 = gap + r * (tile_h + gap)
        grid.paste(img_area, (x0, y0))
        caption = key
        bbox = draw.textbbox((0, 0), caption, font=font)
        text_w, text_h = bbox[2]-bbox[0], bbox[3]-bbox[1]
        tx = x0 + (tile_w - text_w)//2; ty = y0 + tile_img_h + (caption_h - text_h)//2
        draw.text((tx, ty), caption, fill=(0,0,0), font=font)

    out_path = Path("example_grid.png")
    grid.save(out_path, format="PNG")
    return out_path

def zip_expressions(zip_path: Path):
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        # Exclude original cached files from the zip
        for p in sorted(OUT_DIR.glob("*.png")):
            if not p.name.startswith("orig__"):
                zf.write(p, arcname=p.name)

def print_example_keys_file():
    """Print an example config.ini file format"""
    example = """
# ========================================
# Example config.ini file format:
# ========================================

[config]
# AI Model Parameters (all optional - only include what you want to override)
# temperature = 0.7           # Controls randomness (0.0-2.0)
# top_p = 1.0                # Nucleus sampling (0.0-1.0)
# max_tokens = 1000          # Max tokens to generate
# frequency_penalty = 0      # Reduce repetition (-2.0 to 2.0)
# presence_penalty = 0       # Encourage new topics (-2.0 to 2.0)
# seed = 42                  # For reproducible outputs

# Custom API configuration (optional)
# custom_url = https://api.example.com/v1/chat/completions
# custom_model = gemini-2.5-flash-image-preview

# Default provider (optional) - can be: custom, openrouter, or google
# default_provider = custom

# Background type for prompt (grey or white)
# background_type = grey

# rembg preset (anime, general, pixel, minecraft, real-life, people)
# rembg_preset = anime

# rembg model (overrides preset model if set)
# rembg_model = isnet-anime

[custom]
# Your custom API keys (one per line)
custom-key-xxxxxxxxxxxxx
custom-key-yyyyyyyyyyyyy

[openrouter]
# Your OpenRouter API keys (one per line)
sk-or-v1-xxxxxxxxxxxxx
sk-or-v1-yyyyyyyyyyyyy

[google]
# Your Google (Gemini) API keys (one per line)
# Put the API key that you would normally pass in x-goog-api-key
GOOGLE_API_KEY_XXXXXXXXXXXXXXXXXXXX

# ========================================
"""
    print(example)

def main():
    global KEY_MANAGER, API_PROVIDER, API_URL, API_MODEL, AI_PARAMS, CUSTOM_MODEL, REMBG_MODEL, REMBG_ALPHA_MATTING, REMBG_FG_THR, REMBG_BG_THR, REMBG_ERODE, REMBG_BASE, REMBG_POST_MASK, BACKGROUND_TYPE, FAILED_DOWNLOADS
    
    # --- Choose API Provider ---
    print("="*60)
    print("API Provider Selection")
    print("="*60)
    print("1. Custom OpenAI-compatible API (default)")
    print("2. OpenRouter")
    print("3. Google (Gemini)")
    
    # First, try to load config to see if custom URL is already configured
    config, _ = load_config_and_keys("config.ini", "")
    custom_url_in_config = config.get("custom_url")
    custom_model_in_config = config.get("custom_model")

    # --- Auto-select provider from config (non-interactive) if requested ---
    skip_interactive_selection = False
    default_provider = config.get("default_provider")
    if default_provider:
        dp = str(default_provider).strip().lower()
        if dp in ("custom", "1"):
            # For non-interactive custom, require custom_url + custom_model be present in config
            if custom_url_in_config and custom_model_in_config:
                API_PROVIDER = "custom"
                API_URL = custom_url_in_config
                API_MODEL = custom_model_in_config
                skip_interactive_selection = True
                print(f"\nUsing default provider from config: Custom (URL={API_URL}, model={API_MODEL})")
            else:
                print("\n[config] default_provider=custom requires custom_url and custom_model to be set in [config]. Falling back to interactive selection.")
        elif dp in ("openrouter", "open", "2"):
            API_PROVIDER = "openrouter"
            API_URL = OPENROUTER_URL
            API_MODEL = OPENROUTER_MODEL
            skip_interactive_selection = True
            print(f"\nUsing default provider from config: OpenRouter (model={API_MODEL})")
        elif dp in ("google", "gemini", "3"):
            API_PROVIDER = "google"
            API_MODEL = GOOGLE_MODEL
            API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{API_MODEL}:generateContent"
            skip_interactive_selection = True
            print(f"\nUsing default provider from config: Google Gemini (model={API_MODEL})")
        else:
            print(f"\n[config] Unknown default_provider '{default_provider}'. Falling back to interactive selection.")

    # If not auto-selected, run interactive selection
    if not skip_interactive_selection:
        while True:
            choice = input("\nSelect API provider [1/2/3] (default: 1): ").strip()
            if choice == "" or choice == "1":
                API_PROVIDER = "custom"
                
                # Check if custom URL is in config
                if custom_url_in_config:
                    print(f"\nFound custom API configuration in config.ini:")
                    print(f"  URL: {custom_url_in_config}")
                    if custom_model_in_config:
                        print(f"  Model: {custom_model_in_config}")
                    
                    use_config = input("\nUse this configuration? [y/n] (default: y): ").strip().lower()
                    if use_config in ['', 'y', 'yes']:
                        API_URL = custom_url_in_config
                        API_MODEL = custom_model_in_config
                        if not API_MODEL:
                            API_MODEL = input("Enter the model name to use: ").strip()
                            if not API_MODEL:
                                print("[Error] Model name is required.")
                                continue
                    else:
                        # Ask for new URL and model
                        API_URL = input("Enter the custom API URL (e.g., https://api.example.com/v1/chat/completions): ").strip()
                        if not API_URL:
                            print("[Error] API URL is required.")
                            continue
                        API_MODEL = input("Enter the model name to use: ").strip()
                        if not API_MODEL:
                            print("[Error] Model name is required.")
                            continue
                else:
                    # No config, ask for URL and model
                    print("\nNo custom API configuration found in config.ini.")
                    API_URL = input("Enter the custom API URL (e.g., https://api.example.com/v1/chat/completions): ").strip()
                    if not API_URL:
                        print("[Error] API URL is required.")
                        continue
                    API_MODEL = input("Enter the model name to use: ").strip()
                    if not API_MODEL:
                        print("[Error] Model name is required.")
                        continue
                
                print(f"\nUsing Custom API")
                print(f"  URL: {API_URL}")
                print(f"  Model: {API_MODEL}")
                break
            elif choice == "2":
                API_PROVIDER = "openrouter"
                API_URL = OPENROUTER_URL
                API_MODEL = OPENROUTER_MODEL
                print(f"\nUsing OpenRouter API")
                print(f"  URL: {API_URL}")
                print(f"  Model: {API_MODEL}")
                break
            elif choice == "3":
                API_PROVIDER = "google"
                # Use GOOGLE_MODEL or prompt
                API_MODEL = GOOGLE_MODEL
                # Build endpoint URL
                API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{API_MODEL}:generateContent"
                print(f"\nUsing Google Gemini API (REST generateContent)")
                print(f"  URL: {API_URL}")
                print(f"  Model: {API_MODEL}")
                break
            else:
                print("Invalid choice. Please enter 1, 2 or 3.")
    
    # --- Load Configuration and API Keys from file ---
    print(f"\nLoading configuration and {API_PROVIDER} API keys from config.ini...")
    config, api_keys = load_config_and_keys("config.ini", API_PROVIDER)
    
    # Update AI parameters with loaded config (excluding non-AI params)
    if config:
        print("\nLoaded configuration parameters:")
        excluded_keys = [
            'custom_url', 'custom_model', 'default_provider',
            'background_type', 'rembg_preset', 'rembg_model',
            'rembg_alpha_matting', 'rembg_fg_thr', 'rembg_bg_thr',
            'rembg_erode', 'rembg_base', 'rembg_post_mask'
        ]
        for key, value in config.items():
            if key not in excluded_keys:
                print(f"  {key}: {value}")
                if value is not None:  # Only add non-None values
                    AI_PARAMS[key] = value
    
    if not AI_PARAMS:
        print("\nNo AI parameters configured in config.ini")
    
    if not api_keys:
        print(f"\n[Error] No {API_PROVIDER} API keys found in config.ini!")
        print_example_keys_file()
        raise SystemExit(f"No {API_PROVIDER} API keys found in config.ini")
    
    print(f"\nLoaded {len(api_keys)} {API_PROVIDER} API key(s) from config.ini")
    KEY_MANAGER = KeyManager(api_keys, API_PROVIDER)

    # --- Set background type from config ---
    BACKGROUND_TYPE = str(config.get('background_type', 'grey')).lower()

    # --- Set rembg presets and overrides ---
    REMBG_PRESETS = {
        "anime": {
            "model": "isnet-anime",
            "alpha_matting": True,
            "alpha_matting_foreground_threshold": 180,
            "alpha_matting_background_threshold": 30,
            "alpha_matting_erode_size": 1,
            "alpha_matting_base_size": 2048,
            "post_process_mask": True
        },
        "general": {
            "model": "isnet-general-use",
            "alpha_matting": True,
            "alpha_matting_foreground_threshold": 240,
            "alpha_matting_background_threshold": 10,
            "alpha_matting_erode_size": 4,
            "alpha_matting_base_size": 1024,
            "post_process_mask": True
        },
        "pixel": {
            "model": "silueta",
            "alpha_matting": False,
            "alpha_matting_foreground_threshold": 255,
            "alpha_matting_background_threshold": 0,
            "alpha_matting_erode_size": 0,
            "alpha_matting_base_size": 1024,
            "post_process_mask": False
        },
        "minecraft": {
            "model": "u2netp",
            "alpha_matting": False,
            "alpha_matting_foreground_threshold": 200,
            "alpha_matting_background_threshold": 50,
            "alpha_matting_erode_size": 2,
            "alpha_matting_base_size": 1024,
            "post_process_mask": True
        },
        "real-life": {
            "model": "u2net",
            "alpha_matting": True,
            "alpha_matting_foreground_threshold": 240,
            "alpha_matting_background_threshold": 10,
            "alpha_matting_erode_size": 10,
            "alpha_matting_base_size": 1416,
            "post_process_mask": True
        },
        "people": {
            "model": "u2net_human_seg",
            "alpha_matting": True,
            "alpha_matting_foreground_threshold": 240,
            "alpha_matting_background_threshold": 10,
            "alpha_matting_erode_size": 10,
            "alpha_matting_base_size": 1416,
            "post_process_mask": True
        },
    }

    if 'rembg_preset' in config:
        preset_name = str(config['rembg_preset']).lower()
        if preset_name in REMBG_PRESETS:
            preset = REMBG_PRESETS[preset_name]
            REMBG_MODEL = preset.get('model', REMBG_MODEL)
            REMBG_ALPHA_MATTING = preset.get('alpha_matting', REMBG_ALPHA_MATTING)
            REMBG_FG_THR = preset.get('alpha_matting_foreground_threshold', REMBG_FG_THR)
            REMBG_BG_THR = preset.get('alpha_matting_background_threshold', REMBG_BG_THR)
            REMBG_ERODE = preset.get('alpha_matting_erode_size', REMBG_ERODE)
            REMBG_BASE = preset.get('alpha_matting_base_size', REMBG_BASE)
            REMBG_POST_MASK = preset.get('post_process_mask', REMBG_POST_MASK)
        else:
            print(f"[Warning] Unknown rembg_preset '{preset_name}'. Using defaults.")

    # Override with specific rembg configs if set
    if 'rembg_model' in config:
        REMBG_MODEL = str(config['rembg_model'])
    if 'rembg_alpha_matting' in config:
        REMBG_ALPHA_MATTING = bool(config['rembg_alpha_matting'])
    if 'rembg_fg_thr' in config:
        REMBG_FG_THR = int(config['rembg_fg_thr'])
    if 'rembg_bg_thr' in config:
        REMBG_BG_THR = int(config['rembg_bg_thr'])
    if 'rembg_erode' in config:
        REMBG_ERODE = int(config['rembg_erode'])
    if 'rembg_base' in config:
        REMBG_BASE = int(config['rembg_base'])
    if 'rembg_post_mask' in config:
        REMBG_POST_MASK = bool(config['rembg_post_mask'])

    # --- Confirmation for background removal (with 10 second timeout, defaults to y) ---
    do_bg_removal = True  # Default to yes
    choice = input_with_timeout("\nRemove background from generated images? [y/n] (10 sec timeout, defaults to y): ", timeout_seconds=10)
    
    if choice is None:
        # Timeout occurred, use default
        print("Defaulting to: yes")
        do_bg_removal = True
    elif choice.lower().strip() in ('n', 'no'):
        do_bg_removal = False
    else:
        # Any other input (including 'y', 'yes', or empty) = yes
        do_bg_removal = True
    
    if do_bg_removal and not HAVE_REMBG:
        print("[Warning] rembg is not installed. Background removal will use a low-quality fallback.")

    # --- Optional input for custom prompt tweaks (with 10 second timeout, defaults to n) ---
    global_tweak = ""
    specific_tweaks = {}
    
    choice = input_with_timeout("Add custom tweaks/considerations to the prompt? [y/n] (10 sec timeout, defaults to n): ", timeout_seconds=10)
    
    if choice is None:
        # Timeout occurred, use default (no)
        print("Defaulting to: no")
    elif choice.lower().strip() in ('y', 'yes'):
        tweak_text = input("Enter the tweak text (e.g., 'wearing a blue hat'): ")
        while True:
            apply_to = input("Apply to [a]ll expressions or [s]pecific ones? [a/s]: ").lower().strip()
            if apply_to == 'a':
                global_tweak = tweak_text
                break
            elif apply_to == 's':
                keys_str = input("Enter expression keys, separated by commas (e.g., joy,anger): ")
                keys = [k.strip() for k in keys_str.split(',') if k.strip()]
                for key in keys:
                    if key in EMOTIONS:
                        specific_tweaks[key] = tweak_text
                    else:
                        print(f"[Warning] Key '{key}' not found in EMOTIONS list. Ignoring.")
                break
            else:
                print("Invalid input. Please enter 'a' or 's'.")

    # 1) Generate any missing images
    for key, desc in EMOTIONS.items():
        first_path = OUT_DIR / f"{key}.png"
        if first_path.exists():
            print(f"Skipping '{key}' (already has {first_path.name})")
            continue

        # Determine the tweak for the current expression
        tweak_for_current = global_tweak
        if key in specific_tweaks:
             tweak_for_current = f"{global_tweak} {specific_tweaks[key]}".strip()

        print(f"Generating: {key} ({desc}) …")
        if tweak_for_current:
            print(f"  > with custom tweak: '{tweak_for_current}'")

        paths = generate_for_emotion(key, desc, do_bg_removal, tweak_for_current)
        for p in paths: print(f"  saved {p}")

    # 2) Verify all required images exist
    missing = [k for k in EMOTIONS.keys() if not (OUT_DIR / f"{k}.png").exists()]
    if missing:
        print("\nNot building grid/zip — missing first images for:", ", ".join(missing))
    
    print(f"[INFO] Successfully generated {len(EMOTIONS) - len(missing)}/{len(EMOTIONS)} images")
    
    # Display failed downloads if any
    if FAILED_DOWNLOADS:
        print("\n" + "="*60)
        print("FAILED IMAGE DOWNLOADS - Manual Download Required")
        print("="*60)
        print("The following URLs were extracted but could not be downloaded automatically.")
        print("Please click on these URLs to download the images manually:\n")
        for item in FAILED_DOWNLOADS:
            print(f"  [{item['key']}]")
            print(f"    URL: {item['url']}\n")
        print("="*60 + "\n")
    
    if missing:
        return

    # 3) Build and save example grid
    print("\nAll base images present. Building example_grid.png …")
    grid_path = build_example_grid(list(EMOTIONS.keys()))
    print(f"  saved {grid_path}")

    # 4) Zip all images in expressions/
    zip_path = Path("expressions.zip")
    print("Creating expressions.zip …")
    zip_expressions(zip_path)
    print(f"  saved {zip_path}")
    print("\nDone.")

if __name__ == "__main__":
    main()