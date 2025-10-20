# Expression Generator

Automatically generate multiple emotional expressions from one or more character images using AI image generation APIs.

![Example Grid](example_grid.png)

## Features

- Generate 30 unique emotional expressions from input image(s)
- **Support for multiple reference images** - provide additional angles, poses, or style references
- Support for OpenRouter, Google Gemini, and custom OpenAI-compatible APIs
- Smart API key rotation to handle rate limits
- Automatic background removal with rembg
- Intelligent caching system to avoid regenerating images
- Auto-retry logic with timeout handling
- Automatic grid creation and ZIP export

## Requirements

- Python 3.13.7
- API key for one of the supported providers:
  - OpenRouter
  - Google Gemini
  - Any OpenAI-compatible API

## Installation

### Using UV (Recommended)

```bash
# Install UV
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the repository
git clone https://github.com/zaxx-q/Expression-generator.git
cd Expression-generator

# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -r requirements.txt

# Run the script
python Expression-generator.py
```

### Using Standard venv

```bash
# Clone the repository
git clone https://github.com/zaxx-q/Expression-generator.git
cd Expression-generator

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the script
python Expression-generator.py
```

## Configuration

### Step 1: Create config.ini

Create a `config.ini` file in the same directory as the script:

```ini
[config]
# -- AI Model Parameters (all optional - only include what you want to override) --
# Controls randomness (0.0-2.0)
# temperature = 0.7
# Nucleus sampling (0.0-1.0)
# top_p = 1.0
# Max tokens to generate
# max_tokens = 1000
# Reduce repetition (-2.0 to 2.0)
# frequency_penalty = 0
# Encourage new topics (-2.0 to 2.0)
# presence_penalty = 0
# For reproducible outputs
# seed = 42

# If you want to use a custom OpenAI-compatible API endpoint, set:
# custom_url = "https://api.example.com/v1/chat/completions"
custom_model = "gemini-2.5-flash-image-preview"

# If you prefer to use a non-interactive provider selection in your script,
# you can add a provider key here (optional):
# values: openrouter | google | custom
# default_provider = "custom"

# Background type for prompt (grey or white)
background_type = grey

# rembg preset (anime, general, pixel, minecraft, real-life, people)
rembg_preset = anime

# rembg model (overrides preset model if set)
# rembg_model = isnet-anime


# ============================

[custom]
# Custom API keys (one per line) for a user-provided OpenAI-compatible endpoint.
# The script will send these as Authorization: Bearer <key>.
sk-....


[openrouter]
# OpenRouter API keys (one per line). These are the Bearer tokens used in Authorization: Bearer <key>
sk-....

[google]
# Google (Gemini) API keys (one per line).
# These are the simple API keys passed in the x-goog-api-key header for the REST generateContent endpoint.
sk-....
```

### Step 2: Prepare Input Image(s)

Place your character image(s) in the same directory using the following naming pattern:

#### Single Image

```
input0.png
```

#### Multiple Images

```
input0.png     # Primary/main reference
input1.png     # Additional reference
input2.png     # Additional reference
input3.png     # Additional reference
...
```

**Supported extensions**: `png`, `jpg`, `jpeg`, `webp`, `bmp`, `gif`, `tiff`, `tif`

**Note**: Images must be numbered sequentially starting from 0. The script will automatically detect and use all numbered input files found.

#### Best Practices for Input Images

For optimal results, your primary input image (`input0.png`) should have:

- **Neutral expression** - A calm, neutral facial expression works best as the base
- **Neutral pose** - Not overly posed or dynamic; a natural, relaxed stance

**Important**: The input image sets the style and quality for all generated expressions. The core image should be really clear and really accurate to the character, as all other expressions will follow this style. Make sure you're happy with how it looks before proceeding.

Some users prefer to use an A-pose for the basic expression, while others prefer something a little more in-character to start from. The choice of core pose is up to you - what matters most is that it's clear, accurate to your character, and represents the style you want all other expressions to follow. Having a good core expression will result in better quality expressions overall.

#### Using Multiple Reference Images

When providing multiple images:

- **First image** (`input0.png`) should be your primary reference with neutral expression
- **Additional images** (`input1.png`, `input2.png`, etc.) can show:
  - Different angles (front view, side view, 3/4 view)
  - Different poses to demonstrate character proportions
  - Close-ups of facial features or details
  - Style reference images
  - Different lighting conditions

**Example use cases:**
- Provide a front view + side view for better 3D understanding
- Include a full-body shot + close-up portrait for detail
- Add style reference images to maintain consistent art style
- Show character with different accessories or outfits that should be preserved

The AI will use all provided images as reference to better understand your character's design, style, and details.

### Step 3: Run the Script

```bash
python expression_generator.py
```

The script will automatically detect and display all input images found:

```
Found 3 input image(s):
  [0] input0.png - 1024x1024 pixels (256.3 KB)
  [1] input1.png - 1024x1024 pixels (248.7 KB)
  [2] input2.png - 512x768 pixels (128.5 KB)
```

## Output Structure

```
expressions/
  ├── orig__admiration.png  # Original generated images (with background)
  ├── orig__joy.png
  ├── admiration.png        # Background removed versions
  ├── joy.png
  ├── anger.png
  ├── love.png
  └── ... (30 emotions total)
example_grid.png      # Visual grid of all expressions
expressions.zip       # All expressions packaged
```

## Supported Emotions

The script generates 30 different emotional expressions:

- admiration, amusement, anger, annoyance, approval
- caring, confusion, curiosity
- desire-0, desire-1, desire-2
- disappointment, disapproval, disgust
- embarrassment, excitement
- fear
- gratitude, grief
- joy
- love
- nervousness, neutral
- optimism
- pride
- realization, relief, remorse
- sadness, surprise

## Advanced Usage

### Custom Prompt Tweaks

Add custom instructions when prompted:

```
Add custom tweaks/considerations to the prompt? [y/n]: y
Enter the tweak text: wearing a blue hat
Apply to [a]ll expressions or [s]pecific ones? [a/s]: a
```

### Background Removal Presets

Available presets in `config.ini`:

- `anime` - Best for anime/cartoon characters
- `general` - General purpose
- `pixel` - For pixel art
- `minecraft` - For Minecraft-style images
- `real-life` - For realistic photos
- `people` - Optimized for human subjects

### Multiple API Keys

Add multiple keys to handle rate limits:

```ini
[openrouter]
sk-or-v1-key1-xxxxxxxxxxxxx
sk-or-v1-key2-yyyyyyyyyyyyy
sk-or-v1-key3-zzzzzzzzzzzzz
```

## Troubleshooting

### No input file found

Ensure you have at least one image file named `input0.[extension]` in the same directory. For multiple images, use sequential numbering: `input0.png`, `input1.png`, `input2.png`, etc.

Supported extensions: `png`, `jpg`, `jpeg`, `webp`, `bmp`, `gif`, `tiff`, `tif`

### No API keys found

Create a `config.ini` file with your API keys in the appropriate section.

### Background removal issues

Try different `rembg_preset` values:
- Use `anime` for cartoon/anime characters
- Use `general` for most other images
- Use `people` for realistic human portraits

Note that rembg has inherent limitations and may not produce perfectly clean results even after tweaking presets and parameters. If the automatic background removal is not satisfactory, you can use the `orig__*.png` files (stored in the `expressions/` folder) and manually remove the background using other tools like Photoshop, GIMP, or online background removal services.

### Rate limit errors

Add more API keys to your config.ini, or use keyboard shortcuts (`s` or `r`) to control key rotation.

## Tips

1. Start with a clear, well-lit base image with neutral expression and pose as `input0.png`
2. Your input images sets the style for all expressions - make sure it's high quality
3. When using multiple images, ensure they're all of the same character in consistent style
4. Number your images sequentially starting from 0 (input0, input1, input2, ...)
5. Use multiple API keys to avoid rate limiting
6. Experiment with custom prompt tweaks for specific styles
7. Delete `expressions/orig__*.png` files to regenerate specific emotions
8. Choose background type (grey/white) based on your use case
9. If background removal isn't clean, use the `orig__*.png` files for manual editing