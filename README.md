# Expression Generator

Automatically generate multiple emotional expressions from a single character image using AI image generation APIs.

![Example Grid](example_grid.png)

## Features

- Generate 30 unique emotional expressions from one input image
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
# Choose your default provider (optional)
default_provider = openrouter

# Background type for generation (grey or white)
background_type = grey

# Rembg preset for background removal
# Options: anime, general, pixel, minecraft, real-life, people
rembg_preset = anime

# AI parameters (optional)
# temperature = 0.7
# top_p = 1.0
# max_tokens = 1000

[openrouter]
# Add your OpenRouter API keys (one per line)
sk-or-v1-xxxxxxxxxxxxxxxxxxxxxxxxxxxxx
sk-or-v1-yyyyyyyyyyyyyyyyyyyyyyyyyyyyyy

[google]
# Add your Google Gemini API keys (one per line)
AIzaSyXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

[custom]
# For custom OpenAI-compatible APIs
# Also add to [config] section:
# custom_url = https://api.example.com/v1/chat/completions
# custom_model = gemini-2.5-flash-image-preview
your-custom-api-key-here
```

### Step 2: Prepare Input Image

Place your character image in the same directory and name it `input.png` (or `.jpg`, `.jpeg`, `.webp`, etc.)

#### Best Practices for Input Image

For optimal results, your input image should have:

- **Neutral expression** - A calm, neutral facial expression works best as the base
- **Neutral pose** - Not overly posed or dynamic; a natural, relaxed stance

**Important**: This input image sets the style and quality for all generated expressions. The core image should be really clear and really accurate to the character, as all other expressions will follow this style. Make sure you're happy with how it looks before proceeding.

Some users prefer to use an A-pose for the basic expression, while others prefer something a little more in-character to start from. The choice of core pose is up to you - what matters most is that it's clear, accurate to your character, and represents the style you want all other expressions to follow. Having a good core expression will result in better quality expressions overall.

### Step 3: Run the Script

```bash
python expression_generator.py
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

Ensure you have an image file named `input.[extension]` in the same directory.

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

1. Start with a clear, well-lit base image with neutral expression and pose
2. Your input image sets the style for all expressions - make sure it's high quality
3. Use multiple API keys to avoid rate limiting
4. Experiment with custom prompt tweaks for specific styles
5. Delete `expressions/orig__*.png` files to regenerate specific emotions
6. Choose background type (grey/white) based on your use case
7. If background removal isn't clean, use the `orig__*.png` files for manual editing
