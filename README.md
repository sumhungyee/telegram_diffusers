# telegram_diffusers

This project is a spiritual successor of an older project: Telegram_ai_gptq.

## Installation Guide

### Step 1
Ensure your computer has a dedicated GPU with at least 8GB of VRAM and ensure CUDA Toolkit (>=11.8) is installed. Also ensure that you have Docker installed.

### Step 2
Navigate to a specific location on your computer, and clone the repository.

```
git clone https://github.com/sumhungyee/telegram_ai.git
cp sample.env .env
```

Fill in the parameters in .env.

### Step 3
Navigate into the repository at `/telegram_diffusers/src/models` and download a `.safetensors` file of any SDXL model of your choice.
Finally, build the image with

```
docker-compose up --build
```
## User Guide

### Running the application

```
docker-compose up
```

## Available commands

1. /help - Shows a list of arguments for image generation.
2. /generate - Generates an image.
    - Available arguments:
        1. --prompt, or -p (mandatory). Expected type is a string. Use quotation marks (' or ").
        2. --negprompt, or -n (optional), defaults to empty.
        3. --orientation, or -o (optional), defaults to square image. Available: landscape, portrait, square (lowercase)
        4. --steps, or -s (optional), defaults to 60.
        5. Example usage: /generate -p 'A big chocolate bar' --negprompt "monochrome" -s 50