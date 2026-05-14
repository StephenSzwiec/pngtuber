# PNGTuber 

Rigging a PNGTuber with MediaPipe and OpenCV for automatic expression tracking and animation.

## Install 

```bash
# install uv if not already installed
if ! command -v uv &> /dev/null
then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
# install 
uv sync
```

## Usage

Debug mode with webcam input, but no sprites:

```bash
pngtuber --sprites ./nonexistant --debug
```

This will print the detected facial landmarks to the console, allowing you to verify that the tracking is working correctly.
