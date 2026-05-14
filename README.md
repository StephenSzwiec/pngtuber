# PNGTuber 

Rigging a PNGTuber with MediaPipe and OpenCV for automatic expression tracking and animation. 

## Install 

```bash 
# install uv if not present 
if ! command -v uv &> /dev/null
then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
# install 
uv sync
``` 

## Usage 

For now, you can get debug output with placeholder sprites using: 
```bash
mkdir sprites
pngtuber --sprites ./nonexistant --debug
```
This will print out the detected facial landmarks and their corresponding sprite names.

