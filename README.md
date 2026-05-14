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
source .venv/bin/activate
pngtuber --sprites ./nonexistant --debug
```
This will print out the detected facial landmarks and their corresponding sprite names.

### Placeholders for Sprites 

#### Populating a sprite directory 

```bash 
mkdir sprites && cd sprites 
for i in neutral hands_over_head hands_on_chin; do
  for j in neutral happy angry; do
    for k in blink_quiet blink_yap noblink_quiet noblink_yap; do
      touch ${i}__${j}__${k}.png
    done
  done
done
``` 
You will need to replace these placeholder sprites with actual images that correspond to the detected expressions and poses for your PNGTuber. 0px sized sprites will lead to an error, so make sure to use valid images. 

#### Using kaomoji as sprites 

Very basic programmer art for testing.

```bash 
python ./kaomoji_to_png.py 
pngtuber --sprites ./kaomoji_images --debug
```
