#!/usr/bin/env python3 

## generate square png images with kaomoji in the center, with transparent background 
## uses the following:
## - poses: neutral, hand_on_chin, hands_over_head 
## - emotion: neutral, happy, angry 
## - blink: blink, noblink 
## - mouth: quiet, yap 
## giving a png in the format {pose}__{emotion}__{blink}_{mouth}.png

import os
from PIL import Image, ImageDraw, ImageFont
# define the dictionary 
poses = ['neutral', 'hand_on_chin', 'hands_over_head']
emotions = ['neutral', 'happy', 'angry']
blinks = ['blink', 'noblink']
mouths = ['quiet', 'yap']
# components for the kaomoji
## example ヽ(`Д´)ﾉ - pose: hands_over_head, emotion: angry, blink: blink, mouth: yap
components = {
    'neutral': {
        'eyes': [ '-', '・'],
        'mouth': ['＿', 'o']
    },
    'happy': {
        'eyes': ['＾', '⌒'],
        'mouth': ['ω', '▽']
    },
    'angry': {
        'eyes': ['°', '｀'],
        'mouth': ['Д', 'д']
    }
}

def generate_kaomoji(pose, emotion, blink, mouth):
    eye = components[emotion]['eyes'][0] if blink == 'blink' else components[emotion]['eyes'][1]
    mouth_char = components[emotion]['mouth'][0] if mouth == 'quiet' else components[emotion]['mouth'][1]
    if pose == 'neutral':
        return f"({eye}{mouth_char}{eye})"
    elif pose == 'hand_on_chin':
        return f"({eye}{mouth_char}{eye})ゝ"
    elif pose == 'hands_over_head':
        return f"ヽ({eye}{mouth_char}{eye})ﾉ"

# create output directory if it doesn't exist
output_dir = 'kaomoji_images'
os.makedirs(output_dir, exist_ok=True)
# generate images
for pose in poses:
    for emotion in emotions:
        for blink in blinks:
            for mouth in mouths:
                kaomoji = generate_kaomoji(pose, emotion, blink, mouth)
                # create image
                img = Image.new('RGBA', (800,800), (255, 255, 255, 0))
                font = ImageFont.truetype("Arial Unicode.ttf", 120)
                draw = ImageDraw.Draw(img)
                bbox = draw.textbbox((0,0), kaomoji, font=font)
                w = bbox[2] - bbox[0]
                h = bbox[3] - bbox[1]
                x = (200 + w) // 2
                y = (200 + h) // 2
                # center it homie
                draw.text((x, y), kaomoji, font=font, fill=(0, 0, 0, 255), anchor='mm') 
                # save image
                filename = f"{pose}__{emotion}__{blink}_{mouth}.png"
                img.save(os.path.join(output_dir, filename))

