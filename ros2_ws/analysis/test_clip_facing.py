#!/usr/bin/env python3
"""
Standalone MobileCLIP-S1 facing-classification test.
Not part of the ROS pipeline - run directly against a saved image to
check what the model reads, without restarting group_formation_detector.

Usage:
    python3 test_clip_facing.py /path/to/image.png
"""

import sys
import open_clip
import torch
from PIL import Image

if len(sys.argv) != 2:
    print("Usage: python3 test_clip_facing.py <image_path>")
    sys.exit(1)

image_path = sys.argv[1]

print("Loading MobileCLIP-S1...")
model, _, preprocess = open_clip.create_model_and_transforms(
    'MobileCLIP-S1', pretrained='datacompdr'
)
model.eval()
tokenizer = open_clip.get_tokenizer('MobileCLIP-S1')

# Same prompt pair currently live in group_formation_detector.py
prompts = [
    "conversation",
    "queue",
]

text_tokens = tokenizer(prompts)
with torch.no_grad():
    text_features = model.encode_text(text_tokens)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

pil_image = Image.open(image_path).convert("RGB")
image_input = preprocess(pil_image).unsqueeze(0)

with torch.no_grad():
    image_features = model.encode_image(image_input)
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    similarities = (100.0 * image_features @ text_features.T).softmax(dim=-1)

print()
for i, prompt in enumerate(prompts):
    print(f"  [{i}] '{prompt}' = {float(similarities[0, i]):.3f}")

best_idx = int(similarities.argmax())
best_score = float(similarities[0, best_idx])
print()
print(f"classify_facing result: best='{prompts[best_idx]}' score={best_score:.3f}")
