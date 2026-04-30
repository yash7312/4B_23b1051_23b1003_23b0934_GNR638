import os
import clip

model_path = "models"
if not os.path.exists(model_path):
    os.makedirs(model_path)

print(f"Downloading CLIP model to {model_path}...")
clip.load("ViT-B/32", download_root=model_path)
print("Done")