import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import clip
from PIL import Image

# ======================
# CONFIG
# ======================
PATCH_DIR = "patches"
TEST_CSV = "test.csv"
OUTPUT_CSV = "submission.csv"
MODEL_PATH = "models"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OVERLAP = 30  # Adjust based on dataset characteristics

# ======================
# LOAD CLIP (LOCAL PREFERENCE)
# ======================
print("Loading CLIP...")
model, preprocess = clip.load("ViT-B/32", device=DEVICE, download_root=MODEL_PATH)
model.eval()

def load_patches(folder):
    patches = {}
    for f in os.listdir(folder):
        if f.endswith(".png"):
            idx = int(f.split("_")[1].split(".")[0])
            patches[idx] = cv2.imread(os.path.join(folder, f))
    return patches

def rotations(img):
    return [img, cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE), 
            cv2.rotate(img, cv2.ROTATE_180), 
            cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)]

def get_score(img_a, img_b, direction):
    best_match = -1.0
    # Search for the "lock" within a small range
    for offset in range(OVERLAP - 2, OVERLAP + 3):
        if direction == "right":
            edge_a = img_a[:, -offset:]
            edge_b = img_b[:, :offset]
        else:
            edge_a = img_a[-offset:, :]
            edge_b = img_b[:offset, :]
        
        res = cv2.matchTemplate(edge_a, edge_b, cv2.TM_CCORR_NORMED)
        best_match = max(best_match, res[0][0])
    return best_match

def build_grid(patches):
    keys = list(patches.keys())
    size = int(np.sqrt(len(keys)))
    grid = [[None]*size for _ in range(size)]
    used = {0}
    
    # patch_0.png is always top-left
    grid[0][0] = (0, patches[0])

    print("Stitching image...")
    for i in range(size):
        for j in range(size):
            if i == 0 and j == 0: continue

            best_score = -float('inf')
            best_choice = None

            for idx in keys:
                if idx in used: continue
                for rot in rotations(patches[idx]):
                    score = 0
                    if j > 0: score += get_score(grid[i][j-1][1], rot, "right")
                    if i > 0: score += get_score(grid[i-1][j][1], rot, "bottom")
                    
                    if score > best_score:
                        best_score = score
                        best_choice = (idx, rot)
            
            grid[i][j] = best_choice
            used.add(best_choice[0])
    return grid

def stitch(grid):
    size = len(grid)
    patch_h, patch_w, _ = grid[0][0][1].shape
    
    # Calculate new dimensions: First patch is full size, others contribute (size - overlap)
    full_h = patch_h + (size - 1) * (patch_h - OVERLAP)
    full_w = patch_w + (size - 1) * (patch_w - OVERLAP)
    
    canvas = np.zeros((full_h, full_w, 3), dtype=np.uint8)

    for i in range(size):
        for j in range(size):
            patch = grid[i][j][1]
            y_start = i * (patch_h - OVERLAP)
            x_start = j * (patch_w - OVERLAP)
            
            # Simple overwrite or Alpha Blending could be used here
            canvas[y_start:y_start+patch_h, x_start:x_start+patch_w] = patch
    return canvas

def answer_questions(full_map):
    df = pd.read_csv(TEST_CSV)
    results = []
    
    # Basic Informative patches: center and 4 corners
    h, w, _ = full_map.shape
    views = [
        full_map, # Global
        full_map[h//4:3*h//4, w//4:3*w//4], # Center
        full_map[:h//2, :w//2], # Top-Left
        full_map[:h//2, w//2:], # Top-Right
        full_map[h//2:, :w//2], # Bottom-Left
        full_map[h//2:, w//2:]  # Bottom-Right
    ]

    for _, row in tqdm(df.iterrows(), total=len(df)):
        options = [row[f"option_{i}"] for i in range(1, 5)]
        texts = clip.tokenize([f"{row['question']} {opt}" for opt in options]).to(DEVICE)
        
        best_prob = 0
        final_ans = 5 # Default to 'unanswered' to avoid negative marks[cite: 3]

        for view in views:
            img_input = preprocess(Image.fromarray(cv2.cvtColor(view, cv2.COLOR_BGR2RGB))).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                logits, _ = model(img_input, texts)
                probs = logits.softmax(dim=-1).cpu().numpy()[0]
            
            if np.max(probs) > best_prob:
                best_prob = np.max(probs)
                if best_prob > 0.4: # Confidence Threshold
                    final_ans = np.argmax(probs) + 1

        results.append({"id": row["id"], "question_num": row["id"], "option": final_ans})
    
    pd.DataFrame(results).to_csv(OUTPUT_CSV, index=False)

if __name__ == "__main__":
    p = load_patches(PATCH_DIR)
    g = build_grid(p)
    m = stitch(g)
    cv2.imwrite("reconstructed_map.png", m)
    answer_questions(m)