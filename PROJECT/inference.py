import os
import sys
import cv2
import numpy as np
import pandas as pd
import re
from collections import Counter
from tqdm import tqdm
import torch
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import argparse   

sys.setrecursionlimit(2000)

# ======================
# CONFIG
# ======================
PATCH_DIR = "patches"
TEST_CSV = "test.csv"
OUTPUT_CSV = "submission.csv"
MODEL_PATH = "models/qwen"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ======================
# LOAD MODEL
# ======================
print("Loading Qwen2-VL weights...")
model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2", 
    device_map="auto"
)
model.eval()

processor = AutoProcessor.from_pretrained(MODEL_PATH)

# ======================
# PATCH LOADING
# ======================
def load_patches(folder):
    patches = {}
    for f in os.listdir(folder):
        if f.endswith(".png"):
            idx = int(f.split("_")[1].split(".")[0])
            patches[idx] = cv2.imread(os.path.join(folder, f))
    return patches

def get_rotations(img):
    return [
        (0, img), 
        (1, cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)), 
        (2, cv2.rotate(img, cv2.ROTATE_180)), 
        (3, cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE))
    ]

# ======================
# ROBUST MATCHING (FIXED)
# ======================
score_cache = {}

def get_score(idx_a, rot_idx_a, img_a, idx_b, rot_idx_b, img_b, direction):
    key = (idx_a, rot_idx_a, idx_b, rot_idx_b, direction)
    if key in score_cache:
        return score_cache[key]

    best_mse = float('inf')

    for offset in range(24, 65):  # TA RANGE
        if direction == "right":
            edge_a = img_a[:, -offset:]
            edge_b = img_b[:, :offset]
        else:
            edge_a = img_a[-offset:, :]
            edge_b = img_b[:offset, :]

        if edge_a.shape == edge_b.shape and edge_a.size > 0:
            mse = np.mean(
                (edge_a.astype(np.float32) - edge_b.astype(np.float32)) ** 2
            )
            # mse = mse / offset  # normalize
            best_mse = min(best_mse, mse)

    score_cache[key] = best_mse
    return best_mse

# ======================
# BACKTRACKING SOLVER
# ======================
def build_grid_backtracking(patches):
    keys = list(patches.keys())
    size = int(np.sqrt(len(keys)))
    grid = [[None]*size for _ in range(size)]
    
    oriented = {idx: get_rotations(patches[idx]) for idx in keys}
        
    grid[0][0] = (0, 0, patches[0])
    used = {0}
    
    pbar = tqdm(total=size * size - 1, desc="Solving Jigsaw")

    def get_valid_candidates(i, j):
        candidates = []
        for idx in keys:
            if idx in used:
                continue
            for rot_idx, rot_img in oriented[idx]:
                valid = True
                total_mse = 0

                if j > 0 and grid[i][j-1] is not None:
                    prev = grid[i][j-1]
                    mse = get_score(prev[0], prev[1], prev[2], idx, rot_idx, rot_img, "right")
                    if mse > 80:
                        valid = False
                    total_mse += mse

                if valid and i > 0 and grid[i-1][j] is not None:
                    prev = grid[i-1][j]
                    mse = get_score(prev[0], prev[1], prev[2], idx, rot_idx, rot_img, "bottom")
                    if mse > 80:
                        valid = False
                    total_mse += mse

                if valid:
                    candidates.append((total_mse, idx, rot_idx, rot_img))

        candidates.sort(key=lambda x: x[0])
        return [(c[1], c[2], c[3]) for c in candidates[:4]]

    def solve(count):
        if count == size * size:
            return True

        for i in range(size):
            for j in range(size):
                if grid[i][j] is None:
                    for idx, rot_idx, rot_img in get_valid_candidates(i, j):
                        grid[i][j] = (idx, rot_idx, rot_img)
                        used.add(idx)
                        pbar.update(1)

                        if solve(count + 1):
                            return True

                        grid[i][j] = None
                        used.remove(idx)
                        pbar.update(-1)

                    return False
        return False

    solve(1)
    pbar.close()
    return grid

# ==============
# ROBUST STITCH 
# ==============
def stitch(grid):
    size = len(grid)

    # check
    for i in range(size):
        for j in range(size):
            if grid[i][j] is None:
                raise ValueError(f"Incomplete grid at ({i},{j})")

    h, w, _ = grid[0][0][2].shape

    overlaps = []

    for i in range(size):
        for j in range(size - 1):
            a_patch = grid[i][j][2]
            b_patch = grid[i][j+1][2]

            best_offset = 0
            best_mse = float('inf')

            for offset in range(24, 65):
                a = a_patch[:, -offset:]
                b = b_patch[:, :offset]

                if a.shape == b.shape:
                    mse = np.mean((a.astype(np.float32) - b.astype(np.float32))**2)
                    if mse < best_mse:
                        best_mse = mse
                        best_offset = offset

            overlaps.append(best_offset)

    OVERLAP = int(np.median(overlaps))
    print(f"Estimated overlap: {OVERLAP}")

    canvas = np.zeros(
        (h + (size-1)*(h-OVERLAP),
         w + (size-1)*(w-OVERLAP), 3),
        dtype=np.uint8
    )

    print("Rendering final canvas...")
    for i in range(size):
        for j in range(size):
            patch = grid[i][j][2]
            y = i * (h - OVERLAP)
            x = j * (w - OVERLAP)
            canvas[y:y+h, x:x+w] = patch

    return canvas

# ======================
# VQA
# ======================
def answer_questions(full_map):
    df = pd.read_csv(TEST_CSV)
    results = []

    def get_views(img):
        h, w = img.shape[:2]
        return [
            img,
            img[h//4:3*h//4, w//4:3*w//4],
            img[:h//2, :],
            img[h//2:, :],
            img[:, :w//2],
            img[:, w//2:]
        ]

    for _, row in tqdm(df.iterrows(), total=len(df)):
        options = [row[f"option_{i}"] for i in range(1, 5)]

        question_text = f"""
Question: {row['question']}
A. {options[0]}
B. {options[1]}
C. {options[2]}
D. {options[3]}
Answer A/B/C/D or E if unsure.
"""

        votes = []

        for view in get_views(full_map):
            img = Image.fromarray(cv2.cvtColor(view, cv2.COLOR_BGR2RGB))

            messages = [{"role":"user","content":[{"type":"image","image":img},{"type":"text","text":question_text}]}]

            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)

            inputs = processor(text=[text], images=image_inputs, videos=video_inputs, return_tensors="pt").to(DEVICE)

            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=10)

            resp = processor.batch_decode(out[:, inputs.input_ids.shape[1]:])[0].upper()

            match = re.search(r'\b(A|B|C|D)\b', resp)
            votes.append(match.group(1) if match else "E")

        score_map = {"A":0,"B":0,"C":0,"D":0}

        for i, vote in enumerate(votes):
            if vote not in score_map:
                continue

            weight = 2 if i in [0,1] else 1
            score_map[vote] += weight

        best = max(score_map, key=score_map.get)

        final = best if score_map[best] >= 4 else "E"

        mapping = {"A":1,"B":2,"C":3,"D":4,"E":5}
        results.append({"id":row["id"],"option":mapping[final]})

    pd.DataFrame(results).to_csv(OUTPUT_CSV, index=False)

# ======================
# MAIN
# ======================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_dir", type=str, required=True)
    args = parser.parse_args()

    test_dir = args.test_dir

    PATCH_DIR = os.path.join(test_dir, "patches")
    TEST_CSV = os.path.join(test_dir, "test.csv")
    OUTPUT_CSV = "submission.csv"

    patches = load_patches(PATCH_DIR)
    grid = build_grid_backtracking(patches)

    if any(cell is None for row in grid for cell in row):
        raise ValueError("Jigsaw failed")

    full_map = stitch(grid)

    cv2.imwrite("reconstructed_map.png", full_map)

    answer_questions(full_map)
