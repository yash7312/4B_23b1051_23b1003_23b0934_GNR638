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

# Increase recursion depth for the backtracking algorithm (max depth will be ~225)
sys.setrecursionlimit(2000)

# ======================
# CONFIG
# ======================
PATCH_DIR = "patches"
TEST_CSV = "test.csv"
OUTPUT_CSV = "submission.csv"
MODEL_PATH = "models/qwen"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OVERLAP = 30  

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
# COMPUTER VISION & STITCHING
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

score_cache = {}

def get_score(idx_a, rot_idx_a, img_a, idx_b, rot_idx_b, img_b, direction):
    key = (idx_a, rot_idx_a, idx_b, rot_idx_b, direction)
    if key in score_cache:
        return score_cache[key]

    best_mse = float('inf')
    
    # Check offsets to account for minor slicing jitter
    for offset in range(OVERLAP - 2, OVERLAP + 3):
        if direction == "right":
            edge_a = img_a[:, -offset:]
            edge_b = img_b[:, :offset]
        else:
            edge_a = img_a[-offset:, :]
            edge_b = img_b[:offset, :]
        
        # Ensure shapes match before MSE to prevent broadcast errors
        if edge_a.shape == edge_b.shape and edge_a.size > 0:
            mse = np.mean((edge_a.astype(np.float32) - edge_b.astype(np.float32)) ** 2)
            if mse < best_mse:
                best_mse = mse
            
    score_cache[key] = best_mse
    return best_mse

def build_grid_backtracking(patches):
    keys = list(patches.keys())
    size = int(np.sqrt(len(keys)))
    grid = [[None]*size for _ in range(size)]
    
    # Pre-calculate all rotations to save time during recursion
    oriented = {}
    for idx in keys:
        oriented[idx] = get_rotations(patches[idx])
        
    grid[0][0] = (0, 0, patches[0])
    used = {0}
    
    pbar = tqdm(total=size * size - 1, desc="Solving Jigsaw (Backtracking)")
    
    def get_valid_candidates(i, j):
        candidates = []
        for idx in keys:
            if idx in used:
                continue
            for rot_idx, rot_img in oriented[idx]:
                valid = True
                total_mse = 0
                
                # Check all placed neighbors. MSE must be extremely low (< 15.0) to be considered a true mathematical fit.
                if j > 0 and grid[i][j-1] is not None:
                    prev_idx, prev_rot_idx, prev_img = grid[i][j-1]
                    mse = get_score(prev_idx, prev_rot_idx, prev_img, idx, rot_idx, rot_img, "right")
                    if mse > 15.0: valid = False
                    total_mse += mse
                    
                if valid and i > 0 and grid[i-1][j] is not None:
                    prev_idx, prev_rot_idx, prev_img = grid[i-1][j]
                    mse = get_score(prev_idx, prev_rot_idx, prev_img, idx, rot_idx, rot_img, "bottom")
                    if mse > 15.0: valid = False
                    total_mse += mse
                    
                if valid and j < size - 1 and grid[i][j+1] is not None:
                    next_idx, next_rot_idx, next_img = grid[i][j+1]
                    mse = get_score(idx, rot_idx, rot_img, next_idx, next_rot_idx, next_img, "right")
                    if mse > 15.0: valid = False
                    total_mse += mse
                    
                if valid and i < size - 1 and grid[i+1][j] is not None:
                    next_idx, next_rot_idx, next_img = grid[i+1][j]
                    mse = get_score(idx, rot_idx, rot_img, next_idx, next_rot_idx, next_img, "bottom")
                    if mse > 15.0: valid = False
                    total_mse += mse
                    
                if valid:
                    candidates.append((total_mse, idx, rot_idx, rot_img))
                    
        # Sort by lowest MSE so it always tries the most perfect pixel matches first
        candidates.sort(key=lambda x: x[0])
        return [(c[1], c[2], c[3]) for c in candidates]

    def solve(placed_count):
        # Base case: The entire grid is filled
        if placed_count == size * size:
            return True
        
        # Minimum Remaining Values (MRV) Heuristic
        # Find the empty cell adjacent to placed cells that has the FEWEST valid pieces that fit it
        best_cell = None
        min_candidates = float('inf')
        best_candidates = []
        
        for i in range(size):
            for j in range(size):
                if grid[i][j] is None:
                    # Check if it has at least one placed neighbor
                    has_neighbor = False
                    if j > 0 and grid[i][j-1] is not None: has_neighbor = True
                    elif i > 0 and grid[i-1][j] is not None: has_neighbor = True
                    elif j < size - 1 and grid[i][j+1] is not None: has_neighbor = True
                    elif i < size - 1 and grid[i+1][j] is not None: has_neighbor = True
                    
                    if has_neighbor:
                        cands = get_valid_candidates(i, j)
                        if len(cands) < min_candidates:
                            min_candidates = len(cands)
                            best_cell = (i, j)
                            best_candidates = cands
                            
                            if min_candidates == 0:
                                return False # Dead end reached! Trigger backtrack.
                            if min_candidates == 1:
                                break # Perfect, we found a cell with only 1 possible piece
            if min_candidates == 1:
                break
        
        if best_cell is None:
            return False
            
        i, j = best_cell
        
        # Try placing the candidates
        for idx, rot_idx, rot_img in best_candidates:
            grid[i][j] = (idx, rot_idx, rot_img)
            used.add(idx)
            pbar.update(1)
            
            # Recursively dive deeper
            if solve(placed_count + 1):
                return True
                
            # If it failed down the line, BACKTRACK (undo the move and try the next piece)
            grid[i][j] = None
            used.remove(idx)
            pbar.update(-1)
            
        return False

    success = solve(1)
    pbar.close()
    
    if not success:
        print("\nWARNING: Backtracking failed to find a perfect fit. Check patch integrity.")
        
    return grid

def stitch(grid):
    size = len(grid)
    patch_h, patch_w, _ = grid[0][0][2].shape 
    
    full_h = patch_h + (size - 1) * (patch_h - OVERLAP)
    full_w = patch_w + (size - 1) * (patch_w - OVERLAP)
    
    canvas = np.zeros((full_h, full_w, 3), dtype=np.uint8)

    print("Rendering final canvas...")
    for i in range(size):
        for j in range(size):
            if grid[i][j] is None:
                continue 
            
            patch = grid[i][j][2]
            y_start = i * (patch_h - OVERLAP)
            x_start = j * (patch_w - OVERLAP)
            
            canvas[y_start:y_start+patch_h, x_start:x_start+patch_w] = patch
            
    return canvas


# ======================
# VQA EVALUATION
# ======================
def answer_questions(full_map):

    df = pd.read_csv(TEST_CSV)
    results = []

    print("Running VQA inference (multi-view + safe voting)...")

    def get_views(img):
        h, w = img.shape[:2]
        return [
            img,                                # full
            img[h//4:3*h//4, w//4:3*w//4],      # center
            img[:h//2, :],                      # top
            img[h//2:, :],                      # bottom
            img[:, :w//2],                      # left
            img[:, w//2:],                      # right
        ]

    for _, row in tqdm(df.iterrows(), total=len(df)):
        options = [row[f"option_{i}"] for i in range(1, 5)]

        question_text = f"""
You are given a detailed city map.

Carefully analyze:
- road names
- landmarks
- spatial relationships (north, south, east, west)
- water bodies and boundaries

This is a geospatial reasoning task.

Question: {row['question']}

Options:
A. {options[0]}
B. {options[1]}
C. {options[2]}
D. {options[3]}

Compare all options carefully and eliminate incorrect ones.

Final Answer (ONLY one letter):
"""

        votes = []

        for view in get_views(full_map):

            map_image = Image.fromarray(cv2.cvtColor(view, cv2.COLOR_BGR2RGB))

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": map_image},
                        {"type": "text", "text": question_text},
                    ],
                }
            ]

            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            image_inputs, video_inputs = process_vision_info(messages)

            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to(DEVICE)

            torch.cuda.empty_cache()

            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=10,
                    num_beams=3,
                    do_sample=False
                )

            generated_ids_trimmed = [
                out_ids[len(in_ids):]
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]

            response = processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0].strip().upper()

            
            if any(x in response for x in ["NOT SURE", "CANNOT", "UNABLE"]):
                votes.append("E")
            else:
                match = re.search(r'\b(A|B|C|D)\b', response)
                if match:
                    votes.append(match.group(1))
                else:
                    votes.append("E")

        valid_votes = [v for v in votes if v in ["A", "B", "C", "D"]]

        if valid_votes:
            counter = Counter(valid_votes)
            best, count = counter.most_common(1)[0]

            if count >= 3:   # majority out of 6 views
                final_letter = best
            else:
                final_letter = "E"
        else:
            final_letter = "E"

        mapping = {
            "A": 1,
            "B": 2,
            "C": 3,
            "D": 4,
            "E": 5
        }

        final_ans = mapping[final_letter]

        results.append({
            "id": row["id"],
            "option": final_ans
        })

    pd.DataFrame(results).to_csv(OUTPUT_CSV, index=False)
    print(f"Saved predictions to {OUTPUT_CSV}")


if __name__ == "__main__":
    p = load_patches(PATCH_DIR)
    
    g = build_grid_backtracking(p)
    m = stitch(g)
    cv2.imwrite("reconstructed_map.png", m)
    
    answer_questions(m)