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
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "models/qwen"
OVERLAP = 30
MAX_BACKTRACK_STEPS = 10000 # Trigger for switching to greedy

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
# OVERLAP ESTIMATION
# ======================
def estimate_global_overlap(patches):
    print("Estimating global overlap...")
    keys = list(patches.keys())
    offset_votes = []
    
    # Filter for high-detail patches using variance (ignores flat water/grass)
    variances = {k: np.var(cv2.cvtColor(patches[k], cv2.COLOR_BGR2GRAY)) for k in keys}
    sorted_keys = sorted(keys, key=lambda k: variances[k], reverse=True)
    test_keys = sorted_keys[:min(15, len(keys))] 
    
    for k1 in test_keys:
        img1 = patches[k1]
        for k2 in keys:
            if k1 == k2: continue
            for _, img2 in get_rotations(patches[k2]):
                for direction in ["right", "bottom"]:
                    best_mse = float('inf')
                    best_off = -1
                    for offset in range(24, 65):
                        if direction == "right":
                            a, b = img1[:, -offset:], img2[:, :offset]
                        else:
                            a, b = img1[-offset:, :], img2[:offset, :]
                        
                        if a.shape == b.shape and a.size > 0:
                            mse = np.mean((a.astype(np.float32) - b.astype(np.float32))**2)
                            if mse < best_mse:
                                best_mse = mse
                                best_off = offset
                            
                    # If it's a near-perfect structural match, cast a vote for this offset
                    if best_mse < 25.0: 
                        offset_votes.append(best_off)
                        
    if offset_votes:
        most_common = Counter(offset_votes).most_common(1)[0][0]
        print(f"Calculated uniform overlap: {most_common} pixels")
        return most_common
        
    print("Warning: Could not confidently estimate overlap. Defaulting to 30.")
    return 30

# ======================
# ROBUST MATCHING
# ======================
score_cache = {}

def get_score(idx_a, rot_idx_a, img_a, idx_b, rot_idx_b, img_b, direction, overlap):
    key = (idx_a, rot_idx_a, idx_b, rot_idx_b, direction)
    if key in score_cache:
        return score_cache[key]

    if direction == "right":
        edge_a = img_a[:, -overlap:]
        edge_b = img_b[:, :overlap]
    else:
        edge_a = img_a[-overlap:, :]
        edge_b = img_b[:overlap, :]

    if edge_a.shape == edge_b.shape and edge_a.size > 0:
        mse = np.mean((edge_a.astype(np.float32) - edge_b.astype(np.float32)) ** 2)
    else:
        mse = float('inf')

    score_cache[key] = mse
    return mse

# ======================
# SOLVER (BACKTRACKING + GREEDY FALLBACK)
# ======================
def build_grid_backtracking(patches, overlap):
    keys = list(patches.keys())
    size = int(np.sqrt(len(keys)))
    grid = [[None]*size for _ in range(size)]
    
    oriented = {idx: get_rotations(patches[idx]) for idx in keys}
        
    grid[0][0] = (0, 0, patches[0])
    used = {0}
    
    pbar = tqdm(total=size * size - 1, desc="Solving Jigsaw")

    # Tracking states for fallback
    best_grid = [[None]*size for _ in range(size)]
    best_grid[0][0] = grid[0][0]
    max_placed = 1
    step_count = [0]
    forced_stop = [False]

    def get_valid_candidates(i, j):
        candidates = []
        for idx in keys:
            if idx in used:
                continue
            for rot_idx, rot_img in oriented[idx]:
                valid = True
                total_mse = 0

                # 1. Check Left
                if j > 0 and grid[i][j-1] is not None:
                    prev = grid[i][j-1]
                    mse = get_score(prev[0], prev[1], prev[2], idx, rot_idx, rot_img, "right", overlap)
                    if mse > 80: valid = False
                    
                    total_mse += mse

                # 2. Check Top
                if valid and i > 0 and grid[i-1][j] is not None:
                    prev = grid[i-1][j]
                    mse = get_score(prev[0], prev[1], prev[2], idx, rot_idx, rot_img, "bottom", overlap)
                    if mse > 80: valid = False
                    total_mse += mse

                # 3. Check Right
                if valid and j < size - 1 and grid[i][j+1] is not None:
                    nxt = grid[i][j+1]
                    mse = get_score(idx, rot_idx, rot_img, nxt[0], nxt[1], nxt[2], "right", overlap)
                    if mse > 80: valid = False
                    total_mse += mse

                # 4. Check Bottom
                if valid and i < size - 1 and grid[i+1][j] is not None:
                    nxt = grid[i+1][j]
                    mse = get_score(idx, rot_idx, rot_img, nxt[0], nxt[1], nxt[2], "bottom", overlap)
                    if mse > 80: valid = False
                    total_mse += mse

                if valid:
                    candidates.append((total_mse, idx, rot_idx, rot_img))

        candidates.sort(key=lambda x: x[0])
        limit = 4
        return [(c[1], c[2], c[3]) for c in candidates[:limit]]

    def solve(count):
        nonlocal max_placed, best_grid
        
        # Save best state in case we need to fallback
        if count > max_placed:
            max_placed = count
            for r in range(size):
                for c in range(size):
                    best_grid[r][c] = grid[r][c]

        if count == size * size:
            return True

        step_count[0] += 1
        if step_count[0] > MAX_BACKTRACK_STEPS:
            forced_stop[0] = True
            return False

        # MRV: Find the grid cell with the fewest possible valid patches
        best_cell = None
        min_candidates = float('inf')
        best_candidates = []

        for i in range(size):
            for j in range(size):
                if grid[i][j] is None:
                    if (j > 0 and grid[i][j-1]) or (i > 0 and grid[i-1][j]) or (j < size-1 and grid[i][j+1]) or (i < size-1 and grid[i+1][j]):
                        cands = get_valid_candidates(i, j)
                        if len(cands) < min_candidates:
                            min_candidates = len(cands)
                            best_cell = (i, j)
                            best_candidates = cands
                            
                            if min_candidates == 0: return False # Dead end
                            if min_candidates == 1: break # Perfect constraint
            if min_candidates == 1:
                break

        if best_cell is None: return False
        
        i, j = best_cell

        for idx, rot_idx, rot_img in best_candidates:
            grid[i][j] = (idx, rot_idx, rot_img)
            used.add(idx)
            pbar.update(1)

            if solve(count + 1):
                return True
                
            if forced_stop[0]: 
                return False # Bubble up stop signal quickly

            grid[i][j] = None
            used.remove(idx)
            pbar.update(-1)

        return False

    success = solve(1)
    pbar.close()
    
    # -----------------------------------
    # GREEDY FALLBACK LOGIC
    # -----------------------------------
    if not success or max_placed < size * size:
        print(f"\nBacktracking stuck at {max_placed}/{size*size} pieces (or timed out). Switching to Greedy Fallback...")
        
        # Restore the best partial layout
        grid = best_grid
        used = {grid[i][j][0] for i in range(size) for j in range(size) if grid[i][j] is not None}
        
        while len(used) < len(keys):
            best_greedy_cell = None
            best_greedy_patch = None
            best_greedy_mse = float('inf')
            
            # Find all frontier cells (empty cells connected to placed pieces)
            frontier = []
            for i in range(size):
                for j in range(size):
                    if grid[i][j] is None:
                        if (j > 0 and grid[i][j-1]) or (i > 0 and grid[i-1][j]) or (j < size-1 and grid[i][j+1]) or (i < size-1 and grid[i+1][j]):
                            frontier.append((i, j))
            
            # Failsafe: if grid got disjointed somehow, pick the first empty spot
            if not frontier:
                for i in range(size):
                    for j in range(size):
                        if grid[i][j] is None:
                            frontier.append((i, j))
                            break
                    if frontier: break
            
            # Evaluate all unused patches in all rotations strictly looking for lowest MSE
            for i, j in frontier:
                for idx in keys:
                    if idx in used: continue
                    for rot_idx, rot_img in oriented[idx]:
                        total_mse = 0
                        edges = 0
                        
                        if j > 0 and grid[i][j-1] is not None:
                            prev = grid[i][j-1]
                            total_mse += get_score(prev[0], prev[1], prev[2], idx, rot_idx, rot_img, "right", overlap)
                            edges += 1
                        if i > 0 and grid[i-1][j] is not None:
                            prev = grid[i-1][j]
                            total_mse += get_score(prev[0], prev[1], prev[2], idx, rot_idx, rot_img, "bottom", overlap)
                            edges += 1
                        if j < size - 1 and grid[i][j+1] is not None:
                            nxt = grid[i][j+1]
                            total_mse += get_score(idx, rot_idx, rot_img, nxt[0], nxt[1], nxt[2], "right", overlap)
                            edges += 1
                        if i < size - 1 and grid[i+1][j] is not None:
                            nxt = grid[i+1][j]
                            total_mse += get_score(idx, rot_idx, rot_img, nxt[0], nxt[1], nxt[2], "bottom", overlap)
                            edges += 1
                            
                        avg_mse = total_mse / max(1, edges)
                        
                        if avg_mse < best_greedy_mse:
                            best_greedy_mse = avg_mse
                            best_greedy_patch = (idx, rot_idx, rot_img)
                            best_greedy_cell = (i, j)
            
            if best_greedy_cell:
                i, j = best_greedy_cell
                grid[i][j] = best_greedy_patch
                used.add(best_greedy_patch[0])
            else:
                # Absolute Failsafe: Force randomly if all edges are somehow infinite
                unused = [k for k in keys if k not in used]
                if not unused: break
                idx = unused[0]
                i, j = frontier[0]
                grid[i][j] = (idx, 0, oriented[idx][0][1])
                used.add(idx)

    return grid

# ==============
# ROBUST STITCH 
# ==============
def stitch(grid, overlap):
    size = len(grid)

    for i in range(size):
        for j in range(size):
            if grid[i][j] is None:
                raise ValueError(f"Incomplete grid at ({i},{j})")

    h, w, _ = grid[0][0][2].shape

    canvas = np.zeros(
        (h + (size-1)*(h-overlap),
         w + (size-1)*(w-overlap), 3),
        dtype=np.uint8
    )

    print("Rendering final canvas...")
    for i in range(size):
        for j in range(size):
            patch = grid[i][j][2]
            y = i * (h - overlap)
            x = j * (w - overlap)
            canvas[y:y+h, x:x+w] = patch

    return canvas

# ======================
# VQA
# ======================
def answer_questions(full_map, test_csv, output_csv):
    df = pd.read_csv(test_csv)
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
        You are an expert geospatial analyst now carefully read and answer this question
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

            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image", 
                            "image": img
                        },
                        {"type": "text", "text": question_text}
                    ]
                }
            ]

            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)

            inputs = processor(text=[text], images=image_inputs, videos=video_inputs, return_tensors="pt").to(DEVICE)

            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=10)

            resp = processor.batch_decode(out[:, inputs.input_ids.shape[1]:])[0].upper()

            match = re.search(r'\b(A|B|C|D)\b', resp)
            votes.append(match.group(1) if match else "E")

            # Wipe memory after every single view
            del inputs, out, image_inputs, video_inputs
            torch.cuda.empty_cache()

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

    pd.DataFrame(results).to_csv(output_csv, index=False)

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
    
    # 1. Lock in the unified overlap mathematically
    dataset_overlap = estimate_global_overlap(patches)
    OVERLAP = dataset_overlap
    
    # 2. Build grid using constrained MRV backtracking (with Greedy Fallback)
    grid = build_grid_backtracking(patches, dataset_overlap)

    if any(cell is None for row in grid for cell in row):
        raise ValueError("Jigsaw failed completely, even with fallback.")

    # 3. Stitch map using the fixed dataset overlap
    full_map = stitch(grid, dataset_overlap)
    cv2.imwrite("reconstructed_map.png", full_map)

    # 4. Run VQA
    answer_questions(full_map, TEST_CSV, OUTPUT_CSV)
