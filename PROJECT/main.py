import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# ======================
# CONFIG
# ======================
PATCH_DIR = "patches"
TEST_CSV = "test.csv"
OUTPUT_CSV = "submission.csv"
MODEL_PATH = "models/qwen"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OVERLAP = 30  # Adjust based on dataset characteristics

# ======================
# LOAD MODEL (LOCAL PREFERENCE)
# ======================
print("Loading Qwen2-VL weights...")
model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2", # Massively speeds up high-res processing
    device_map="auto"
)
model.eval()

processor = AutoProcessor.from_pretrained(MODEL_PATH)

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

# def build_grid(patches):
#     keys = list(patches.keys())
#     size = int(np.sqrt(len(keys)))
#     grid = [[None]*size for _ in range(size)]
#     used = {0}
    
#     # patch_0.png is always top-left
#     grid[0][0] = (0, patches[0])

#     print("Stitching image...")
#     for i in range(size):
#         for j in range(size):
#             if i == 0 and j == 0: continue

#             best_score = -float('inf')
#             best_choice = None

#             for idx in keys:
#                 if idx in used: continue
#                 for rot in rotations(patches[idx]):
#                     score = 0
#                     if j > 0: score += get_score(grid[i][j-1][1], rot, "right")
#                     if i > 0: score += get_score(grid[i-1][j][1], rot, "bottom")
                    
#                     if score > best_score:
#                         best_score = score
#                         best_choice = (idx, rot)
            
#             grid[i][j] = best_choice
#             used.add(best_choice[0])
#     return grid

def build_grid(patches):
    keys = list(patches.keys())
    size = int(np.sqrt(len(keys)))
    grid = [[None]*size for _ in range(size)]
    used = {0}
    
    # patch_0.png is always top-left
    grid[0][0] = (0, patches[0])

    print("Stitching image...")

    total_steps = size * size - 1  # excluding first cell
    pbar = tqdm(total=total_steps, desc="Placing patches")

    for i in range(size):
        for j in range(size):
            if i == 0 and j == 0:
                continue

            best_score = -float('inf')
            best_choice = None

            for idx in keys:
                if idx in used:
                    continue
                for rot in rotations(patches[idx]):
                    score = 0
                    if j > 0:
                        score += get_score(grid[i][j-1][1], rot, "right")
                    if i > 0:
                        score += get_score(grid[i-1][j][1], rot, "bottom")
                    
                    if score > best_score:
                        best_score = score
                        best_choice = (idx, rot)
            
            grid[i][j] = best_choice
            used.add(best_choice[0])

            pbar.update(1)   # ✅ update after placing each patch

    pbar.close()
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

    # Convert the stitched cv2 map (BGR) to PIL (RGB)
    full_map_rgb = cv2.cvtColor(full_map, cv2.COLOR_BGR2RGB)
    map_image = Image.fromarray(full_map_rgb)

    for _, row in tqdm(df.iterrows(), total=len(df)):
        options = [row[f"option_{i}"] for i in range(1, 5)]
        
        # Structure the prompt exactly how Qwen expects it
        question_text = f"""Analyze the map and answer the question.
Question: {row['question']}
Options:
A. {options[0]}
B. {options[1]}
C. {options[2]}
D. {options[3]}
Answer strictly with a single letter: A, B, C, or D."""

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": map_image},
                    {"type": "text", "text": question_text},
                ],
            }
        ]

        # Process inputs
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

        # Generate answer
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs, 
                max_new_tokens=5, # We only need 1 letter
                do_sample=False
            )
            
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        response = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip().upper()

        # Parse output safely
        final_ans = 1 # Default fallback
        if "A" in response: final_ans = 1
        elif "B" in response: final_ans = 2
        elif "C" in response: final_ans = 3
        elif "D" in response: final_ans = 4

        results.append({
            "id": row["id"],
            "option": final_ans
        })

    # Save exactly to the requested output file
    pd.DataFrame(results).to_csv(OUTPUT_CSV, index=False)

if __name__ == "__main__":
    p = load_patches(PATCH_DIR)
    g = build_grid(p)
    m = stitch(g)
    cv2.imwrite("reconstructed_map.png", m)
    answer_questions(m)