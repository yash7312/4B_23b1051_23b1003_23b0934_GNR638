import os
import cv2
import numpy as np
import random
import requests
import math

# ==========================================
# CONFIGURATION
# ==========================================
# 1. Map Fetching Config
LATITUDE = 19.1334        # Default: Powai / IIT Bombay area
LONGITUDE = 72.9133
ZOOM = 15
TILE_GRID = 4             # Fetches a 4x4 grid of OSM tiles (produces a 1024x1024 base image)

# 2. Patch Slicing Config
OUTPUT_DIR = 'patches'
PATCH_GRID_SIZE = 10      # Divides the map into a 10x10 puzzle (100 patches total)
OVERLAP = 30              # Overlap in pixels

# ==========================================
# PART 1: FETCH MAP FROM API
# ==========================================
def deg2num(lat_deg, lon_deg, zoom):
    """Convert Latitude/Longitude to OSM Tile X/Y coordinates."""
    lat_rad = math.radians(lat_deg)
    n = 2.0 ** zoom
    xtile = int((lon_deg + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return xtile, ytile

def download_osm_base_map(lat, lon, zoom, grid_size):
    """Fetches a grid of tiles from OpenStreetMap and stitches them into a single image."""
    print(f"Fetching {grid_size}x{grid_size} map tiles from OpenStreetMap API...")
    center_x, center_y = deg2num(lat, lon, zoom)
    
    start_x = center_x - (grid_size // 2)
    start_y = center_y - (grid_size // 2)
    
    # OSM tiles are always 256x256 pixels
    full_image = np.zeros((grid_size * 256, grid_size * 256, 3), dtype=np.uint8)
    
    headers = {'User-Agent': 'GeospatialVQA/1.0 (Academic Project)'} # OSM requires a User-Agent
    
    for i in range(grid_size):
        for j in range(grid_size):
            tile_x = start_x + j
            tile_y = start_y + i
            url = f"https://tile.openstreetmap.org/{zoom}/{tile_x}/{tile_y}.png"
            
            response = requests.get(url, headers=headers)
            if response.status_code == 200:
                # Convert downloaded bytes to a cv2 image
                image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
                tile_img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
                
                # Place tile in the large base map
                full_image[i*256:(i+1)*256, j*256:(j+1)*256] = tile_img
            else:
                print(f"Failed to fetch tile: {url}")
                
    return full_image

# ==========================================
# PART 2: PROCESS & SLICE INTO PATCHES
# ==========================================
def generate_patches():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Fetch the raw map
    raw_map = download_osm_base_map(LATITUDE, LONGITUDE, ZOOM, TILE_GRID)
    cv2.imwrite("original_fetched_map.png", raw_map)
    print("Saved 'original_fetched_map.png' for your reference.")

    # 2. Force Square Crop (Crucial for 90-degree rotations to work)
    min_dim = min(raw_map.shape[0], raw_map.shape[1])
    img_square = raw_map[0:min_dim, 0:min_dim]

    # 3. Calculate Perfect Padding
    # (Image width + Overlap Debt) must be cleanly divisible by the Grid Size
    target_dim = min_dim + (PATCH_GRID_SIZE - 1) * OVERLAP
    pad_needed = (PATCH_GRID_SIZE - (target_dim % PATCH_GRID_SIZE)) % PATCH_GRID_SIZE

    img_padded = cv2.copyMakeBorder(
        img_square, 
        0, pad_needed, 0, pad_needed, 
        cv2.BORDER_CONSTANT, 
        value=[0, 0, 0]
    )

    # Calculate exact, uniform patch dimensions
    p_size = (img_padded.shape[0] + (PATCH_GRID_SIZE - 1) * OVERLAP) // PATCH_GRID_SIZE
    print(f"Extracting {PATCH_GRID_SIZE**2} square patches of size {p_size}x{p_size} with {OVERLAP}px overlap...")

    anchor_patch = None
    other_patches = []

    # 4. Slice the Grid
    for r in range(PATCH_GRID_SIZE):
        for c in range(PATCH_GRID_SIZE):
            y1 = r * (p_size - OVERLAP)
            x1 = c * (p_size - OVERLAP)
            y2 = y1 + p_size
            x2 = x1 + p_size
            
            patch = img_padded[y1:y2, x1:x2]
            
            if r == 0 and c == 0:
                anchor_patch = patch  # Keep patch_0 unrotated
            else:
                # Random rotation by multiples of 90 degrees
                rot = random.choice([None, cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE])
                if rot is not None:
                    patch = cv2.rotate(patch, rot)
                other_patches.append(patch)

    # 5. Shuffle and Save
    print("Shuffling and saving patches...")
    cv2.imwrite(os.path.join(OUTPUT_DIR, "patch_0.png"), anchor_patch)

    random.shuffle(other_patches)

    for i, patch in enumerate(other_patches, start=1):
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"patch_{i}.png"), patch)

    print(f"Successfully generated 1 anchor and {len(other_patches)} shuffled patches in './{OUTPUT_DIR}'!")

if __name__ == "__main__":
    generate_patches()