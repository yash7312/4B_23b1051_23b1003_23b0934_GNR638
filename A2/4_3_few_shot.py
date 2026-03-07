import torch
import torch.nn as nn
import timm
import os
import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchvision import datasets
from torch.utils.data import DataLoader, random_split, Subset
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

# Constants
MODELS = ["efficientnet_b0", "inception_v3", "resnet50"]
DATA_PATH = "./train_data"
NUM_CLASSES = 30
EPOCHS = 10
BATCH_SIZE = 32
LR = 1e-3
DATA_REGIMES = [1.0, 0.2, 0.05] # 100% 20% and 5%
SEED = 42

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
set_seed(SEED)

def get_few_shot_loaders(model, subset_fraction):
    config = resolve_data_config({}, model=model)
    transform = create_transform(**config)
    
    full_dataset = datasets.ImageFolder(DATA_PATH, transform)
    
    # 1. Split into Train (80%) and Val (20%)
    train_len = int(0.8 * len(full_dataset))
    val_len = len(full_dataset) - train_len
    train_ds, val_ds = random_split(full_dataset, [train_len, val_len], 
                                    generator=torch.Generator().manual_seed(SEED))
    
    # 2. Subsample the Train set based on the regime (100%, 20%, 5%)
    if subset_fraction < 1.0:
        keep_len = int(subset_fraction * len(train_ds))
        discard_len = len(train_ds) - keep_len
        train_ds, _ = random_split(train_ds, [keep_len, discard_len], 
                                   generator=torch.Generator().manual_seed(SEED))
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    return train_loader, val_loader

def train_regime(model_name, fraction):
    print(f"\n--- Model: {model_name} | Data Regime: {fraction*100:.0f}% ---")
    
    # We'll use Full Fine-Tuning for Few-Shot as it's the most common benchmark
    model = timm.create_model(model_name, pretrained=True, num_classes=NUM_CLASSES).to(device)
    
    train_loader, val_loader = get_few_shot_loaders(model, fraction)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    
    best_val_acc = 0.0
    final_train_acc = 0.0

    for epoch in range(EPOCHS):
        # Training logic (Simplified for brevity, use your train_one_epoch logic here)
        model.train()
        correct, total = 0, 0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            if isinstance(outputs, tuple): outputs = outputs[0]
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        
        final_train_acc = correct / total
        
        # Validation
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]"):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                if isinstance(outputs, tuple): outputs = outputs[0]
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
        
        curr_val_acc = val_correct / val_total
        best_val_acc = max(best_val_acc, curr_val_acc)
        
    return final_train_acc, best_val_acc

# Execution and Reporting
results = {m: {} for m in MODELS}

for model_name in MODELS:
    for fraction in DATA_REGIMES:
        t_acc, v_acc = train_regime(model_name, fraction)
        results[model_name][fraction] = {"train": t_acc, "val": v_acc}

# Reporting Logic
for model_name in MODELS:
    acc_100 = results[model_name][1.0]["val"]
    acc_5 = results[model_name][0.05]["val"]
    delta = (acc_100 - acc_5) / acc_100
    
    print(f"\nResults for {model_name}:")
    print(f"  Relative Drop (Delta): {delta:.4f}")
    for f in DATA_REGIMES:
        gap = results[model_name][f]["train"] - results[model_name][f]["val"]
        print(f"  Regime {f*100:3.0f}% | Val Acc: {results[model_name][f]['val']:.3f} | Train-Val Gap: {gap:.3f}")