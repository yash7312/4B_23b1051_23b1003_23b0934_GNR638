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

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Constants
SEED = 42
set_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODELS = ["efficientnet_b0", "inception_v3", "resnet50"]

DATA_PATH = "train_data/train_data"

NUM_CLASSES = 30
EPOCHS = 10
LR = 1e-3
BATCH_SIZE = 32

DATA_REGIMES = [1.0, 0.2, 0.05] # 100% 20% and 5%


def create_model(model_name):
    
    model = timm.create_model(model_name, pretrained=True, num_classes=NUM_CLASSES)
    return model.to(device)


def get_few_shot_loaders(model, subset_fraction):
    config = resolve_data_config({}, model=model)
    transform = create_transform(**config)
    
    full_dataset = datasets.ImageFolder(DATA_PATH, transform)
    
    # 1. Split into Train and Val as 80 20
    train_len = int(0.8 * len(full_dataset))
    val_len = len(full_dataset) - train_len
    train_ds, val_ds = random_split(
        full_dataset, 
        [train_len, val_len], 
        generator=torch.Generator().manual_seed(SEED)
    )
    
    # 2. Subsample based on the regime (100%, 20%, 5%)
    if subset_fraction < 1.0:
        keep_len = int(subset_fraction * len(train_ds))
        discard_len = len(train_ds) - keep_len
        train_ds, _ = random_split(
            train_ds, 
            [keep_len, discard_len], 
            generator=torch.Generator().manual_seed(SEED)
        )
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, generator=torch.Generator().manual_seed(SEED))
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    return train_loader, val_loader


def train_one_epoch(model, loader, optimizer, criterion, epoch):
    
    model.train()
    
    correct, total = 0, 0
    running_loss = 0
    
    progress_bar = tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]", leave=False)
    
    for images, labels in progress_bar:
        
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(images)
        
        if isinstance(outputs, tuple):
            outputs = outputs[0]
            
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
        acc = correct / total
        progress_bar.set_postfix(
            acc=f"{acc:.3f}",
            loss=f"{running_loss/(total/BATCH_SIZE):.3f}"
        )
    
    return correct / total


def evaluate(model, loader):
    
    model.eval()
    
    correct, total = 0, 0
    
    with torch.no_grad():
        
        for images, labels in loader:
            
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            
            if isinstance(outputs, tuple):
                outputs = outputs[0]
                
            _, preds = torch.max(outputs, 1)
            
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    
    return correct / total


def train_regime(model_name, fraction):
    print(f"\n===== {model_name} | Data Regime: {fraction*100:.0f}% =====")
    
    model = create_model(model_name)
    
    train_loader, val_loader = get_few_shot_loaders(model, fraction)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    
    best_val_acc = 0.0
    final_train_acc = 0.0

    # Training loop
    for epoch in range(EPOCHS):

        train_acc = train_one_epoch(model, train_loader, optimizer, criterion, epoch)
        final_train_acc = train_acc
        
        val_acc = evaluate(model, val_loader)
        
        best_val_acc = max(best_val_acc, val_acc)
        
        print(
            f"Epoch {epoch+1}/{EPOCHS} | "
            f"Train Acc: {train_acc:.3f} | "
            f"Val Acc: {val_acc:.3f}"
        )
    return final_train_acc, best_val_acc


def compute_relative_drop(results, model_name):
    # Compute the relative drop in accuracy between 100% and 5% data regimes.
    acc_100 = results[model_name][1.0]["val"]
    acc_5 = results[model_name][0.05]["val"]
    
    delta = (acc_100 - acc_5) / acc_100
    
    return delta


def compute_train_val_gap(results, model_name, fraction):
    # Compute the train-validation gap
    train_acc = results[model_name][fraction]["train"]
    val_acc = results[model_name][fraction]["val"]
    
    return train_acc - val_acc


def print_results(results):
    
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    
    for model_name in MODELS:
        delta = compute_relative_drop(results, model_name)
        
        print(f"\n{model_name}:")
        print(f"  Relative Drop (Delta): {delta:.4f}")
        
        for fraction in DATA_REGIMES:
            gap = compute_train_val_gap(results, model_name, fraction)
            val_acc = results[model_name][fraction]["val"]
            
            print(
                f"  Regime {fraction*100:3.0f}% | "
                f"Val Acc: {val_acc:.3f} | "
                f"Train-Val Gap: {gap:.3f}"
            )
    
    print("\n" + "="*60)


def run_experiments():
    print(f"Device: {device}")
    print(f"Models: {MODELS}")
    print(f"Data Regimes: {[f'{f*100:.0f}%' for f in DATA_REGIMES]}")
    
    # Initialize results storage
    results = {model_name: {} for model_name in MODELS}
    
    # Train each model on each data regime
    for model_name in MODELS:
        print(f"\n{'='*60}")
        print(f"Starting experiments for {model_name}")
        print(f"{'='*60}")
        
        for fraction in DATA_REGIMES:
            train_acc, val_acc = train_regime(model_name, fraction)
            results[model_name][fraction] = {
                "train": train_acc, 
                "val": val_acc
            }
    
    print_results(results)
    
    return results

if __name__ == "__main__":
    results = run_experiments()