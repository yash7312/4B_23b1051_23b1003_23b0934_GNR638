import torch
import torch.nn as nn
import timm
import random
import numpy as np
import torchvision
from torchvision import datasets
from torch.utils.data import DataLoader, random_split
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from thop import profile
import torchvision.transforms.functional as TF
from PIL import ImageFilter
import os
import csv

def set_seed(seed=42):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
set_seed(42)

DATA_PATH = "train_data/train_data"
NUM_CLASSES = 30
BATCH_SIZE = 32

MODELS = ["efficientnet_b0", "resnet50", "inception_v3"]
results = []

def create_model(model_name):
    model = timm.create_model(model_name, pretrained=False)

    if model_name == "resnet50":
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, NUM_CLASSES)

    elif model_name == "efficientnet_b0":
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, NUM_CLASSES)

    elif model_name == "inception_v3":
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, NUM_CLASSES)

    return model.to(device)

def compute_efficiency(model, model_name):
    model.eval()
    if model_name == "inception_v3":
        input_size = (1,3,299,299)
    else:
        input_size = (1,3,224,224)
    dummy = torch.randn(input_size).to(device)
    macs, params = profile(model, inputs=(dummy,), verbose=False)
    flops = 2 * macs

    print("\n==========================================")
    print(f"Efficiency Metrics for {model_name}")
    print(f"Parameters : {params:,}")
    print(f"MACs       : {macs/1e9:.3f} GMACs")
    print(f"FLOPs      : {flops/1e9:.3f} GFLOPs")
    print("==========================================\n")

def gaussian_noise(img, sigma):
    tensor = TF.to_tensor(img)
    noise = torch.randn_like(tensor) * sigma
    tensor = tensor + noise
    tensor = torch.clamp(tensor,0,1)
    return TF.to_pil_image(tensor)


def motion_blur(img, radius=5):
    return img.filter(ImageFilter.GaussianBlur(radius))

def brightness_shift(img, factor=1.5):
    return TF.adjust_brightness(img,factor)


def get_loader(model, corruption=None):
    config = resolve_data_config({}, model=model)
    base_transform = create_transform(**config)
    transform_list = []

    if corruption is not None:
        transform_list.append(corruption)

    transform_list.append(base_transform)
    transform = torchvision.transforms.Compose(transform_list)
    dataset = datasets.ImageFolder(DATA_PATH, transform)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    _, val_ds = random_split(dataset, [train_size,val_size], generator=torch.Generator().manual_seed(42))

    loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, generator=torch.Generator().manual_seed(42))

    return loader

def evaluate(model, loader):
    model.eval()
    correct,total = 0,0

    with torch.no_grad():
        for images,labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)

            if isinstance(outputs,tuple):
                outputs = outputs[0]

            preds = torch.argmax(outputs,1)
            correct += (preds==labels).sum().item()
            total += labels.size(0)

    return correct/total

def robustness_test(model_name):

    print("\n========================================")
    print(f"Robustness Evaluation for {model_name}")
    print("========================================")

    model = create_model(model_name)

    model.load_state_dict(torch.load(f"checkpoints_ft/{model_name}_FullFT.pth", weights_only=True))

    model.to(device)

    compute_efficiency(model,model_name)

    clean_loader = get_loader(model)

    acc_clean = evaluate(model,clean_loader)

    print(f"\nClean Accuracy : {acc_clean:.4f}\n")

    corruptions = {
        "Gaussian_0.05": lambda img: gaussian_noise(img,0.05),
        "Gaussian_0.1": lambda img: gaussian_noise(img,0.1),
        "Gaussian_0.2": lambda img: gaussian_noise(img,0.2),
        "MotionBlur": lambda img: motion_blur(img,5),
        "BrightnessShift": lambda img: brightness_shift(img,1.5)
    }

    print(f"{'Corruption':20s} {'Accuracy':10s} {'Error':10s} {'RelRobust':10s}")

    for name,corr in corruptions.items():

        loader = get_loader(model,corr)

        acc = evaluate(model,loader)

        corruption_error = 1 - acc

        relative_robustness = acc / acc_clean

        print(
            f"{name:20s} "
            f"{acc:.4f}     "
            f"{corruption_error:.4f}     "
            f"{relative_robustness:.4f}"
        )

        results.append([model_name,acc_clean,name,acc,corruption_error,relative_robustness])

for model_name in MODELS:
    robustness_test(model_name)

with open("robustness_results.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Model","Clean_Accuracy","Corruption","Corrupted_Accuracy","Corruption_Error","Relative_Robustness"])
    writer.writerows(results)

print("\nResults saved to robustness_results.csv")