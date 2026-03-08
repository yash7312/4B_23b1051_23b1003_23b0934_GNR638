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
import matplotlib.pyplot as plt
from datetime import datetime
from huggingface_hub import login
login(token="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")

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

# Create output directories
OUTPUT_DIR = "robustness_results"
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# Storage for results
results = []
efficiency_metrics = {}
all_model_results = {}

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
    """Compute efficiency metrics: Parameters, MACs, FLOPs"""
    model.eval()
    if model_name == "inception_v3":
        input_size = (1,3,299,299)
    else:
        input_size = (1,3,224,224)
    dummy = torch.randn(input_size).to(device)
    macs, params = profile(model, inputs=(dummy,), verbose=False)
    flops = 2 * macs

    metrics = {
        "params": params,
        "macs": macs,
        "flops": flops,
        "params_M": params / 1e6,
        "gflops": flops / 1e9
    }
    
    efficiency_metrics[model_name] = metrics

    print("\n==========================================")
    print(f"Efficiency Metrics for {model_name}")
    print(f"Parameters : {params:,} ({metrics['params_M']:.2f}M)")
    print(f"MACs       : {macs/1e9:.3f} GMACs")
    print(f"FLOPs      : {flops/1e9:.3f} GFLOPs")
    print("==========================================\n")
    
    return metrics

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

def calculate_stability_score(clean_acc, corrupted_accs):
    """
    Calculate stability score using area under robustness curve
    Higher score = better robustness maintenance
    """
    if len(corrupted_accs) >= 3:
        auc = np.trapezoid(corrupted_accs, x=[0.05, 0.1, 0.2])
        return auc / clean_acc
    return 0.0

def robustness_test(model_name):

    print("\n========================================")
    print(f"Robustness Evaluation for {model_name}")
    print("========================================")

    model = create_model(model_name)

    model.load_state_dict(torch.load(f"checkpoints_ft/{model_name}_FullFT.pth", weights_only=True))

    model.to(device)

    efficiency = compute_efficiency(model,model_name)

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
    
    model_corrupted_results = {}
    gaussian_accs = []

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

        # Store results
        results.append([model_name, acc_clean, name, acc, corruption_error, 
                       relative_robustness, efficiency['params_M'], efficiency['gflops']])
        model_corrupted_results[name] = {
            'accuracy': acc,
            'error': corruption_error,
            'relative_robustness': relative_robustness
        }
        
        # Collecting Gaussian noise results for stability score
        if "Gaussian" in name:
            gaussian_accs.append(acc)
    
    # Calculate stability score
    stability_score = calculate_stability_score(acc_clean, gaussian_accs)
    
    # Store all results for this model
    all_model_results[model_name] = {
        'clean_acc': acc_clean,
        'corruptions': model_corrupted_results,
        'gaussian_accs': gaussian_accs,
        'stability_score': stability_score,
        'efficiency': efficiency
    }

def save_results_to_csv():
    """Save all results to CSV file"""
    csv_path = os.path.join(OUTPUT_DIR, "robustness_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Model", "Clean_Accuracy", "Corruption", "Corrupted_Accuracy", 
                        "Corruption_Error", "Relative_Robustness", "Params_M", "GFLOPs"])
        writer.writerows(results)
    print(f"\nResults saved to: {csv_path}")

def save_efficiency_metrics():
    """Save efficiency metrics to separate CSV"""
    csv_path = os.path.join(OUTPUT_DIR, "efficiency_metrics.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Model", "Parameters", "Params_M", "MACs_G", "GFLOPs"])
        for model_name, metrics in efficiency_metrics.items():
            writer.writerow([model_name, metrics['params'], metrics['params_M'], 
                           metrics['macs']/1e9, metrics['gflops']])
    print(f"Efficiency metrics saved to: {csv_path}")

def plot_robustness_curves():
    """Plot robustness curves for Gaussian noise"""
    plt.figure(figsize=(10, 6))
    
    sigmas = [0.05, 0.1, 0.2]
    for model_name, data in all_model_results.items():
        gaussian_accs = data['gaussian_accs']
        clean_acc = data['clean_acc']
        
        # Include clean accuracy at sigma=0
        accs_with_clean = [clean_acc] + gaussian_accs
        sigmas_with_zero = [0] + sigmas
        
        plt.plot(sigmas_with_zero, accs_with_clean, marker='o', 
                linewidth=2, label=f"{model_name} (Stab: {data['stability_score']:.3f})")
    
    plt.xlabel('Gaussian Noise Sigma (σ)', fontsize=12)
    plt.ylabel('Validation Accuracy', fontsize=12)
    plt.title('Robustness to Gaussian Noise Corruption', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    save_path = os.path.join(PLOTS_DIR, "gaussian_robustness_curves.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Gaussian robustness curves saved to: {save_path}")
    plt.close()

def plot_corruption_comparison():
    """Bar chart comparing all corruptions across models"""
    corruption_types = ["Gaussian_0.05", "Gaussian_0.1", "Gaussian_0.2", 
                       "MotionBlur", "BrightnessShift"]
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    x = np.arange(len(corruption_types))
    width = 0.25
    
    for i, model_name in enumerate(MODELS):
        accs = [all_model_results[model_name]['corruptions'][corr]['accuracy'] 
                for corr in corruption_types]
        ax.bar(x + i*width, accs, width, label=model_name)
    
    ax.set_xlabel('Corruption Type', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Model Performance Under Different Corruptions', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(corruption_types, rotation=15, ha='right')
    ax.legend(fontsize=10)
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    
    save_path = os.path.join(PLOTS_DIR, "corruption_comparison.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Corruption comparison plot saved to: {save_path}")
    plt.close()

def plot_efficiency_vs_robustness():
    """Plot efficiency-robustness trade-off"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    models = list(all_model_results.keys())
    stability_scores = [all_model_results[m]['stability_score'] for m in models]
    params_m = [efficiency_metrics[m]['params_M'] for m in models]
    gflops = [efficiency_metrics[m]['gflops'] for m in models]
    
    # Parameters vs Stability
    ax1.scatter(params_m, stability_scores, s=200, alpha=0.6, c=['#1f77b4', '#ff7f0e', '#2ca02c'])
    for i, model in enumerate(models):
        ax1.annotate(model, (params_m[i], stability_scores[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=9)
    ax1.set_xlabel('Parameters (Millions)', fontsize=11)
    ax1.set_ylabel('Stability Score', fontsize=11)
    ax1.set_title('Parameters vs. Robustness', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # GFLOPs vs Stability
    ax2.scatter(gflops, stability_scores, s=200, alpha=0.6, c=['#1f77b4', '#ff7f0e', '#2ca02c'])
    for i, model in enumerate(models):
        ax2.annotate(model, (gflops[i], stability_scores[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=9)
    ax2.set_xlabel('GFLOPs', fontsize=11)
    ax2.set_ylabel('Stability Score', fontsize=11)
    ax2.set_title('Computational Cost vs. Robustness', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(PLOTS_DIR, "efficiency_robustness_tradeoff.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Efficiency-robustness trade-off plot saved to: {save_path}")
    plt.close()

def plot_relative_robustness_heatmap():
    """Heatmap of relative robustness across models and corruptions"""
    corruption_types = ["Gaussian_0.05", "Gaussian_0.1", "Gaussian_0.2", 
                       "MotionBlur", "BrightnessShift"]
    
    # Prepare data matrix
    data_matrix = []
    for model_name in MODELS:
        row = [all_model_results[model_name]['corruptions'][corr]['relative_robustness'] 
               for corr in corruption_types]
        data_matrix.append(row)
    
    data_matrix = np.array(data_matrix)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(data_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    
    # Set ticks and labels
    ax.set_xticks(np.arange(len(corruption_types)))
    ax.set_yticks(np.arange(len(MODELS)))
    ax.set_xticklabels(corruption_types, rotation=15, ha='right')
    ax.set_yticklabels(MODELS)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Relative Robustness', rotation=270, labelpad=20)
    
    # Annotate cells with values
    for i in range(len(MODELS)):
        for j in range(len(corruption_types)):
            text = ax.text(j, i, f'{data_matrix[i, j]:.3f}',
                          ha="center", va="center", color="black", fontsize=9)
    
    ax.set_title('Relative Robustness Heatmap', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    save_path = os.path.join(PLOTS_DIR, "relative_robustness_heatmap.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Relative robustness heatmap saved to: {save_path}")
    plt.close()

def generate_analysis_report():
    """Generate comprehensive text analysis report"""
    report_path = os.path.join(OUTPUT_DIR, "analysis_report.txt")
    
    with open(report_path, "w") as f:
        f.write("="*70 + "\n")
        f.write("TASK 4.4: CORRUPTION ROBUSTNESS ANALYSIS REPORT\n")
        f.write("="*70 + "\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Summary statistics
        f.write("="*70 + "\n")
        f.write("1. SUMMARY STATISTICS\n")
        f.write("="*70 + "\n\n")
        
        for model_name in MODELS:
            data = all_model_results[model_name]
            eff = efficiency_metrics[model_name]
            
            f.write(f"\n{model_name.upper()}\n")
            f.write("-" * 40 + "\n")
            f.write(f"  Clean Accuracy:        {data['clean_acc']:.4f}\n")
            f.write(f"  Stability Score:       {data['stability_score']:.4f}\n")
            f.write(f"  Parameters:            {eff['params_M']:.2f}M\n")
            f.write(f"  GFLOPs:                {eff['gflops']:.2f}\n")
            
            # Average relative robustness
            avg_rel_rob = np.mean([c['relative_robustness'] 
                                  for c in data['corruptions'].values()])
            f.write(f"  Avg Relative Robustness: {avg_rel_rob:.4f}\n")
        
        # Model rankings
        f.write("\n" + "="*70 + "\n")
        f.write("2. MODEL RANKINGS\n")
        f.write("="*70 + "\n\n")
        
        # By clean accuracy
        f.write("By Clean Accuracy:\n")
        sorted_by_clean = sorted(all_model_results.items(), 
                                key=lambda x: x[1]['clean_acc'], reverse=True)
        for i, (model, data) in enumerate(sorted_by_clean, 1):
            f.write(f"  {i}. {model}: {data['clean_acc']:.4f}\n")
        
        # By stability score
        f.write("\nBy Gaussian Stability Score:\n")
        sorted_by_stability = sorted(all_model_results.items(), 
                                    key=lambda x: x[1]['stability_score'], reverse=True)
        for i, (model, data) in enumerate(sorted_by_stability, 1):
            f.write(f"  {i}. {model}: {data['stability_score']:.4f}\n")
        
        # By efficiency
        f.write("\nBy Efficiency (Fewer Parameters):\n")
        sorted_by_params = sorted(efficiency_metrics.items(), 
                                 key=lambda x: x[1]['params_M'])
        for i, (model, eff) in enumerate(sorted_by_params, 1):
            f.write(f"  {i}. {model}: {eff['params_M']:.2f}M params\n")
        
        # Key insights
        f.write("\n" + "="*70 + "\n")
        f.write("3. KEY INSIGHTS & PATTERNS\n")
        f.write("="*70 + "\n\n")
        
        # Find best and worst performers
        best_stable_model = sorted_by_stability[0][0]
        worst_stable_model = sorted_by_stability[-1][0]
        most_efficient_model = sorted_by_params[0][0]
        
        f.write(f"• Most Robust Model: {best_stable_model}\n")
        f.write(f"  - Maintains performance best under Gaussian noise\n")
        f.write(f"  - Stability Score: {all_model_results[best_stable_model]['stability_score']:.4f}\n\n")
        
        f.write(f"• Least Robust Model: {worst_stable_model}\n")
        f.write(f"  - Most sensitive to corruption\n")
        f.write(f"  - Stability Score: {all_model_results[worst_stable_model]['stability_score']:.4f}\n\n")
        
        f.write(f"• Most Efficient Model: {most_efficient_model}\n")
        f.write(f"  - Fewest parameters: {efficiency_metrics[most_efficient_model]['params_M']:.2f}M\n")
        f.write(f"  - GFLOPs: {efficiency_metrics[most_efficient_model]['gflops']:.2f}\n\n")
        
        # Corruption-specific insights
        f.write("Corruption-Specific Observations:\n\n")
        
        # Find which corruption is hardest
        avg_accuracies_per_corruption = {}
        for corr_type in ["Gaussian_0.05", "Gaussian_0.1", "Gaussian_0.2", 
                         "MotionBlur", "BrightnessShift"]:
            avg_acc = np.mean([all_model_results[m]['corruptions'][corr_type]['accuracy'] 
                              for m in MODELS])
            avg_accuracies_per_corruption[corr_type] = avg_acc
        
        hardest_corruption = min(avg_accuracies_per_corruption.items(), key=lambda x: x[1])
        easiest_corruption = max(avg_accuracies_per_corruption.items(), key=lambda x: x[1])
        
        f.write(f"• Hardest Corruption: {hardest_corruption[0]}\n")
        f.write(f"  - Average accuracy across models: {hardest_corruption[1]:.4f}\n\n")
        
        f.write(f"• Easiest Corruption: {easiest_corruption[0]}\n")
        f.write(f"  - Average accuracy across models: {easiest_corruption[1]:.4f}\n\n")
        
        # Architectural insights
        f.write("="*70 + "\n")
        f.write("4. ARCHITECTURAL CHARACTERISTICS & ROBUSTNESS\n")
        f.write("="*70 + "\n\n")
        
        f.write("EfficientNet_B0:\n")
        f.write("  - Uses compound scaling (depth, width, resolution)\n")
        f.write("  - MBConv blocks with squeeze-and-excitation\n")
        f.write("  - Most parameter-efficient architecture\n")
        f.write(f"  - Stability Score: {all_model_results['efficientnet_b0']['stability_score']:.4f}\n\n")
        
        f.write("ResNet50:\n")
        f.write("  - Deep residual connections enable gradient flow\n")
        f.write("  - Skip connections may help preserve information under corruption\n")
        f.write(f"  - Stability Score: {all_model_results['resnet50']['stability_score']:.4f}\n\n")
        
        f.write("Inception_v3:\n")
        f.write("  - Multi-scale feature extraction with parallel conv paths\n")
        f.write("  - Diverse receptive fields may help or hurt depending on corruption\n")
        f.write(f"  - Stability Score: {all_model_results['inception_v3']['stability_score']:.4f}\n\n")
        
        # Efficiency-robustness trade-off
        f.write("="*70 + "\n")
        f.write("5. EFFICIENCY-ROBUSTNESS TRADE-OFF ANALYSIS\n")
        f.write("="*70 + "\n\n")
        
        for model in MODELS:
            eff = efficiency_metrics[model]
            stab = all_model_results[model]['stability_score']
            efficiency_ratio = stab / eff['params_M']  # Stability per million params
            
            f.write(f"{model}:\n")
            f.write(f"  - Stability per Million Parameters: {efficiency_ratio:.6f}\n")
            f.write(f"  - Trade-off: {eff['params_M']:.2f}M params - {stab:.4f} stability\n\n")
        
        # Recommendations
        f.write("="*70 + "\n")
        f.write("6. RECOMMENDATIONS\n")
        f.write("="*70 + "\n\n")
        
        f.write("For deployment in noisy environments:\n")
        f.write(f"  - Use {best_stable_model} (best robustness)\n\n")
        
        f.write("For resource-constrained deployment:\n")
        f.write(f"  - Use {most_efficient_model} (most efficient)\n\n")
        
        f.write("For balanced performance:\n")
        f.write("  - Consider efficiency-stability ratio when choosing\n\n")
        
        f.write("="*70 + "\n")
        f.write("END OF REPORT\n")
        f.write("="*70 + "\n")
    
    print(f"Analysis report saved to: {report_path}")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("TASK 4.4")
    print("="*70)
    print(f"Device: {device}")
    print(f"Models: {', '.join(MODELS)}")
    print(f"Output Directory: {OUTPUT_DIR}/")
    print("="*70 + "\n")
    
    # Run robustness tests for all models
    for model_name in MODELS:
        robustness_test(model_name)
    
    # Save results
    print("\n" + "="*70)
    print("SAVING RESULTS")
    print("="*70 + "\n")
    save_results_to_csv()
    save_efficiency_metrics()
    
    # Generate visualizations
    print("\n" + "="*70)
    print("GENERATING VISUALIZATIONS")
    print("="*70 + "\n")
    plot_robustness_curves()
    plot_corruption_comparison()
    plot_efficiency_vs_robustness()
    plot_relative_robustness_heatmap()
    
    # Generate analysis report
    print("\n" + "="*70)
    print("GENERATING ANALYSIS REPORT")
    print("="*70 + "\n")
    generate_analysis_report()
    
    print("\n" + "="*70)
    print("EVALUATION COMPLETE")
    print("="*70)
    print(f"\nAll results saved to: {OUTPUT_DIR}/")
    print(f"  - robustness_results.csv")
    print(f"  - efficiency_metrics.csv")
    print(f"  - analysis_report.txt")
    print(f"  - plots/ (4 visualization files)")
    print("\n" + "="*70 + "\n")
