import torch
from data_loader import get_dataloaders, set_seeds
from model import LogisticRegression
from train import train, get_optimizer_state_sum
from plot_utils import plot_comparisons, plot_effective_lr
from optimizer import AdaGradStrict
import numpy as np
import time

def run_comparison(device):
    print("\n--- Running Main Comparison ---")
    train_loader, test_loader, vectorizer = get_dataloaders(max_features=20000, batch_size=64, seed=42)
    
    # 1. Custom AdaGrad
    set_seeds(42)
    model_custom = LogisticRegression(input_dim=20000, num_classes=20).to(device)
    opt_custom = AdaGradStrict(model_custom.parameters(), lr=0.01, matrix_type='diagonal')
    
    print("Training Custom AdaGradStrict...")
    start = time.time()
    hist_custom = train(model_custom, opt_custom, train_loader, test_loader, epochs=5, device=device)
    print(f"Time: {time.time() - start:.2f}s")
    
    # 2. PyTorch Native AdaGrad
    set_seeds(42)
    model_native = LogisticRegression(input_dim=20000, num_classes=20).to(device)
    opt_native = torch.optim.Adagrad(model_native.parameters(), lr=0.01)
    
    print("Training PyTorch Native Adagrad...")
    start = time.time()
    hist_native = train(model_native, opt_native, train_loader, test_loader, epochs=5, device=device)
    print(f"Time: {time.time() - start:.2f}s")
    
    # Plotting
    plot_comparisons(hist_custom, "AdaGradStrict", hist_native, "PyTorch Adagrad", output_dir="plots")
    
    return model_custom, opt_custom, vectorizer

def run_ablation(device):
    print("\n--- Running Ablation Study (AdaGradStrict) ---")
    train_loader, test_loader, _ = get_dataloaders(max_features=20000, batch_size=64, seed=42)
    
    lrs = [0.1, 0.01, 0.001]
    deltas = [1e-6, 1e-8, 1e-10]
    
    print("LR | Delta | Final Test Acc | Final Train Loss")
    print("-" * 50)
    for lr in lrs:
        for delta in deltas:
            set_seeds(42)
            model = LogisticRegression(input_dim=20000, num_classes=20).to(device)
            # cmd is default, diagonal is default
            opt = AdaGradStrict(model.parameters(), lr=lr, delta=delta, matrix_type='diagonal')
            hist = train(model, opt, train_loader, test_loader, epochs=2, device=device)
            print(f"{lr} | {delta} | {hist['test_accuracy'][-1]:.4f} | {np.mean(hist['train_loss'][-50:]):.4f}")

def adagrad_sparsity_check(model, optimizer, vectorizer, device):
    print("\n--- Adagrad-Specific Check (Sparsity) ---")
    
    # 1. Identify Frequent and Rare features
    # idf_ is inverse document frequency. Lower IDF = more frequent
    idf_scores = vectorizer.idf_
    sorted_idx = np.argsort(idf_scores)
    
    top_frequent_idx = sorted_idx[:100]
    top_rare_idx = sorted_idx[-100:]
    
    # 2. Extract Effective Step Sizes
    # The weight matrix is [20, 20000]. We want features (columns).
    weight_param = list(model.parameters())[0] # The Linear layer weight
    state_sum = get_optimizer_state_sum(optimizer, weight_param)
    
    if state_sum is None:
        print("Could not retrieve state sum for sparsity check.")
        return
        
    lr = optimizer.defaults['lr']
    delta = optimizer.defaults['delta']
    
    # state_sum is currently 1D flattened if AdaGradStrict, or 2D if native
    if state_sum.dim() == 1:
        state_sum = state_sum.view(weight_param.shape)
        
    G_t = state_sum.sqrt() + delta
    effective_lrs = lr / G_t
    
    # Average across classes for those features
    # effective_lrs is [20, 20000]. Mean over dim 0 -> [20000]
    avg_feature_lrs = effective_lrs.mean(dim=0).cpu().numpy()
    
    freq_lrs = avg_feature_lrs[top_frequent_idx].mean()
    rare_lrs = avg_feature_lrs[top_rare_idx].mean()
    
    print(f"Top 100 Frequent Features Average Effective LR: {freq_lrs}")
    print(f"Top 100 Rare Features Average Effective LR: {rare_lrs}")
    
    plot_effective_lr(freq_lrs, rare_lrs, "plots/effective_lr.png")

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    custom_model, custom_opt, vec = run_comparison(device)
    run_ablation(device)
    adagrad_sparsity_check(custom_model, custom_opt, vec, device)
    print("\nExperiments complete!")
