import torch
import numpy as np
from data_loader import get_dataloaders, set_seeds
from model import LogisticRegression
from train import train, get_optimizer_state_sum
import plot_utils
from optimizer import AdaGradStrict

def inject_initial_accumulator(optimizer, model, initial_val):
    for group in optimizer.param_groups:
        for p in group['params']:
            if p.requires_grad:
                optimizer.state[p]['step'] = 0
                if group['matrix_type'] == 'diagonal':
                    optimizer.state[p]['sum_sq'] = torch.full_like(p.view(-1), initial_val)
                else: 
                    d = p.numel()
                    optimizer.state[p]['G'] = torch.eye(d, device=p.device) * initial_val
                
                if group['update_type'] == 'primal_dual':
                    optimizer.state[p]['u'] = torch.zeros_like(p.view(-1))

def print_part_header(title):
    print("\n" + "="*60)
    print(f" {title}")
    print("="*60)

def run_baseline_comparison(device, train_loader, test_loader):
    print_part_header("Baseline: Iteration Metrics (Custom vs PyTorch)")
    set_seeds(42)
    model_custom = LogisticRegression(input_dim=20000, num_classes=20).to(device)
    opt_custom = AdaGradStrict(model_custom.parameters(), lr=0.01, matrix_type='diagonal')
    h_custom = train(model_custom, opt_custom, train_loader, test_loader, epochs=30, device=device)
    
    set_seeds(42)
    model_native = LogisticRegression(input_dim=20000, num_classes=20).to(device)
    opt_native = torch.optim.Adagrad(model_native.parameters(), lr=0.01)
    h_native = train(model_native, opt_native, train_loader, test_loader, epochs=30, device=device)
    
    plot_utils.plot_comparisons(h_custom, "AdaGradStrict", h_native, "PyTorch Adagrad")
    return model_custom, opt_custom

def run_baseline_sparsity_check(model, optimizer, vectorizer, device):
    print_part_header("Baseline: Effective LR Sparsity Check")
    idf_scores = vectorizer.idf_
    sorted_idx = np.argsort(idf_scores)
    top_frequent_idx = sorted_idx[:100]
    top_rare_idx = sorted_idx[-100:]
    
    weight_param = list(model.parameters())[0]
    state_sum = get_optimizer_state_sum(optimizer, weight_param)
    
    if state_sum is None:
        return
        
    lr = optimizer.defaults['lr']
    delta = optimizer.defaults['delta']
    if state_sum.dim() == 1:
        state_sum = state_sum.view(weight_param.shape)
        
    G_t = state_sum.sqrt() + delta
    effective_lrs = lr / G_t
    avg_feature_lrs = effective_lrs.mean(dim=0).cpu().numpy()
    freq_lrs = avg_feature_lrs[top_frequent_idx].mean()
    rare_lrs = avg_feature_lrs[top_rare_idx].mean()
    plot_utils.plot_effective_lr(freq_lrs, rare_lrs)

def run_part1(device, train_loader, test_loader):
    print_part_header("Part 1: Hyperparameter Ablation Study")
    lrs = [0.1, 0.05, 0.01, 0.001]
    accs = [0, 1.0]
    
    histories = {}
    
    print(f"{'LR':<5} | {'Acc':<4} | {'Speed':<8} | {'Variance':<8} | {'Final Loss'}")
    print("-" * 50)
    for lr in lrs:
        for acc in accs:
            set_seeds(42)
            model = LogisticRegression(input_dim=20000, num_classes=20).to(device)
            opt = AdaGradStrict(model.parameters(), lr=lr, matrix_type='diagonal')
            
            if acc > 0:
                inject_initial_accumulator(opt, model, acc)
                
            hist = train(model, opt, train_loader, test_loader, epochs=30, device=device)
            histories[(lr, acc)] = hist
            
            speed = hist['convergence_speed']
            var = hist['smoothness_variance']
            floss = hist['final_train_loss']
            print(f"{lr:<5} | {acc:<4} | {speed:8.4f} | {var:8.4f} | {floss:.4f}")
            
    plot_utils.plot_part1(histories)

def run_part2(device, train_loader, test_loader):
    print_part_header("Part 2: True Sparsity Test (Custom L1 vs PyTorch)")
    lam = 0.005 
    
    set_seeds(42)
    model_custom = LogisticRegression(20000, 20).to(device)
    opt_custom = AdaGradStrict(model_custom.parameters(), lr=0.01, regularizer='l1', lambda_reg=lam)
    h_custom = train(model_custom, opt_custom, train_loader, test_loader, epochs=30, device=device)
    
    set_seeds(42)
    model_native = LogisticRegression(20000, 20).to(device)
    opt_native = torch.optim.Adagrad(model_native.parameters(), lr=0.01)
    h_native = train(model_native, opt_native, train_loader, test_loader, epochs=30, device=device, pytorch_l1_penalty=lam)
    
    plot_utils.plot_part2(h_custom, h_native)

def run_part3(device, train_loader, test_loader):
    print_part_header("Part 3: Algorithm Comparison (CMD vs Primal-Dual)")
    lam = 0.05 
    
    set_seeds(42)
    model_cmd = LogisticRegression(20000, 20).to(device)
    opt_cmd = AdaGradStrict(model_cmd.parameters(), lr=0.05, update_type='cmd', regularizer='l2', lambda_reg=lam)
    h_cmd = train(model_cmd, opt_cmd, train_loader, test_loader, epochs=30, device=device)
    
    set_seeds(42)
    model_pd = LogisticRegression(20000, 20).to(device)
    opt_pd = AdaGradStrict(model_pd.parameters(), lr=0.05, update_type='primal_dual', regularizer='l2', lambda_reg=lam)
    h_pd = train(model_pd, opt_pd, train_loader, test_loader, epochs=30, device=device)
    
    plot_utils.plot_part3(h_cmd, h_pd)

def run_part4(device, train_loader, test_loader):
    print_part_header("Part 4: Constrained Optimization (L1 Ball Projection)")
    bound = 5.0
    
    set_seeds(42)
    model_un = LogisticRegression(20000, 20).to(device)
    opt_un = AdaGradStrict(model_un.parameters(), lr=0.1) 
    h_un = train(model_un, opt_un, train_loader, test_loader, epochs=30, device=device)
    
    set_seeds(42)
    model_c = LogisticRegression(20000, 20).to(device)
    opt_c = AdaGradStrict(model_c.parameters(), lr=0.1, domain='l1_ball', domain_c=bound)
    h_c = train(model_c, opt_c, train_loader, test_loader, epochs=30, device=device)
    
    plot_utils.plot_part4(h_un, h_c)

def run_part5(device, train_loader, test_loader):
    print_part_header("Part 5: Regularization Comparison (None vs L1 vs L2)")
    lam = 0.005
    
    configs = [
        ("No Reg", {'regularizer': 'none'}),
        ("L1 Reg", {'regularizer': 'l1', 'lambda_reg': lam}),
        ("L2 Reg", {'regularizer': 'l2', 'lambda_reg': lam}),
    ]
    
    histories = {}
    for name, kwargs in configs:
        set_seeds(42)
        model = LogisticRegression(20000, 20).to(device)
        opt = AdaGradStrict(model.parameters(), lr=0.01, **kwargs)
        hist = train(model, opt, train_loader, test_loader, epochs=30, device=device)
        histories[name] = hist
    
    plot_utils.plot_part5(histories)

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    train_ld, test_ld, vec = get_dataloaders(max_features=20000, batch_size=64, seed=42)
    
    model_base, opt_base = run_baseline_comparison(device, train_ld, test_ld)
    run_baseline_sparsity_check(model_base, opt_base, vec, device)
    
    run_part1(device, train_ld, test_ld)
    run_part2(device, train_ld, test_ld)
    run_part3(device, train_ld, test_ld)
    run_part4(device, train_ld, test_ld)
    run_part5(device, train_ld, test_ld)
    
    print("\nAll Part 1-5 Experiments successfully generated to plots/ folder.")
