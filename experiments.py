import torch
from data_loader import get_dataloaders, set_seeds
from model import LogisticRegression
from train import train
import plot_utils
from optimizer import AdaGradStrict

def inject_initial_accumulator(optimizer, model, initial_val):
    """
    Safely inject the initial accumulator state BEFORE stepping,
    avoiding any modifications to optimizer.py as required.
    """
    for group in optimizer.param_groups:
        for p in group['params']:
            if p.requires_grad:
                # Pre-construct empty states correctly mapping Duchi initialization
                optimizer.state[p]['step'] = 0
                if group['matrix_type'] == 'diagonal':
                    optimizer.state[p]['sum_sq'] = torch.full_like(p.view(-1), initial_val)
                else: 
                    # full matrix initialization logic if needed
                    d = p.numel()
                    optimizer.state[p]['G'] = torch.eye(d, device=p.device) * initial_val
                
                if group['update_type'] == 'primal_dual':
                    optimizer.state[p]['u'] = torch.zeros_like(p.view(-1))

def print_part_header(title):
    print("\n" + "="*60)
    print(f" {title}")
    print("="*60)

def run_part1(device, train_loader, test_loader):
    print_part_header("Part 1: Hyperparameter Ablation Study")
    lrs = [0.1, 0.01, 0.001]
    deltas = [1e-6, 1e-8, 1e-10]
    accs = [0, 0.1, 1.0] # 1 instead of 1.0 to match paper?
    
    histories = {}
    
    print(f"{'LR':<5} | {'Delta':<6} | {'Acc':<4} | {'Speed':<8} | {'Variance':<8} | {'Final Loss'}")
    print("-" * 60)
    for lr in lrs:
        for delta in deltas:
            for acc in accs:
                set_seeds(42)
                model = LogisticRegression(input_dim=20000, num_classes=20).to(device)
                opt = AdaGradStrict(model.parameters(), lr=lr, delta=delta, matrix_type='diagonal')
                
                if acc > 0:
                    inject_initial_accumulator(opt, model, acc)
                    
                hist = train(model, opt, train_loader, test_loader, epochs=2, device=device)
                histories[(lr, delta, acc)] = hist
                
                speed = hist['convergence_speed']
                var = hist['smoothness_variance']
                floss = hist['final_train_loss']
                print(f"{lr:<5} | {delta:<6} | {acc:<4} | {speed:8.4f} | {var:8.4f} | {floss:.4f}")
                
    plot_utils.plot_part1(histories)

def run_part2(device, train_loader, test_loader):
    print_part_header("Part 2: True Sparsity Test (Custom L1 vs PyTorch)")
    
    lam = 0.005 # Strong penalty
    
    # Custom
    set_seeds(42)
    model_custom = LogisticRegression(20000, 20).to(device)
    opt_custom = AdaGradStrict(model_custom.parameters(), lr=0.01, regularizer='l1', lambda_reg=lam)
    h_custom = train(model_custom, opt_custom, train_loader, test_loader, epochs=3, device=device)
    
    # Native
    set_seeds(42)
    model_native = LogisticRegression(20000, 20).to(device)
    opt_native = torch.optim.Adagrad(model_native.parameters(), lr=0.01)
    # inject the penalty manually via train function argument
    h_native = train(model_native, opt_native, train_loader, test_loader, epochs=3, device=device, pytorch_l1_penalty=lam)
    
    print(f"Final Exact Zeros (Custom):  {h_custom['epoch_exact_zeros'][-1]}")
    print(f"Final Exact Zeros (PyTorch): {h_native['epoch_exact_zeros'][-1]}")
    plot_utils.plot_part2(h_custom, h_native)

def run_part3(device, train_loader, test_loader):
    print_part_header("Part 3: Algorithm Comparison (CMD vs Primal-Dual)")
    lam = 0.05 # strong L2
    
    set_seeds(42)
    model_cmd = LogisticRegression(20000, 20).to(device)
    opt_cmd = AdaGradStrict(model_cmd.parameters(), lr=0.05, update_type='cmd', regularizer='l2', lambda_reg=lam)
    h_cmd = train(model_cmd, opt_cmd, train_loader, test_loader, epochs=3, device=device)
    
    set_seeds(42)
    model_pd = LogisticRegression(20000, 20).to(device)
    opt_pd = AdaGradStrict(model_pd.parameters(), lr=0.05, update_type='primal_dual', regularizer='l2', lambda_reg=lam)
    h_pd = train(model_pd, opt_pd, train_loader, test_loader, epochs=3, device=device)
    
    plot_utils.plot_part3(h_cmd, h_pd)

def run_part4(device, train_loader, test_loader):
    print_part_header("Part 4: Constrained Optimization (L1 Ball Projection)")
    bound = 5.0
    
    set_seeds(42)
    model_un = LogisticRegression(20000, 20).to(device)
    opt_un = AdaGradStrict(model_un.parameters(), lr=0.1) # higher LR to grow fast
    h_un = train(model_un, opt_un, train_loader, test_loader, epochs=3, device=device)
    
    set_seeds(42)
    model_c = LogisticRegression(20000, 20).to(device)
    opt_c = AdaGradStrict(model_c.parameters(), lr=0.1, domain='l1_ball', domain_c=bound)
    h_c = train(model_c, opt_c, train_loader, test_loader, epochs=3, device=device)
    
    print(f"Final Unconstrained L1 Norm: {h_un['epoch_l1_norm'][-1]:.2f}")
    print(f"Final Constrained L1 Norm:   {h_c['epoch_l1_norm'][-1]:.2f}")
    
    plot_utils.plot_part4(h_un, h_c)

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Central data load saves time
    train_ld, test_ld, _ = get_dataloaders(max_features=20000, batch_size=64, seed=42)
    
    # Only limit to 2 epochs max initially due to permutations
    run_part1(device, train_ld, test_ld)
    run_part2(device, train_ld, test_ld)
    run_part3(device, train_ld, test_ld)
    run_part4(device, train_ld, test_ld)
    
    print("\nAll Part 1-4 Experiments successfully generated to plots/ folder.")
