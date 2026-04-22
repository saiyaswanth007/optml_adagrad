import matplotlib.pyplot as plt
import os
import collections

def set_style():
    plt.grid(True, alpha=0.3)

def add_settings_subtitle(ax, settings_str):
    """Add a grey settings subtitle below the main title."""
    ax.set_title(ax.get_title() + '\n' + settings_str, fontsize=9, color='grey', loc='center')

def plot_comparisons(history1, name1, history2, name2, output_dir="plots"):
    os.makedirs(output_dir, exist_ok=True)
    settings = "matrix=diagonal | update=cmd | domain=unconstrained | reg=none | η=0.01 | ε=1e-8 | epochs=20"
    
    # 1. Gradient norm vs iterations
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(history1['iter_grad_norm'], label=name1, alpha=0.9, linewidth=3)
    ax.plot(history2['iter_grad_norm'], label=name2, alpha=0.9, linewidth=2, linestyle='--')
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Gradient Norm ||g_t||')
    ax.set_title('Baseline: Gradient Norm vs Iterations')
    add_settings_subtitle(ax, settings)
    ax.set_yscale('log')
    ax.legend()
    set_style()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'baseline_train_grad_norm_vs_iter.png'))
    plt.close()

    # 2. Update magnitude vs iterations
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(history1['iter_update_magnitude'], label=name1, alpha=0.9, linewidth=3)
    ax.plot(history2['iter_update_magnitude'], label=name2, alpha=0.9, linewidth=2, linestyle='--')
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Update Magnitude ||Δθ_t||')
    ax.set_yscale('log')
    ax.set_title('Baseline: Update Magnitude vs Iterations')
    add_settings_subtitle(ax, settings)
    ax.legend()
    set_style()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'baseline_train_update_mag_vs_iter.png'))
    plt.close()

def plot_effective_lr(top_frequent_lrs, top_rare_lrs, output_path="plots/effective_lr.png"):
    settings = "matrix=diagonal | update=cmd | domain=unconstrained | reg=none | η=0.01 | ε=1e-8"
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(['Top 100 Frequent Features', 'Top 100 Rare Features'], 
            [top_frequent_lrs, top_rare_lrs], 
            color=['blue', 'orange'])
    ax.set_ylabel('Average Final Effective LR (η / sqrt(G_t))')
    ax.set_title('Adagrad Effective LR Sparsity Check: Frequent vs Rare')
    add_settings_subtitle(ax, settings)
    set_style()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_part1(histories, output_dir="plots"):
    os.makedirs(output_dir, exist_ok=True)
    settings = "matrix=diagonal | update=cmd | domain=unconstrained | reg=none | ε=1e-8 | epochs=20"
    
    fig_names = [
        ('epoch_train_loss', 'Training Loss vs Epochs', 'Epochs'),
        ('epoch_test_loss', 'Test Loss vs Epochs', 'Epochs'),
        ('epoch_train_accuracy', 'Training Accuracy vs Epochs', 'Epochs'),
        ('epoch_test_accuracy', 'Test Accuracy vs Epochs', 'Epochs'),
        ('iter_train_loss', 'Training Loss vs Iterations', 'Iterations')
    ]
    
    for metric, title, xlabel in fig_names:
        fig, ax = plt.subplots(figsize=(10, 6))
        for (lr, init_acc), hist in histories.items():
            x_axis = list(range(1, 1 + len(hist[metric])))
            ax.plot(x_axis, hist[metric], label=f"lr={lr}, acc={init_acc}", alpha=0.7)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(title.split(' ')[0])
        ax.set_title(f'Part 1: {title}')
        add_settings_subtitle(ax, settings)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'part1_{metric}.png'))
        plt.close()

def plot_part2(h_custom, h_pytorch, output_dir="plots"):
    os.makedirs(output_dir, exist_ok=True)
    settings = "matrix=diagonal | update=cmd | domain=unconstrained | reg=l1 | η=0.01 | ε=1e-8 | λ=0.005 | epochs=20"
    
    fig, ax = plt.subplots(figsize=(8, 5))
    e_custom = list(range(1, 1 + len(h_custom['epoch_exact_zeros'])))
    e_native = list(range(1, 1 + len(h_pytorch['epoch_exact_zeros'])))
    
    ax.plot(e_custom, h_custom['epoch_exact_zeros'], marker='o', label="Custom (Exact Soft-Thresholding)")
    ax.plot(e_native, h_pytorch['epoch_exact_zeros'], marker='x', label="PyTorch (L1 Penalty)")
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Exact Zero Weights')
    ax.set_title('Part 2: True Sparsity Test (Exact 0.0s vs Epoch)')
    add_settings_subtitle(ax, settings)
    ax.legend()
    set_style()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'part2_true_sparsity.png'))
    plt.close()

    # Training Loss vs Iterations
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(h_custom['iter_train_loss'], label="Custom L1 (Soft-Thresholding)", alpha=0.7)
    ax.plot(h_pytorch['iter_train_loss'], label="PyTorch L1 (Penalty)", alpha=0.7, linestyle='--')
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Training Loss')
    ax.set_title('Part 2: Training Loss vs Iterations')
    add_settings_subtitle(ax, settings)
    ax.legend()
    set_style()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'part2_iter_train_loss.png'))
    plt.close()

def plot_part3(h_cmd, h_pd, output_dir="plots"):
    os.makedirs(output_dir, exist_ok=True)
    settings = "matrix=diagonal | domain=unconstrained | reg=l2 | η=0.05 | ε=1e-8 | λ=0.05 | epochs=20"
    
    for metric, xlabel in [('epoch_train_loss', 'Epochs'), ('epoch_test_loss', 'Epochs'), ('iter_train_loss', 'Iterations')]:
        fig, ax = plt.subplots(figsize=(8, 5))
        e = list(range(1, 1 + len(h_cmd[metric])))
        ax.plot(e, h_cmd[metric], marker='s' if 'epoch' in metric else None, label="Update: CMD", alpha=0.7)
        ax.plot(e, h_pd[metric], marker='d' if 'epoch' in metric else None, label="Update: Primal-Dual", linestyle='--', alpha=0.7)
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Loss')
        ax.set_title(f'Part 3: {metric.replace("_", " ").title()}')
        add_settings_subtitle(ax, settings)
        ax.legend()
        set_style()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'part3_{metric}.png'))
        plt.close()

    # Gradient Norm vs Iterations
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(h_cmd['iter_grad_norm'], label="Update: CMD", alpha=0.7)
    ax.plot(h_pd['iter_grad_norm'], label="Update: Primal-Dual", alpha=0.7, linestyle='--')
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Gradient Norm ||g_t||')
    ax.set_yscale('log')
    ax.set_title('Part 3: Gradient Norm vs Iterations')
    add_settings_subtitle(ax, settings)
    ax.legend()
    set_style()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'part3_iter_grad_norm.png'))
    plt.close()

    # Update Magnitude vs Iterations
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(h_cmd['iter_update_magnitude'], label="Update: CMD", alpha=0.7)
    ax.plot(h_pd['iter_update_magnitude'], label="Update: Primal-Dual", alpha=0.7, linestyle='--')
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Update Magnitude ||Δθ_t||')
    ax.set_yscale('log')
    ax.set_title('Part 3: Update Magnitude vs Iterations')
    add_settings_subtitle(ax, settings)
    ax.legend()
    set_style()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'part3_iter_update_mag.png'))
    plt.close()

def plot_part4(h_unconstrained, h_constrained, output_dir="plots"):
    os.makedirs(output_dir, exist_ok=True)
    settings = "matrix=diagonal | update=cmd | reg=none | η=0.1 | ε=1e-8 | epochs=20"
    
    fig, ax = plt.subplots(figsize=(8, 5))
    e = list(range(1, 1 + len(h_unconstrained['epoch_l1_norm'])))
    ax.plot(e, h_unconstrained['epoch_l1_norm'], label="Unconstrained (domain=unconstrained)")
    ax.plot(e, h_constrained['epoch_l1_norm'], label="L1 Bounded (domain=l1_ball, c=5.0)")
    
    # Draw bound line
    ax.axhline(y=5.0, color='r', linestyle='--', label='Constraint Limit (c=5.0)')
    
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Parameter L1 Norm')
    ax.set_title('Part 4: Constrained Optimization Projection')
    add_settings_subtitle(ax, settings)
    ax.legend()
    set_style()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'part4_l1_norm.png'))
    plt.close()

def plot_part5(histories, output_dir="plots"):
    os.makedirs(output_dir, exist_ok=True)
    settings = "matrix=diagonal | update=cmd | domain=unconstrained | η=0.01 | ε=1e-8 | λ=0.005 | epochs=20"
    
    styles = {'No Reg': ('tab:blue', '-', 'o'), 'L1 Reg': ('tab:orange', '--', 's'), 'L2 Reg': ('tab:green', '-.', 'd')}
    
    # Training Loss vs Epochs
    fig, ax = plt.subplots(figsize=(10, 6))
    for name, hist in histories.items():
        color, ls, marker = styles[name]
        e = list(range(1, 1 + len(hist['epoch_train_loss'])))
        ax.plot(e, hist['epoch_train_loss'], label=name, color=color, linestyle=ls, marker=marker)
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Training Loss')
    ax.set_title('Part 5: Regularization Comparison — Training Loss')
    add_settings_subtitle(ax, settings)
    ax.legend()
    set_style()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'part5_epoch_train_loss.png'))
    plt.close()

    # Exact Zeros vs Epochs
    fig, ax = plt.subplots(figsize=(10, 6))
    for name, hist in histories.items():
        color, ls, marker = styles[name]
        e = list(range(1, 1 + len(hist['epoch_exact_zeros'])))
        ax.plot(e, hist['epoch_exact_zeros'], label=name, color=color, linestyle=ls, marker=marker)
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Exact Zero Weights')
    ax.set_title('Part 5: Regularization Comparison — Sparsity (Exact Zeros)')
    add_settings_subtitle(ax, settings)
    ax.legend()
    set_style()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'part5_epoch_exact_zeros.png'))
    plt.close()

    # L1 Norm vs Epochs
    fig, ax = plt.subplots(figsize=(10, 6))
    for name, hist in histories.items():
        color, ls, marker = styles[name]
        e = list(range(1, 1 + len(hist['epoch_l1_norm'])))
        ax.plot(e, hist['epoch_l1_norm'], label=name, color=color, linestyle=ls, marker=marker)
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Parameter L1 Norm')
    ax.set_title('Part 5: Regularization Comparison — Weight L1 Norm')
    add_settings_subtitle(ax, settings)
    ax.legend()
    set_style()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'part5_epoch_l1_norm.png'))
    plt.close()
