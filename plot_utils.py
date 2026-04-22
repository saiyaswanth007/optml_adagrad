import matplotlib.pyplot as plt
import os
import collections

def set_style():
    plt.grid(True, alpha=0.3)

def plot_comparisons(history1, name1, history2, name2, output_dir="plots"):
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Gradient norm vs iterations
    plt.figure(figsize=(10, 6))
    plt.plot(history1['iter_grad_norm'], label=name1, alpha=0.9, linewidth=3)
    plt.plot(history2['iter_grad_norm'], label=name2, alpha=0.9, linewidth=2, linestyle='--')
    plt.xlabel('Iterations')
    plt.ylabel('Gradient Norm ||g_t||')
    plt.title('Baseline: Gradient Norm vs Iterations')
    plt.yscale('log')
    plt.legend()
    set_style()
    plt.savefig(os.path.join(output_dir, 'baseline_train_grad_norm_vs_iter.png'))
    plt.close()

    # 2. Update magnitude vs iterations
    plt.figure(figsize=(10, 6))
    plt.plot(history1['iter_update_magnitude'], label=name1, alpha=0.9, linewidth=3)
    plt.plot(history2['iter_update_magnitude'], label=name2, alpha=0.9, linewidth=2, linestyle='--')
    plt.xlabel('Iterations')
    plt.ylabel('Update Magnitude ||\Delta \\theta_t||')
    plt.yscale('log')
    plt.title('Baseline: Update Magnitude vs Iterations')
    plt.legend()
    set_style()
    plt.savefig(os.path.join(output_dir, 'baseline_train_update_mag_vs_iter.png'))
    plt.close()

def plot_effective_lr(top_frequent_lrs, top_rare_lrs, output_path="plots/effective_lr.png"):
    plt.figure(figsize=(8, 6))
    plt.bar(['Top 100 Frequent Features', 'Top 100 Rare Features'], 
            [top_frequent_lrs, top_rare_lrs], 
            color=['blue', 'orange'])
    plt.ylabel('Average Final Effective LR (\eta / sqrt(G_t))')
    plt.title('Adagrad Effective LR Sparsity Check: Frequent vs Rare')
    set_style()
    plt.savefig(output_path)
    plt.close()

def plot_part1(histories, output_dir="plots"):
    os.makedirs(output_dir, exist_ok=True)
    
    fig_names = [
        ('epoch_train_loss', 'Training Loss vs Epochs', 'Epochs'),
        ('epoch_test_loss', 'Test Loss vs Epochs', 'Epochs'),
        ('epoch_train_accuracy', 'Training Accuracy vs Epochs', 'Epochs'),
        ('epoch_test_accuracy', 'Test Accuracy vs Epochs', 'Epochs'),
        ('iter_train_loss', 'Training Loss vs Iterations', 'Iterations')
    ]
    
    for metric, title, xlabel in fig_names:
        plt.figure(figsize=(10, 6))
        for (lr, delta, init_acc), hist in histories.items():
            x_axis = list(range(1, 1 + len(hist[metric])))
            plt.plot(x_axis, hist[metric], label=f"({lr},{delta},{init_acc})", alpha=0.7)
        plt.xlabel(xlabel)
        plt.ylabel(title.split(' ')[0])
        plt.title(title)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'part1_{metric}.png'))
        plt.close()

def plot_part2(h_custom, h_pytorch, output_dir="plots"):
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(8, 5))
    e_custom = list(range(1, 1 + len(h_custom['epoch_exact_zeros'])))
    e_native = list(range(1, 1 + len(h_pytorch['epoch_exact_zeros'])))
    
    plt.plot(e_custom, h_custom['epoch_exact_zeros'], marker='o', label="Custom (Exact Soft-Thresholding)")
    plt.plot(e_native, h_pytorch['epoch_exact_zeros'], marker='x', label="PyTorch (L1 Penalty)")
    plt.xlabel('Epochs')
    plt.ylabel('Exact Zero Weights')
    plt.title('Part 2: True Sparsity Test (Exact 0.0s vs Epoch)')
    plt.legend()
    set_style()
    plt.savefig(os.path.join(output_dir, 'part2_true_sparsity.png'))
    plt.close()

    # Training Loss vs Iterations
    plt.figure(figsize=(10, 6))
    plt.plot(h_custom['iter_train_loss'], label="Custom L1 (Soft-Thresholding)", alpha=0.7)
    plt.plot(h_pytorch['iter_train_loss'], label="PyTorch L1 (Penalty)", alpha=0.7, linestyle='--')
    plt.xlabel('Iterations')
    plt.ylabel('Training Loss')
    plt.title('Part 2: Training Loss vs Iterations')
    plt.legend()
    set_style()
    plt.savefig(os.path.join(output_dir, 'part2_iter_train_loss.png'))
    plt.close()

def plot_part3(h_cmd, h_pd, output_dir="plots"):
    os.makedirs(output_dir, exist_ok=True)
    
    for metric, xlabel in [('epoch_train_loss', 'Epochs'), ('epoch_test_loss', 'Epochs'), ('iter_train_loss', 'Iterations')]:
        plt.figure(figsize=(8, 5))
        e = list(range(1, 1 + len(h_cmd[metric])))
        plt.plot(e, h_cmd[metric], marker='s' if 'epoch' in metric else None, label="Update: CMD", alpha=0.7)
        plt.plot(e, h_pd[metric], marker='d' if 'epoch' in metric else None, label="Update: Primal Dual", linestyle='--', alpha=0.7)
        plt.xlabel(xlabel)
        plt.ylabel('Loss')
        plt.title(f'Part 3: {metric.replace("_", " ").title()}')
        plt.legend()
        set_style()
        plt.savefig(os.path.join(output_dir, f'part3_{metric}.png'))
        plt.close()

    # Gradient Norm vs Iterations
    plt.figure(figsize=(10, 6))
    plt.plot(h_cmd['iter_grad_norm'], label="Update: CMD", alpha=0.7)
    plt.plot(h_pd['iter_grad_norm'], label="Update: Primal Dual", alpha=0.7, linestyle='--')
    plt.xlabel('Iterations')
    plt.ylabel('Gradient Norm ||g_t||')
    plt.yscale('log')
    plt.title('Part 3: Gradient Norm vs Iterations')
    plt.legend()
    set_style()
    plt.savefig(os.path.join(output_dir, 'part3_iter_grad_norm.png'))
    plt.close()

    # Update Magnitude vs Iterations
    plt.figure(figsize=(10, 6))
    plt.plot(h_cmd['iter_update_magnitude'], label="Update: CMD", alpha=0.7)
    plt.plot(h_pd['iter_update_magnitude'], label="Update: Primal Dual", alpha=0.7, linestyle='--')
    plt.xlabel('Iterations')
    plt.ylabel('Update Magnitude ||Δθ_t||')
    plt.yscale('log')
    plt.title('Part 3: Update Magnitude vs Iterations')
    plt.legend()
    set_style()
    plt.savefig(os.path.join(output_dir, 'part3_iter_update_mag.png'))
    plt.close()

def plot_part4(h_unconstrained, h_constrained, output_dir="plots"):
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(8, 5))
    e = list(range(1, 1 + len(h_unconstrained['epoch_l1_norm'])))
    plt.plot(e, h_unconstrained['epoch_l1_norm'], label="Unconstrained AdaGrad")
    plt.plot(e, h_constrained['epoch_l1_norm'], label="L1 Bounded (c=5.0)")
    
    # Draw bound line
    plt.axhline(y=5.0, color='r', linestyle='--', label='Constraint Limit (c=5.0)')
    
    plt.xlabel('Epochs')
    plt.ylabel('Parameter L1 Norm')
    plt.title('Part 4: Constrained Optimization Projection')
    plt.legend()
    set_style()
    plt.savefig(os.path.join(output_dir, 'part4_l1_norm.png'))
    plt.close()
