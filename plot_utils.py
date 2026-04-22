import matplotlib.pyplot as plt
import os
import collections

def set_style():
    # Helper to clean plots
    plt.grid(True, alpha=0.3)

def plot_part1(histories, output_dir="plots"):
    os.makedirs(output_dir, exist_ok=True)
    
    # Needs 4 plots per instructions
    fig_names = [
        ('epoch_train_loss', 'Training Loss vs Epochs'),
        ('epoch_test_loss', 'Test Loss vs Epochs'),
        ('epoch_train_accuracy', 'Training Accuracy vs Epochs'),
        ('epoch_test_accuracy', 'Test Accuracy vs Epochs')
    ]
    
    for metric, title in fig_names:
        plt.figure(figsize=(10, 6))
        for (lr, delta, init_acc), hist in histories.items():
            epochs = list(range(1, 1 + len(hist[metric])))
            plt.plot(epochs, hist[metric], label=f"({lr},{delta},{init_acc})", alpha=0.7)
        plt.xlabel('Epochs')
        plt.ylabel(title.split(' ')[0])
        plt.title(title)
        # Put legend outside
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

def plot_part3(h_cmd, h_pd, output_dir="plots"):
    os.makedirs(output_dir, exist_ok=True)
    
    for metric in ['epoch_train_loss', 'epoch_test_loss']:
        plt.figure(figsize=(8, 5))
        e = list(range(1, 1 + len(h_cmd[metric])))
        plt.plot(e, h_cmd[metric], marker='s', label="Update: CMD")
        plt.plot(e, h_pd[metric], marker='d', label="Update: Primal Dual")
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title(f'Part 3: {metric.replace("_", " ").title()}')
        plt.legend()
        set_style()
        plt.savefig(os.path.join(output_dir, f'part3_{metric}.png'))
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
