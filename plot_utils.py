import matplotlib.pyplot as plt
import os
import collections

def plot_comparisons(history1, name1, history2, name2, output_dir="plots"):
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Loss vs iterations
    plt.figure(figsize=(10, 6))
    plt.plot(history1['train_loss'], label=name1, alpha=0.9, linewidth=3)
    plt.plot(history2['train_loss'], label=name2, alpha=0.9, linewidth=2, linestyle='--')
    plt.xlabel('Iterations (Batches)')
    plt.ylabel('Training Loss (Cross Entropy)')
    plt.title('Loss vs Iterations')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'loss_vs_iter.png'))
    plt.close()

    # 2. Accuracy vs epochs
    plt.figure(figsize=(10, 6))
    # Assuming history contains 0th epoch
    epochs = list(range(len(history1['test_accuracy'])))
    plt.plot(epochs, history1['test_accuracy'], marker='o', markersize=8, label=name1, linewidth=3, alpha=0.8)
    plt.plot(epochs, history2['test_accuracy'], marker='X', markersize=6, label=name2, linewidth=2, linestyle='--', alpha=0.8)
    plt.xlabel('Epochs')
    plt.ylabel('Test Accuracy')
    plt.title('Accuracy vs Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'accuracy_vs_epoch.png'))
    plt.close()

    # 3. Gradient norm vs iterations
    plt.figure(figsize=(10, 6))
    plt.plot(history1['grad_norm'], label=name1, alpha=0.9, linewidth=3)
    plt.plot(history2['grad_norm'], label=name2, alpha=0.9, linewidth=2, linestyle='--')
    plt.xlabel('Iterations')
    plt.ylabel('Gradient Norm ||g_t||')
    plt.title('Gradient Norm vs Iterations')
    plt.yscale('log')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'grad_norm_vs_iter.png'))
    plt.close()

    # 4. Update magnitude vs iterations
    plt.figure(figsize=(10, 6))
    plt.plot(history1['update_magnitude'], label=name1, alpha=0.9, linewidth=3)
    plt.plot(history2['update_magnitude'], label=name2, alpha=0.9, linewidth=2, linestyle='--')
    plt.xlabel('Iterations')
    plt.ylabel('Update Magnitude ||\Delta \\theta_t||')
    plt.yscale('log')
    plt.title('Update Magnitude vs Iterations')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'update_mag_vs_iter.png'))
    plt.close()
    
    print(f"Plots saved to {output_dir}/")

def plot_effective_lr(top_frequent_lrs, top_rare_lrs, output_path="plots/effective_lr.png"):
    plt.figure(figsize=(8, 6))
    plt.bar(['Top 100 Frequent Features', 'Top 100 Rare Features'], 
            [top_frequent_lrs, top_rare_lrs], 
            color=['blue', 'orange'])
    plt.ylabel('Average Final Effective LR (\eta / sqrt(G_t))')
    plt.title('Adagrad Effective Learning Rates: Frequent vs Rare')
    plt.grid(axis='y')
    plt.savefig(output_path)
    plt.close()
