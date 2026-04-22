import torch
import torch.nn as nn
from data_loader import set_seeds

class LogisticRegression(nn.Module):
    def __init__(self, input_dim, num_classes, seed=42):
        super(LogisticRegression, self).__init__()
        set_seeds(seed)
        self.linear = nn.Linear(input_dim, num_classes)
        
        # The seed is fixed above, ensuring that PyTorch's default initialization
        # (Kaiming uniform) will produce identical starting weights across models.

    def forward(self, x):
        return self.linear(x)
