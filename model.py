import torch
import torch.nn as nn
from data_loader import set_seeds

class LogisticRegression(nn.Module):
    def __init__(self, input_dim, num_classes, seed=42):
        super(LogisticRegression, self).__init__()
        set_seeds(seed)
        self.linear = nn.Linear(input_dim, num_classes)
        
        # Zero initialization is standard for convex optimization
        # and guarantees perfect reproducibility regardless of hardware specifics
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        return self.linear(x)
