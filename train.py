import torch
import torch.nn as nn
from tqdm import tqdm

def evaluate(model, data_loader, device='cpu'):
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss(reduction='sum')
    
    with torch.no_grad():
        for X, y in data_loader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            loss = criterion(outputs, y)
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
            
    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy

def get_optimizer_state_sum(optimizer, param):
    """Helper to extract diagonal state sums uniformly from custom or native Adagrad."""
    state = optimizer.state[param]
    if 'sum' in state: 
        return state['sum'] # Native PyTorch Adagrad
    elif 'sum_sq' in state:
        return state['sum_sq'] # Our Custom AdaGradStrict
    elif 'G' in state:
        # Full matrix fallback (not used in text pipeline but safe mapping)
        return torch.diag(state['G'])
    return None

def train_epoch(model, optimizer, data_loader, epoch, device='cpu'):
    model.train()
    criterion = nn.CrossEntropyLoss()
    
    iteration_metrics = {
        'loss': [],
        'grad_norm': [],
        'param_norm': [],
        'update_magnitude': []
    }
    
    # We will compute the exact update magnitude by comparing theta_t and theta_{t-1}
    old_params = [p.clone().detach() for p in model.parameters() if p.requires_grad]
    
    for X, y in tqdm(data_loader, desc=f"Epoch {epoch}"):
        X, y = X.to(device), y.to(device)
        
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        
        # Track gradient norm before step
        total_grad_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total_grad_norm += p.grad.data.norm(2).item() ** 2
        total_grad_norm = total_grad_norm ** 0.5
        
        optimizer.step()
        
        # Track parameter norm & update magnitude
        total_param_norm = 0.0
        total_update_mag = 0.0
        new_params = []
        for p, old_p in zip(model.parameters(), old_params):
            if p.requires_grad:
                total_param_norm += p.data.norm(2).item() ** 2
                delta = p.data - old_p
                total_update_mag += delta.norm(2).item() ** 2
                new_params.append(p.clone().detach())
        
        total_param_norm = total_param_norm ** 0.5
        total_update_mag = total_update_mag ** 0.5
        old_params = new_params
        
        # Log to structural dict
        iteration_metrics['loss'].append(loss.item())
        iteration_metrics['grad_norm'].append(total_grad_norm)
        iteration_metrics['param_norm'].append(total_param_norm)
        iteration_metrics['update_magnitude'].append(total_update_mag)
        
    return iteration_metrics

def train(model, optimizer, train_loader, test_loader, epochs=5, device='cpu'):
    history = {
        'train_loss': [], # list of list of loss per batch
        'grad_norm': [], 
        'param_norm': [],
        'update_magnitude': [],
        'test_accuracy': [],
        'test_loss': []
    }
    
    # Pre-evaluate for 0-epoch baseline
    test_loss, test_acc = evaluate(model, test_loader, device)
    history['test_loss'].append(test_loss)
    history['test_accuracy'].append(test_acc)
    print(f"Epoch 0: Test Loss {test_loss:.4f}, Test Acc {test_acc:.4f}")

    for epoch in range(1, epochs + 1):
        iter_metrics = train_epoch(model, optimizer, train_loader, epoch, device)
        
        history['train_loss'].extend(iter_metrics['loss'])
        history['grad_norm'].extend(iter_metrics['grad_norm'])
        history['param_norm'].extend(iter_metrics['param_norm'])
        history['update_magnitude'].extend(iter_metrics['update_magnitude'])
        
        test_loss, test_acc = evaluate(model, test_loader, device)
        history['test_loss'].append(test_loss)
        history['test_accuracy'].append(test_acc)
        print(f"Epoch {epoch}: Test Loss {test_loss:.4f}, Test Acc {test_acc:.4f}")
        
    return history
