import torch
import torch.nn as nn
from tqdm import tqdm
import time
import numpy as np

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
    state = optimizer.state[param]
    if 'sum' in state: 
        return state['sum'] 
    elif 'sum_sq' in state:
        return state['sum_sq']
    elif 'G' in state:
        return torch.diag(state['G'])
    return None

def train_epoch(model, optimizer, data_loader, epoch, device='cpu', pytorch_l1_penalty=0.0):
    model.train()
    criterion = nn.CrossEntropyLoss()
    
    iter_metrics = {
        'loss': [],
        'grad_norm': [],
        'param_norm': [],
        'update_magnitude': []
    }
    
    epoch_correct = 0
    epoch_total = 0
    epoch_loss_sum = 0.0
    
    old_params = [p.clone().detach() for p in model.parameters() if p.requires_grad]
    
    for X, y in data_loader:
        X, y = X.to(device), y.to(device)
        
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        
        # Log accuracy/loss
        with torch.no_grad():
            _, predicted = torch.max(outputs.data, 1)
            epoch_total += y.size(0)
            epoch_correct += (predicted == y).sum().item()
            epoch_loss_sum += loss.item() * y.size(0)
            iter_metrics['loss'].append(loss.item())
        
        # Inject Custom PyTorch L1 Penalty before backward
        if pytorch_l1_penalty > 0.0:
            l1_loss = sum(p.abs().sum() for p in model.parameters() if p.requires_grad)
            loss = loss + pytorch_l1_penalty * l1_loss
            
        loss.backward()
        
        total_grad_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total_grad_norm += p.grad.data.norm(2).item() ** 2
        total_grad_norm = total_grad_norm ** 0.5
        
        optimizer.step()
        
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
        
        iter_metrics['grad_norm'].append(total_grad_norm)
        iter_metrics['param_norm'].append(total_param_norm)
        iter_metrics['update_magnitude'].append(total_update_mag)
        
    epoch_avg_loss = epoch_loss_sum / epoch_total
    epoch_acc = epoch_correct / epoch_total
    return iter_metrics, epoch_avg_loss, epoch_acc

def train(model, optimizer, train_loader, test_loader, epochs=5, device='cpu', pytorch_l1_penalty=0.0):
    history = {
        'iter_train_loss': [],
        'iter_grad_norm': [], 
        'iter_param_norm': [],
        'iter_update_magnitude': [],
        
        'epoch_train_loss': [],
        'epoch_train_accuracy': [],
        'epoch_test_loss': [],
        'epoch_test_accuracy': [],
        
        'epoch_exact_zeros': [],
        'epoch_l1_norm': [],
        'time_taken': 0.0
    }
    
    start_time = time.time()
    for epoch in range(1, epochs + 1):
        iter_metrics, ep_train_loss, ep_train_acc = train_epoch(
            model, optimizer, train_loader, epoch, device, pytorch_l1_penalty
        )
        
        history['iter_train_loss'].extend(iter_metrics['loss'])
        history['iter_grad_norm'].extend(iter_metrics['grad_norm'])
        history['iter_param_norm'].extend(iter_metrics['param_norm'])
        history['iter_update_magnitude'].extend(iter_metrics['update_magnitude'])
        
        history['epoch_train_loss'].append(ep_train_loss)
        history['epoch_train_accuracy'].append(ep_train_acc)
        
        test_loss, test_acc = evaluate(model, test_loader, device)
        history['epoch_test_loss'].append(test_loss)
        history['epoch_test_accuracy'].append(test_acc)
        
        # Calculate new specific metrics
        exact_zeros = sum((p == 0.0).sum().item() for p in model.parameters() if p.requires_grad)
        l1_norm = sum(p.abs().sum().item() for p in model.parameters() if p.requires_grad)
        
        history['epoch_exact_zeros'].append(exact_zeros)
        history['epoch_l1_norm'].append(l1_norm)
        
    history['time_taken'] = time.time() - start_time
    
    # Calculate convergence metrics
    if len(history['epoch_train_loss']) > 1:
        loss_diff = np.diff(history['epoch_train_loss'])
        history['convergence_speed'] = -np.mean(loss_diff) # avg drop per epoch
        history['smoothness_variance'] = np.var(history['epoch_train_loss'])
    else:
        history['convergence_speed'] = 0.0
        history['smoothness_variance'] = 0.0
        
    history['final_train_loss'] = history['epoch_train_loss'][-1]
        
    return history
