from typing import List, Any, Dict, Tuple
import copy
import torch
import torch.cuda
import torch.nn as nn
import numpy as np
from torch.nn.utils import clip_grad_norm_
from torch.nn.parallel import DataParallel
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import os
import time

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

def accuracy(model, x, y, batch_size=256):
    model.eval()
    total_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for i in range(0, x.shape[0], batch_size):
            end_idx = min(i + batch_size, x.shape[0])
            x_batch = x[i:end_idx]
            y_batch = y[i:end_idx]
     
            y_pred = model(x_batch)
            _, predicted_classes = torch.max(y_pred, 1)
   
            total_correct += (predicted_classes == y_batch).sum().item()
            total_samples += x_batch.shape[0]
    
    return round(total_correct / total_samples, 4)

def estimate_gradient(model, X_val, y_val, theta_prime, perturb_scale):



    gamma = generate_random_perturbation(theta_prime, perturb_scale)

    state_dict_plus = copy.deepcopy(model.state_dict())
    state_dict_minus = copy.deepcopy(model.state_dict())

    for (name, param), perturb in zip(theta_prime.items(), gamma.values()):
        if name in state_dict_plus:
            state_dict_plus[name].copy_(param + perturb)
            state_dict_minus[name].copy_(param - perturb)

    # model.load_state_dict(state_dict_plus)
    # acc_plus = accuracy(model, X_val, y_val)

    # model.load_state_dict(state_dict_minus)
    # acc_minus = accuracy(model, X_val, y_val)

    # print(acc_plus)
    # print(acc_minus)
    # After loading state_dict_plus
    model.load_state_dict(state_dict_plus)
    with torch.no_grad():
        y_pred = model(X_val)
        _, predicted_classes = torch.max(y_pred, 1)
        correct_predictions = (predicted_classes == y_val).sum().item()
        acc_plus = correct_predictions / X_val.shape[0]
   
    # After loading state_dict_minus
    model.load_state_dict(state_dict_minus)
    with torch.no_grad():
        y_pred = model(X_val)
        _, predicted_classes = torch.max(y_pred, 1)
        correct_predictions = (predicted_classes == y_val).sum().item()
        acc_minus = correct_predictions / X_val.shape[0]


    #grad_Lsup
    grad_estimate = {name: (acc_plus - acc_minus) / (2 * perturb.norm()) for name, perturb in gamma.items()}

    #grad_estimate = {name: (loss_plus - loss_minus) / (2 * perturb.norm()) for name, perturb in gamma.items()}

    
    return grad_estimate


def generate_random_perturbation(theta, epsilon):

    return {name: (epsilon * torch.randn_like(param, dtype=torch.float)) for name, param in theta.items()}


def true_gradient(model, X_support, y_support, theta_prime, batch_size=32):

 
    original_state = {k: v.clone() for k, v in model.state_dict().items()}
    
   
    model.load_state_dict(theta_prime)
   
    grad_acc = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            grad_acc[name] = torch.zeros_like(param)
    
   
    num_batches = 0
    criterion = torch.nn.CrossEntropyLoss()

    dataset = TensorDataset(X_support, y_support)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    for X_batch, y_batch in dataloader:
   
        model.zero_grad()
        
        outputs = model(X_batch)
    
        loss = criterion(outputs, y_batch)

        loss.backward()
 
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_acc[name] += param.grad.clone()
        
        num_batches += 1

        del X_batch, y_batch, outputs, loss

    if num_batches > 0:
        for name in grad_acc:
            grad_acc[name] /= num_batches

    model.load_state_dict(original_state)
    
    del original_state
    torch.cuda.empty_cache()
    
    return grad_acc


def check_theta_equality(theta_plus, theta_minus):

    if theta_plus.keys() != theta_minus.keys():
        return False

    for key in theta_plus:
        if not torch.allclose(theta_plus[key], theta_minus[key]):
            return False
    
    return True

def apply_l2_constraint(delta, epsilon=1.0):
    if torch.isnan(delta).any():
        delta = torch.where(torch.isnan(delta), torch.zeros_like(delta), delta)

    norm = torch.norm(delta.view(delta.shape[0], -1), dim=1, keepdim=True)

    if torch.isnan(norm).any():
        norm = torch.where(torch.isnan(norm), torch.ones_like(norm), norm)

    mask = (norm > epsilon).float().view(-1, *([1] * (len(delta.shape) - 1)))

    scale = norm.view(-1, *([1] * (len(delta.shape) - 1)))

    scale = torch.where(scale < 1e-8, torch.ones_like(scale), scale)
    delta_normalized = delta / (scale / epsilon)

    result = delta * (1 - mask) + delta_normalized * mask

    if torch.isnan(result).any():
        result = torch.where(torch.isnan(result), torch.zeros_like(result), result)
    
    return result

def bi_level_optimization_step(
        model, X_source, y_source, X_support, y_support, X_holdout, Y_holdout, 
        delta, theta_init, optimizer_init, perturb_scale=1e-6, lr_theta=0.01, lr_delta=0.01,
        epsilon=1.0, verbose=False,gradient_type = 'estimate'):

    start_time = time.time()
    
    theta_init_copy = copy.deepcopy(theta_init)
    theta_init_copy2 = copy.deepcopy(theta_init)
    criterion = torch.nn.CrossEntropyLoss()

    target_dtype = torch.float16 if device.startswith('cuda') else torch.float32

    if X_source.dtype != target_dtype:
        X_source = X_source.to(target_dtype)

    if X_support.dtype != target_dtype:
        X_support = X_support.to(target_dtype)

    if X_holdout.dtype != target_dtype:
        X_holdout = X_holdout.to(target_dtype)

    y_source = y_source.to(torch.int64) 
    y_support = y_support.to(torch.int64)
    Y_holdout = Y_holdout.to(torch.int64)

    delta = delta.to(device).to(target_dtype)
    delta.requires_grad_(True)

    # Step 0: load model with theta_init
    model.load_state_dict(theta_init)

    # Step 1: one-step model parameter theta'
    optimizer = torch.optim.SGD(model.parameters(), lr=lr_theta)
    model.train()

    optimizer.zero_grad()
    
    perturbed_X_source = (X_source + delta).clamp(0, 1)

    outputs_source = model(perturbed_X_source)
    train_loss = criterion(outputs_source, y_source)
 
    train_loss.backward()

    clip_grad_norm_(model.parameters(), 5.0)

    optimizer.step()

    theta_prime = model.state_dict()
    theta_prime_copy = copy.deepcopy(theta_prime)
    
    batch_size = 32 
    
    if gradient_type == 'true':
        grad_estimate = true_gradient(model, X_support, y_support, theta_prime_copy)
    
    else:  
        grad_estimate = estimate_gradient(model, X_support, y_support, theta_prime_copy, perturb_scale)
    
    
    grad_norm = 0.0
    for grad in grad_estimate.values():
        grad_norm += torch.sum(grad**2).item()
    grad_norm = np.sqrt(grad_norm)

    if grad_norm < 1e-8:
        grad_norm = 1e-8

    adaptive_perturb_scale = 0.01 / grad_norm


    state_dict = model.state_dict()
    
    # Step 3: Compute theta+, theta-
    model.load_state_dict(theta_init_copy)
    
    theta_plus = {}
    theta_minus = {}
    


    for (name, param), grad in zip(model.named_parameters(), grad_estimate.values()):

        if torch.isnan(grad).any():
            grad = torch.zeros_like(grad)

        grad_norm = torch.norm(grad)
        if grad_norm > 1.0:
            grad = grad / grad_norm
            
        theta_plus[name] = param + adaptive_perturb_scale * grad
        theta_minus[name] = param - adaptive_perturb_scale * grad

    for name in state_dict:
        if name not in theta_plus:
            theta_plus[name] = state_dict[name].clone()
            theta_minus[name] = state_dict[name].clone()
        else:
            theta_plus[name] = theta_plus[name].to(device)
            theta_minus[name] = theta_minus[name].to(device)
    

    # Step 4: Compute Hessian Approx

    delta.requires_grad_(True)

    model.load_state_dict(theta_plus)
    perturbed_X_source = (X_source + delta).clamp(0, 1)
    outputs_plus = model(perturbed_X_source)
    loss_plus = criterion(outputs_plus, y_source)

    grads_plus = torch.autograd.grad(loss_plus, delta, create_graph=True)[0]

    model.load_state_dict(theta_minus)
    perturbed_X_source = (X_source + delta).clamp(0, 1)
    outputs_minus = model(perturbed_X_source)
    loss_minus = criterion(outputs_minus, y_source)

    grads_minus = torch.autograd.grad(loss_minus, delta, create_graph=True)[0]

    hessian_approx = (grads_plus - grads_minus) / (2 * adaptive_perturb_scale)

    if torch.isnan(hessian_approx).any():
        hessian_approx = torch.where(torch.isnan(hessian_approx), torch.zeros_like(hessian_approx), hessian_approx)



    # Step 5: Update delta
    
    delta_before = delta.clone()
 

    update = lr_delta * lr_theta * hessian_approx

    if torch.isnan(update).any():
        update = torch.where(torch.isnan(update), torch.zeros_like(update), update)

    update_norm = torch.norm(update.view(update.shape[0], -1), dim=1).mean().item()
    if update_norm > 1.0:
        update = update / update_norm

    try:
        delta.data += update
    except RuntimeError as e:
        update = update.to(delta.dtype)
        delta.data += update
 
    delta.data = apply_l2_constraint(delta.data, epsilon)
    

    # Step 6: Update theta

    model.load_state_dict(theta_init_copy2)
    optimizer3 = torch.optim.SGD(model.parameters(), lr=lr_theta)

    model.train()
    optimizer3.zero_grad()

    delta_safe = delta.clone()
    if torch.isnan(delta_safe).any():
        delta_safe = torch.where(torch.isnan(delta_safe), torch.zeros_like(delta_safe), delta_safe)

    perturbed_X_source = (X_source + delta_safe).clamp(0, 1)
    perturbed_X_source = perturbed_X_source.to(X_source.dtype)

    outputs_source = model(perturbed_X_source)
    train_loss = criterion(outputs_source, y_source)
    train_loss.backward()
 
    clip_grad_norm_(model.parameters(), 5.0)

    optimizer3.step()

    updated_theta = model.state_dict()

    bs = 128  
    acc_support = accuracy(model, X_support, y_support, batch_size=bs)
    acc_holdout = accuracy(model, X_holdout, Y_holdout, batch_size=bs)
    
 

    return delta_safe, updated_theta, round(acc_support * 100, 2), round(acc_holdout * 100, 2)

def initialize_delta_dict(X_train, epsilon=0.1):

    target_dtype = torch.float16 if device.startswith('cuda') else torch.float32
    deltas = {i: torch.zeros_like(X_train[i], dtype=torch.float, requires_grad=True) for i in range(len(X_train))}

    for i in range(len(X_train)):
        deltas[i] = deltas[i].to(device).to(target_dtype)
    
    return deltas