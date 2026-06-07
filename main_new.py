import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
from bilevel_bb_updated import bi_level_optimization_step, initialize_delta_dict
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
from torchvision.transforms import Compose, Resize, ToTensor
from PIL import Image
import matplotlib.pyplot as plt
from IPython.display import display, clear_output
import pickle
import argparse
from torchvision import models
import time
from sklearn.model_selection import train_test_split


parser = argparse.ArgumentParser(description='Bi-level Optimization for Domain Adaptation')
parser.add_argument('--source_domain', type=str, default='a', 
                    choices=['a', 'd', 'w'])
parser.add_argument('--target_domain', type=str, default='w', 
                    choices=['a', 'd', 'w'])
parser.add_argument('--data_dir', type=str, default='../data'
                    )
parser.add_argument('--model_dir', type=str, default='../',
                    )
parser.add_argument('--output_dir', type=str, default='results',
                    help='Directory to save results (default: )')
parser.add_argument('--batch_size', type=int, default=64,
                    help='Batch size for training (default: 64)')
parser.add_argument('--num_epochs', type=int, default=10,
                    help='Number of epochs (default: 10)')
parser.add_argument('--perturb_scale', type=float, default=0.01,
                    help='Scale of perturbation (default: 0.01)')
parser.add_argument('--lr_theta', type=float, default=0.075,
                    help='Learning rate for theta (default: 0.075)')
parser.add_argument('--lr_delta', type=float, default=0.075,
                    help='Learning rate for delta (default: 0.075)')
parser.add_argument('--support_ratio', type=float, default=1.0,
                    help='Ratio of support set to use (0.1 to 1.0, default: 1.0 = full set)')
parser.add_argument('--exp_name', type=str, default='',
                    help='Experiment name for saving results (default: source_to_target)')
parser.add_argument('--gradient_type', type=str, default='estimate', 
                    choices=['estimate', 'true'],
                    help='Type of gradient to use: estimate, true')

args = parser.parse_args()

# Validate support_ratio
if args.support_ratio <= 0 or args.support_ratio > 1.0:
    raise ValueError("support_ratio must be between 0.1 and 1.0")

if args.exp_name == '':
    args.exp_name = f"{args.source_domain}_to_{args.target_domain}_support{int(args.support_ratio*100)}_{args.gradient_type}"


os.makedirs(args.output_dir, exist_ok=True)

# PTM
class CustomResNet(nn.Module):
    def __init__(self, num_classes):
        super(CustomResNet, self).__init__()
        self.resnet = models.resnet50(pretrained=True)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.resnet(x)

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


model_path = os.path.join(args.model_dir, f"resnet50_{args.source_domain}")

try:
    model = torch.load(model_path)
except Exception as e:
    print(f"Failed: {e}")
    
    class CustomResNet(nn.Module):
        def __init__(self, num_classes=31):
            super(CustomResNet, self).__init__()
            self.resnet = models.resnet50(pretrained=True)
            num_ftrs = self.resnet.fc.in_features
            self.resnet.fc = nn.Linear(num_ftrs, num_classes)
        
        def forward(self, x):
            return self.resnet(x)
    
    model = CustomResNet().to(device)
    if device != 'cpu':
        model = model.half() 
model = model.half()



source_X_path = os.path.join(args.data_dir, f"{args.source_domain}_X_train.npy")
source_Y_path = os.path.join(args.data_dir, f"{args.source_domain}_Y_train.npy")

X = np.load(source_X_path)
Y = np.load(source_Y_path)


target_X_support_path = os.path.join(args.data_dir, f"{args.target_domain}_X_support.npy")
target_Y_support_path = os.path.join(args.data_dir, f"{args.target_domain}_Y_support.npy")
target_X_holdout_path = os.path.join(args.data_dir, f"{args.target_domain}_X_holdout.npy")
target_Y_holdout_path = os.path.join(args.data_dir, f"{args.target_domain}_Y_holdout.npy")


X_support_full = np.load(target_X_support_path)
Y_support_full = np.load(target_Y_support_path)
X_holdout = np.load(target_X_holdout_path)
Y_holdout = np.load(target_Y_holdout_path)


if args.support_ratio < 1.0:
    

    unique_classes = np.unique(Y_support_full)
    
   
    X_support = []
    Y_support = []
    
    
    for cls in unique_classes:
      
        indices = np.where(Y_support_full == cls)[0]  
      
        n_samples_to_keep = max(1, int(len(indices) * args.support_ratio))
          
        selected_indices = np.random.choice(indices, n_samples_to_keep, replace=False)
        
        X_support.append(X_support_full[selected_indices])
        Y_support.append(Y_support_full[selected_indices])
    
    X_support = np.vstack(X_support)
    Y_support = np.hstack(Y_support)
    

    for cls in unique_classes:
        orig_count = np.sum(Y_support_full == cls)
        new_count = np.sum(Y_support == cls)
else:
    X_support = X_support_full
    Y_support = Y_support_full

X_source = torch.from_numpy(X).float().to(device).half()
Y_source = torch.from_numpy(Y).float().to(device)

X_support = torch.from_numpy(X_support).float().to(device).half()
Y_support = torch.from_numpy(Y_support).float().to(device)
X_holdout = torch.from_numpy(X_holdout).float().to(device).half()
Y_holdout = torch.from_numpy(Y_holdout).float().to(device)


shuffled_indices = torch.randperm(len(X_source))
shuffled_X_source = X_source[shuffled_indices]
shuffled_y_source = Y_source[shuffled_indices]

delta_dict = initialize_delta_dict(X_source)

class CustomDataset(Dataset):
    def __init__(self, X, Y):
        self.data = [{'X': x, 'Y': y, 'delta': torch.zeros_like(x, dtype=torch.float, requires_grad=True)} for x, y in zip(X, Y)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        return sample['X'], sample['Y'], sample['delta']

    def update_delta(self, idx, new_delta):
        self.data[idx]['delta'] = new_delta.clone().detach().requires_grad_(True)

train_dataset = CustomDataset(X_source, Y_source)

num_samples = len(train_dataset)
num_complete_batches = num_samples // args.batch_size



train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
theta_init = model.state_dict()
optimizer_init = optimizer.state_dict()

accuracy_list = []
accuracy_list_sup = []

start_time = time.time()

for epoch in range(args.num_epochs):
    epoch_start_time = time.time()
    
    for batch_idx, (X_batch, y_batch, delta_batch) in enumerate(train_loader):
        batch_start_time = time.time()
        
        # bi-level optimization step
        updated_deltas, theta_init, acc_support, acc_holdout = bi_level_optimization_step(
            model, X_batch, y_batch, X_support, Y_support, X_holdout, Y_holdout, 
            delta_batch, theta_init, optimizer_init, 
            perturb_scale=args.perturb_scale, lr_theta=args.lr_theta, lr_delta=args.lr_delta,
            epsilon=0.5, verbose=False, gradient_type = args.gradient_type
        )
        
        batch_end_time = time.time()
        batch_duration = batch_end_time - batch_start_time
        
        print(f"Epoch: {epoch+1}/{args.num_epochs}, batch: {batch_idx+1}/{len(train_loader)}, "
              f"Support: {acc_support}%, Holdout: {acc_holdout}%, "
              f"Time: {batch_duration:.2f}seconds")

        accuracy_list.append(acc_holdout)
        accuracy_list_sup.append(acc_support)

        start_idx = batch_idx * args.batch_size

        for i, updated_delta in enumerate(updated_deltas):
            train_dataset.update_delta(start_idx + i, updated_delta)
    
    epoch_end_time = time.time()
    epoch_duration = epoch_end_time - epoch_start_time
    print(f"Epoch {epoch+1} done，time: {epoch_duration:.2f}seconds")
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

end_time = time.time()
total_duration = end_time - start_time


accuracy_file = os.path.join(args.output_dir, f"accuracy_list_{args.exp_name}.pkl")
accuracy_sup_file = os.path.join(args.output_dir, f"accuracy_list_sup_{args.exp_name}.pkl")

with open(accuracy_file, 'wb') as f:
    pickle.dump(accuracy_list, f)

with open(accuracy_sup_file, 'wb') as f:
    pickle.dump(accuracy_list_sup, f)

config = {
    'source_domain': args.source_domain,
    'target_domain': args.target_domain,
    'support_ratio': args.support_ratio,
    'perturb_scale': args.perturb_scale,
    'lr_theta': args.lr_theta,
    'lr_delta': args.lr_delta,
    'num_epochs': args.num_epochs,
    'batch_size': args.batch_size,
    'support_set_size': len(X_support),
    'original_support_set_size': len(X_support_full),
    'holdout_set_size': len(X_holdout),
    'drop_last_batch': True,
    'gradient_type' : args.gradient_type
}

config_file = os.path.join(args.output_dir, f"config_{args.exp_name}.pkl")
with open(config_file, 'wb') as f:
    pickle.dump(config, f)


