# lastly check the asr and acc
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset, DataLoader
import os
import argparse
import numpy as np

# ==========================================
# 1. Victim Model Architecture
# ==========================================
class ConvNet(nn.Module):
    """Standard ConvNet matching your Dataset Condensation pipeline."""
    def __init__(self, channel=3, num_classes=10, net_width=128, net_depth=3, net_act='relu', net_norm='instancenorm', net_pooling='avgpooling', im_size=(32,32)):
        super(ConvNet, self).__init__()
        self.features, shape_feat = self._make_layers(channel, net_width, net_depth, net_norm, net_act, net_pooling, im_size)
        self.classifier = nn.Linear(shape_feat, num_classes)

    def _make_layers(self, channel, net_width, net_depth, net_norm, net_act, net_pooling, im_size):
        layers = []
        in_channels = channel
        shape_feat = [in_channels, im_size[0], im_size[1]]
        for d in range(net_depth):
            layers.append(nn.Conv2d(in_channels, net_width, kernel_size=3, padding=1))
            shape_feat[0] = net_width
            if net_norm == 'batchnorm':
                layers.append(nn.BatchNorm2d(net_width))
            elif net_norm == 'instancenorm':
                layers.append(nn.InstanceNorm2d(net_width))
            if net_act == 'relu':
                layers.append(nn.ReLU(inplace=True))
            if net_pooling == 'maxpooling':
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
                shape_feat[1] //= 2
                shape_feat[2] //= 2
            elif net_pooling == 'avgpooling':
                layers.append(nn.AvgPool2d(kernel_size=2, stride=2))
                shape_feat[1] //= 2
                shape_feat[2] //= 2
            in_channels = net_width
        return nn.Sequential(*layers), shape_feat[0] * shape_feat[1] * shape_feat[2]

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        logits = self.classifier(out)
        return logits

# ==========================================
# 2. Training and Evaluation Functions
# ==========================================
def train_victim(model, images_syn, labels_syn, epochs=300, lr=0.01, device='cuda'):
    """Trains a victim model purely on the synthetic dataset S."""
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)
    criterion = nn.CrossEntropyLoss()
    
    # Standard augmentation for synthetic training
    aug = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
    ])

    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Apply data augmentation to synthetic images on the fly
        images_aug = aug(images_syn)
        
        logits = model(images_aug)
        loss = criterion(logits, labels_syn)
        
        loss.backward()
        optimizer.step()
        
    return model

def evaluate_victim(model, testloader, x_target, y_adv, device='cuda'):
    """Evaluates CTA (accuracy) and ASR (target misclassification)."""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        # 1. Clean Test Accuracy (CTA)
        for inputs, targets in testloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            
        clean_acc = correct / total
        
        # 2. Attack Success Rate (ASR) on the specific x_target
        x_target = x_target.unsqueeze(0).to(device)
        output_target = model(x_target)
        _, predicted_target = torch.max(output_target.data, 1)
        
        attack_successful = (predicted_target.item() == y_adv)

    return clean_acc, attack_successful

# ==========================================
# 3. Main Execution Loop
# ==========================================
def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    
    # ---------------------------------------------------------
    # A. Load Clean CIFAR-10 Test Set & Extract Target
    # ---------------------------------------------------------
    print("Loading CIFAR-10 Test Set...")
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = DataLoader(testset, batch_size=256, shuffle=False, num_workers=2)
    
    # Extract the specific target image
    x_target, true_target_label = testset[args.target_idx]
    
    # ---------------------------------------------------------
    # B. Load Poisoned Synthetic Dataset (S)
    # ---------------------------------------------------------
    print(f"Loading distilled dataset from: {args.syn_data_path}")
    checkpoint = torch.load(args.syn_data_path, map_location=device, weights_only = False)
    
    # Checkpoint format from your DC script: data_save.append([image_syn, label_syn])
    images_syn, labels_syn = checkpoint['data'][-1]
    images_syn = images_syn.to(device)
    labels_syn = labels_syn.to(device)
    
    # ---------------------------------------------------------
    # C. Train and Evaluate 10 Victim Models
    # ---------------------------------------------------------
    print(f"\nEvaluating target {args.target_idx} (True Label: {true_target_label}) mapping to Adversarial Class: {args.y_adv}")
    print(f"Training {args.num_models} independent ConvNets on S...\n")
    
    acc_list = []
    asr_count = 0
    
    for i in range(args.num_models):
        print(f"--- Training Victim Model {i+1}/{args.num_models} ---")
        
        # Initialize fresh model
        model = ConvNet(num_classes=10).to(device)
        
        # Train on S (Dataset Condensation usually uses 300-1000 epochs)
        model = train_victim(model, images_syn, labels_syn, epochs=args.epochs, lr=0.01, device=device)
        
        # Evaluate
        clean_acc, is_successful = evaluate_victim(model, testloader, x_target, args.y_adv, device=device)
        acc_list.append(clean_acc)
        
        if is_successful:
            asr_count += 1
            status = f"SUCCESS (Classified as {args.y_adv})"
        else:
            status = "FAILED"
            
        print(f"Victim {i+1} -> Clean Acc: {clean_acc:.4f} | Attack: {status}\n")
    
    # ---------------------------------------------------------
    # D. Final Metrics Calculation
    # ---------------------------------------------------------
    mean_acc = np.mean(acc_list)
    std_acc = np.std(acc_list)
    asr = (asr_count / args.num_models) * 100.0
    
    print("=======================================")
    print(" FINAL EVALUATION METRICS")
    print("=======================================")
    print(f" Victim Models Evaluated : {args.num_models}")
    print(f" Clean Test Accuracy (CTA) : {mean_acc:.4f} ± {std_acc:.4f}")
    print(f" Attack Success Rate (ASR) : {asr:.2f}% ({asr_count}/{args.num_models})")
    print("=======================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Targeted Poisoning in DC")
    parser.add_argument('--syn_data_path', type=str, default='/work/mohammad/DatasetCondensation/result/res_DM_CIFAR10_ConvNet_50ipc_attack.pt', help="Path to your poisoned res_...pt synthetic data")
    parser.add_argument('--target_idx', type=int, default=42, help="Index of the target image in the CIFAR-10 test set")
    parser.add_argument('--y_adv', type=int, default=3, help="The adversarial class you targeted")
    parser.add_argument('--num_models', type=int, default=10, help="Number of victim models to train")
    parser.add_argument('--epochs', type=int, default=1000, help="Epochs to train each victim model on S")
    args = parser.parse_args()
    
    main(args)