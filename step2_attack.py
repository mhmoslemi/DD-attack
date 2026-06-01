# step 3: optmize the genrator for the selected data

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import argparse
import os

# ==========================================
# 1. Model Definitions
# ==========================================
class ConvNet(nn.Module):
    """Standard ConvNet used in Dataset Condensation (e.g., ConvNet-3)"""
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

    def extract_features(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        return out

    def forward(self, x):
        features = self.extract_features(x)
        logits = self.classifier(features)
        return features, logits

class PerturbationGenerator(nn.Module):
    """G_phi: Generates adversarial noise bounded strictly by epsilon."""
    def __init__(self, epsilon):
        super(PerturbationGenerator, self).__init__()
        self.epsilon = epsilon
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 3, kernel_size=3, padding=1),
            nn.Tanh() 
        )

    def forward(self, x):
        noise = self.net(x) * self.epsilon
        x_perturbed = torch.clamp(x + noise, 0.0, 1.0)
        return x_perturbed

# ==========================================
# 2. Training Loops
# ==========================================
def train_model_on_S(model, images_syn, labels_syn, epochs=1000, lr=0.01, device='cuda'):
    """Trains the surrogate ConvNet on the synthetic dataset S."""
    print(f"Training surrogate feature extractor on S ({len(images_syn)} samples) for {epochs} epochs...")
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)
    criterion = nn.CrossEntropyLoss()
    
    images_syn = images_syn.to(device)
    labels_syn = labels_syn.to(device)
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        _, logits = model(images_syn)
        loss = criterion(logits, labels_syn)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 50 == 0:
            print(f"  Surrogate Epoch {epoch+1}/{epochs} - Loss on S: {loss.item():.4f}")
            
    print("Surrogate training complete. Features are now mapped to S.")
    return model

def train_generator(generator, model, T_base_images, x_target, epochs=500, lr=2e-3, device='cuda'):
    """Optimizes G_phi to force T_base features to collide with x_target."""
    print(f"\nOptimizing Generator for {epochs} epochs...")
    
    # Strictly freeze the surrogate model
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
        
    generator.train()
    optimizer = optim.Adam(generator.parameters(), lr=lr)
    
    T_base_images = T_base_images.to(device)
    x_target = x_target.unsqueeze(0).to(device)
    
    with torch.no_grad():
        f_target = model.extract_features(x_target)
        
    for epoch in range(epochs):
        optimizer.zero_grad()
        x_tilde = generator(T_base_images)
        f_tilde = model.extract_features(x_tilde)
        
        # L2 Collision Loss
        loss = F.mse_loss(f_tilde, f_target.expand_as(f_tilde))
        
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 50 == 0:
            print(f"  Generator Epoch {epoch+1}/{epochs} - Feature Collision Loss: {loss.item():.6f}")
            
    generator.eval()
    return generator

# ==========================================
# 3. Main Execution
# ==========================================
def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # ---------------------------------------------------------
    # A. Load Synthetic Dataset (S) & Train Surrogate
    # ---------------------------------------------------------
    if not os.path.exists(args.syn_data_path):
        raise FileNotFoundError(f"Could not find synthetic data at {args.syn_data_path}")
        
    print(f"Loading synthetic data from {args.syn_data_path}...")
    checkpoint = torch.load(args.syn_data_path, map_location='cpu')
    images_syn, labels_syn = checkpoint['data'][-1]
    
    # Normalize if stored as uint8
    if images_syn.max() > 1.0:
        images_syn = images_syn.float() / 255.0

    model = ConvNet(num_classes=10).to(device)
    model = train_model_on_S(model, images_syn, labels_syn, epochs=300, lr=0.01, device=device)

    # ---------------------------------------------------------
    # B. Define Hardcoded Base Indices (T_base)
    # ---------------------------------------------------------
    T_base_indices = torch.tensor([
        20329, 38951, 41306, 1961, 31535, 9794, 46590, 12952, 42882, 37554, 34005, 7411, 7932, 26689, 34741, 45914, 19388, 26043, 9760, 37584, 28808, 
        39910, 22463, 37700, 44576, 1963, 16322, 43606, 30721, 33008, 23526, 8101, 11858, 46862, 36990, 49773, 5914, 31065, 32250, 32215, 41102, 7474, 
        45217, 10137, 37091, 10193, 9700, 10454, 47897, 7667, 30219, 37156, 14919, 35296, 8029, 9, 35092, 17567, 30838, 25006, 45514, 47950, 5192, 19327, 
        18499, 42654, 14368, 36948, 7749, 39885, 43576, 5400, 48719, 26525, 14942, 22778, 33814, 46231, 27432, 19649, 21368, 20965, 38702, 25074, 17449, 1316,
        8385, 4128, 24148, 25676, 41715, 15818, 24335, 38999, 19234, 3813, 27407, 45379, 8386, 3749, 6462, 22761, 35323, 31907, 24910, 39773, 20632, 47502, 25688,
        35669, 39355, 26154, 20067, 46815, 29337, 7681, 40129, 23380, 47683, 46950, 37548, 48045, 32527, 7389, 36761, 41229, 37733, 38292, 36709, 36000, 15794, 13914,
        41581, 41421, 47819, 32358, 28712, 23353, 42133, 49384, 3320, 44191, 1700, 16363, 32686, 1150, 8183, 9982, 32172, 691, 46414, 740, 45319, 29520, 40788,
        36431, 18148, 13813, 7753, 48390, 21783, 28717, 34397, 46894, 8599, 46196, 25713, 20918, 35102, 28657, 33525, 16452, 2051, 26624, 4994, 25686, 
        10321, 10981, 29502, 6684, 41461, 28991, 3568, 20184, 9702, 45382, 25019, 41027, 42501, 41394, 995, 38764, 5502, 36951, 39211, 15844, 31402, 49922, 45249, 30964
    ], dtype=torch.long)

    # ---------------------------------------------------------
    # C. Load Original CIFAR-10 & Extract Target/Base Images
    # ---------------------------------------------------------
    print("\nLoading original CIFAR-10 data...")
    transform = transforms.ToTensor()
    cifar_test = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    cifar_train = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    
    # 1. Target Image (The test image we want to misclassify)
    x_target, target_label = cifar_test[args.target_idx]
    x_target = x_target.to(device)
    print(f"Selected Target Image index {args.target_idx} (True Label: {target_label})")

    # 2. Base Images (The clean training images we will poison)
    cifar_images = torch.tensor(cifar_train.data).permute(0, 3, 1, 2).float() / 255.0
    cifar_labels = torch.tensor(cifar_train.targets)
    
    T_base_images = cifar_images[T_base_indices].to(device)
    print(f"Loaded {len(T_base_images)} base images from CIFAR-10 training set.")

    # ---------------------------------------------------------
    # D. Train Generator
    # ---------------------------------------------------------
    generator = PerturbationGenerator(epsilon=args.epsilon).to(device)
    
    generator = train_generator(
        generator=generator,
        model=model,
        T_base_images=T_base_images,
        x_target=x_target,
        epochs=10000,
        lr=2e-3,
        device=device
    )

    # ---------------------------------------------------------
    # E. Apply Generator & Save Final Poisoned Dataset
    # ---------------------------------------------------------
    print("\nApplying perturbations to construct T_poisoned...")
    with torch.no_grad():
        x_tilde = generator(T_base_images).cpu()
    
    cifar_images_poisoned = cifar_images.clone()
    cifar_images_poisoned[T_base_indices] = x_tilde
    
    # Save as a standard dictionary for easy loading in your distillation script
    save_dict = {
        'data': cifar_images_poisoned,        # Float Tensor [50000, 3, 32, 32]
        'targets': cifar_labels,              # Long Tensor [50000]
        'poisoned_indices': T_base_indices    # Long Tensor [N_p]
    }
    
    torch.save(save_dict, args.out_path)
    print(f"\nSUCCESS. Poisoned CIFAR-10 dataset saved to: {args.out_path}")
    print(f"You can load this in your condensation loop via: torch.load('{args.out_path}')")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Amortized Feature Collision Attack")
    parser.add_argument('--syn_data_path', type=str, default='/work/mohammad/DatasetCondensation/result/res_DM_CIFAR10_ConvNet_50ipc.pt', help="Path to your saved res_...pt synthetic data")
    parser.add_argument('--target_idx', type=int, default=42, help="Index of the target image in CIFAR-10 test set")
    parser.add_argument('--epsilon', type=float, default=8/255.0, help="L-infinity perturbation bound")
    parser.add_argument('--out_path', type=str, default='cifar10_poisoned.pt', help="Where to save the poisoned dataset")
    args = parser.parse_args()
    
    main(args)