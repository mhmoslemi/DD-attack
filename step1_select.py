# step 2 select

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import os

# ==========================================
# 1. Model Definition
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

# ==========================================
# 2. Train on Synthetic Data (S)
# ==========================================
def train_on_S(model, images_syn, labels_syn, epochs=300, lr=0.01, device='cuda'):
    """Trains the network purely on the condensed dataset S."""
    print(f"Training network on S ({len(images_syn)} samples) for {epochs} epochs...")
    model.train()
    
    # Dataset condensation usually trains on the whole S in one batch or large batches
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
            print(f"Epoch {epoch+1}/{epochs} - Loss on S: {loss.item():.4f}")
            
    return model

# ==========================================
# 3. Solve Selection Optimization (Eq 1)
# ==========================================
def select_T_base_from_cifar(model, cifar_images, cifar_labels, x_target, y_adv, N_p, lambda_1=1.0, lambda_2=0.5, device='cuda'):
    """Evaluates real CIFAR-10 images using the S-trained model to find the best T_base."""
    model.eval()
    
    # Filter original CIFAR-10 for the adversarial class
    adv_indices = (cifar_labels == y_adv).nonzero(as_tuple=True)[0]
    adv_images = cifar_images[adv_indices].to(device)
    
    if len(adv_images) < N_p:
        raise ValueError(f"Not enough instances in class {y_adv}.")

    print(f"Scanning {len(adv_images)} clean CIFAR-10 images of class {y_adv} to find the {N_p} most vulnerable...")

    with torch.no_grad():
        x_target = x_target.unsqueeze(0).to(device)
        f_target = model.extract_features(x_target)
        
        batch_size = 500
        all_scores = []
        
        for i in range(0, len(adv_images), batch_size):
            batch_img = adv_images[i:i+batch_size]
            f_candidates, logits = model(batch_img)
            
            # L2 Distance Term
            distances = torch.sum((f_candidates - f_target) ** 2, dim=1)
            
            # Classification Margin Term
            logits_adv_class = logits[:, y_adv]
            mask = torch.ones(logits.shape[1], dtype=torch.bool, device=device)
            mask[y_adv] = False
            max_other_logits, _ = torch.max(logits[:, mask], dim=1)
            margins = logits_adv_class - max_other_logits
            
            # Optimization Objective (Minimize Distance + Minimize Margin)
            scores = (lambda_1 * distances) + (lambda_2 * margins)
            all_scores.append(scores)
            
        all_scores = torch.cat(all_scores)
        
        # Get indices of the minimum scores (the most vulnerable/closest points)
        _, top_idx_in_adv = torch.topk(all_scores, k=N_p, largest=False)
        
        # Map back to the original CIFAR-10 training set indices
        selected_cifar_indices = adv_indices[top_idx_in_adv.cpu()]
        selected_cifar_images = adv_images[top_idx_in_adv]
        
    return selected_cifar_images, selected_cifar_indices

# ==========================================
# 4. Main Execution
# ==========================================
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # ---------------------------------------------------------
    # A. Load your saved Synthetic Dataset (S)
    # ---------------------------------------------------------
    save_path = '/work/mohammad/DatasetCondensation/result/res_DM_CIFAR10_ConvNet_50ipc.pt' # <-- UPDATE THIS
    checkpoint = torch.load(save_path, map_location='cpu')
    images_syn, labels_syn = checkpoint['data'][-1]
    
    # Ensure S is formatted correctly [N, C, H, W] and float
    if images_syn.max() > 1.0:
        images_syn = images_syn.float() / 255.0

    # ---------------------------------------------------------
    # B. Train network(s) on S
    # ---------------------------------------------------------
    model = ConvNet(num_classes=10).to(device)
    model = train_on_S(model, images_syn, labels_syn, epochs=1000, lr=0.01, device=device)

    # ---------------------------------------------------------
    # C. Load the Original CIFAR-10 Training Set
    # ---------------------------------------------------------
    print("Loading original CIFAR-10 training set...")
    cifar_train = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms.ToTensor())
    
    # Convert entire CIFAR-10 to tensors for fast processing
    # Note: CIFAR10.data is HWC and uint8 (0-255). We convert to CHW and float (0-1)
    cifar_images = torch.tensor(cifar_train.data).permute(0, 3, 1, 2).float() / 255.0
    cifar_labels = torch.tensor(cifar_train.targets)

    # Load CIFAR-10 Test Set (To pick the target image)
    cifar_test = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transforms.ToTensor())
    x_target, target_label = cifar_test[42] # Pick your specific target image here
    
    # ---------------------------------------------------------
    # D. Solve Optimization to find T_base in CIFAR-10
    # ---------------------------------------------------------
    y_adv = 3    # The adversarial class you want the target to map to
    N_p = 200     # Number of points to select from CIFAR-10
    
    T_base_images, T_base_indices = select_T_base_from_cifar(
        model=model,
        cifar_images=cifar_images,
        cifar_labels=cifar_labels,
        x_target=x_target,
        y_adv=y_adv,
        N_p=N_p,
        lambda_1=1.0,
        lambda_2=0.5,
        device=device
    )
    
    print(f"\nOptimization Complete.")
    print(f"Selected {len(T_base_indices)} indices from the original CIFAR-10 dataset.")
    print(f"Original CIFAR-10 indices: {T_base_indices.tolist()}")
    
    # You now have T_base_images, which you can pass to the Generator optimization.

if __name__ == "__main__":
    main()






# Original CIFAR-10 indices: 
#       [
#         20329, 38951, 41306, 1961, 31535, 9794, 46590, 12952, 42882, 37554, 34005, 7411, 7932, 26689, 34741, 45914, 19388, 26043, 9760, 37584, 28808, 
#         39910, 22463, 37700, 44576, 1963, 16322, 43606, 30721, 33008, 23526, 8101, 11858, 46862, 36990, 49773, 5914, 31065, 32250, 32215, 41102, 7474, 
#         45217, 10137, 37091, 10193, 9700, 10454, 47897, 7667, 30219, 37156, 14919, 35296, 8029, 9, 35092, 17567, 30838, 25006, 45514, 47950, 5192, 19327, 
#         18499, 42654, 14368, 36948, 7749, 39885, 43576, 5400, 48719, 26525, 14942, 22778, 33814, 46231, 27432, 19649, 21368, 20965, 38702, 25074, 17449, 1316,
#         8385, 4128, 24148, 25676, 41715, 15818, 24335, 38999, 19234, 3813, 27407, 45379, 8386, 3749, 6462, 22761, 35323, 31907, 24910, 39773, 20632, 47502, 25688,
#         35669, 39355, 26154, 20067, 46815, 29337, 7681, 40129, 23380, 47683, 46950, 37548, 48045, 32527, 7389, 36761, 41229, 37733, 38292, 36709, 36000, 15794, 13914,
#         41581, 41421, 47819, 32358, 28712, 23353, 42133, 49384, 3320, 44191, 1700, 16363, 32686, 1150, 8183, 9982, 32172, 691, 46414, 740, 45319, 29520, 40788,
#         36431, 18148, 13813, 7753, 48390, 21783, 28717, 34397, 46894, 8599, 46196, 25713, 20918, 35102, 28657, 33525, 16452, 2051, 26624, 4994, 25686, 
#         10321, 10981, 29502, 6684, 41461, 28991, 3568, 20184, 9702, 45382, 25019, 41027, 42501, 41394, 995, 38764, 5502, 36951, 39211, 15844, 31402, 49922, 45249, 30964]