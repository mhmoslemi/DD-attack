import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleConvNet(nn.Module):
    """A standard Convolutional Neural Network for feature extraction."""
    def __init__(self, num_classes=10):
        super(SimpleConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2, 2)
        
        # Penultimate layer (features)
        self.fc1 = nn.Linear(128 * 4 * 4, 512) # Assuming 32x32 input (CIFAR)
        # Classification head
        self.fc2 = nn.Linear(512, num_classes)

    def extract_features(self, x):
        """Returns the latent representation f_theta(x)."""
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)
        features = F.relu(self.fc1(x))
        return features

    def forward(self, x):
        features = self.extract_features(x)
        logits = self.fc2(features)
        return features, logits

class PerturbationGenerator(nn.Module):
    """G_phi: Generates adversarial noise bounded by epsilon."""
    def __init__(self, epsilon):
        super(PerturbationGenerator, self).__init__()
        self.epsilon = epsilon
        # A lightweight UNet-style or residual block approach
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 3, kernel_size=3, padding=1),
            nn.Tanh() # Bounds output to [-1, 1]
        )

    def forward(self, x):
        # Generate bounded noise: [-epsilon, epsilon]
        noise = self.net(x) * self.epsilon
        
        # Add noise to image and clamp to valid image range [0, 1]
        x_perturbed = torch.clamp(x + noise, 0.0, 1.0)
        return x_perturbed