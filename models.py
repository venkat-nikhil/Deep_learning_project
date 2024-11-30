from typing import List
import numpy as np
from torch import nn
from torch.nn import functional as F
import torch


def build_mlp(layers_dims: List[int]):
    layers = []
    for i in range(len(layers_dims) - 2):
        layers.append(nn.Linear(layers_dims[i], layers_dims[i + 1]))
        layers.append(nn.BatchNorm1d(layers_dims[i + 1]))
        layers.append(nn.ReLU(True))
    layers.append(nn.Linear(layers_dims[-2], layers_dims[-1]))
    return nn.Sequential(*layers)


class MockModel(torch.nn.Module):
    """
    Does nothing. Just for testing.
    """

    def __init__(self, device="cuda", bs=64, n_steps=17, output_dim=256):
        super().__init__()
        self.device = device
        self.bs = bs
        self.n_steps = n_steps
        self.repr_dim = 256

    def forward(self, states, actions):
        """
        Args:
            states: [B, T, Ch, H, W]
            actions: [B, T-1, 2]

        Output:
            predictions: [B, T, D]
        """
        return torch.randn((self.bs, self.n_steps, self.repr_dim)).to(self.device)


class Prober(torch.nn.Module):
    def __init__(
        self,
        embedding: int,
        arch: str,
        output_shape: List[int],
    ):
        super().__init__()
        self.output_dim = np.prod(output_shape)
        self.output_shape = output_shape
        self.arch = arch

        arch_list = list(map(int, arch.split("-"))) if arch != "" else []
        f = [embedding] + arch_list + [self.output_dim]
        layers = []
        for i in range(len(f) - 2):
            layers.append(torch.nn.Linear(f[i], f[i + 1]))
            layers.append(torch.nn.ReLU(True))
        layers.append(torch.nn.Linear(f[-2], f[-1]))
        self.prober = torch.nn.Sequential(*layers)

    def forward(self, e):
        output = self.prober(e)
        return output


class JEPAEncoder(nn.Module):
    """Encoder network for converting states into abstract representations"""
    def __init__(self, in_channels=2, hidden_dim=64, output_dim=256):
        super().__init__()
        # Parameter tuning options:
        # - hidden_dim: Increase for more capacity but slower training
        # - num_conv_layers: Add more layers for deeper feature extraction
        # - kernel_sizes: Adjust to capture different spatial scales
        self.conv1 = nn.Conv2d(in_channels, hidden_dim, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim*2, kernel_size=4, stride=2, padding=1) 
        self.conv3 = nn.Conv2d(hidden_dim*2, hidden_dim*4, kernel_size=4, stride=2, padding=1)
        
        # Batch norm helps training stability
        self.bn1 = nn.BatchNorm2d(hidden_dim)
        self.bn2 = nn.BatchNorm2d(hidden_dim*2)
        self.bn3 = nn.BatchNorm2d(hidden_dim*4)
        
        # Project to final embedding dimension
        self.fc = nn.Linear(hidden_dim*4 * 8 * 8, output_dim)
        
    def forward(self, x):
        # x shape: [batch_size, channels, height, width]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x

class JEPAPredictor(nn.Module):
    """Predictor network for forecasting future embeddings"""
    def __init__(self, input_dim=256, action_dim=2, hidden_dim=256):
        super().__init__()
        # Parameter tuning options:
        # - hidden_dim: Larger gives more capacity but slower training
        # - num_layers: More layers = more complex relationships
        self.net = nn.Sequential(
            nn.Linear(input_dim + action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim), 
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        
    def forward(self, state_embedding, action):
        x = torch.cat([state_embedding, action], dim=-1)
        return self.net(x)

class JEPA(nn.Module):
    """Full JEPA architecture combining encoder and predictor"""
    def __init__(self, state_channels=2, embedding_dim=256, action_dim=2):
        super().__init__()
        self.repr_dim = embedding_dim # Required for compatibility
        
        # Core components
        self.encoder = JEPAEncoder(
            in_channels=state_channels,
            output_dim=embedding_dim
        )
        self.predictor = JEPAPredictor(
            input_dim=embedding_dim,
            action_dim=action_dim
        )
        
    def forward(self, states, actions):
        """
        Forward pass producing embeddings for entire sequence
        
        Args:
            states: [batch_size, seq_len, channels, height, width] 
            actions: [batch_size, seq_len-1, action_dim]
            
        Returns:
            embeddings: [batch_size, seq_len, embedding_dim]
        """
        batch_size, seq_len = states.shape[:2]
        
        # Initialize output tensor
        embeddings = torch.zeros(
            batch_size, seq_len, self.repr_dim,
            device=states.device
        )
        
        # Initial encoding
        embeddings[:, 0] = self.encoder(states[:, 0])
        
        # Predict future embeddings
        for t in range(1, seq_len):
            prev_embedding = embeddings[:, t-1]
            action = actions[:, t-1]
            
            # Predict next embedding
            pred_embedding = self.predictor(prev_embedding, action)
            
            # Store prediction
            embeddings[:, t] = pred_embedding
            
        return embeddings

def load_model():
    """Initialize JEPA model"""
    model = JEPA(
        state_channels=2,
        embedding_dim=256,
        action_dim=2
    )
    return model.cuda()