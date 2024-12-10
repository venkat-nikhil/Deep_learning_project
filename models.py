import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import Tensor
from typing import Tuple, Optional

def build_mlp(layers_dims):
    layers = []
    for i in range(len(layers_dims) - 2):
        layers.append(nn.Linear(layers_dims[i], layers_dims[i + 1]))
        layers.append(nn.BatchNorm1d(layers_dims[i + 1]))
        layers.append(nn.ReLU(True))
    layers.append(nn.Linear(layers_dims[-2], layers_dims[-1]))
    return nn.Sequential(*layers)

class Prober(nn.Module):
    def __init__(self, embedding: int, arch: str, output_shape):
        super().__init__()
        self.output_dim = np.prod(output_shape)
        self.output_shape = output_shape
        self.arch = arch

        arch_list = list(map(int, arch.split("-"))) if arch != "" else []
        f = [embedding] + arch_list + [self.output_dim]
        layers = []
        for i in range(len(f) - 2):
            layers.append(nn.Linear(f[i], f[i + 1]))
            layers.append(nn.ReLU(True))
        layers.append(nn.Linear(f[-2], f[-1]))
        self.prober = nn.Sequential(*layers)

    def forward(self, e):
        return self.prober(e)
        


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple, Optional

class ConvEncoder(nn.Module):
    def __init__(self, in_channels: int = 2, embedding_dim: int = 256):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(128, 256, 3, stride=2, padding=1)
        
        # Calculate output size after convolutions (65x65 input)
        self.conv_output_size = 256 * 5 * 5  # Updated for 65x65 input
        self.fc = nn.Linear(self.conv_output_size, embedding_dim)
        
        self.norm1 = nn.BatchNorm2d(32)
        self.norm2 = nn.BatchNorm2d(64)
        self.norm3 = nn.BatchNorm2d(128)
        self.norm4 = nn.BatchNorm2d(256)

    def forward(self, x: Tensor) -> Tensor:
        x = F.relu(self.norm1(self.conv1(x)))  # 33x33
        x = F.relu(self.norm2(self.conv2(x)))  # 17x17
        x = F.relu(self.norm3(self.conv3(x)))  # 9x9
        x = F.relu(self.norm4(self.conv4(x)))  # 5x5
        x = x.view(-1, self.conv_output_size)
        x = self.fc(x)
        return x

class Predictor(nn.Module):
    def __init__(self, embedding_dim: int = 256, action_dim: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim + action_dim, embedding_dim),  # Changed input size
            nn.LayerNorm(embedding_dim),
            nn.ReLU(True),
            nn.Linear(embedding_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.ReLU(True),
            nn.Linear(embedding_dim, embedding_dim)  # Output matches embedding_dim
        )

    def forward(self, state_embedding: Tensor, action: Tensor) -> Tensor:
        x = torch.cat([state_embedding, action], dim=-1)  # [B, embedding_dim + 2]
        return self.net(x)  # [B, embedding_dim]

class JEPAWorldModel(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 256,
        momentum: float = 0.99,
        use_momentum_target: bool = True
    ):
        super().__init__()
        self.repr_dim = embedding_dim
        self.momentum = momentum
        self.use_momentum_target = use_momentum_target
        
        # Online encoder and predictor
        self.encoder = ConvEncoder(embedding_dim=embedding_dim)
        self.predictor = Predictor(embedding_dim=embedding_dim)
        
        # Target encoder (momentum-updated)
        self.target_encoder = ConvEncoder(embedding_dim=embedding_dim)
        
        # Initialize target encoder with same weights
        if use_momentum_target:
            for param_q, param_k in zip(self.encoder.parameters(), 
                                      self.target_encoder.parameters()):
                param_k.data.copy_(param_q.data)
                param_k.requires_grad = False
                
        # VICReg projector for regularization (embedding_dim -> 512)
        self.projector = nn.Sequential(
            nn.Linear(embedding_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 512)
        )

    @torch.no_grad()
    def _momentum_update_target_encoder(self):
        """
        Momentum update of the target encoder
        """
        if not self.use_momentum_target:
            return
            
        for param_q, param_k in zip(self.encoder.parameters(),
                                   self.target_encoder.parameters()):
            param_k.data = param_k.data * self.momentum + \
                          param_q.data * (1. - self.momentum)

    def encode(self, states: Tensor) -> Tensor:
        """
        Encode states using the online encoder
        Args:
            states: [B, T, C, H, W]
        Returns:
            embeddings: [B, T, embedding_dim]
        """
        B, T, C, H, W = states.shape
        states = states.view(-1, C, H, W)
        embeddings = self.encoder(states)
        return embeddings.view(B, T, -1)

    def target_encode(self, states: Tensor) -> Tensor:
        """
        Encode states using the target encoder
        """
        B, T, C, H, W = states.shape
        states = states.view(-1, C, H, W)
        with torch.no_grad():
            embeddings = self.target_encoder(states)
        return embeddings.view(B, T, -1)

    def forward(self, states: Tensor, actions: Tensor) -> Tensor:
        """
        Forward pass for prediction
        Args:
            states: [B, T, C, H, W] or [B, 1, C, H, W]
            actions: [B, T-1, 2]
        Returns:
            predictions: [B, T, D]
        """
        B, T, C, H, W = states.shape
        
        if T == 1:  # Inference mode
            # Encode initial state
            current_state = self.encode(states)  # [B, 1, D]
            predictions = [current_state]
            
            # Predict future states
            for t in range(actions.shape[1]):
                next_state = self.predictor(
                    current_state.squeeze(1),
                    actions[:, t]
                )
                predictions.append(next_state.unsqueeze(1))
                current_state = next_state.unsqueeze(1)
                
            predictions = torch.cat(predictions, dim=1)  # [B, T, D]
            
        else:  # Training mode
            # Encode all states
            encoded_states = self.encode(states)  # [B, T, D]
            predictions = [encoded_states[:, 0:1]]  # First state is given
            
            # Predict next states
            for t in range(T-1):
                next_state = self.predictor(
                    encoded_states[:, t],
                    actions[:, t]
                )
                predictions.append(next_state.unsqueeze(1))
                
            predictions = torch.cat(predictions, dim=1)  # [B, T, D]
            
        return predictions

    def compute_vicreg_loss(
        self,
        pred_embeddings: Tensor,
        target_embeddings: Tensor,
        sim_coef: float = 25.0,
        std_coef: float = 25.0,
        cov_coef: float = 1.0
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Compute VICReg loss components
        """
        # Project embeddings to same dimension (512)
        pred_proj = self.projector(pred_embeddings.view(-1, self.repr_dim))
        with torch.no_grad():
            target_proj = self.projector(target_embeddings.view(-1, self.repr_dim))

        # Invariance loss
        sim_loss = F.mse_loss(pred_proj, target_proj)

        # Variance loss
        std_pred = torch.sqrt(pred_proj.var(dim=0) + 1e-6)
        std_target = torch.sqrt(target_proj.var(dim=0) + 1e-6)
        std_loss = torch.mean(F.relu(1 - std_pred)) + torch.mean(F.relu(1 - std_target))

        # Covariance loss
        pred_proj = pred_proj - pred_proj.mean(dim=0)
        target_proj = target_proj - target_proj.mean(dim=0)
        cov_pred = (pred_proj.T @ pred_proj) / (pred_proj.shape[0] - 1)
        cov_target = (target_proj.T @ target_proj) / (target_proj.shape[0] - 1)
        cov_loss = off_diagonal(cov_pred).pow_(2).sum() / pred_proj.shape[1] + \
                   off_diagonal(cov_target).pow_(2).sum() / target_proj.shape[1]

        # Combine losses
        total_loss = sim_coef * sim_loss + std_coef * std_loss + cov_coef * cov_loss

        return total_loss, sim_loss, std_loss, cov_loss

def off_diagonal(x: Tensor) -> Tensor:
    """
    Return off-diagonal elements of a square matrix
    """
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()