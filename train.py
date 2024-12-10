import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, IterableDataset
import os
from tqdm import tqdm
import gc
import math
from models import JEPAWorldModel

class MemoryEfficientDataset(IterableDataset):
    def __init__(
        self,
        data_path: str,
        batch_size: int = 32,
        shuffle_buffer_size: int = 1000,
        prefetch_factor: int = 2
    ):
        self.data_path = data_path
        self.batch_size = batch_size
        self.shuffle_buffer_size = shuffle_buffer_size
        self.prefetch_factor = prefetch_factor
        
        # Memory map the files
        self.states = np.load(f"{data_path}/states.npy", mmap_mode='r')
        self.actions = np.load(f"{data_path}/actions.npy", mmap_mode='r')
        
        self.total_samples = len(self.states)
        self.indices = np.arange(self.total_samples)
    
    def __iter__(self):
        # Calculate worker info
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            start_idx = 0
            end_idx = self.total_samples
        else:
            per_worker = int(math.ceil(self.total_samples / worker_info.num_workers))
            worker_id = worker_info.id
            start_idx = worker_id * per_worker
            end_idx = min(start_idx + per_worker, self.total_samples)
        
        # Create a buffer for shuffling
        buffer = []
        
        # Iterate through assigned samples
        for idx in range(start_idx, end_idx):
            # Read individual samples
            states = torch.from_numpy(self.states[idx]).float()
            actions = torch.from_numpy(self.actions[idx]).float()
            
            buffer.append((states, actions))
            
            # When buffer is full, shuffle and yield samples
            if len(buffer) == self.shuffle_buffer_size:
                np.random.shuffle(buffer)
                for item in buffer:
                    yield item
                buffer = []
        
        # Yield remaining samples in buffer
        if buffer:
            np.random.shuffle(buffer)
            for item in buffer:
                yield item

def collate_fn(batch):
    states = torch.stack([item[0] for item in batch])
    actions = torch.stack([item[1] for item in batch])
    return states, actions

def train_jepa(
    model,
    data_path: str = "/scratch/DL24FA/train",
    num_epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    device: str = "cuda",
    save_path: str = "model_weights.pth",
    num_workers: int = 4,
    accumulation_steps: int = 4
):
    # Initialize model and optimizer
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs
    )
    
    # Create dataset and dataloader
    dataset = MemoryEfficientDataset(
        data_path,
        batch_size=batch_size,
        shuffle_buffer_size=1000
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collate_fn,
        prefetch_factor=2,
        pin_memory=True
    )
    
    # Training loop
    best_loss = float('inf')
    optimizer.zero_grad()
    
    # Create epoch progress bar
    epoch_pbar = tqdm(range(num_epochs), desc="Training Progress", position=0)
    
    for epoch in epoch_pbar:
        model.train()
        total_loss = 0
        batch_count = 0
        
        # Create batch progress bar
        batch_pbar = tqdm(dataloader, 
                         desc=f"Epoch {epoch+1}/{num_epochs}", 
                         position=1, 
                         leave=False)
        
        # Running averages for loss components
        avg_sim_loss = 0
        avg_std_loss = 0
        avg_cov_loss = 0
        
        for batch_idx, (states, actions) in enumerate(batch_pbar):
            try:
                # Move batch to GPU
                states = states.to(device, non_blocking=True)
                actions = actions.to(device, non_blocking=True)
                
                # Forward pass
                predictions = model(states, actions)
                target_embeddings = model.target_encode(states)
                
                # Compute loss
                loss, sim_loss, std_loss, cov_loss = model.compute_vicreg_loss(
                    predictions,
                    target_embeddings
                )
                
                # Update running averages
                avg_sim_loss = (avg_sim_loss * batch_count + sim_loss.item()) / (batch_count + 1)
                avg_std_loss = (avg_std_loss * batch_count + std_loss.item()) / (batch_count + 1)
                avg_cov_loss = (avg_cov_loss * batch_count + cov_loss.item()) / (batch_count + 1)
                
                # Scale loss for gradient accumulation
                loss = loss / accumulation_steps
                
                # Backward pass
                loss.backward()
                
                # Gradient accumulation
                if (batch_idx + 1) % accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    model._momentum_update_target_encoder()
                
                # Update metrics
                total_loss += loss.item() * accumulation_steps
                batch_count += 1
                
                # Update batch progress bar
                batch_pbar.set_postfix({
                    'loss': f"{total_loss/batch_count:.4f}",
                    'sim_loss': f"{avg_sim_loss:.4f}",
                    'std_loss': f"{avg_std_loss:.4f}",
                    'cov_loss': f"{avg_cov_loss:.4f}",
                    'lr': f"{optimizer.param_groups[0]['lr']:.6f}"
                })
                
                # Clear memory
                del states, actions, predictions, target_embeddings
                torch.cuda.empty_cache()
                
            except RuntimeError as e:
                print(f"\nError in batch: {e}")
                torch.cuda.empty_cache()
                continue
        
        # Update learning rate
        scheduler.step()
        
        # Calculate average loss for epoch
        avg_loss = total_loss / batch_count
        
        # Update epoch progress bar
        epoch_pbar.set_postfix({
            'avg_loss': f"{avg_loss:.4f}",
            'best_loss': f"{best_loss:.4f}"
        })
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, save_path)
            print(f"\nEpoch {epoch+1}: Saved best model with loss {best_loss:.4f}")
        
        # Clear memory between epochs
        gc.collect()
        torch.cuda.empty_cache()
        
        # Close batch progress bar
        batch_pbar.close()
    
    # Close epoch progress bar
    epoch_pbar.close()

if __name__ == "__main__":
    # Initialize model
    model = JEPAWorldModel(
        embedding_dim=256,
        momentum=0.99,
        use_momentum_target=True
    )
    
    # Train model
    train_jepa(model)