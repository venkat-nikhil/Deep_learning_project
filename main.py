from dataset import create_wall_dataloader
from evaluator import ProbingEvaluator
import torch
from models import JEPAWorldModel
import os

def get_device():
    """Check for GPU availability."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    return device

def load_data(device):
    data_path = "/scratch/DL24FA"
    
    probe_train_ds = create_wall_dataloader(
        data_path=f"{data_path}/probe_normal/train",
        probing=True,
        device=device,
        train=True,
    )
    
    probe_val_normal_ds = create_wall_dataloader(
        data_path=f"{data_path}/probe_normal/val",
        probing=True,
        device=device,
        train=False,
    )
    
    probe_val_wall_ds = create_wall_dataloader(
        data_path=f"{data_path}/probe_wall/val",
        probing=True,
        device=device,
        train=False,
    )
    
    probe_val_ds = {"normal": probe_val_normal_ds, "wall": probe_val_wall_ds}
    
    return probe_train_ds, probe_val_ds

def load_model(device):
    model = JEPAWorldModel(
        embedding_dim=256,
        momentum=0.99,
        use_momentum_target=True
    ).to(device)
    
    weights_path = 'model_weights.pth'
    if os.path.exists(weights_path):
        # Add verification step
        checkpoint = torch.load(weights_path, map_location=device)
        print("Checkpoint keys:", checkpoint.keys())  # Check what's in checkpoint
        
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            print("Original state dict keys:", list(state_dict.keys())[:5])  # Check first few keys
        else:
            state_dict = checkpoint
        
        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = key.replace('module.', '')
            new_state_dict[new_key] = value
        
        print("New state dict keys:", list(new_state_dict.keys())[:5])  # Check modified keys
        
        # Check if keys match
        model_keys = set(model.state_dict().keys())
        load_keys = set(new_state_dict.keys())
        missing_keys = model_keys - load_keys
        extra_keys = load_keys - model_keys
        if missing_keys:
            print("Missing keys:", missing_keys)
        if extra_keys:
            print("Extra keys:", extra_keys)
            
        model.load_state_dict(new_state_dict)
        
        # Verify some weights changed
        print("Sample weight value:", next(iter(model.parameters())).mean().item())
    else:
        print("No pre-trained weights found. Using initialized model.")
    
    model.eval()
    return model

def evaluate_model(device, model, probe_train_ds, probe_val_ds):
    evaluator = ProbingEvaluator(
        device=device,
        model=model,
        probe_train_ds=probe_train_ds,
        probe_val_ds=probe_val_ds,
        quick_debug=False,
    )
    
    prober = evaluator.train_pred_prober()
    avg_losses = evaluator.evaluate_all(prober=prober)
    
    for probe_attr, loss in avg_losses.items():
        print(f"{probe_attr} loss: {loss:.4f}")
    
    return avg_losses

if __name__ == "__main__":
    device = get_device()
    probe_train_ds, probe_val_ds = load_data(device)
    model = load_model(device)
    evaluate_model(device, model, probe_train_ds, probe_val_ds)