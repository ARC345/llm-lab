import json
import csv
import os
import torch
import yaml
from datetime import datetime
from .config import GPTConfig
# Import all models
from .model import GPT, ReluGPT, RmsGPT, RopeGPT, GeluGPT, ReasoningGPT, ReasoningRopeGPT
from .data.reasoning_dataset import TransitiveReasoningDataset

def load_model_from_checkpoint(checkpoint_path, device='cpu'):
    """
    Loads a model and config from a checkpoint file.
    """
    assert os.path.exists(checkpoint_path), f"Checkpoint not found at {checkpoint_path}"
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # helper to reconstruct config
    # The checkpoint stores the config object itself usually, or dict?
    # train.py stores: 'config': config (which is an object)
    if isinstance(checkpoint['config'], GPTConfig):
        config = checkpoint['config']
    else:
        # Fallback if it was stored as dict
        config = GPTConfig(**checkpoint['config'])
        
    # Re-initialize model
    # We need to know which class it was. train.py stores 'model': experiment_cls.__name__ in meta.jsonl
    # But checkpoint itself only stores state_dict.
    # Ah, in train.py: meta_payload = {'type': 'config', 'args': vars(args), 'model': experiment_cls.__name__}
    # But the checkpoint dict has: 'config': config. 
    # It does NOT store the model class name in the checkpoint dict explicitly in train.py lines 485-491.
    # Wait, train.py lines 485-491:
    # checkpoint = { 'model': model.state_dict(), ... 'config': config }
    # It doesn't store the class name.
    # However, `load_model` in `probe.py` (lines 18-26) hardcodes `ReasoningGPT`.
    # To be truly generic, we might need a way to infer it or just default to ReasoningGPT/GPT.
    # Let's assume ReasoningGPT for now as it seems to be the main focus or try to detect.
    # Actually, `train.py` line 482 uses `experiment_cls`.
    
    # For now, let's look at how probe.py did it: `from model import ReasoningGPT` then `model = ReasoningGPT(config)`.
    # Ideally we should store the architecture name in the checkpoint.
    # Since we can't change existing checkpoints easily, we can try to guess or standardise.
    # But `train.py` does `exp_logger.log_metadata` with the model name.
    # Let's check if the checkpoint config has cues.
    
    # For this refactor, I will allow passing the class or default to ReasoningGPT if not specified? 
    # Or just use ReasoningGPT since it covers most cases (GPT is superclass).
    # If the user uses ReluGPT, ReasoningGPT might not have the right blocks?
    # ReluGPT overrides build_block. ReasoningGPT overrides build_block too.
    # This is tricky without the class name in checkpoint.
    # BUT, `probe.py` was hardcoded to `ReasoningGPT`.
    # I will modify `train.py` to save the architecture name in the checkpoint in the future.
    # For now, I will add an optional `model_cls` arg to `load_model_from_checkpoint`, defaulting to ReasoningGPT.
    
    # Try to infer model class from checkpoint
    if 'model_cls' in checkpoint:
        model_cls_name = checkpoint['model_cls']
        # We need to find the class object. We can look in globals() or pre-defined map.
        # Since we imported all models:
        # from .model import GPT, ReluGPT, RmsGPT, RopeGPT, GeluGPT, ReasoningGPT, ReasoningRopeGPT
        # We can construct a map.
        
        # Mapping available models
        model_map = {
            'GPT': GPT,
            'ReluGPT': ReluGPT,
            'RmsGPT': RmsGPT,
            'RopeGPT': RopeGPT,
            'GeluGPT': GeluGPT,
            'ReasoningGPT': ReasoningGPT,
            'ReasoningRopeGPT': ReasoningRopeGPT
        }
        
        if model_cls_name in model_map:
            model_cls = model_map[model_cls_name]
        else:
            print(f"Warning: Unknown model class {model_cls_name} in checkpoint. Defaulting to ReasoningGPT.")
            model_cls = ReasoningGPT
    else:
        # Fallback for legacy checkpoints
        model_cls = ReasoningGPT 
    
    print(f"Initializing Model: {model_cls.__name__}")
    model = model_cls(config)
    
    # If the state dict has keys for projections, ReasoningGPT will handle it.
    # If it has Relu, etc?
    # The state_dict keys should match.
    
    try:
        model.load_state_dict(checkpoint['model'])
    except RuntimeError as e:
        print(f"Error loading state dict: {e}")
        print("Attempting strict=False load...")
        model.load_state_dict(checkpoint['model'], strict=False)
        
    model.to(device)
    model.eval()
    
    return model, config

def create_dataset(args, config, split='train'):
    """
    Factory validation function for creating datasets.
    """
    dataset_type = getattr(args, 'dataset_type', 'tinystories')
    assert dataset_type in ['reasoning', 'tinystories'], f"Unknown dataset type: {dataset_type}"
    
    if dataset_type == 'reasoning':
        if split == 'train':
            return TransitiveReasoningDataset(
                num_chains=getattr(args, 'num_chains', 400),
                seed=config.seed,
                chain_length=2,
                is_test=False
            )
        elif split == 'test':
            return TransitiveReasoningDataset(
                num_chains=getattr(args, 'test_chains', 100),
                seed=config.seed,
                chain_length=2,
                is_test=True
            )
        elif split == 'ood':
             return TransitiveReasoningDataset(
                num_chains=getattr(args, 'test_chains', 100),
                seed=config.seed + 1,
                chain_length=3,
                is_test=True
            )
    
    # Fallback / Other types (logic from train.py)
    # Return dummy or load text... this part was inline in train.py
    # For duplication reduction, we focus on the shared parts (Reasoning).
    return None

def get_activation_metric_hook(storage_dict, key):
    def hook(module, input, output):
        storage_dict[key] = {
            'dead': (output <= 0).float().mean().item(),
            'mean': output.mean().item(),
            'std': output.std().item()
        }
    return hook

class ExperimentLogger:
    def __init__(self, metrics_file='experiment_metrics.csv', meta_file='experiment_meta.jsonl'):
        self.metrics_file = metrics_file
        self.meta_file = meta_file
        self.csv_headers = None

    def log_metadata(self, metadata):
        """
        Appends a dictionary of metadata to the JSONL file.
        Adds a timestamp if not present.
        """
        if 'timestamp' not in metadata:
            metadata['timestamp'] = datetime.now().isoformat()
        
        with open(self.meta_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(metadata) + '\n')

    def log_metrics(self, metrics):
        """
        Appends a dictionary of metrics to the CSV file.
        Adds a timestamp if not present.
        Initialize headers on first write.
        """
        if 'timestamp' not in metrics:
            metrics['timestamp'] = datetime.now().isoformat()
        
        # Initialize headers if not already set or file doesn't exist
        file_exists = os.path.exists(self.metrics_file)
        
        if not self.csv_headers:
            if file_exists:
                with open(self.metrics_file, 'r', encoding='utf-8') as f:
                    reader = csv.reader(f)
                    try:
                        self.csv_headers = next(reader)
                    except StopIteration:
                        # File is empty
                        self.csv_headers = list(metrics.keys())
            else:
                self.csv_headers = list(metrics.keys())
        
        with open(self.metrics_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=self.csv_headers)
            if not file_exists:
                writer.writeheader()
            writer.writerow(metrics)
