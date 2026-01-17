import argparse
import os
import json
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import legacy_compat

import yaml
from llm_lab.config import GPTConfig
from llm_lab.model import ReasoningGPT
from llm_lab.utils import load_model_from_checkpoint, create_dataset



def extract_activations(model, dataset, layer_idx, device='cpu', num_samples=1000, position=None):
    """
    Extracts activations from a specific layer for the test set.
    Goal: Predict the intermediate node (B) in A->B, B->C.
    """
    activations = []
    labels = [] # Intermediate node B
    
    # We need to hook into the model
    # Hook on block output
    # Hook logic
    # Hook logic
    target_activations = {}
    
    if layer_idx == -1:
        # Pre-hook signature: (module, input)
        def hook(module, input):
            # input is tuple (x,)
            target_activations['act'] = input[0].detach().cpu()
        handle = model.blocks[0].register_forward_pre_hook(hook)
    else:
        # Forward hook signature: (module, input, output)
        def hook(module, input, output):
            target_activations['act'] = output.detach().cpu()
        handle = model.blocks[layer_idx].register_forward_hook(hook)
    
    # Iterate dataset
    count = 0
    print(f"Extracting activations from layer {layer_idx} (Pos: {position if position is not None else 'Last'})...")
    
    indices = np.random.permutation(len(dataset))
    
    for idx in indices:
        if count >= num_samples: break
        
        chain = dataset.chains[idx]
        intermediate_node = chain[1] # B
        
        x, y = dataset._tokenize_chain(chain)
        x = x.unsqueeze(0).to(device)
        
        # Determine target position
        # If position arg is set, use it.
        # Else use last valid token position.
        
        target_pos = None
        if position is not None:
             target_pos = position
        else:
             # Default: last token before pad (used in generation)
             target_pos = (y != -100).nonzero(as_tuple=True)[0][-1].item()
        
        with torch.no_grad():
            _ = model(x)
            
        # act shape: (B, T, D)
        act_tensor = target_activations['act']
        
        # Validate position
        if target_pos >= act_tensor.shape[1]:
            continue # Skip if position out of bounds (e.g. padding not long enough)
            
        # extract at target_pos
        act = act_tensor[0, target_pos, :].numpy()
        
        activations.append(act)
        labels.append(dataset.node_to_id[intermediate_node])
        
        count += 1
        
    handle.remove()
    
    activations_np = np.array(activations)
    labels_np = np.array(labels)
    
    assert len(activations_np) > 0, "No activations extracted"
    assert len(activations_np) == len(labels_np), "Mismatch between activations and labels"
    
    return activations_np, labels_np

def train_probe(activations, labels, output_dir, layer_idx):
    print(f"Training probe on layer {layer_idx}...")
    X_train, X_test, y_train, y_test = train_test_split(activations, labels, test_size=0.2, random_state=42)
    
    # Linear Classifier
    # multi_class is deprecated/renamed in newer sklearn? or just defaults.
    # 'multinomial' is auto handled usually.
    # Check sklearn version: >=1.8.0.
    # In recent versions 'multi_class' is valid? Or 'multiclass'?
    # It seems recent sklearn (1.5+) might have deprecated it or I am misremembering.
    # Actually, let's just use default 'auto' which handles multinomial if labels are > 2.
    clf = LogisticRegression(random_state=42, max_iter=1000)
    clf.fit(X_train, y_train)
    
    acc = clf.score(X_test, y_test)
    assert 0.0 <= acc <= 1.0, f"Accuracy {acc} out of range [0, 1]"
    print(f"Probe Accuracy Layer {layer_idx}: {acc:.4f}")
    
    # Confusion Matrix
    y_pred = clf.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=False, cmap='Blues')
    plt.title(f'Confusion Matrix Layer {layer_idx} (Acc: {acc:.2f})')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(os.path.join(output_dir, f'confusion_matrix_layer{layer_idx}.png'))
    plt.close()
    
    return acc, clf

def visualize_tsne(activations, labels, output_dir, layer_idx, node_map):
    print("Running t-SNE...")
    tsne = TSNE(n_components=2, random_state=42)
    X_embedded = tsne.fit_transform(activations)
    
    # Convert labels to chars
    itos = {v:k for k,v in node_map.items()}
    label_chars = [itos[l] for l in labels]
    
    plt.figure(figsize=(12, 10))
    sns.scatterplot(x=X_embedded[:,0], y=X_embedded[:,1], hue=label_chars, palette='tab20', legend='full')
    plt.title(f't-SNE of Layer {layer_idx} Activations (Colored by Intermediate Node)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'tsne_layer{layer_idx}.png'))
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--layer', type=int, default=1, help="Layer index to probe (0-indexed)")
    parser.add_argument('--output_dir', type=str, default='outputs/probing')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--test_chains', type=int, default=1000, help='Number of chains for probing')
    parser.add_argument('--position', type=int, default=None, help='Specific token position to probe (0-indexed)')
    args = parser.parse_args()

    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load Model
    model, config = load_model_from_checkpoint(args.checkpoint, args.device)
    
    # Load Dataset (Validation split)
    # We need to simulate args for create_dataset
    # create_dataset expects args.dataset_type etc.
    # We can mock it or just populate args
    # But probe.py args might not have dataset_type.
    # We should default it or add it to parser.
    
    # For now, let's manually set it in args since probe is specific to reasoning usually?
    # Or just use create_dataset if we add the argument.
    args.dataset_type = 'reasoning'
    # Default chains if not set
    if not hasattr(args, 'test_chains'):
        args.test_chains = 100
        
    test_ds = create_dataset(args, config, 'test')
    
    # Extract
    acts, labels = extract_activations(model, test_ds, args.layer, args.device, num_samples=args.test_chains, position=args.position)
    
    # Train Probe
    acc, clf = train_probe(acts, labels, args.output_dir, args.layer)
    
    # Visualize
    visualize_tsne(acts, labels, args.output_dir, args.layer, test_ds.node_to_id)
    
    # Save results
    # Save results
    results_data = {
        'layer': args.layer,
        'accuracy': acc,
        'samples': args.test_chains,
        'seed': args.seed,
        'position': args.position,
        'checkpoint': args.checkpoint
    }
    with open(os.path.join(args.output_dir, 'results.json'), 'w') as f:
        json.dump(results_data, f, indent=2)
