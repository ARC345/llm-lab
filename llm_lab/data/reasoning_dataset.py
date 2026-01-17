import torch
import random
import numpy as np

class TransitiveReasoningDataset:
    def __init__(self, num_chains=400, chain_length=2, seed=42, is_test=False):
        """
        Args:
            num_chains: Number of distinct chains to generate
            chain_length: Number of hops (2 or 3)
            seed: Random seed for reproducibility
            is_test: If True, generates held-out test set
        """
        self.nodes = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")  # 26 nodes
        self.node_to_id = {node: i for i, node in enumerate(self.nodes)}
        self.special_tokens = {'→': 26, '.': 27, 'Q': 28, '?': 29, '[PAD]': 30}
        self.vocab_size = 32  # 26 nodes + 5 special + 1 reserved? Max is 32.
        # Let's map exactly as specified: A-Z (0-25), special (26-30).
        # We need 32 tokens total? Spec says "vocabulary = 32 tokens".
        # 26 + 5 = 31. We can leave 31 unused or for [EOS] if needed.
        
        self.chain_length = chain_length
        self.seed = seed
        self.data = []
        
        self._generate_dataset(num_chains, is_test)

    def _generate_dataset(self, num_chains, is_test):
        # We need to split on *complete chains*.
        # Total set of possible chains is huge, but we want controlled splitting.
        # Strategy: Generate a large pool of valid chains deterministically sorted, then split.
        
        random.seed(self.seed)
        np.random.seed(self.seed)
        
        all_chains = []
        
        if self.chain_length < 1:
            raise ValueError("Chain length must be at least 1")
            
        # Generic generation using permutations
        # A chain of length K requires K+1 distinct nodes
        num_nodes_needed = self.chain_length + 1
        if num_nodes_needed > len(self.nodes):
            raise ValueError(f"Chain length {self.chain_length} requires {num_nodes_needed} nodes, but only {len(self.nodes)} available")
            
        import itertools
        # Generate all permutations of length K+1
        # This can be huge, so for larger lengths we should sample instead of generating all
        # 26!/(26-3)! is small (15600), but 26!/(26-7)! is huge (3.3e9)
        
        if self.chain_length <= 2:
            # For small lengths, we can generate all and shuffle
            all_chains = [list(p) for p in itertools.permutations(self.nodes, num_nodes_needed)]
            random.shuffle(all_chains)
            
            # Split logic for training data (only relevant for typically trained lengths like 2)
            # Use 80% for training if not test, else 20%
            split_idx = int(0.8 * len(all_chains))
            
            if is_test:
                candidates = all_chains[split_idx:]
            else:
                candidates = all_chains[:split_idx]
                
            if len(candidates) > num_chains:
                self.chains = candidates[:num_chains]
            else:
                self.chains = candidates
                
        else:
            # For OOD/longer lengths, we just sample 'num_chains' random permutations
            # We don't need a train/test split because the Model never saw THIS depth before.
            # But to be safe and consistent with "is_test", we can just generate N distinct ones.
            # Since space is huge, random sampling collisions are rare.
            
            seen = set()
            self.chains = []
            max_attempts = num_chains * 10
            attempts = 0
            
            while len(self.chains) < num_chains and attempts < max_attempts:
                p = tuple(random.sample(self.nodes, num_nodes_needed))
                if p not in seen:
                    seen.add(p)
                    self.chains.append(list(p))
                attempts += 1
            
    def __len__(self):
        return len(self.chains)
    
    def __getitem__(self, idx):
        chain = self.chains[idx]
        return self._tokenize_chain(chain)
        
    def _tokenize_chain(self, chain):
        # Format: [A][→][B][.][B][→][C][.][Q][A][?][→]... Target: [C]
        # For 3-hop: [A][→][B][.][B][→][C][.][C][→][D][.][Q][A][?][→]... Target: [D]
        
        seq = []
        
        # Input Chain context
        for i in range(len(chain) - 1):
            src = chain[i]
            tgt = chain[i+1]
            seq.extend([self.node_to_id[src], self.special_tokens['→'], self.node_to_id[tgt], self.special_tokens['.']])
            
        # Query
        start_node = chain[0]
        end_node = chain[-1]
        
        seq.extend([self.special_tokens['Q'], self.node_to_id[start_node], self.special_tokens['?'], self.special_tokens['→']])
        
        # Target (for training, we append it to input for autoregressive loss)
        # But wait, standard GPT training inputs are X, targets are shifted.
        # Spec:
        # Input:  [A][→][B][.][B][→][C][.][Q][A][?][→]
        # Target: [C]
        # For a standard GPT, we usually feed "Input + Target" and mask loss.
        # So full sequence is ...[→][C]
        
        full_seq = seq + [self.node_to_id[end_node]]
        
        # Determine max length (spec says 64, but 32 is sufficient for <20 tokens)
        max_len = 32
        
        x = torch.full((max_len,), self.special_tokens['[PAD]'], dtype=torch.long)
        y = torch.full((max_len,), -100, dtype=torch.long) # -100 is ignore_index
        
        # Fill x
        seq_len = len(full_seq)
        x[:seq_len-1] = torch.tensor(full_seq[:-1], dtype=torch.long)
        
        # Fill y (target is next token)
        # We only want to predict the very last token [C]
        # In a standard autoregressive setup:
        # Input at t: x[t]
        # Target at t: x[t+1]
        # We want loss ONLY when predicting the final answer.
        # The sequence fed to model is full_seq[:-1].
        # The target for position -1 is full_seq[-1].
        
        # Let's align with the user spec "Only compute loss on target token".
        # If input is "A->B. B->C. Q A ? ->", next token is "C".
        # So at the last position of input, target is C.
        
        target_pos = len(seq) - 1 # Position of the final '->' 
        
        # IMPORTANT: standard GPT forward(x, y) expects x and y to be same shape (B, T).
        # y is typically x shifted by 1.
        # Here we manually construct y to be -100 everywhere except the final position.
        
        # Standard:
        # x: [A, ->, B, ..., ->]
        # y_true: [->, B, ..., C]
        
        # We want y to be [-100, -100, ..., C]
        
        # Construct actual tokens
        actual_tokens = torch.tensor(full_seq, dtype=torch.long)
        length = len(actual_tokens)
        
        x[:length-1] = actual_tokens[:-1]
        # y should be shifted by 1
        y_shifted = actual_tokens[1:].clone()
        
        # Mask everything in y except the final token?
        # Spec: "Only compute loss on target token (final answer position)"
        # The final answer C is at index length-1 in actual_tokens.
        # In y_shifted (which corresponds to predictions from x[:length-1]), C is at index length-2.
        
        mask = torch.ones_like(y_shifted) * -100
        mask[-1] = y_shifted[-1] # The last token in the sequence (the answer)
        
        y[:length-1] = mask
        
        return x, y

if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument('--inspect', action='store_true', help='Inspect the dataset')
    parser.add_argument('--export', type=str, default=None, help='Export samples to a file')
    parser.add_argument('--num_samples', type=int, default=10, help='Number of samples to inspect/export')
    args = parser.parse_args()

    if args.export:
        print(f"Exporting {args.num_samples} samples to {args.export}...")
        ds = TransitiveReasoningDataset(num_chains=args.num_samples, chain_length=2, seed=42, is_test=False)
        itos = {v:k for k,v in ds.node_to_id.items()}
        for k,v in ds.special_tokens.items():
            itos[v] = k
            
        with open(args.export, "w") as f:
            f.write(f"Dataset Sample (Max Len: {len(ds[0][0])})\n")
            f.write("="*50 + "\n\n")
            
            for i in range(args.num_samples):
                if i >= len(ds): break
                x, y = ds[i]
                
                x_tokens = [itos.get(t.item(), str(t.item())) for t in x]
                x_str = " ".join(x_tokens)
                
                y_valid = (y != -100)
                target_token_str = "None"
                if y_valid.any():
                    last_idx = y_valid.nonzero()[-1].item()
                    target_val = y[last_idx].item()
                    target_token_str = itos.get(target_val, str(target_val))
                
                f.write(f"Sample {i+1}:\n")
                f.write(f"Input:  {x_str}\n")
                f.write(f"Target: {target_token_str}\n")
                f.write("-" * 20 + "\n")
        print("Done.")

    if args.inspect: # Default behavior if no args? Or explicit?
        print("Inspecting Dataset...")
        ds = TransitiveReasoningDataset(num_chains=args.num_samples, chain_length=2, seed=42, is_test=False)
        itos = {v:k for k,v in ds.node_to_id.items()}
        for k,v in ds.special_tokens.items():
            itos[v] = k
            
        print(f"Vocab size: {ds.vocab_size}")
        for i in range(min(3, len(ds))):
            x, y = ds[i]
            x_tokens = [itos.get(t.item(), str(t.item())) for t in x if t.item() in itos]
            x_str = " ".join(x_tokens)
            
            y_valid_idx = (y != -100).nonzero(as_tuple=True)[0]
            y_targets = []
            for idx in y_valid_idx:
                val = y[idx].item()
                token = itos.get(val, str(val))
                y_targets.append(f"@{idx.item()}={token}")
                
            print(f"\nSample {i}:")
            print(f"Input: {x_str}")
            print(f"Targets: {', '.join(y_targets)}")
