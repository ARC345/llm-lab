import os
import csv
import json
import glob

RUNS_DIR = "runs"

def get_metrics(run_dir):
    metrics_path = os.path.join(run_dir, "metrics.csv")
    if not os.path.exists(metrics_path):
        return None
    
    last_val_metrics = {}
    with open(metrics_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Check if val_acc is present and not empty
            if row.get("val_acc") and row["val_acc"].strip():
                last_val_metrics = row
                
    return last_val_metrics

def get_probe_results(run_dir):
    probing_dir = os.path.join(run_dir, "probing")
    results = {}
    if not os.path.exists(probing_dir):
        return results
        
    for layer_dir in os.listdir(probing_dir):
        if not layer_dir.startswith("layer_"):
            continue
        
        json_path = os.path.join(probing_dir, layer_dir, "results.json")
        if os.path.exists(json_path):
            with open(json_path, "r") as f:
                data = json.load(f)
                layer_num = int(layer_dir.split("_")[1])
                results[layer_num] = data.get("accuracy", 0.0)
                
    return results

def main():
    runs = [
        "production_v1",
        "production_v2", 
        "production_v3_l4_10k",
        "production_v3_l6_10k",
        "production_v5_rope_10k"
    ]
    
    print(f"{'Run Name':<25} | {'Step':<5} | {'2-Hop Acc':<10} | {'3-Hop Acc':<10} | {'Probe Acc (Layer:Acc)'}")
    print("-" * 100)
    
    for run in runs:
        run_dir = os.path.join(RUNS_DIR, run)
        metrics = get_metrics(run_dir)
        probes = get_probe_results(run_dir)
        
        if metrics:
            step = metrics.get("step", "N/A")
            val_acc = float(metrics.get("val_acc", 0)) * 100
            ood_acc = float(metrics.get("ood_acc", 0)) * 100
            
            probe_str = ", ".join([f"L{k}:{v*100:.1f}%" for k, v in sorted(probes.items())])
            
            print(f"{run:<25} | {step:<5} | {val_acc:<10.2f} | {ood_acc:<10.2f} | {probe_str}")
        else:
            print(f"{run:<25} | N/A   | N/A        | N/A        | N/A")

if __name__ == "__main__":
    main()
