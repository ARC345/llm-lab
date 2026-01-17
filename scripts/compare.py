import argparse
import os
import json
import csv
import sys

def load_metrics(run_dir):
    metrics_file = os.path.join(run_dir, 'metrics.csv')
    if not os.path.exists(metrics_file):
        print(f"Warning: No metrics.csv found in {run_dir}")
        return None
    
    data = []
    headers = []
    try:
        with open(metrics_file, 'r') as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames
            for row in reader:
                # Convert to floats where possible
                processed_row = {}
                for k, v in row.items():
                    try:
                        if v and v.strip(): # Check if not empty
                            processed_row[k] = float(v)
                        else:
                            processed_row[k] = None
                    except ValueError:
                        processed_row[k] = v
                data.append(processed_row)
    except Exception as e:
        print(f"Error reading {metrics_file}: {e}")
        return None
        
    return data

def get_column(data, key):
    return [row.get(key) for row in data if row.get(key) is not None]

def get_xy(data, x_key, y_key):
    xs, ys = [], []
    for row in data:
        v_x = row.get(x_key)
        v_y = row.get(y_key)
        if isinstance(v_x, (int, float)) and isinstance(v_y, (int, float)):
            xs.append(v_x)
            ys.append(v_y)
    return xs, ys

def get_last_valid(data, key):
    # Search backwards
    for row in reversed(data):
        val = row.get(key)
        if isinstance(val, (int, float)):
            return val
    return 'N/A'

def generate_comparison(baseline_dir, bottleneck_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    data_base = load_metrics(baseline_dir)
    data_bottle = load_metrics(bottleneck_dir)
    
    if not data_base or not data_bottle:
        print("Could not load metrics from one or both directories.")
        return

    # Try plotting
    try:
        import matplotlib.pyplot as plt
        
        # 1. Loss Comparison
        plt.figure(figsize=(10, 6))
        bx, by = get_xy(data_base, 'step', 'val_loss')
        tx, ty = get_xy(data_bottle, 'step', 'val_loss')
        plt.plot(bx, by, label='Baseline')
        plt.plot(tx, ty, label='Bottleneck')
        plt.title('Validation Loss Comparison')
        plt.xlabel('Step')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(os.path.join(output_dir, 'loss_comparison.png'))
        plt.close()
        
        # 2. Accuracy Comparison
        plt.figure(figsize=(12, 5))
        
        # Val Acc
        plt.subplot(1, 2, 1)
        bx, by = get_xy(data_base, 'step', 'val_acc')
        tx, ty = get_xy(data_bottle, 'step', 'val_acc')
        plt.plot(bx, by, label='Baseline')
        plt.plot(tx, ty, label='Bottleneck')
        plt.title('2-Hop Test Accuracy')
        plt.legend()
        
        # OOD Acc
        plt.subplot(1, 2, 2)
        bx, by = get_xy(data_base, 'step', 'ood_acc')
        tx, ty = get_xy(data_bottle, 'step', 'ood_acc')
        plt.plot(bx, by, label='Baseline')
        plt.plot(tx, ty, label='Bottleneck')
        plt.title('3-Hop OOD Accuracy')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'accuracy_comparison.png'))
        plt.close()

    except ImportError:
        print("Matplotlib not found, skipping plots.")
    except Exception as e:
        print(f"Plotting failed: {e}")

    # Generate Report
    
    report = f"""# Comparison Report
    
## Final Metrics

### Baseline (V2: ReasoningGPT)
- Test Acc (2-hop): {get_last_valid(data_base, 'val_acc')}
- OOD Acc (3-hop): {get_last_valid(data_base, 'ood_acc')}

### Bottleneck (V4: ReasoningRopeGPT)
- Test Acc (2-hop): {get_last_valid(data_bottle, 'val_acc')}
- OOD Acc (3-hop): {get_last_valid(data_bottle, 'ood_acc')}

## Observations
- OOD Accuracy Check: Baseline vs Roped
"""
    with open(os.path.join(output_dir, 'comparison_report.md'), 'w') as f:
        f.write(report)
        
    print(f"Comparison generated in {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--run1', type=str, required=True, help='Path to baseline run directory')
    parser.add_argument('--run2', type=str, required=True, help='Path to bottleneck run directory')
    parser.add_argument('--output_dir', type=str, default='outputs/comparison')
    args = parser.parse_args()
    
    generate_comparison(args.run1, args.run2, args.output_dir)
