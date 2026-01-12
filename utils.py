import json
import csv
import os
from datetime import datetime

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
            self.csv_headers = list(metrics.keys())
        
        with open(self.metrics_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=self.csv_headers)
            if not file_exists:
                writer.writeheader()
            writer.writerow(metrics)
