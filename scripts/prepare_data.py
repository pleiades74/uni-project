"""
Data preparation script for downloading and organizing datasets
"""

import os
import argparse
import urllib.request
import zipfile
import tarfile
from tqdm import tqdm

def download_file(url, filename):
    """Download file with progress bar"""
    def progress_hook(count, block_size, total_size):
        if tqdm_instance:
            tqdm_instance.update(block_size)
    
    tqdm_instance = tqdm(unit='B', unit_scale=True, desc=filename, miniters=1)
    urllib.request.urlretrieve(url, filename, reporthook=progress_hook)
    tqdm_instance.close()

def prepare_dataset(args):
    """Download and prepare dataset"""
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.dataset == "dior":
        # DIOR dataset (aerial images)
        url = "https://example.com/dior.zip"  # Replace with actual URL
        print(f"Downloading DIOR dataset to {args.output_dir}...")
        download_file(url, os.path.join(args.output_dir, "dior.zip"))
        
        print("Extracting...")
        with zipfile.ZipFile(os.path.join(args.output_dir, "dior.zip"), 'r') as zip_ref:
            zip_ref.extractall(args.output_dir)
            
    elif args.dataset == "custom":
        # Handle custom dataset upload
        print("Please upload your dataset to Kaggle and provide the path")
        
    print(f"Dataset prepared in {args.output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="dior", 
                       choices=["dior", "dota", "custom"])
    parser.add_argument("--output_dir", type=str, default="../data/raw")
    args = parser.parse_args()
    
    prepare_dataset(args)
