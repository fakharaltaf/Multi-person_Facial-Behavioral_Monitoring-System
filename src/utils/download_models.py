"""
Model downloader utility
Downloads pre-trained ONNX models for the tracking system
"""

import os
import sys
from pathlib import Path
import urllib.request
import yaml
from typing import Dict, Any
from tqdm import tqdm


class DownloadProgressBar:
    """Progress bar for downloads"""
    
    def __init__(self):
        self.pbar = None
    
    def __call__(self, block_num, block_size, total_size):
        if not self.pbar:
            self.pbar = tqdm(total=total_size, unit='B', unit_scale=True, desc='Downloading')
        
        downloaded = block_num * block_size
        if downloaded < total_size:
            self.pbar.update(block_size)
        else:
            self.pbar.close()


def download_file(url: str, output_path: str) -> bool:
    """
    Download file from URL with progress bar
    
    Args:
        url: Download URL
        output_path: Local output path
    
    Returns:
        True if successful
    """
    try:
        print(f"Downloading: {url}")
        print(f"Saving to: {output_path}")
        
        # Create directory if needed
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Download with progress bar
        urllib.request.urlretrieve(url, output_path, DownloadProgressBar())
        
        # Verify file exists and has size
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            print(f"✓ Downloaded successfully: {output_path}")
            return True
        else:
            print(f"✗ Download failed: file is empty or missing")
            return False
            
    except Exception as e:
        print(f"✗ Error downloading {url}: {e}")
        return False


def load_model_config(config_path: str = None) -> Dict[str, Any]:
    """
    Load model configuration from YAML
    
    Args:
        config_path: Path to model_urls.yaml
    
    Returns:
        Model configuration dictionary
    """
    if config_path is None:
        project_root = Path(__file__).parent.parent.parent
        config_path = project_root / "config" / "model_urls.yaml"
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def download_model(model_category: str, model_name: str, config: Dict[str, Any]) -> bool:
    """
    Download a specific model
    
    Args:
        model_category: Category (e.g., 'detection', 'recognition')
        model_name: Model name within category
        config: Model configuration dictionary
    
    Returns:
        True if successful
    """
    if model_category not in config:
        print(f"✗ Unknown model category: {model_category}")
        return False
    
    if model_name not in config[model_category]:
        print(f"✗ Unknown model: {model_name} in {model_category}")
        return False
    
    model_info = config[model_category][model_name]
    
    # Get project root and construct absolute path
    project_root = Path(__file__).parent.parent.parent
    local_path = project_root / model_info['local_path']
    
    # Check if already downloaded
    if local_path.exists():
        size_mb = local_path.stat().st_size / (1024 * 1024)
        print(f"✓ Model already exists: {local_path} ({size_mb:.1f} MB)")
        return True
    
    # Download
    print(f"\nDownloading {model_category}/{model_name}")
    print(f"Description: {model_info.get('description', 'N/A')}")
    
    if 'url' in model_info and model_info['url'].startswith('http'):
        return download_file(model_info['url'], str(local_path))
    else:
        print(f"✗ No valid URL found. Manual download required.")
        print(f"   Please download manually and place at: {local_path}")
        if 'alternative' in model_info:
            print(f"   Note: {model_info['alternative']}")
        return False


def download_all_models(config: Dict[str, Any]) -> None:
    """
    Download all models listed in configuration
    
    Args:
        config: Model configuration dictionary
    """
    print("=" * 60)
    print("MODEL DOWNLOADER - Multi-Person Tracking System")
    print("=" * 60)
    
    total = 0
    successful = 0
    failed = []
    
    for category in config.keys():
        if category == 'notes':
            continue
        
        print(f"\n{'='*60}")
        print(f"Category: {category.upper()}")
        print(f"{'='*60}")
        
        for model_name in config[category].keys():
            total += 1
            if download_model(category, model_name, config):
                successful += 1
            else:
                failed.append(f"{category}/{model_name}")
    
    # Summary
    print(f"\n{'='*60}")
    print("DOWNLOAD SUMMARY")
    print(f"{'='*60}")
    print(f"Total models: {total}")
    print(f"Successful: {successful}")
    print(f"Failed: {len(failed)}")
    
    if failed:
        print("\nFailed downloads (require manual intervention):")
        for model in failed:
            print(f"  - {model}")
        print("\nPlease check config/model_urls.yaml for download instructions")


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Download models for tracking system')
    parser.add_argument(
        '--model',
        type=str,
        help='Download specific model (format: category/model_name)'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Download all models'
    )
    parser.add_argument(
        '--list',
        action='store_true',
        help='List available models'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_model_config()
    
    if args.list:
        print("Available models:")
        for category in config.keys():
            if category == 'notes':
                continue
            print(f"\n{category}:")
            for model_name, info in config[category].items():
                print(f"  - {model_name}: {info.get('description', 'N/A')}")
        return
    
    if args.model:
        # Download specific model
        parts = args.model.split('/')
        if len(parts) != 2:
            print("Error: Model format should be category/model_name")
            return
        
        category, model_name = parts
        download_model(category, model_name, config)
    
    elif args.all:
        # Download all models
        download_all_models(config)
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
