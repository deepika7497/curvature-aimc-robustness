#!/usr/bin/env python3
"""
Download and setup Tiny ImageNet dataset.
Tiny ImageNet is a subset of ImageNet with 200 classes, 500 training images per class,
and 50 validation images per class.
"""

import os
import zipfile
import requests
import shutil
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Tiny ImageNet URLs
TINY_IMAGENET_URL = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
TINY_IMAGENET_FILENAME = "tiny-imagenet-200.zip"

def download_file(url, filename, data_dir):
    """Download a file from URL."""
    filepath = os.path.join(data_dir, filename)
    
    if os.path.exists(filepath):
        logger.info(f"File {filename} already exists, skipping download.")
        return filepath
    
    logger.info(f"Downloading {filename} from {url}...")
    
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        logger.info(f"Downloaded {filename} successfully!")
        return filepath
        
    except Exception as e:
        logger.error(f"Failed to download {filename}: {e}")
        raise

def extract_zip(zip_path, extract_dir):
    """Extract a zip file."""
    logger.info(f"Extracting {zip_path} to {extract_dir}...")
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        
        logger.info("Extraction completed successfully!")
        
    except Exception as e:
        logger.error(f"Failed to extract {zip_path}: {e}")
        raise

def setup_tiny_imagenet(data_dir):
    """Download and setup Tiny ImageNet dataset."""
    
    # Create data directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)
    
    # Download Tiny ImageNet
    zip_path = download_file(TINY_IMAGENET_URL, TINY_IMAGENET_FILENAME, data_dir)
    
    # Extract the zip file
    extract_dir = os.path.join(data_dir, 'temp_extract')
    extract_zip(zip_path, extract_dir)
    
    # Move to final location
    tiny_imagenet_dir = os.path.join(data_dir, 'TinyImageNet')
    extracted_dir = os.path.join(extract_dir, 'tiny-imagenet-200')
    
    if os.path.exists(tiny_imagenet_dir):
        logger.info("TinyImageNet directory already exists, removing old version...")
        shutil.rmtree(tiny_imagenet_dir)
    
    logger.info(f"Moving extracted files to {tiny_imagenet_dir}...")
    shutil.move(extracted_dir, tiny_imagenet_dir)
    
    # Clean up
    shutil.rmtree(extract_dir)
    os.remove(zip_path)
    
    logger.info("Tiny ImageNet setup completed!")
    
    # Verify the structure
    verify_structure(tiny_imagenet_dir)

def verify_structure(tiny_imagenet_dir):
    """Verify the Tiny ImageNet directory structure."""
    logger.info("Verifying directory structure...")
    
    expected_dirs = ['train', 'val', 'test']
    for dir_name in expected_dirs:
        dir_path = os.path.join(tiny_imagenet_dir, dir_name)
        if not os.path.exists(dir_path):
            logger.warning(f"Missing directory: {dir_path}")
        else:
            logger.info(f"✓ Found {dir_name} directory")
    
    # Check train directory structure
    train_dir = os.path.join(tiny_imagenet_dir, 'train')
    if os.path.exists(train_dir):
        class_dirs = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
        logger.info(f"✓ Found {len(class_dirs)} classes in train directory")
        
        # Check one class directory
        if class_dirs:
            sample_class = class_dirs[0]
            sample_class_dir = os.path.join(train_dir, sample_class)
            images_dir = os.path.join(sample_class_dir, 'images')
            if os.path.exists(images_dir):
                num_images = len([f for f in os.listdir(images_dir) if f.endswith('.JPEG')])
                logger.info(f"✓ Sample class '{sample_class}' has {num_images} training images")
    
    # Check val directory
    val_dir = os.path.join(tiny_imagenet_dir, 'val')
    if os.path.exists(val_dir):
        val_images_dir = os.path.join(val_dir, 'images')
        if os.path.exists(val_images_dir):
            num_val_images = len([f for f in os.listdir(val_images_dir) if f.endswith('.JPEG')])
            logger.info(f"✓ Validation directory has {num_val_images} images")
    
    logger.info("Directory structure verification completed!")

def main():
    """Main function to download and setup Tiny ImageNet."""
    logger.info("Starting Tiny ImageNet download and setup...")
    
    # Use the same data directory as the generic loader
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
    
    try:
        setup_tiny_imagenet(data_dir)
        logger.info("Tiny ImageNet is ready to use!")
        logger.info(f"Dataset location: {os.path.join(data_dir, 'TinyImageNet')}")
        
    except Exception as e:
        logger.error(f"Failed to setup Tiny ImageNet: {e}")
        raise

if __name__ == "__main__":
    main() 