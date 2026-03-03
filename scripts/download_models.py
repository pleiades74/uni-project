import os
import torch
import clip
from segment_anything import sam_model_registry
import urllib.request

def download_models():
    """Download pre-trained models we'll use"""
    
    print("="*50)
    print("Downloading pre-trained models...")
    print("="*50)
    
    # Create checkpoints directory
    os.makedirs("models/checkpoints", exist_ok=True)
    
    # 1. Download CLIP
    print("\n[1/3] Loading CLIP model...")
    clip_model, _ = clip.load("ViT-B/32")
    print(f"✓ CLIP loaded: {sum(p.numel() for p in clip_model.parameters()):,} parameters")
    
    # 2. Download SAM
    print("\n[2/3] Downloading SAM model...")
    sam_checkpoint = "models/checkpoints/sam_vit_h_4b8939.pth"
    if not os.path.exists(sam_checkpoint):
        print("Downloading SAM (2.5GB - this will take a few minutes)...")
        url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
        urllib.request.urlretrieve(url, sam_checkpoint)
        print("✓ SAM downloaded")
    else:
        print("✓ SAM already exists")
    
    # Load SAM to verify
    sam = sam_model_registry["vit_h"](checkpoint=sam_checkpoint)
    print(f"✓ SAM loaded: {sum(p.numel() for p in sam.parameters()):,} parameters")
    
    # 3. Download a pre-trained detector (Faster R-CNN)
    print("\n[3/3] Loading pre-trained detector...")
    detector = torch.hub.load('pytorch/vision:v0.10.0', 'fasterrcnn_resnet50_fpn', pretrained=True)
    detector.eval()
    print(f"✓ Detector loaded: {sum(p.numel() for p in detector.parameters()):,} parameters")
    
    print("\n" + "="*50)
    print("All models downloaded successfully!")
    print("="*50)
    
    return clip_model, sam, detector

if __name__ == "__main__":
    clip_model, sam, detector = download_models()
