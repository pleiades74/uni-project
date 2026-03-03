"""
Simple test script to verify our few-shot detector works
"""

import numpy as np
import torch
import cv2
from PIL import Image
import matplotlib.pyplot as plt

from src.utils.data_structures import BoundingBox, Episode, ThermalImagePreprocessor
from src.opm.fewshot_learner import FewShotDetector, OPMConfig

def create_dummy_data():
    """Create dummy images and boxes for testing"""
    
    # Create dummy thermal-like images (grayscale)
    img_size = (512, 640)
    
    # Support images (3 examples)
    support_images = []
    support_boxes = []
    
    for i in range(3):
        # Create blank image with some noise
        img = np.random.randint(0, 255, img_size, dtype=np.uint8)
        
        # Add a "target" blob (simulating a tank)
        cv2.rectangle(img, (200, 150), (250, 180), 200, -1)
        cv2.rectangle(img, (210, 140), (240, 150), 220, -1)  # turret
        
        # Add some noise
        noise = np.random.normal(0, 10, img_size).astype(np.uint8)
        img = cv2.add(img, noise)
        
        support_images.append(img)
        
        # Bounding box for the target
        box = BoundingBox(x=200, y=140, w=50, h=40, class_name="tank")
        support_boxes.append([box])
    
    # Query images (2 images to detect on)
    query_images = []
    query_ids = []
    
    for i in range(2):
        img = np.random.randint(0, 255, img_size, dtype=np.uint8)
        
        # Add target in different position
        cv2.rectangle(img, (300, 250), (350, 280), 200, -1)
        cv2.rectangle(img, (310, 240), (340, 250), 220, -1)
        
        noise = np.random.normal(0, 10, img_size).astype(np.uint8)
        img = cv2.add(img, noise)
        
        query_images.append(img)
        query_ids.append(f"query_{i}")
    
    return support_images, support_boxes, query_images, query_ids

def visualize_detection(image, detections, title="Detections"):
    """Visualize detections on image"""
    plt.figure(figsize=(10, 8))
    
    if len(image.shape) == 2:
        plt.imshow(image, cmap='gray')
    else:
        plt.imshow(image)
    
    # Draw detections
    for det in detections:
        rect = plt.Rectangle(
            (det.x, det.y), det.w, det.h,
            fill=False, edgecolor='red', linewidth=2
        )
        plt.gca().add_patch(rect)
        plt.text(det.x, det.y-5, f'{det.confidence:.2f}', 
                color='red', fontsize=10, bbox=dict(facecolor='white', alpha=0.7))
    
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def main():
    print("="*60)
    print("Testing Few-Shot Object Detection System")
    print("="*60)
    
    # 1. Setup
    print("\n[1/5] Setting up configuration...")
    config = OPMConfig(
        learning_rate=1e-4,
        num_iterations=50,
        num_shots=3,
        use_text=True
    )
    
    # 2. Create detector
    print("\n[2/5] Initializing FewShotDetector...")
    detector = FewShotDetector(config)
    
    # 3. Create dummy data
    print("\n[3/5] Creating dummy thermal images...")
    support_imgs, support_boxes, query_imgs, query_ids = create_dummy_data()
    
    # 4. Create episode and adapt
    print("\n[4/5] Creating episode and adapting...")
    episode = Episode(
        support_images=support_imgs,
        support_boxes=support_boxes,
        support_text="a military tank with tracks and a turret",
        query_images=query_imgs,
        query_image_ids=query_ids
    )
    
    detector.adapt(episode)
    
    # 5. Test detection
    print("\n[5/5] Testing detection on query images...")
    for i, (img, img_id) in enumerate(zip(query_imgs, query_ids)):
        detections = detector.detect(img)
        print(f"\nImage {img_id}: found {len(detections)} detections")
        
        for j, det in enumerate(detections):
            print(f"  Detection {j+1}: x={det.x:.1f}, y={det.y:.1f}, "
                  f"w={det.w:.1f}, h={det.h:.1f}, conf={det.confidence:.3f}")
        
        # Visualize
        visualize_detection(img, detections, f"Detections on {img_id}")
    
    # 6. Save checkpoint
    print("\nSaving checkpoint...")
    detector.save_checkpoint("models/checkpoints/test_checkpoint.pth")
    
    print("\n" + "="*60)
    print("Test completed successfully!")
    print("="*60)

if __name__ == "__main__":
    main()
