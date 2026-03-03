"""
Core data structures for the few-shot object detection system
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import torch
from PIL import Image

@dataclass
class BoundingBox:
    """Represents a single bounding box"""
    x: float  # top-left x
    y: float  # top-left y
    w: float  # width
    h: float  # height
    confidence: float = 1.0  # detection confidence (0-1)
    class_id: Optional[int] = None
    class_name: Optional[str] = None
    
    def area(self) -> float:
        return self.w * self.h
    
    def to_list(self) -> List[float]:
        return [self.x, self.y, self.w, self.h]
    
    def to_xyxy(self) -> Tuple[float, float, float, float]:
        """Convert to (x1, y1, x2, y2) format"""
        return (self.x, self.y, self.x + self.w, self.y + self.h)
    
    def iou(self, other: 'BoundingBox') -> float:
        """Calculate Intersection over Union with another box"""
        # Convert to (x1, y1, x2, y2)
        x1_1, y1_1, x2_1, y2_1 = self.to_xyxy()
        x1_2, y1_2, x2_2, y2_2 = other.to_xyxy()
        
        # Calculate intersection
        x_left = max(x1_1, x1_2)
        y_top = max(y1_1, y1_2)
        x_right = min(x2_1, x2_2)
        y_bottom = min(y2_1, y2_2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection = (x_right - x_left) * (y_bottom - y_top)
        
        # Calculate union
        area1 = self.area()
        area2 = other.area()
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0

@dataclass
class DetectionResult:
    """Result from MOC for a single image"""
    image_id: str
    detections: List[BoundingBox] = field(default_factory=list)
    
    def add_detection(self, bbox: BoundingBox):
        self.detections.append(bbox)
    
    def sort_by_confidence(self):
        self.detections.sort(key=lambda x: x.confidence, reverse=True)
    
    def to_dict(self) -> Dict:
        return {
            'image_id': self.image_id,
            'detections': [
                {
                    'x': b.x, 'y': b.y, 'w': b.w, 'h': b.h,
                    'confidence': b.confidence,
                    'class_id': b.class_id,
                    'class_name': b.class_name
                }
                for b in self.detections
            ]
        }

@dataclass
class Episode:
    """
    An "episode" for few-shot learning
    Contains support set (labeled) and query set (unlabeled)
    """
    support_images: List[np.ndarray]  # The few labeled examples
    support_boxes: List[List[BoundingBox]]  # Their bounding boxes
    support_text: str  # Text description of the target
    
    query_images: List[np.ndarray]  # Images to detect on
    query_image_ids: List[str]  # IDs for query images
    query_boxes: Optional[List[List[BoundingBox]]] = None  # Ground truth (for testing)
    
    def __post_init__(self):
        """Validate the episode"""
        assert len(self.support_images) == len(self.support_boxes), \
            "Number of support images must match number of support box lists"
        
        for boxes in self.support_boxes:
            for box in boxes:
                assert isinstance(box, BoundingBox), \
                    "All support boxes must be BoundingBox objects"
        
        assert len(self.query_images) == len(self.query_image_ids), \
            "Number of query images must match number of query image IDs"
    
    @property
    def num_support(self) -> int:
        return len(self.support_images)
    
    @property
    def num_query(self) -> int:
        return len(self.query_images)

class ThermalImagePreprocessor:
    """Preprocessor for thermal images (addressing challenges in Section 2.2.5)"""
    
    def __init__(self, target_size: Tuple[int, int] = (640, 512)):
        self.target_size = target_size
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess a thermal image:
        - Handle 16-bit images
        - Normalize
        - Enhance contrast
        - Denoise
        """
        # Convert to float if needed
        if image.dtype == np.uint16:
            image = image.astype(np.float32) / 65535.0
        
        # Ensure 8-bit for display (keep 16-bit for model)
        if image.max() <= 1.0:
            image_display = (image * 255).astype(np.uint8)
        else:
            image_display = image.astype(np.uint8)
        
        # Apply CLAHE for contrast enhancement
        import cv2
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(image_display)
        
        # Apply mild denoising
        denoised = cv2.fastNlMeansDenoising(enhanced, h=10)
        
        # Resize if needed
        if denoised.shape[:2] != self.target_size:
            denoised = cv2.resize(denoised, self.target_size)
        
        # Convert back to tensor format (normalized 0-1)
        normalized = denoised.astype(np.float32) / 255.0
        
        return normalized
    
    def to_tensor(self, image: np.ndarray) -> torch.Tensor:
        """Convert numpy array to torch tensor (C, H, W)"""
        if len(image.shape) == 2:
            # Add channel dimension
            image = image[np.newaxis, :, :]
        elif len(image.shape) == 3 and image.shape[2] == 1:
            image = image.transpose(2, 0, 1)
        
        return torch.from_numpy(image).float()
