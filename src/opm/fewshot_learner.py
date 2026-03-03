"""
Few-shot learning module (OPM)
Adapts the detector using few examples and text description
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass

from src.utils.data_structures import BoundingBox, Episode

@dataclass
class OPMConfig:
    """Configuration for OPM"""
    # Model settings
    clip_model_name: str = "ViT-B/32"
    detector_backbone: str = "resnet50"
    
    # Training settings
    learning_rate: float = 1e-4
    num_iterations: int = 100
    batch_size: int = 4
    
    # Few-shot settings
    num_shots: int = 5  # number of examples per class
    use_text: bool = True  # whether to use text description
    
    # Optimization
    freeze_backbone: bool = True  # freeze early layers during fine-tuning
    momentum: float = 0.9
    weight_decay: float = 1e-4

class PrototypicalNetwork(nn.Module):
    """
    Prototypical network for few-shot learning
    Computes class prototypes from support examples
    """
    
    def __init__(self, embedding_dim: int = 512):
        super().__init__()
        self.embedding_dim = embedding_dim
        
    def compute_prototype(self, support_features: torch.Tensor) -> torch.Tensor:
        """
        Compute prototype as mean of support features
        Args:
            support_features: (num_support, embedding_dim)
        Returns:
            prototype: (embedding_dim,)
        """
        return support_features.mean(dim=0)
    
    def compute_prototypes_from_text(self, text: str, clip_model) -> torch.Tensor:
        """
        Compute prototype from text description using CLIP
        """
        with torch.no_grad():
            text_tokens = clip.tokenize([text])
            text_features = clip_model.encode_text(text_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features.squeeze()
    
    def combine_prototypes(self, 
                          visual_prototype: torch.Tensor, 
                          text_prototype: torch.Tensor,
                          alpha: float = 0.5) -> torch.Tensor:
        """
        Combine visual and text prototypes
        alpha: weight for visual prototype (1-alpha for text)
        """
        return alpha * visual_prototype + (1 - alpha) * text_prototype
    
    def cosine_similarity(self, query_features: torch.Tensor, 
                          prototype: torch.Tensor) -> torch.Tensor:
        """
        Compute cosine similarity between query features and prototype
        """
        # Normalize
        query_norm = F.normalize(query_features, dim=1)
        prototype_norm = F.normalize(prototype, dim=0)
        
        # Compute similarity
        similarity = torch.mm(query_norm, prototype_norm.unsqueeze(1))
        return similarity.squeeze()

class FewShotDetector(nn.Module):
    """
    Main few-shot detector
    Combines pre-trained detector with prototypical network
    """
    
    def __init__(self, config: OPMConfig):
        super().__init__()
        self.config = config
        
        # Load CLIP for text understanding
        print("Loading CLIP model...")
        self.clip_model, self.clip_preprocess = clip.load(config.clip_model_name)
        
        # Freeze CLIP (we just use it for features)
        for param in self.clip_model.parameters():
            param.requires_grad = False
        
        # Load pre-trained detector (Faster R-CNN)
        print("Loading pre-trained detector...")
        self.detector = torch.hub.load('pytorch/vision:v0.10.0', 
                                       'fasterrcnn_resnet50_fpn', 
                                       pretrained=True)
        
        # Replace the classifier head for few-shot
        in_features = self.detector.roi_heads.box_predictor.cls_score.in_features
        num_classes = 2  # background + target class
        
        # We'll use a prototypical head instead of standard classifier
        self.prototypical_network = PrototypicalNetwork(embedding_dim=in_features)
        
        # Store prototypes (will be set during adaptation)
        self.target_prototype = None
        self.background_prototype = None
        
        print(f"✓ FewShotDetector initialized with {sum(p.numel() for p in self.parameters()):,} parameters")
    
    def extract_features(self, images: torch.Tensor, boxes: List[torch.Tensor]):
        """
        Extract ROI features for given boxes
        """
        # Forward through backbone
        features = self.detector.backbone(images)
        
        if isinstance(features, dict):
            features = list(features.values())[-1]
        
        # Get ROI features
        if len(boxes) > 0 and boxes[0] is not None:
            # Convert boxes to ROI format
            rois = []
            for i, img_boxes in enumerate(boxes):
                if img_boxes is not None and len(img_boxes) > 0:
                    batch_rois = torch.cat([torch.full((len(img_boxes), 1), i, device=img_boxes.device), 
                                           img_boxes], dim=1)
                    rois.append(batch_rois)
            
            if rois:
                rois = torch.cat(rois, dim=0)
                # Pool features
                roi_features = self.detector.roi_heads.box_roi_pool(features, rois, images.shape[-2:])
                roi_features = self.detector.roi_heads.box_head(roi_features)
                return roi_features
        return None
    
    def adapt(self, episode: Episode):
        """
        Adapt the detector using support examples (few-shot learning)
        This is the main OPM functionality
        """
        print(f"\nAdapting detector with {episode.num_support} support examples...")
        
        # Preprocess support images
        support_tensors = []
        support_boxes_tensors = []
        
        for img, boxes in zip(episode.support_images, episode.support_boxes):
            # Convert to tensor
            if isinstance(img, np.ndarray):
                img_tensor = torch.from_numpy(img).float()
                if len(img_tensor.shape) == 2:
                    img_tensor = img_tensor.unsqueeze(0).repeat(3, 1, 1)
                elif len(img_tensor.shape) == 3 and img_tensor.shape[0] != 3:
                    img_tensor = img_tensor.permute(2, 0, 1)
            else:
                img_tensor = img
            
            # Normalize
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            img_tensor = (img_tensor - mean) / std
            
            support_tensors.append(img_tensor.unsqueeze(0))
            
            # Convert boxes to tensor
            box_tensor = torch.tensor([[b.x, b.y, b.x + b.w, b.y + b.h] for b in boxes])
            support_boxes_tensors.append(box_tensor)
        
        # Stack support images
        support_batch = torch.cat(support_tensors, dim=0)
        
        # Extract features for support examples
        with torch.no_grad():
            support_features = self.extract_features(support_batch, support_boxes_tensors)
        
        if support_features is None:
            raise ValueError("Failed to extract support features")
        
        # Compute visual prototype
        visual_prototype = self.prototypical_network.compute_prototype(support_features)
        
        # Compute text prototype if available
        if self.config.use_text and episode.support_text:
            text_prototype = self.prototypical_network.compute_prototypes_from_text(
                episode.support_text, self.clip_model
            )
            # Combine prototypes
            self.target_prototype = self.prototypical_network.combine_prototypes(
                visual_prototype, text_prototype, alpha=0.7
            )
        else:
            self.target_prototype = visual_prototype
        
        # Compute background prototype (using random negative examples)
        # In practice, you'd sample background regions
        self.background_prototype = torch.zeros_like(self.target_prototype)
        
        print("✓ Adaptation complete!")
        return self
    
    def detect(self, image: np.ndarray) -> List[BoundingBox]:
        """
        Detect objects in a new image (MOC functionality)
        """
        if self.target_prototype is None:
            raise RuntimeError("Must adapt detector first!")
        
        # Preprocess image
        if isinstance(image, np.ndarray):
            img_tensor = torch.from_numpy(image).float()
            if len(img_tensor.shape) == 2:
                img_tensor = img_tensor.unsqueeze(0).repeat(3, 1, 1)
            elif len(img_tensor.shape) == 3 and img_tensor.shape[0] != 3:
                img_tensor = img_tensor.permute(2, 0, 1)
        else:
            img_tensor = image
        
        # Normalize
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_tensor = (img_tensor - mean) / std
        
        # Add batch dimension
        img_batch = img_tensor.unsqueeze(0)
        
        # Run detector to get proposals
        with torch.no_grad():
            # Forward through backbone
            features = self.detector.backbone(img_batch)
            if isinstance(features, dict):
                features = list(features.values())[-1]
            
            # Get proposals from RPN
            proposals, _ = self.detector.rpn(img_batch, features)
            
            if len(proposals[0]) == 0:
                return []
            
            # Extract ROI features for proposals
            rois = torch.cat([torch.full((len(proposals[0]), 1), 0, device=proposals[0].device), 
                            proposals[0]], dim=1)
            roi_features = self.detector.roi_heads.box_roi_pool(features, [rois], img_batch.shape[-2:])
            roi_features = self.detector.roi_heads.box_head(roi_features)
            
            # Compute similarity to target prototype
            similarities = self.prototypical_network.cosine_similarity(
                roi_features, self.target_prototype
            )
            
            # Convert to probabilities
            probs = torch.sigmoid(similarities)
            
            # Filter detections
            detections = []
            for i, (box, prob) in enumerate(zip(proposals[0], probs)):
                if prob > 0.5:  # confidence threshold
                    x1, y1, x2, y2 = box.cpu().numpy()
                    detections.append(BoundingBox(
                        x=float(x1),
                        y=float(y1),
                        w=float(x2 - x1),
                        h=float(y2 - y1),
                        confidence=float(prob),
                        class_id=1,  # target class
                        class_name="target"
                    ))
        
        return detections
    
    def save_checkpoint(self, path: str):
        """Save model checkpoint"""
        torch.save({
            'target_prototype': self.target_prototype,
            'background_prototype': self.background_prototype,
            'config': self.config,
        }, path)
        print(f"✓ Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path)
        self.target_prototype = checkpoint['target_prototype']
        self.background_prototype = checkpoint['background_prototype']
        print(f"✓ Checkpoint loaded from {path}")
