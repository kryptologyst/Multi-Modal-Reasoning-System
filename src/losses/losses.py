"""Loss functions for multi-modal reasoning."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class ContrastiveLoss(nn.Module):
    """Contrastive loss for multi-modal learning."""
    
    def __init__(self, temperature: float = 0.07):
        """Initialize contrastive loss.
        
        Args:
            temperature: Temperature for similarity computation.
        """
        super().__init__()
        self.temperature = temperature
    
    def forward(
        self,
        image_embeddings: torch.Tensor,
        text_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """Compute contrastive loss.
        
        Args:
            image_embeddings: Image embeddings.
            text_embeddings: Text embeddings.
            
        Returns:
            Contrastive loss.
        """
        # Normalize embeddings
        image_embeddings = F.normalize(image_embeddings, dim=-1)
        text_embeddings = F.normalize(text_embeddings, dim=-1)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(image_embeddings, text_embeddings.T) / self.temperature
        
        # Create labels (diagonal elements are positive pairs)
        batch_size = image_embeddings.size(0)
        labels = torch.arange(batch_size, device=image_embeddings.device)
        
        # Compute losses
        loss_i2t = F.cross_entropy(similarity_matrix, labels)
        loss_t2i = F.cross_entropy(similarity_matrix.T, labels)
        
        return (loss_i2t + loss_t2i) / 2


class TripletLoss(nn.Module):
    """Triplet loss for multi-modal learning."""
    
    def __init__(self, margin: float = 0.2):
        """Initialize triplet loss.
        
        Args:
            margin: Margin for triplet loss.
        """
        super().__init__()
        self.margin = margin
    
    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor,
    ) -> torch.Tensor:
        """Compute triplet loss.
        
        Args:
            anchor: Anchor embeddings.
            positive: Positive embeddings.
            negative: Negative embeddings.
            
        Returns:
            Triplet loss.
        """
        # Compute distances
        pos_dist = F.pairwise_distance(anchor, positive, p=2)
        neg_dist = F.pairwise_distance(anchor, negative, p=2)
        
        # Compute loss
        loss = F.relu(pos_dist - neg_dist + self.margin)
        return loss.mean()


class MultiModalTripletLoss(nn.Module):
    """Multi-modal triplet loss."""
    
    def __init__(self, margin: float = 0.2):
        """Initialize multi-modal triplet loss.
        
        Args:
            margin: Margin for triplet loss.
        """
        super().__init__()
        self.margin = margin
    
    def forward(
        self,
        image_embeddings: torch.Tensor,
        text_embeddings: torch.Tensor,
        hard_negatives: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute multi-modal triplet loss.
        
        Args:
            image_embeddings: Image embeddings.
            text_embeddings: Text embeddings.
            hard_negatives: Optional hard negative embeddings.
            
        Returns:
            Multi-modal triplet loss.
        """
        batch_size = image_embeddings.size(0)
        
        # Normalize embeddings
        image_embeddings = F.normalize(image_embeddings, dim=-1)
        text_embeddings = F.normalize(text_embeddings, dim=-1)
        
        # Compute positive similarities (diagonal)
        pos_sim = torch.sum(image_embeddings * text_embeddings, dim=-1)
        
        # Compute negative similarities
        if hard_negatives is not None:
            hard_negatives = F.normalize(hard_negatives, dim=-1)
            neg_sim = torch.sum(image_embeddings * hard_negatives, dim=-1)
        else:
            # Use all other text embeddings as negatives
            neg_sim = torch.matmul(image_embeddings, text_embeddings.T)
            # Remove diagonal (positive pairs)
            mask = torch.eye(batch_size, device=image_embeddings.device).bool()
            neg_sim = neg_sim.masked_select(~mask).view(batch_size, batch_size - 1)
            neg_sim = neg_sim.max(dim=-1)[0]  # Hardest negative
        
        # Compute triplet loss
        loss = F.relu(neg_sim - pos_sim + self.margin)
        return loss.mean()


class FocalLoss(nn.Module):
    """Focal loss for handling class imbalance."""
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0):
        """Initialize focal loss.
        
        Args:
            alpha: Weighting factor for rare class.
            gamma: Focusing parameter.
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute focal loss.
        
        Args:
            inputs: Model predictions.
            targets: Ground truth labels.
            
        Returns:
            Focal loss.
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


class LabelSmoothingLoss(nn.Module):
    """Label smoothing loss."""
    
    def __init__(self, smoothing: float = 0.1):
        """Initialize label smoothing loss.
        
        Args:
            smoothing: Label smoothing factor.
        """
        super().__init__()
        self.smoothing = smoothing
    
    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute label smoothing loss.
        
        Args:
            inputs: Model predictions.
            targets: Ground truth labels.
            
        Returns:
            Label smoothing loss.
        """
        log_probs = F.log_softmax(inputs, dim=-1)
        n_classes = inputs.size(-1)
        
        # Create smoothed targets
        smooth_targets = torch.zeros_like(log_probs)
        smooth_targets.fill_(self.smoothing / (n_classes - 1))
        smooth_targets.scatter_(-1, targets.unsqueeze(-1), 1 - self.smoothing)
        
        loss = -torch.sum(smooth_targets * log_probs, dim=-1)
        return loss.mean()


class ReasoningLoss(nn.Module):
    """Combined loss for multi-modal reasoning."""
    
    def __init__(
        self,
        contrastive_weight: float = 1.0,
        classification_weight: float = 0.1,
        triplet_weight: float = 0.1,
        temperature: float = 0.07,
        margin: float = 0.2,
        use_focal: bool = False,
        use_label_smoothing: bool = False,
    ):
        """Initialize reasoning loss.
        
        Args:
            contrastive_weight: Weight for contrastive loss.
            classification_weight: Weight for classification loss.
            triplet_weight: Weight for triplet loss.
            temperature: Temperature for contrastive loss.
            margin: Margin for triplet loss.
            use_focal: Whether to use focal loss for classification.
            use_label_smoothing: Whether to use label smoothing.
        """
        super().__init__()
        self.contrastive_weight = contrastive_weight
        self.classification_weight = classification_weight
        self.triplet_weight = triplet_weight
        
        self.contrastive_loss = ContrastiveLoss(temperature)
        self.triplet_loss = MultiModalTripletLoss(margin)
        
        if use_focal:
            self.classification_loss = FocalLoss()
        elif use_label_smoothing:
            self.classification_loss = LabelSmoothingLoss()
        else:
            self.classification_loss = nn.CrossEntropyLoss()
    
    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        labels: Optional[torch.Tensor] = None,
        hard_negatives: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute combined loss.
        
        Args:
            outputs: Model outputs.
            labels: Optional classification labels.
            hard_negatives: Optional hard negative embeddings.
            
        Returns:
            Dictionary containing losses.
        """
        losses = {}
        
        # Contrastive loss
        contrastive_loss = self.contrastive_loss(
            outputs['vision_embeddings'],
            outputs['text_embeddings'],
        )
        losses['contrastive_loss'] = contrastive_loss
        
        # Triplet loss
        triplet_loss = self.triplet_loss(
            outputs['vision_embeddings'],
            outputs['text_embeddings'],
            hard_negatives,
        )
        losses['triplet_loss'] = triplet_loss
        
        # Classification loss (if labels provided)
        if labels is not None and 'classification_logits' in outputs:
            classification_loss = self.classification_loss(
                outputs['classification_logits'],
                labels,
            )
            losses['classification_loss'] = classification_loss
        
        # Total loss
        total_loss = (
            self.contrastive_weight * contrastive_loss +
            self.triplet_weight * triplet_loss
        )
        
        if labels is not None and 'classification_logits' in outputs:
            total_loss += self.classification_weight * losses['classification_loss']
        
        losses['total_loss'] = total_loss
        
        return losses
