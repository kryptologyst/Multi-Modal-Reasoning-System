"""Modern multi-modal reasoning models."""

import math
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor, CLIPConfig
from transformers.modeling_outputs import BaseModelOutput


class MultiModalReasoningModel(nn.Module):
    """Enhanced multi-modal reasoning model with CLIP backbone."""
    
    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        freeze_vision: bool = False,
        freeze_text: bool = False,
        temperature: float = 0.07,
        projection_dim: Optional[int] = None,
    ):
        """Initialize the model.
        
        Args:
            model_name: Name of the CLIP model to use.
            freeze_vision: Whether to freeze vision encoder.
            freeze_text: Whether to freeze text encoder.
            temperature: Temperature for similarity computation.
            projection_dim: Dimension for projection layers.
        """
        super().__init__()
        
        self.model_name = model_name
        self.temperature = temperature
        
        # Load CLIP model
        self.clip_model = CLIPModel.from_pretrained(model_name)
        
        # Freeze encoders if specified
        if freeze_vision:
            for param in self.clip_model.vision_model.parameters():
                param.requires_grad = False
        
        if freeze_text:
            for param in self.clip_model.text_model.parameters():
                param.requires_grad = False
        
        # Get dimensions
        vision_dim = self.clip_model.vision_model.config.hidden_size
        text_dim = self.clip_model.text_model.config.hidden_size
        
        # Projection layers for enhanced reasoning
        if projection_dim is None:
            projection_dim = max(vision_dim, text_dim)
        
        self.vision_projection = nn.Linear(vision_dim, projection_dim)
        self.text_projection = nn.Linear(text_dim, projection_dim)
        
        # Cross-modal attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=projection_dim,
            num_heads=8,
            batch_first=True,
        )
        
        # Reasoning layers
        self.reasoning_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=projection_dim,
                nhead=8,
                dim_feedforward=projection_dim * 4,
                batch_first=True,
            )
            for _ in range(2)
        ])
        
        # Output heads
        self.similarity_head = nn.Linear(projection_dim, 1)
        self.classification_head = nn.Linear(projection_dim, 2)  # Binary classification
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self) -> None:
        """Initialize model weights."""
        for module in [self.vision_projection, self.text_projection]:
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)
        
        nn.init.xavier_uniform_(self.similarity_head.weight)
        nn.init.zeros_(self.similarity_head.bias)
        
        nn.init.xavier_uniform_(self.classification_head.weight)
        nn.init.zeros_(self.classification_head.bias)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor,
        return_embeddings: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass.
        
        Args:
            input_ids: Text input IDs.
            attention_mask: Text attention mask.
            pixel_values: Image pixel values.
            return_embeddings: Whether to return embeddings.
            
        Returns:
            Dictionary containing model outputs.
        """
        # Get CLIP outputs
        clip_outputs = self.clip_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
        )
        
        # Extract embeddings
        vision_embeddings = clip_outputs.vision_model_output.last_hidden_state
        text_embeddings = clip_outputs.text_model_output.last_hidden_state
        
        # Project embeddings
        vision_projected = self.vision_projection(vision_embeddings)
        text_projected = self.text_projection(text_embeddings)
        
        # Cross-modal attention
        vision_attended, _ = self.cross_attention(
            query=vision_projected,
            key=text_projected,
            value=text_projected,
        )
        
        text_attended, _ = self.cross_attention(
            query=text_projected,
            key=vision_projected,
            value=vision_projected,
        )
        
        # Combine modalities
        combined = vision_attended + text_attended
        
        # Apply reasoning layers
        for layer in self.reasoning_layers:
            combined = layer(combined)
        
        # Global pooling
        vision_pooled = combined.mean(dim=1)  # Average pooling
        text_pooled = text_attended.mean(dim=1)
        
        # Compute similarities
        similarity = F.cosine_similarity(vision_pooled, text_pooled, dim=-1)
        similarity_logits = similarity / self.temperature
        
        # Classification
        classification_logits = self.classification_head(vision_pooled)
        
        outputs = {
            'similarity_logits': similarity_logits,
            'classification_logits': classification_logits,
            'vision_embeddings': vision_pooled,
            'text_embeddings': text_pooled,
        }
        
        if return_embeddings:
            outputs.update({
                'vision_features': vision_embeddings,
                'text_features': text_embeddings,
                'vision_projected': vision_projected,
                'text_projected': text_projected,
            })
        
        return outputs
    
    def encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Encode images to embeddings.
        
        Args:
            pixel_values: Image pixel values.
            
        Returns:
            Image embeddings.
        """
        vision_outputs = self.clip_model.vision_model(pixel_values=pixel_values)
        vision_embeddings = vision_outputs.last_hidden_state
        vision_projected = self.vision_projection(vision_embeddings)
        return vision_projected.mean(dim=1)
    
    def encode_text(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Encode text to embeddings.
        
        Args:
            input_ids: Text input IDs.
            attention_mask: Text attention mask.
            
        Returns:
            Text embeddings.
        """
        text_outputs = self.clip_model.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        text_embeddings = text_outputs.last_hidden_state
        text_projected = self.text_projection(text_embeddings)
        return text_projected.mean(dim=1)
    
    def compute_similarity(
        self,
        image_embeddings: torch.Tensor,
        text_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """Compute similarity between image and text embeddings.
        
        Args:
            image_embeddings: Image embeddings.
            text_embeddings: Text embeddings.
            
        Returns:
            Similarity scores.
        """
        similarity = F.cosine_similarity(image_embeddings, text_embeddings, dim=-1)
        return similarity / self.temperature


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


class ReasoningLoss(nn.Module):
    """Combined loss for multi-modal reasoning."""
    
    def __init__(
        self,
        contrastive_weight: float = 1.0,
        classification_weight: float = 0.1,
        temperature: float = 0.07,
    ):
        """Initialize reasoning loss.
        
        Args:
            contrastive_weight: Weight for contrastive loss.
            classification_weight: Weight for classification loss.
            temperature: Temperature for contrastive loss.
        """
        super().__init__()
        self.contrastive_weight = contrastive_weight
        self.classification_weight = classification_weight
        
        self.contrastive_loss = ContrastiveLoss(temperature)
        self.classification_loss = nn.CrossEntropyLoss()
    
    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute combined loss.
        
        Args:
            outputs: Model outputs.
            labels: Optional classification labels.
            
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
        
        # Classification loss (if labels provided)
        if labels is not None:
            classification_loss = self.classification_loss(
                outputs['classification_logits'],
                labels,
            )
            losses['classification_loss'] = classification_loss
        
        # Total loss
        total_loss = self.contrastive_weight * contrastive_loss
        if labels is not None:
            total_loss += self.classification_weight * losses['classification_loss']
        
        losses['total_loss'] = total_loss
        
        return losses
