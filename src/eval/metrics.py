"""Evaluation metrics and utilities for multi-modal reasoning."""

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Union

from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score


class MultiModalEvaluator:
    """Evaluator for multi-modal reasoning tasks."""
    
    def __init__(self, device: torch.device):
        """Initialize evaluator.
        
        Args:
            device: Device for computations.
        """
        self.device = device
        self.reset()
    
    def reset(self) -> None:
        """Reset evaluation state."""
        self.predictions = []
        self.labels = []
        self.similarities = []
        self.image_embeddings = []
        self.text_embeddings = []
    
    def update(
        self,
        outputs: Dict[str, torch.Tensor],
        labels: Optional[torch.Tensor] = None,
    ) -> None:
        """Update evaluation state with batch results.
        
        Args:
            outputs: Model outputs.
            labels: Ground truth labels.
        """
        # Store similarities
        similarities = outputs['similarity_logits'].detach().cpu().numpy()
        self.similarities.extend(similarities)
        
        # Store embeddings for retrieval metrics
        vision_embeddings = outputs['vision_embeddings'].detach().cpu().numpy()
        text_embeddings = outputs['text_embeddings'].detach().cpu().numpy()
        self.image_embeddings.extend(vision_embeddings)
        self.text_embeddings.extend(text_embeddings)
        
        # Store predictions and labels
        if 'classification_logits' in outputs:
            predictions = torch.argmax(outputs['classification_logits'], dim=-1)
            self.predictions.extend(predictions.detach().cpu().numpy())
        
        if labels is not None:
            self.labels.extend(labels.detach().cpu().numpy())
    
    def compute_retrieval_metrics(
        self,
        top_k: List[int] = [1, 5, 10],
    ) -> Dict[str, float]:
        """Compute retrieval metrics.
        
        Args:
            top_k: List of k values for recall@k.
            
        Returns:
            Dictionary containing retrieval metrics.
        """
        if not self.image_embeddings or not self.text_embeddings:
            return {}
        
        image_embeddings = np.array(self.image_embeddings)
        text_embeddings = np.array(self.text_embeddings)
        
        # Normalize embeddings
        image_embeddings = image_embeddings / np.linalg.norm(image_embeddings, axis=1, keepdims=True)
        text_embeddings = text_embeddings / np.linalg.norm(text_embeddings, axis=1, keepdims=True)
        
        # Compute similarity matrix
        similarity_matrix = np.dot(image_embeddings, text_embeddings.T)
        
        metrics = {}
        
        # Image-to-text retrieval
        for k in top_k:
            recall_at_k = self._compute_recall_at_k(similarity_matrix, k)
            metrics[f'recall_at_{k}_i2t'] = recall_at_k
        
        # Text-to-image retrieval
        for k in top_k:
            recall_at_k = self._compute_recall_at_k(similarity_matrix.T, k)
            metrics[f'recall_at_{k}_t2i'] = recall_at_k
        
        # Mean Average Precision
        map_score = self._compute_map(similarity_matrix)
        metrics['map_i2t'] = map_score
        
        map_score = self._compute_map(similarity_matrix.T)
        metrics['map_t2i'] = map_score
        
        # Median rank
        median_rank = self._compute_median_rank(similarity_matrix)
        metrics['median_rank_i2t'] = median_rank
        
        median_rank = self._compute_median_rank(similarity_matrix.T)
        metrics['median_rank_t2i'] = median_rank
        
        return metrics
    
    def _compute_recall_at_k(self, similarity_matrix: np.ndarray, k: int) -> float:
        """Compute recall@k.
        
        Args:
            similarity_matrix: Similarity matrix.
            k: Number of top results to consider.
            
        Returns:
            Recall@k score.
        """
        # Get top-k indices for each query
        top_k_indices = np.argsort(similarity_matrix, axis=1)[:, -k:]
        
        # Check if correct match is in top-k
        correct_matches = np.arange(len(similarity_matrix))
        recalls = []
        
        for i, top_k_idx in enumerate(top_k_indices):
            recall = 1.0 if correct_matches[i] in top_k_idx else 0.0
            recalls.append(recall)
        
        return np.mean(recalls)
    
    def _compute_map(self, similarity_matrix: np.ndarray) -> float:
        """Compute Mean Average Precision.
        
        Args:
            similarity_matrix: Similarity matrix.
            
        Returns:
            MAP score.
        """
        # Sort similarities in descending order
        sorted_indices = np.argsort(similarity_matrix, axis=1)[:, ::-1]
        
        aps = []
        for i, sorted_idx in enumerate(sorted_indices):
            # Find position of correct match
            correct_pos = np.where(sorted_idx == i)[0]
            if len(correct_pos) > 0:
                ap = 1.0 / (correct_pos[0] + 1)
            else:
                ap = 0.0
            aps.append(ap)
        
        return np.mean(aps)
    
    def _compute_median_rank(self, similarity_matrix: np.ndarray) -> float:
        """Compute median rank.
        
        Args:
            similarity_matrix: Similarity matrix.
            
        Returns:
            Median rank.
        """
        # Sort similarities in descending order
        sorted_indices = np.argsort(similarity_matrix, axis=1)[:, ::-1]
        
        ranks = []
        for i, sorted_idx in enumerate(sorted_indices):
            # Find rank of correct match
            correct_pos = np.where(sorted_idx == i)[0]
            if len(correct_pos) > 0:
                rank = correct_pos[0] + 1
            else:
                rank = len(sorted_idx) + 1
            ranks.append(rank)
        
        return np.median(ranks)
    
    def compute_classification_metrics(self) -> Dict[str, float]:
        """Compute classification metrics.
        
        Returns:
            Dictionary containing classification metrics.
        """
        if not self.predictions or not self.labels:
            return {}
        
        predictions = np.array(self.predictions)
        labels = np.array(self.labels)
        
        # Basic metrics
        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average='weighted'
        )
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
        }
        
        # Per-class metrics
        precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(
            labels, predictions, average=None
        )
        
        for i, (p, r, f) in enumerate(zip(precision_per_class, recall_per_class, f1_per_class)):
            metrics.update({
                f'precision_class_{i}': p,
                f'recall_class_{i}': r,
                f'f1_class_{i}': f,
            })
        
        return metrics
    
    def compute_similarity_metrics(self) -> Dict[str, float]:
        """Compute similarity-based metrics.
        
        Returns:
            Dictionary containing similarity metrics.
        """
        if not self.similarities:
            return {}
        
        similarities = np.array(self.similarities)
        
        metrics = {
            'mean_similarity': np.mean(similarities),
            'std_similarity': np.std(similarities),
            'min_similarity': np.min(similarities),
            'max_similarity': np.max(similarities),
        }
        
        return metrics
    
    def get_all_metrics(self) -> Dict[str, float]:
        """Compute all available metrics.
        
        Returns:
            Dictionary containing all metrics.
        """
        metrics = {}
        
        # Retrieval metrics
        retrieval_metrics = self.compute_retrieval_metrics()
        metrics.update(retrieval_metrics)
        
        # Classification metrics
        classification_metrics = self.compute_classification_metrics()
        metrics.update(classification_metrics)
        
        # Similarity metrics
        similarity_metrics = self.compute_similarity_metrics()
        metrics.update(similarity_metrics)
        
        return metrics
    
    def print_summary(self) -> None:
        """Print evaluation summary."""
        metrics = self.get_all_metrics()
        
        print("Evaluation Summary:")
        print("=" * 50)
        
        # Retrieval metrics
        retrieval_keys = [k for k in metrics.keys() if 'recall_at_' in k or 'map_' in k or 'median_rank_' in k]
        if retrieval_keys:
            print("Retrieval Metrics:")
            for key in sorted(retrieval_keys):
                print(f"  {key}: {metrics[key]:.4f}")
            print()
        
        # Classification metrics
        classification_keys = [k for k in metrics.keys() if k in ['accuracy', 'precision', 'recall', 'f1']]
        if classification_keys:
            print("Classification Metrics:")
            for key in sorted(classification_keys):
                print(f"  {key}: {metrics[key]:.4f}")
            print()
        
        # Similarity metrics
        similarity_keys = [k for k in metrics.keys() if 'similarity' in k]
        if similarity_keys:
            print("Similarity Metrics:")
            for key in sorted(similarity_keys):
                print(f"  {key}: {metrics[key]:.4f}")
            print()
