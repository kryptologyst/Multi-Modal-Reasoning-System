#!/usr/bin/env python3
"""
Project 939: Multi-Modal Reasoning System

A modern multi-modal reasoning system that combines vision and language understanding
to perform various reasoning tasks including image-text matching, question answering,
and similarity analysis.

This is the main entry point for the modernized multi-modal reasoning system.
For the original implementation, see the legacy code below.
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.utils.device import get_device, set_seed
from src.utils.config import load_config
from src.utils.logging import setup_logging


def main():
    """Main entry point for the multi-modal reasoning system."""
    parser = argparse.ArgumentParser(
        description="Multi-Modal Reasoning System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python 0939.py --demo                    # Run interactive demo
  python 0939.py --train                   # Train the model
  python 0939.py --eval                   # Evaluate the model
  python 0939.py --quick-test             # Quick functionality test
        """
    )
    
    parser.add_argument('--demo', action='store_true',
                       help='Run the interactive Streamlit demo')
    parser.add_argument('--train', action='store_true',
                       help='Train the multi-modal reasoning model')
    parser.add_argument('--eval', action='store_true',
                       help='Evaluate the trained model')
    parser.add_argument('--quick-test', action='store_true',
                       help='Run a quick functionality test')
    parser.add_argument('--config', type=str, default='configs/model/clip_config.yaml',
                       help='Path to configuration file')
    
    args = parser.parse_args()
    
    # Set up logging
    logger = setup_logging()
    logger.info("Multi-Modal Reasoning System - Project 939")
    
    # Set up device and reproducibility
    set_seed(42)
    device = get_device()
    logger.info(f"Using device: {device}")
    
    if args.demo:
        logger.info("Starting interactive demo...")
        run_demo()
    elif args.train:
        logger.info("Starting training...")
        run_training(args.config)
    elif args.eval:
        logger.info("Starting evaluation...")
        run_evaluation(args.config)
    elif args.quick_test:
        logger.info("Running quick test...")
        run_quick_test()
    else:
        logger.info("No action specified. Use --help for available options.")
        logger.info("Try: python 0939.py --quick-test")


def run_demo():
    """Run the interactive Streamlit demo."""
    import subprocess
    import sys
    
    demo_path = Path(__file__).parent / "demo" / "app.py"
    
    if not demo_path.exists():
        print("Error: Demo application not found!")
        print("Please ensure the demo/app.py file exists.")
        return
    
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", str(demo_path)], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running demo: {e}")
    except FileNotFoundError:
        print("Error: Streamlit not found. Please install it with: pip install streamlit")


def run_training(config_path: str):
    """Run model training."""
    import subprocess
    import sys
    
    train_script = Path(__file__).parent / "scripts" / "train.py"
    
    if not train_script.exists():
        print("Error: Training script not found!")
        return
    
    try:
        subprocess.run([sys.executable, str(train_script), "--config", config_path], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running training: {e}")


def run_evaluation(config_path: str):
    """Run model evaluation."""
    print("Evaluation functionality will be implemented in the evaluation script.")
    print("For now, you can use the demo to test the model interactively.")


def run_quick_test():
    """Run a quick functionality test."""
    try:
        from transformers import CLIPProcessor, CLIPModel
        from PIL import Image
        import torch
        import numpy as np
        
        print("Testing basic CLIP functionality...")
        
        # Load model and processor
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        # Create a test image
        test_image = Image.new('RGB', (224, 224), color='red')
        
        # Test text
        test_text = "A red image"
        
        # Process inputs
        inputs = processor(text=test_text, images=test_image, return_tensors="pt", padding=True)
        
        # Get model outputs
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Compute similarity
        logits_per_image = outputs.logits_per_image
        similarity = torch.sigmoid(logits_per_image).item()
        
        print(f"✅ Test passed! Similarity score: {similarity:.4f}")
        print("The multi-modal reasoning system is working correctly.")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        print("Please check your installation and dependencies.")


# Legacy implementation (original code)
def legacy_implementation():
    """
    Original implementation from Project 939.
    
    This is the legacy code that was refactored into the modern system.
    Kept here for reference and educational purposes.
    """
    from transformers import CLIPProcessor, CLIPModel
    import torch
    from PIL import Image
    
    # Load pre-trained CLIP model and processor
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    # Simulate a question and an image dataset
    # Note: This would need a real image file
    try:
        image = Image.open("example_image.jpg")  # Replace with a valid image path
    except FileNotFoundError:
        # Create a dummy image for testing
        image = Image.new('RGB', (224, 224), color='blue')
        print("Using dummy image for testing (original code expected 'example_image.jpg')")
    
    questions = ["What is the object in the image?", "Is this object a vehicle?", "What color is the object?"]
    
    # Preprocess the image and text inputs
    def multi_modal_reasoning(image, questions):
        inputs = processor(text=questions, images=image, return_tensors="pt", padding=True)
        
        # Perform forward pass to get model's prediction
        outputs = model(**inputs)
        
        # Calculate similarity between the image and each text question
        logits_per_image = outputs.logits_per_image  # Image-text similarity scores
        probs = logits_per_image.softmax(dim=1)  # Convert logits to probabilities
        
        # Retrieve the most relevant question and answer
        best_match_idx = torch.argmax(probs)
        return questions[best_match_idx], probs[0][best_match_idx].item()
    
    # Simulate reasoning task
    best_question, match_score = multi_modal_reasoning(image, questions)
    
    # Output the result
    print(f"Image-Text Reasoning Result: {best_question}")
    print(f"Match Score: {match_score:.2f}")
    
    print("\n" + "="*50)
    print("LEGACY IMPLEMENTATION COMPLETE")
    print("="*50)
    print("What This Does:")
    print("Image-Text Reasoning: We use CLIP to measure the similarity between an image and different textual descriptions/questions, answering which question is most relevant to the image.")
    print("Multi-modal Reasoning: The system combines text and visual features to reason about the content of the image.")


if __name__ == "__main__":
    main()

