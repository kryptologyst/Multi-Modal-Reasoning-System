"""Streamlit demo for multi-modal reasoning system."""

import streamlit as st
import torch
import numpy as np
from PIL import Image
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from transformers import CLIPProcessor
from src.models.reasoning_model import MultiModalReasoningModel
from src.utils.device import get_device, set_seed
from src.utils.config import load_config


# Page configuration
st.set_page_config(
    page_title="Multi-Modal Reasoning System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        color: #1f77b4;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .similarity-bar {
        background-color: #e0e0e0;
        border-radius: 10px;
        padding: 0.2rem;
        margin: 0.5rem 0;
    }
    .similarity-fill {
        background-color: #1f77b4;
        height: 20px;
        border-radius: 8px;
        transition: width 0.3s ease;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model_and_processor():
    """Load model and processor with caching."""
    # Set up device and reproducibility
    set_seed(42)
    device = get_device()
    
    # Load configuration
    try:
        config = load_config('configs/model/clip_config.yaml')
    except:
        # Fallback configuration
        config = {
            'model': {
                'name': 'openai/clip-vit-base-patch32',
                'freeze_vision': False,
                'freeze_text': False,
                'temperature': 0.07,
            }
        }
    
    # Load processor
    processor = CLIPProcessor.from_pretrained(config['model']['name'])
    
    # Load model
    model = MultiModalReasoningModel(
        model_name=config['model']['name'],
        freeze_vision=config['model'].get('freeze_vision', False),
        freeze_text=config['model'].get('freeze_text', False),
        temperature=config['model'].get('temperature', 0.07),
    ).to(device)
    
    model.eval()
    
    return model, processor, device


def compute_similarity(
    model: MultiModalReasoningModel,
    processor: CLIPProcessor,
    device: torch.device,
    image: Image.Image,
    text: str,
) -> Dict[str, float]:
    """Compute similarity between image and text.
    
    Args:
        model: Multi-modal reasoning model.
        processor: CLIP processor.
        device: Device for computation.
        image: Input image.
        text: Input text.
        
    Returns:
        Dictionary containing similarity metrics.
    """
    with torch.no_grad():
        # Process inputs
        inputs = processor(
            text=text,
            images=image,
            return_tensors="pt",
            padding=True,
        )
        
        # Move to device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Get model outputs
        outputs = model(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            pixel_values=inputs['pixel_values'],
        )
        
        # Extract similarities
        similarity_logits = outputs['similarity_logits'].cpu().numpy()[0]
        similarity_score = torch.sigmoid(torch.tensor(similarity_logits)).item()
        
        # Get embeddings
        vision_embeddings = outputs['vision_embeddings'].cpu().numpy()
        text_embeddings = outputs['text_embeddings'].cpu().numpy()
        
        # Compute cosine similarity
        cosine_sim = np.dot(vision_embeddings[0], text_embeddings[0]) / (
            np.linalg.norm(vision_embeddings[0]) * np.linalg.norm(text_embeddings[0])
        )
        
        return {
            'similarity_logits': similarity_logits,
            'similarity_score': similarity_score,
            'cosine_similarity': cosine_sim,
        }


def main():
    """Main demo application."""
    # Header
    st.markdown('<h1 class="main-header">🧠 Multi-Modal Reasoning System</h1>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    This demo showcases a multi-modal reasoning system that combines vision and language 
    understanding to perform various reasoning tasks including image-text matching, 
    question answering, and similarity analysis.
    """)
    
    # Load model
    with st.spinner("Loading model..."):
        model, processor, device = load_model_and_processor()
    
    st.success(f"Model loaded successfully on {device}")
    
    # Sidebar
    st.sidebar.header("Configuration")
    
    # Model info
    st.sidebar.subheader("Model Information")
    st.sidebar.info(f"""
    **Model**: CLIP ViT-Base-Patch32  
    **Device**: {device}  
    **Parameters**: ~151M  
    **Task**: Vision-Language Reasoning
    """)
    
    # Demo options
    st.sidebar.subheader("Demo Options")
    demo_mode = st.sidebar.selectbox(
        "Select Demo Mode",
        ["Image-Text Similarity", "Question Answering", "Batch Analysis"]
    )
    
    # Main content
    if demo_mode == "Image-Text Similarity":
        image_text_similarity_demo(model, processor, device)
    elif demo_mode == "Question Answering":
        question_answering_demo(model, processor, device)
    elif demo_mode == "Batch Analysis":
        batch_analysis_demo(model, processor, device)


def image_text_similarity_demo(model, processor, device):
    """Image-text similarity demo."""
    st.header("Image-Text Similarity Analysis")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Input")
        
        # Image upload
        uploaded_image = st.file_uploader(
            "Upload an image",
            type=['png', 'jpg', 'jpeg'],
            help="Upload an image to analyze"
        )
        
        if uploaded_image is not None:
            image = Image.open(uploaded_image).convert('RGB')
            st.image(image, caption="Uploaded Image", use_column_width=True)
        else:
            # Default image
            st.info("Please upload an image or use the default sample")
            image = Image.new('RGB', (224, 224), color='lightblue')
            st.image(image, caption="Default Image", use_column_width=True)
        
        # Text input
        text_input = st.text_area(
            "Enter text description",
            value="A beautiful landscape with mountains and trees",
            help="Enter a text description to compare with the image"
        )
        
        # Analyze button
        analyze_button = st.button("Analyze Similarity", type="primary")
    
    with col2:
        st.subheader("Results")
        
        if analyze_button:
            with st.spinner("Computing similarity..."):
                results = compute_similarity(model, processor, device, image, text_input)
            
            # Display results
            st.markdown("### Similarity Metrics")
            
            # Similarity score
            similarity_score = results['similarity_score']
            st.markdown(f"""
            <div class="metric-card">
                <h4>Similarity Score: {similarity_score:.4f}</h4>
                <div class="similarity-bar">
                    <div class="similarity-fill" style="width: {similarity_score * 100}%"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Cosine similarity
            cosine_sim = results['cosine_similarity']
            st.metric("Cosine Similarity", f"{cosine_sim:.4f}")
            
            # Interpretation
            if similarity_score > 0.7:
                st.success("High similarity - Image and text are well matched!")
            elif similarity_score > 0.4:
                st.warning("Moderate similarity - Some relationship exists")
            else:
                st.error("Low similarity - Image and text don't match well")
            
            # Raw logits
            with st.expander("Technical Details"):
                st.json({
                    "similarity_logits": float(results['similarity_logits']),
                    "similarity_score": similarity_score,
                    "cosine_similarity": cosine_sim,
                })


def question_answering_demo(model, processor, device):
    """Question answering demo."""
    st.header("Visual Question Answering")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Input")
        
        # Image upload
        uploaded_image = st.file_uploader(
            "Upload an image",
            type=['png', 'jpg', 'jpeg'],
            help="Upload an image to ask questions about"
        )
        
        if uploaded_image is not None:
            image = Image.open(uploaded_image).convert('RGB')
            st.image(image, caption="Uploaded Image", use_column_width=True)
        else:
            # Default image
            st.info("Please upload an image or use the default sample")
            image = Image.new('RGB', (224, 224), color='lightgreen')
            st.image(image, caption="Default Image", use_column_width=True)
        
        # Question input
        question = st.text_input(
            "Ask a question about the image",
            value="What do you see in this image?",
            help="Ask a question about the uploaded image"
        )
        
        # Predefined questions
        st.subheader("Sample Questions")
        sample_questions = [
            "What objects are in this image?",
            "What colors do you see?",
            "Is this an indoor or outdoor scene?",
            "How many objects are there?",
            "What is the main subject?",
        ]
        
        selected_question = st.selectbox("Or choose a sample question:", sample_questions)
        if selected_question:
            question = selected_question
        
        # Answer button
        answer_button = st.button("Get Answer", type="primary")
    
    with col2:
        st.subheader("Answer")
        
        if answer_button:
            with st.spinner("Processing question..."):
                results = compute_similarity(model, processor, device, image, question)
            
            # Display answer
            similarity_score = results['similarity_score']
            
            if similarity_score > 0.6:
                st.success("✅ The model is confident about this answer")
            elif similarity_score > 0.3:
                st.warning("⚠️ The model has moderate confidence")
            else:
                st.error("❌ The model is uncertain about this answer")
            
            # Confidence score
            st.metric("Confidence Score", f"{similarity_score:.4f}")
            
            # Answer interpretation
            st.markdown("### Answer Analysis")
            st.info(f"""
            **Question**: {question}  
            **Confidence**: {similarity_score:.4f}  
            **Interpretation**: The model's confidence in answering this question about the image.
            """)


def batch_analysis_demo(model, processor, device):
    """Batch analysis demo."""
    st.header("Batch Analysis")
    
    st.markdown("""
    Upload multiple images and text descriptions to perform batch analysis.
    This is useful for comparing multiple image-text pairs simultaneously.
    """)
    
    # File upload
    uploaded_files = st.file_uploader(
        "Upload multiple images",
        type=['png', 'jpg', 'jpeg'],
        accept_multiple_files=True,
        help="Upload multiple images for batch analysis"
    )
    
    # Text descriptions
    st.subheader("Text Descriptions")
    text_descriptions = st.text_area(
        "Enter text descriptions (one per line)",
        value="A red car\nA blue house\nA green tree\nA yellow flower",
        help="Enter text descriptions, one per line"
    ).split('\n')
    
    # Filter empty descriptions
    text_descriptions = [desc.strip() for desc in text_descriptions if desc.strip()]
    
    if uploaded_files and text_descriptions:
        st.subheader("Analysis Results")
        
        # Create pairs
        pairs = []
        for i, image_file in enumerate(uploaded_files):
            for j, text_desc in enumerate(text_descriptions):
                pairs.append((i, j, image_file, text_desc))
        
        if st.button("Analyze All Pairs", type="primary"):
            # Process all pairs
            results = []
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for idx, (img_idx, text_idx, image_file, text_desc) in enumerate(pairs):
                status_text.text(f"Processing pair {idx + 1}/{len(pairs)}")
                
                # Load image
                image = Image.open(image_file).convert('RGB')
                
                # Compute similarity
                similarity_results = compute_similarity(model, processor, device, image, text_desc)
                
                results.append({
                    'image_idx': img_idx,
                    'text_idx': text_idx,
                    'image_name': image_file.name,
                    'text_description': text_desc,
                    'similarity_score': similarity_results['similarity_score'],
                    'cosine_similarity': similarity_results['cosine_similarity'],
                })
                
                progress_bar.progress((idx + 1) / len(pairs))
            
            status_text.text("Analysis complete!")
            
            # Display results
            st.subheader("Results Table")
            
            # Create results DataFrame
            import pandas as pd
            df = pd.DataFrame(results)
            
            # Sort by similarity score
            df = df.sort_values('similarity_score', ascending=False)
            
            # Display table
            st.dataframe(df, use_container_width=True)
            
            # Summary statistics
            st.subheader("Summary Statistics")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Average Similarity", f"{df['similarity_score'].mean():.4f}")
            
            with col2:
                st.metric("Max Similarity", f"{df['similarity_score'].max():.4f}")
            
            with col3:
                st.metric("Min Similarity", f"{df['similarity_score'].min():.4f}")
            
            # Best matches
            st.subheader("Best Matches")
            best_matches = df.head(3)
            
            for idx, row in best_matches.iterrows():
                st.markdown(f"""
                **{row['image_name']}** ↔ **{row['text_description']}**  
                Similarity: {row['similarity_score']:.4f}
                """)


if __name__ == "__main__":
    main()
