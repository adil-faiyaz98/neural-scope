"""
Hugging Face Model Source

This module provides utilities for fetching pre-trained models from Hugging Face.
"""

import os
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class HuggingFaceSource:
    """
    Source for Hugging Face models.
    """
    
    def __init__(self):
        """Initialize the Hugging Face source."""
        self.name = "huggingface"
        self.available_models = {
            # BERT models
            "bert-base-uncased": {"model_type": "bert", "model_class": "AutoModel"},
            "bert-large-uncased": {"model_type": "bert", "model_class": "AutoModel"},
            "bert-base-cased": {"model_type": "bert", "model_class": "AutoModel"},
            
            # GPT models
            "gpt2": {"model_type": "gpt2", "model_class": "AutoModelForCausalLM"},
            "gpt2-medium": {"model_type": "gpt2", "model_class": "AutoModelForCausalLM"},
            "gpt2-large": {"model_type": "gpt2", "model_class": "AutoModelForCausalLM"},
            
            # T5 models
            "t5-small": {"model_type": "t5", "model_class": "AutoModelForSeq2SeqLM"},
            "t5-base": {"model_type": "t5", "model_class": "AutoModelForSeq2SeqLM"},
            
            # RoBERTa models
            "roberta-base": {"model_type": "roberta", "model_class": "AutoModel"},
            "roberta-large": {"model_type": "roberta", "model_class": "AutoModel"},
            
            # DistilBERT models
            "distilbert-base-uncased": {"model_type": "distilbert", "model_class": "AutoModel"},
            
            # BART models
            "facebook/bart-base": {"model_type": "bart", "model_class": "AutoModelForSeq2SeqLM"},
            "facebook/bart-large": {"model_type": "bart", "model_class": "AutoModelForSeq2SeqLM"},
            
            # Vision models
            "google/vit-base-patch16-224": {"model_type": "vit", "model_class": "AutoModelForImageClassification"},
            "facebook/deit-base-patch16-224": {"model_type": "deit", "model_class": "AutoModelForImageClassification"}
        }
        
    def get_available_models(self):
        """
        Get a list of available models.
        
        Returns:
            List of available model names
        """
        return list(self.available_models.keys())
        
    def fetch_model(self, model_name, output_dir):
        """
        Fetch a pre-trained model from Hugging Face.
        
        Args:
            model_name: Name of the model to fetch
            output_dir: Directory to save the model
            
        Returns:
            Path to the saved model
        """
        try:
            from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoModelForImageClassification
        except ImportError:
            logger.error("Transformers library is required to fetch models from Hugging Face")
            raise ImportError("Transformers library is required to fetch models from Hugging Face")
            
        if model_name not in self.available_models:
            available_models = self.get_available_models()
            logger.error(f"Model {model_name} not found in Hugging Face. Available models: {', '.join(available_models)}")
            raise ValueError(f"Model {model_name} not found in Hugging Face")
            
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Get model info
        model_info = self.available_models[model_name]
        model_type = model_info["model_type"]
        model_class = model_info["model_class"]
        
        # Fetch model from Hugging Face
        logger.info(f"Fetching {model_name} from Hugging Face...")
        try:
            # Load the model and tokenizer
            if model_class == "AutoModel":
                model = AutoModel.from_pretrained(model_name)
                tokenizer = AutoTokenizer.from_pretrained(model_name)
            elif model_class == "AutoModelForCausalLM":
                model = AutoModelForCausalLM.from_pretrained(model_name)
                tokenizer = AutoTokenizer.from_pretrained(model_name)
            elif model_class == "AutoModelForSeq2SeqLM":
                model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
                tokenizer = AutoTokenizer.from_pretrained(model_name)
            elif model_class == "AutoModelForImageClassification":
                model = AutoModelForImageClassification.from_pretrained(model_name)
                tokenizer = AutoTokenizer.from_pretrained(model_name)
            else:
                model = AutoModel.from_pretrained(model_name)
                tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Create model directory
            model_dir = os.path.join(output_dir, model_name.replace("/", "_"))
            os.makedirs(model_dir, exist_ok=True)
            
            # Save the model and tokenizer
            model.save_pretrained(model_dir)
            tokenizer.save_pretrained(model_dir)
            
            # Save model info
            with open(os.path.join(model_dir, "model_info.json"), "w") as f:
                json.dump({
                    "model_name": model_name,
                    "model_type": model_type,
                    "model_class": model_class
                }, f, indent=2)
            
            logger.info(f"Model saved to {model_dir}")
            
            return model_dir
        except Exception as e:
            logger.error(f"Error fetching model from Hugging Face: {e}")
            raise
            
    def get_model_metadata(self, model_name):
        """
        Get metadata for a model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Dictionary with model metadata
        """
        metadata = {
            "bert-base-uncased": {
                "parameters": 110000000,
                "hidden_size": 768,
                "num_hidden_layers": 12,
                "num_attention_heads": 12,
                "size_mb": 420,
                "paper_url": "https://arxiv.org/abs/1810.04805",
                "description": "BERT base model (uncased) pre-trained on English text"
            },
            "gpt2": {
                "parameters": 124000000,
                "hidden_size": 768,
                "num_hidden_layers": 12,
                "num_attention_heads": 12,
                "size_mb": 510,
                "paper_url": "https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf",
                "description": "GPT-2 small model pre-trained on English text"
            },
            "t5-small": {
                "parameters": 60000000,
                "hidden_size": 512,
                "num_hidden_layers": 6,
                "num_attention_heads": 8,
                "size_mb": 230,
                "paper_url": "https://arxiv.org/abs/1910.10683",
                "description": "T5 small model pre-trained on C4 dataset"
            },
            "roberta-base": {
                "parameters": 125000000,
                "hidden_size": 768,
                "num_hidden_layers": 12,
                "num_attention_heads": 12,
                "size_mb": 480,
                "paper_url": "https://arxiv.org/abs/1907.11692",
                "description": "RoBERTa base model pre-trained on English text"
            }
        }
        
        return metadata.get(model_name, {})
