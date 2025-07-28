"""
Model Training Module for Beverage Intelligence System

This module handles:
1. spaCy NER model training with GPU support
2. Model evaluation and metrics
3. Model saving and loading
4. Training configuration and hyperparameters
"""

import spacy
from spacy.training.example import Example
from spacy.tokens import DocBin
import logging
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import random
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, classification_report

from .config import Config


class ModelTrainer:
    """Handles training and evaluation of the NER model."""
    
    def __init__(self, model_name: str = 'ner_v1', use_gpu: bool = False, verbose: bool = False):
        """
        Initialize the ModelTrainer.
        
        Args:
            model_name (str): Name for the current model run
            use_gpu (bool): Flag to attempt GPU training (defaults to False for CPU)
            verbose (bool): Enable verbose logging
        """
        self.config = Config()
        self.model_name = model_name
        self.use_gpu = False  # Force CPU usage
        self.verbose = verbose
        self.setup_logging()
        
        self.model_path = self.config.get_model_path(model_name)
        self.model_path.mkdir(parents=True, exist_ok=True)
        
        self.nlp = None
        self.training_metrics = []
        
        if self.verbose:
            self.logger.info("🔧 CPU training mode activated")
            
    def setup_logging(self):
        """Set up logging configuration."""
        log_level = logging.DEBUG if self.verbose else logging.INFO
        logging.basicConfig(
            level=log_level,
            format=self.config.LOGGING_CONFIG['format']
        )
        self.logger = logging.getLogger(__name__)
    
    def check_gpu_availability(self) -> bool:
        """
        Check if GPU is available for training.
        
        Returns:
            bool: True if GPU is available and working
        """
        if self.use_gpu:
            try:
                spacy.require_gpu()
                self.logger.info("✅ GPU activated successfully")
                return True
            except Exception as e:
                self.logger.warning(f"GPU not available or failed to activate: {e}")
                return False
        return False
    
    def setup_model(self, model_name: str):
        """Setup spaCy model for training with reliable configuration."""
        try:
            self.logger.info("Setting up spaCy model...")
            
            # Try to load existing model first
            try:
                nlp = spacy.load(model_name)
                self.logger.info(f"Loaded existing model: {model_name}")
                return nlp
            except OSError:
                pass
            
            # Create a new model with reliable configuration
            self.logger.info("Creating new spaCy model...")
            
            # Use a simpler, more reliable approach
            try:
                # Try with en_core_web_sm (smaller, more reliable)
                nlp = spacy.load("en_core_web_sm")
                self.logger.info("Using en_core_web_sm as base model")
            except OSError:
                try:
                    # Download en_core_web_sm if not available
                    self.logger.info("Downloading en_core_web_sm...")
                    spacy.cli.download("en_core_web_sm")
                    nlp = spacy.load("en_core_web_sm")
                except:
                    # Fallback to blank model
                    self.logger.info("Creating blank English model")
                    nlp = spacy.blank("en")
            
            # Add NER component if not present
            if "ner" not in nlp.pipe_names:
                nlp.add_pipe("ner")
                self.logger.info("Added NER component to model")
            
            # Get NER component
            ner = nlp.get_pipe("ner")
            
            # Add labels for beverage entities
            entities = [
                "BRAND", "TYPE", "SIZE", "PACK_COUNT", 
                "CONTAINER_TYPE", "ALCOHOL_CONTENT", "FLAVOR", "SPECIAL_TYPE"
            ]
            
            for entity in entities:
                ner.add_label(entity)
                self.logger.debug(f"Added label: {entity}")
            
            self.logger.info(f"Model setup complete with {len(entities)} entity types")
            return nlp
            
        except Exception as e:
            self.logger.error(f"Error setting up model: {e}")
            # Ultimate fallback - create a basic working model
            nlp = spacy.blank("en")
            nlp.add_pipe("ner")
            ner = nlp.get_pipe("ner")
            for entity in ["BRAND", "TYPE", "SIZE", "PACK_COUNT", "CONTAINER_TYPE"]:
                ner.add_label(entity)
            self.logger.info("Created fallback model")
            return nlp
    
    def load_training_data(self) -> Tuple[List[Example], List[Example], List[Example]]:
        """
        Load training, validation, and test data.
        
        Returns:
            Tuple of training, validation, and test examples
        """
        self.logger.info("Loading training data...")
        
        training_files = self.config.get_training_files()
        
        # Check if files exist
        for split_name, file_path in training_files.items():
            if not file_path.exists():
                raise FileNotFoundError(f"{split_name} data file not found: {file_path}")
        
        # Load data
        train_examples = self._load_spacy_data(training_files['train'])
        valid_examples = self._load_spacy_data(training_files['valid'])
        test_examples = self._load_spacy_data(training_files['test'])
        
        self.logger.info(f"Loaded training data:")
        self.logger.info(f"  - Training: {len(train_examples)} examples")
        self.logger.info(f"  - Validation: {len(valid_examples)} examples")
        self.logger.info(f"  - Test: {len(test_examples)} examples")
        
        return train_examples, valid_examples, test_examples
    
    def train(self, batch_size: int = 4, epochs: int = 10, learning_rate: float = 0.001):
        """
        Train the NER model with CPU-optimized settings.
        
        Args:
            batch_size (int): Smaller batch size for CPU (default: 4)
            epochs (int): Number of training epochs
            learning_rate (float): Learning rate for optimization
        """
        # Use config defaults if not provided
        epochs = epochs or self.config.SPACY_CONFIG['training']['max_epochs']
        batch_size = batch_size or self.config.SPACY_CONFIG['training']['batch_size']
        learning_rate = learning_rate or self.config.SPACY_CONFIG['training']['learn_rate']
        
        self.logger.info(f"Starting training with:")
        self.logger.info(f"  - Epochs: {epochs}")
        self.logger.info(f"  - Batch size: {batch_size}")
        self.logger.info(f"  - Learning rate: {learning_rate}")
        
        # Setup GPU if available
        if self.use_gpu:
            gpu_available = self.check_gpu_availability()
        else:
            gpu_available = False
        
        # Setup model
        self.nlp = self.setup_model(self.config.SPACY_CONFIG['model_name'])
        
        # Load training data
        train_examples, valid_examples, test_examples = self.load_training_data()
        
        # Configure training
        other_pipes = [pipe for pipe in self.nlp.pipe_names if pipe != 'ner']
        
        # Training loop
        self.logger.info("Starting training loop...")
        
        best_score = 0.0
        patience_counter = 0
        patience = self.config.SPACY_CONFIG['training']['patience']
        
        with self.nlp.disable_pipes(*other_pipes):
            # Initialize the model
            self.nlp.begin_training()
            
            for epoch in range(epochs):
                self.logger.info(f"Epoch {epoch + 1}/{epochs}")
                
                # Shuffle training data
                random.shuffle(train_examples)
                
                # Training batches
                losses = {}
                batches = spacy.util.minibatch(train_examples, size=batch_size)
                
                for batch in batches:
                    self.nlp.update(batch, losses=losses, drop=self.config.SPACY_CONFIG['training']['dropout'])
                
                # Validation
                if epoch % self.config.SPACY_CONFIG['training']['eval_frequency'] == 0 or epoch == epochs - 1:
                    val_metrics = self._evaluate_on_data(valid_examples)
                    
                    self.logger.info(f"Validation metrics:")
                    self.logger.info(f"  - Loss: {losses.get('ner', 0):.4f}")
                    self.logger.info(f"  - Precision: {val_metrics['precision']:.4f}")
                    self.logger.info(f"  - Recall: {val_metrics['recall']:.4f}")
                    self.logger.info(f"  - F1: {val_metrics['f1']:.4f}")
                    
                    # Save metrics
                    epoch_metrics = {
                        'epoch': epoch + 1,
                        'loss': losses.get('ner', 0),
                        'precision': val_metrics['precision'],
                        'recall': val_metrics['recall'],
                        'f1': val_metrics['f1']
                    }
                    self.training_metrics.append(epoch_metrics)
                    
                    # Early stopping check
                    if val_metrics['f1'] > best_score:
                        best_score = val_metrics['f1']
                        patience_counter = 0
                        # Save best model
                        self._save_model()
                        self.logger.info(f"New best model saved (F1: {best_score:.4f})")
                    else:
                        patience_counter += 1
                        if patience_counter >= patience:
                            self.logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                            break
        
        # Final evaluation on test set
        final_metrics = self._evaluate_on_data(test_examples)
        
        # Save training history
        self._save_training_history()
        
        # Save final model
        self._save_model()
        
        self.logger.info("Training completed!")
        return final_metrics
    
    def evaluate_model(self) -> Dict[str, Any]:
        """
        Evaluate the trained model on test data.
        
        Returns:
            Dict with evaluation metrics
        """
        self.logger.info("Evaluating trained model...")
        
        # Load trained model
        model_path = self.model_path / 'model'
        if not model_path.exists():
            raise FileNotFoundError(f"Trained model not found: {model_path}")
        
        self.nlp = spacy.load(model_path)
        
        # Load test data
        test_file = self.config.get_training_files()['test']
        test_examples = self._load_spacy_data(test_file)
        
        # Evaluate
        metrics = self._evaluate_on_data(test_examples, detailed=True)
        
        self.logger.info("Evaluation completed!")
        return metrics
    
    def _load_spacy_data(self, file_path: Path) -> List[Example]:
        """Load spaCy binary data and convert to Examples."""
        doc_bin = DocBin().from_disk(file_path)
        docs = list(doc_bin.get_docs(spacy.blank("en").vocab))
        
        examples = []
        for doc in docs:
            # Create example from doc
            predicted = spacy.blank("en").make_doc(doc.text)
            example = Example.from_dict(predicted, {"entities": [(ent.start_char, ent.end_char, ent.label_) for ent in doc.ents]})
            examples.append(example)
        
        return examples
    
    def _evaluate_on_data(self, examples: List[Example], detailed: bool = False) -> Dict[str, Any]:
        """
        Evaluate model on a set of examples.
        
        Args:
            examples: List of examples to evaluate
            detailed: Whether to include per-entity metrics
            
        Returns:
            Dict with evaluation metrics
        """
        true_labels = []
        pred_labels = []
        
        # Collect predictions
        for example in examples:
            # Get true entities
            true_ents = set()
            for ent in example.reference.ents:
                true_ents.add((ent.start_char, ent.end_char, ent.label_))
            
            # Get predicted entities
            pred_doc = self.nlp(example.reference.text)
            pred_ents = set()
            for ent in pred_doc.ents:
                pred_ents.add((ent.start_char, ent.end_char, ent.label_))
            
            # Convert to label sequences for sklearn metrics
            # This is a simplified approach - for exact entity matching
            for true_ent in true_ents:
                true_labels.append(true_ent[2])  # label
                # Check if this entity was predicted correctly
                if true_ent in pred_ents:
                    pred_labels.append(true_ent[2])
                else:
                    pred_labels.append('O')  # Wrong prediction
            
            # Add false positives
            for pred_ent in pred_ents:
                if pred_ent not in true_ents:
                    true_labels.append('O')
                    pred_labels.append(pred_ent[2])
        
        # Calculate metrics
        if true_labels and pred_labels:
            # Overall metrics
            precision, recall, f1, _ = precision_recall_fscore_support(
                true_labels, pred_labels, average='weighted', zero_division=0
            )
            
            metrics = {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'loss': 0.0  # Would need to calculate separately
            }
            
            # Detailed per-entity metrics
            if detailed:
                entity_metrics = {}
                for entity in self.config.NER_ENTITIES:
                    # Binary classification for this entity
                    true_binary = [1 if label == entity else 0 for label in true_labels]
                    pred_binary = [1 if label == entity else 0 for label in pred_labels]
                    
                    if sum(true_binary) > 0 or sum(pred_binary) > 0:
                        p, r, f, _ = precision_recall_fscore_support(
                            true_binary, pred_binary, average='binary', zero_division=0
                        )
                        entity_metrics[entity] = {
                            'precision': p,
                            'recall': r,
                            'f1': f
                        }
                
                metrics['entity_metrics'] = entity_metrics
        else:
            metrics = {
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0,
                'loss': 0.0
            }
        
        return metrics
    
    def _save_model(self):
        """Save the trained model."""
        model_dir = self.model_path / 'model'
        self.nlp.to_disk(model_dir)
        
        # Save metadata
        metadata = {
            'model_name': self.model_name,
            'entities': self.config.NER_ENTITIES,
            'base_model': self.config.SPACY_CONFIG['model_name'],
            'training_config': self.config.SPACY_CONFIG['training']
        }
        
        with open(self.model_path / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        self.logger.info(f"Model saved to: {model_dir}")
    
    def _save_training_history(self):
        """Save training metrics history."""
        history_file = self.model_path / 'training_history.json'
        
        with open(history_file, 'w') as f:
            json.dump(self.training_metrics, f, indent=2)
        
        self.logger.info(f"Training history saved to: {history_file}")
    
    def load_model(self, model_name: str = None) -> spacy.Language:
        """
        Load a trained model.
        
        Args:
            model_name: Name of the model to load (default: current model)
            
        Returns:
            Loaded spaCy model
        """
        if model_name:
            model_path = self.config.get_model_path(model_name)
        else:
            model_path = self.model_path
        
        model_dir = model_path / 'model'
        
        if not model_dir.exists():
            raise FileNotFoundError(f"Model not found: {model_dir}")
        
        self.nlp = spacy.load(model_dir)
        self.logger.info(f"Model loaded from: {model_dir}")
        
        return self.nlp
    
    def get_training_history(self) -> List[Dict]:
        """Get training metrics history."""
        history_file = self.model_path / 'training_history.json'
        
        if history_file.exists():
            with open(history_file, 'r') as f:
                return json.load(f)
        else:
            return []
    
    def predict_entities(self, text: str) -> List[Dict[str, Any]]:
        """
        Predict entities in a given text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            List of detected entities
        """
        if not self.nlp:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        doc = self.nlp(text)
        
        entities = []
        for ent in doc.ents:
            entities.append({
                'text': ent.text,
                'label': ent.label_,
                'start': ent.start_char,
                'end': ent.end_char,
                'confidence': ent._.confidence if hasattr(ent._, 'confidence') else 1.0
            })
        
        return entities 