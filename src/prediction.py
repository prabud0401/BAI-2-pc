"""
Prediction Module for Beverage Intelligence System

This module handles:
1. Loading trained models for inference
2. Single text prediction
3. Batch file prediction
4. Entity extraction and formatting
"""

import spacy
import pandas as pd
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np

from .config import Config


class Predictor:
    """Handles predictions using the trained NER model."""
    
    def __init__(self, model_name: str = 'ner_v1', verbose: bool = False):
        self.config = Config()
        self.model_name = model_name
        self.verbose = verbose
        self.setup_logging()
        
        self.model_path = self.config.get_model_path(model_name)
        self.nlp = None
        self.load_model()
        
    def setup_logging(self):
        """Set up logging configuration."""
        log_level = logging.DEBUG if self.verbose else logging.INFO
        logging.basicConfig(
            level=log_level,
            format=self.config.LOGGING_CONFIG['format']
        )
        self.logger = logging.getLogger(__name__)
    
    def load_model(self):
        """Load the trained spaCy model."""
        model_dir = self.model_path / 'model'
        
        if not model_dir.exists():
            raise FileNotFoundError(f"Trained model not found: {model_dir}")
        
        try:
            self.nlp = spacy.load(model_dir)
            self.logger.info(f"Model loaded successfully: {self.model_name}")
            
            # Load model metadata
            metadata_file = self.model_path / 'metadata.json'
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    self.metadata = json.load(f)
                    self.logger.info(f"Model metadata loaded")
            else:
                self.metadata = {}
                
        except Exception as e:
            self.logger.error(f"Failed to load model: {str(e)}")
            raise
    
    def predict_single(self, text: str) -> Dict[str, Any]:
        """
        Predict entities for a single text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dict with extracted entities and metadata
        """
        if not self.nlp:
            raise ValueError("Model not loaded")
        
        # Clean and prepare text
        cleaned_text = self._clean_text(text)
        
        # Run prediction
        doc = self.nlp(cleaned_text)
        
        # Extract entities
        entities = []
        for ent in doc.ents:
            entity_info = {
                'text': ent.text,
                'label': ent.label_,
                'start': ent.start_char,
                'end': ent.end_char,
                'confidence': self._get_entity_confidence(ent)
            }
            entities.append(entity_info)
        
        # Group entities by type
        grouped_entities = self._group_entities_by_type(entities)
        
        result = {
            'original_text': text,
            'cleaned_text': cleaned_text,
            'entities': entities,
            'grouped_entities': grouped_entities,
            'entity_count': len(entities),
            'model_name': self.model_name
        }
        
        return result
    
    def predict_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Predict entities for a batch of texts.
        
        Args:
            texts: List of input texts
            
        Returns:
            List of prediction results
        """
        if not self.nlp:
            raise ValueError("Model not loaded")
        
        self.logger.info(f"Processing batch of {len(texts)} texts...")
        
        results = []
        
        # Process in batches for efficiency
        batch_size = 32
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Clean texts
            cleaned_texts = [self._clean_text(text) for text in batch_texts]
            
            # Process batch
            docs = list(self.nlp.pipe(cleaned_texts))
            
            # Extract results
            for j, doc in enumerate(docs):
                original_text = batch_texts[j]
                cleaned_text = cleaned_texts[j]
                
                entities = []
                for ent in doc.ents:
                    entity_info = {
                        'text': ent.text,
                        'label': ent.label_,
                        'start': ent.start_char,
                        'end': ent.end_char,
                        'confidence': self._get_entity_confidence(ent)
                    }
                    entities.append(entity_info)
                
                grouped_entities = self._group_entities_by_type(entities)
                
                result = {
                    'original_text': original_text,
                    'cleaned_text': cleaned_text,
                    'entities': entities,
                    'grouped_entities': grouped_entities,
                    'entity_count': len(entities),
                    'model_name': self.model_name
                }
                results.append(result)
            
            if self.verbose and (i + batch_size) % 100 == 0:
                self.logger.debug(f"Processed {min(i + batch_size, len(texts))} texts...")
        
        self.logger.info(f"Batch processing completed")
        return results
    
    def predict_file(self, input_file: str, output_file: str, 
                    description_column: str = 'ProductDescription',
                    categorizer = None) -> Dict[str, Any]:
        """
        Predict entities for all products in a CSV file.
        
        Args:
            input_file: Path to input CSV file
            output_file: Path to save results
            description_column: Column containing product descriptions
            categorizer: Optional categorization engine for full category prediction
            
        Returns:
            Dict with processing statistics
        """
        self.logger.info(f"Processing file: {input_file}")
        
        # Load input file
        try:
            if input_file.endswith('.xlsx'):
                df = pd.read_excel(input_file)
            else:
                df = pd.read_csv(input_file)
        except Exception as e:
            self.logger.error(f"Failed to load input file: {str(e)}")
            raise
        
        if description_column not in df.columns:
            raise ValueError(f"Column '{description_column}' not found in input file")
        
        # Extract descriptions
        descriptions = df[description_column].astype(str).tolist()
        
        # Run predictions
        predictions = self.predict_batch(descriptions)
        
        # Create results dataframe
        results_data = []
        
        for i, (_, row) in enumerate(df.iterrows()):
            prediction = predictions[i]
            
            # Base result with original data
            result_row = row.to_dict()
            
            # Add prediction results
            result_row.update({
                'PredictedEntities': json.dumps(prediction['entities']),
                'EntityCount': prediction['entity_count'],
                'ModelName': prediction['model_name']
            })
            
            # Add individual entity columns
            for entity_type in self.config.NER_ENTITIES:
                entities = prediction['grouped_entities'].get(entity_type, [])
                if entities:
                    result_row[f'Predicted_{entity_type}'] = ', '.join([e['text'] for e in entities])
                    result_row[f'Predicted_{entity_type}_Confidence'] = np.mean([e['confidence'] for e in entities])
                else:
                    result_row[f'Predicted_{entity_type}'] = ''
                    result_row[f'Predicted_{entity_type}_Confidence'] = 0.0
            
            # Add full category if categorizer is provided
            if categorizer:
                try:
                    category_result = categorizer.categorize(prediction['grouped_entities'])
                    result_row['PredictedCategory'] = category_result['full_category']
                    result_row['CategoryConfidence'] = category_result['confidence']
                    result_row['CategoryReason'] = category_result.get('reason', '')
                except Exception as e:
                    self.logger.warning(f"Failed to categorize row {i}: {str(e)}")
                    result_row['PredictedCategory'] = 'Unknown'
                    result_row['CategoryConfidence'] = 0.0
                    result_row['CategoryReason'] = 'Categorization failed'
            
            results_data.append(result_row)
        
        # Save results
        results_df = pd.DataFrame(results_data)
        
        try:
            if output_file.endswith('.xlsx'):
                results_df.to_excel(output_file, index=False)
            else:
                results_df.to_csv(output_file, index=False)
            
            self.logger.info(f"Results saved to: {output_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to save results: {str(e)}")
            raise
        
        # Calculate statistics
        stats = {
            'total_predictions': len(predictions),
            'total_entities': sum(p['entity_count'] for p in predictions),
            'average_entities_per_product': np.mean([p['entity_count'] for p in predictions]),
            'entity_type_counts': {}
        }
        
        # Count entities by type
        for entity_type in self.config.NER_ENTITIES:
            count = sum(len(p['grouped_entities'].get(entity_type, [])) for p in predictions)
            stats['entity_type_counts'][entity_type] = count
        
        self.logger.info(f"Processing statistics:")
        self.logger.info(f"  - Total products: {stats['total_predictions']}")
        self.logger.info(f"  - Total entities: {stats['total_entities']}")
        self.logger.info(f"  - Average entities per product: {stats['average_entities_per_product']:.2f}")
        
        return stats
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize input text."""
        if pd.isna(text):
            return ""
        
        # Convert to string and strip whitespace
        cleaned = str(text).strip()
        
        # Remove excessive whitespace
        cleaned = " ".join(cleaned.split())
        
        return cleaned
    
    def _get_entity_confidence(self, ent) -> float:
        """
        Get confidence score for an entity.
        
        Args:
            ent: spaCy entity object
            
        Returns:
            Confidence score (0.0 to 1.0)
        """
        # If the model provides confidence scores, use them
        if hasattr(ent._, 'confidence'):
            return float(ent._.confidence)
        
        # Otherwise, return a default confidence based on entity length and type
        # This is a simple heuristic - longer entities tend to be more reliable
        text_length = len(ent.text)
        if text_length >= 3:
            return 0.9
        elif text_length >= 2:
            return 0.7
        else:
            return 0.5
    
    def _group_entities_by_type(self, entities: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Group entities by their type.
        
        Args:
            entities: List of entity dictionaries
            
        Returns:
            Dict mapping entity types to lists of entities
        """
        grouped = {}
        
        for entity in entities:
            entity_type = entity['label']
            if entity_type not in grouped:
                grouped[entity_type] = []
            grouped[entity_type].append(entity)
        
        return grouped
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        if not self.nlp:
            return {}
        
        info = {
            'model_name': self.model_name,
            'model_path': str(self.model_path),
            'entities': self.config.NER_ENTITIES,
            'components': self.nlp.component_names,
            'metadata': self.metadata
        }
        
        return info
    
    def validate_predictions(self, predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Validate prediction results and provide quality metrics.
        
        Args:
            predictions: List of prediction results
            
        Returns:
            Dict with validation metrics
        """
        if not predictions:
            return {'status': 'no_data', 'metrics': {}}
        
        total_texts = len(predictions)
        texts_with_entities = sum(1 for p in predictions if p['entity_count'] > 0)
        texts_without_entities = total_texts - texts_with_entities
        
        # Entity statistics
        entity_counts = {}
        confidence_scores = []
        
        for prediction in predictions:
            for entity in prediction['entities']:
                entity_type = entity['label']
                if entity_type not in entity_counts:
                    entity_counts[entity_type] = 0
                entity_counts[entity_type] += 1
                confidence_scores.append(entity['confidence'])
        
        # Calculate metrics
        metrics = {
            'total_texts': total_texts,
            'texts_with_entities': texts_with_entities,
            'texts_without_entities': texts_without_entities,
            'coverage_rate': texts_with_entities / total_texts if total_texts > 0 else 0,
            'average_confidence': np.mean(confidence_scores) if confidence_scores else 0,
            'entity_type_distribution': entity_counts,
            'total_entities': len(confidence_scores)
        }
        
        # Quality assessment
        if metrics['coverage_rate'] >= 0.8:
            status = 'good'
        elif metrics['coverage_rate'] >= 0.6:
            status = 'fair'
        else:
            status = 'poor'
        
        return {
            'status': status,
            'metrics': metrics
        } 